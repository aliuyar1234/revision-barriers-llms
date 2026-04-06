from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.baselines import (
    bootstrap_metric_delta,
    evaluate_logistic_predictions,
    refit_logistic_on_train_dev,
    select_logistic_by_dev,
)
from src.analysis.plots import save_commitment_calibration_plot, save_probe_layer_plot
from src.analysis.probes import (
    add_revisability_labels,
    build_latent_output_margin_control,
    calibration_by_decile,
    choose_d0,
    commitment_separation_by_cell,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, np.ndarray):
        return [sanitize(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sanitize(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: sanitize(row.get(name)) for name in fieldnames})


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(sanitize(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for feature_index, feature_name in enumerate(feature_names):
            matrix[row_index, feature_index] = float(row[feature_name])
    return matrix


def build_split_indices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    split_map: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
    for index, row in enumerate(rows):
        split_map.setdefault(row["split"], []).append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in split_map.items()}


def run_baseline(
    rows: list[dict[str, Any]],
    split_indices: dict[str, np.ndarray],
    *,
    feature_names: list[str],
    c_grid: list[float],
) -> dict[str, Any]:
    x_values = feature_matrix(rows, feature_names)
    y_values = np.asarray([row["z"] for row in rows], dtype=np.int64)

    train_index = split_indices["train"]
    dev_index = split_indices["dev"]
    test_index = split_indices["test"]

    selection = select_logistic_by_dev(
        x_values[train_index],
        y_values[train_index],
        x_values[dev_index],
        y_values[dev_index],
        c_grid=c_grid,
    )
    final_model, final_scaler = refit_logistic_on_train_dev(
        x_values[np.concatenate([train_index, dev_index])],
        y_values[np.concatenate([train_index, dev_index])],
        selection["selected_C"],
    )
    test_metrics = evaluate_logistic_predictions(final_model, final_scaler, x_values[test_index], y_values[test_index])
    all_scores = evaluate_logistic_predictions(final_model, final_scaler, x_values, y_values)["scores"]

    return {
        "feature_names": feature_names,
        "selected_C": selection["selected_C"],
        "dev_auroc": selection["selected_dev_auroc"],
        "dev_auprc": selection["selected_dev_auprc"],
        "candidates": selection["candidates"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "test_scores": test_metrics["scores"],
        "all_scores": all_scores,
        "model": final_model,
        "scaler": final_scaler,
    }


def choose_best_layer(layer_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    best_auroc = max(metric["dev_auroc"] for metric in layer_metrics)
    near_best = [metric for metric in layer_metrics if best_auroc - metric["dev_auroc"] <= 0.01]
    near_best.sort(key=lambda metric: (metric["layer"], metric["selected_C"]))
    return near_best[0]


def run_label_shuffle_control(
    train_states: np.ndarray,
    train_labels: np.ndarray,
    dev_states: np.ndarray,
    dev_labels: np.ndarray,
    *,
    selected_c: float,
    n_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    from src.analysis.baselines import fit_logistic_model, fit_standard_scaler, logistic_scores, safe_auroc

    scaler = fit_standard_scaler(train_states)
    train_scaled = scaler.transform(train_states)
    dev_scaled = scaler.transform(dev_states)

    shuffle_aurocs: list[float] = []
    for _ in range(n_shuffles):
        shuffled_labels = np.asarray(train_labels, copy=True)
        rng.shuffle(shuffled_labels)
        model = fit_logistic_model(train_scaled, shuffled_labels, selected_c)
        dev_scores = logistic_scores(model, dev_scaled)
        shuffle_aurocs.append(safe_auroc(dev_labels, dev_scores))

    mean_auroc = float(np.nanmean(shuffle_aurocs)) if shuffle_aurocs else float("nan")
    pass_band = 0.45 <= mean_auroc <= 0.55 if not math.isnan(mean_auroc) else False
    high_count = sum(1 for value in shuffle_aurocs if value <= 0.60)
    return {
        "n_shuffles": n_shuffles,
        "mean_dev_auroc": mean_auroc,
        "dev_aurocs": shuffle_aurocs,
        "passes_mean_band": pass_band,
        "n_at_or_below_0_60": high_count,
        "passes_count_rule": high_count >= max(0, n_shuffles - 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M4 probe and baseline models.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)

    state_manifest = read_json(ROOT / config["inputs"]["state_manifest_json"])
    state_rows = read_jsonl(ROOT / config["inputs"]["state_rows_jsonl"])
    state_array = np.load(ROOT / state_manifest["state_array_npy"], mmap_mode="r")

    if state_array.shape[0] != len(state_rows):
        raise ValueError("State array row count does not match state_rows_jsonl.")

    rows_with_index = [{**row, "state_index": index} for index, row in enumerate(state_rows)]
    valid_main_rows = [row for row in rows_with_index if row["valid_main"]]
    split_counts = {split: sum(1 for row in valid_main_rows if row["split"] == split) for split in {"train", "dev", "test"}}
    if min(split_counts.values()) == 0:
        raise ValueError(
            "M4 probe training requires valid_main rows in train/dev/test. "
            "Provide an all-splits Qwen behavior row set before training probes."
        )

    dev_rows = [row for row in valid_main_rows if row["split"] == "dev"]
    d0_selection = choose_d0(dev_rows)
    labeled_rows = add_revisability_labels(rows_with_index, d0_selection.d0)
    valid_labeled_rows = [row for row in labeled_rows if row["valid_main"]]
    split_indices = build_split_indices(valid_labeled_rows)

    c_grid = [float(value) for value in config["run"].get("c_grid", [0.01, 0.1, 1.0, 10.0])]
    bootstrap_resamples = int(config["run"].get("bootstrap_resamples", 1000))
    label_shuffle_count = int(config["run"].get("label_shuffle_count", 20))
    min_examples_per_history_type = int(config["run"].get("output_margin_min_per_history_type", 25))

    baseline_specs = {
        "m": ["m"],
        "o": ["o"],
        "m_o": ["m", "o"],
    }
    baseline_results = {
        name: run_baseline(valid_labeled_rows, split_indices, feature_names=feature_names, c_grid=c_grid)
        for name, feature_names in baseline_specs.items()
    }

    train_index = split_indices["train"]
    dev_index = split_indices["dev"]
    test_index = split_indices["test"]
    y_values = np.asarray([row["z"] for row in valid_labeled_rows], dtype=np.int64)

    layer_metrics: list[dict[str, Any]] = []
    selected_models: dict[int, dict[str, Any]] = {}
    for layer in range(state_array.shape[1]):
        layer_states = np.asarray(state_array[[row["state_index"] for row in valid_labeled_rows], layer, :], dtype=np.float32)
        selection = select_logistic_by_dev(
            layer_states[train_index],
            y_values[train_index],
            layer_states[dev_index],
            y_values[dev_index],
            c_grid=c_grid,
        )
        selected_models[layer] = selection
        layer_metrics.append(
            {
                "layer": layer,
                "selected_C": selection["selected_C"],
                "dev_auroc": selection["selected_dev_auroc"],
                "dev_auprc": selection["selected_dev_auprc"],
            }
        )

    selected_layer = choose_best_layer(layer_metrics)
    selected_layer_index = selected_layer["layer"]
    selected_layer_c = float(selected_layer["selected_C"])
    selected_layer_states = np.asarray(state_array[[row["state_index"] for row in valid_labeled_rows], selected_layer_index, :], dtype=np.float32)

    shuffle_control = run_label_shuffle_control(
        selected_layer_states[train_index],
        y_values[train_index],
        selected_layer_states[dev_index],
        y_values[dev_index],
        selected_c=selected_layer_c,
        n_shuffles=label_shuffle_count,
        seed=int(config["run"].get("label_shuffle_seed", 0)),
    )

    final_probe_model, final_probe_scaler = refit_logistic_on_train_dev(
        selected_layer_states[np.concatenate([train_index, dev_index])],
        y_values[np.concatenate([train_index, dev_index])],
        selected_layer_c,
    )
    final_probe_eval = evaluate_logistic_predictions(
        final_probe_model,
        final_probe_scaler,
        selected_layer_states[test_index],
        y_values[test_index],
    )
    all_probe_scores = evaluate_logistic_predictions(
        final_probe_model,
        final_probe_scaler,
        selected_layer_states,
        y_values,
    )["scores"]

    baseline_delta_vs_m = bootstrap_metric_delta(
        y_values[test_index],
        final_probe_eval["scores"],
        baseline_results["m"]["test_scores"],
        n_resamples=bootstrap_resamples,
        seed=int(config["run"].get("bootstrap_seed", 0)),
    )
    baseline_delta_vs_o = bootstrap_metric_delta(
        y_values[test_index],
        final_probe_eval["scores"],
        baseline_results["o"]["test_scores"],
        n_resamples=bootstrap_resamples,
        seed=int(config["run"].get("bootstrap_seed", 0)) + 1,
    )
    baseline_delta_vs_m_o = bootstrap_metric_delta(
        y_values[test_index],
        final_probe_eval["scores"],
        baseline_results["m_o"]["test_scores"],
        n_resamples=bootstrap_resamples,
        seed=int(config["run"].get("bootstrap_seed", 0)) + 2,
    )

    all_valid_rows_with_scores: list[dict[str, Any]] = []
    valid_indices = [row["state_index"] for row in valid_labeled_rows]
    baseline_score_maps = {
        "baseline_m_score": baseline_results["m"]["all_scores"],
        "baseline_o_score": baseline_results["o"]["all_scores"],
        "baseline_m_o_score": baseline_results["m_o"]["all_scores"],
    }
    for position, row in enumerate(valid_labeled_rows):
        all_valid_rows_with_scores.append(
            {
                **row,
                "probe_score": float(all_probe_scores[position]),
                "baseline_m_score": float(baseline_score_maps["baseline_m_score"][position]),
                "baseline_o_score": float(baseline_score_maps["baseline_o_score"][position]),
                "baseline_m_o_score": float(baseline_score_maps["baseline_m_o_score"][position]),
            }
        )

    score_rows: list[dict[str, Any]] = []
    valid_by_example_id = {row["example_id"]: row for row in all_valid_rows_with_scores}
    for row in labeled_rows:
        enriched = valid_by_example_id.get(row["example_id"])
        if enriched is None:
            score_rows.append({**row, "probe_score": None, "baseline_m_score": None, "baseline_o_score": None, "baseline_m_o_score": None})
        else:
            score_rows.append(enriched)

    test_valid_rows = [row for row in all_valid_rows_with_scores if row["split"] == "test"]
    output_margin_control = build_latent_output_margin_control(
        [row for row in all_valid_rows_with_scores if row["split"] == "dev"],
        test_valid_rows,
        probe_score_key="probe_score",
        combined_baseline_score_key="baseline_m_o_score",
        min_examples_per_history_type=min_examples_per_history_type,
    )
    calibration_rows = calibration_by_decile(test_valid_rows, score_key="probe_score")
    separation_rows = commitment_separation_by_cell(
        test_valid_rows,
        score_key="probe_score",
        min_examples_per_history_type=min_examples_per_history_type,
    )

    weights_standardized = final_probe_model.coef_[0]
    scaler_std = final_probe_scaler.std
    scaler_mean = final_probe_scaler.mean
    weights_model_space = weights_standardized / scaler_std
    bias_standardized = float(final_probe_model.intercept_[0])
    bias_model_space = float(bias_standardized - np.dot(weights_standardized, scaler_mean / scaler_std))
    norm = float(np.linalg.norm(weights_model_space))
    weights_model_space_unit_norm = weights_model_space / norm if norm > 0 else weights_model_space

    layer_metric_rows = [
        {
            "layer": row["layer"],
            "C": row["selected_C"],
            "dev_auroc": row["dev_auroc"],
            "dev_auprc": row["dev_auprc"],
            "selected": row["layer"] == selected_layer_index,
        }
        for row in layer_metrics
    ]
    write_csv(
        ROOT / config["outputs"]["layer_metrics_csv"],
        layer_metric_rows,
        ["layer", "C", "dev_auroc", "dev_auprc", "selected"],
    )

    baseline_metrics_payload = {
        "config": str(config_path),
        "run_id": config["run"]["run_id"],
        "d0_selection": {
            "d0": d0_selection.d0,
            "dev_positive_rate": d0_selection.dev_positive_rate,
            "candidate_rates": d0_selection.candidate_rates,
            "used_fallback": d0_selection.used_fallback,
        },
        "split_counts_valid_main": split_counts,
        "baselines": {
            name: {
                "feature_names": result["feature_names"],
                "selected_C": result["selected_C"],
                "dev_auroc": result["dev_auroc"],
                "dev_auprc": result["dev_auprc"],
                "test_auroc": result["test_auroc"],
                "test_auprc": result["test_auprc"],
                "candidates": result["candidates"],
            }
            for name, result in baseline_results.items()
        },
        "output_margin_control": output_margin_control,
        "commitment_separation_by_cell": separation_rows,
        "calibration_rows": calibration_rows,
    }
    write_json(ROOT / config["outputs"]["baseline_metrics_json"], baseline_metrics_payload)

    selected_probe_payload = {
        "run_id": config["run"]["run_id"],
        "config": str(config_path),
        "d0": d0_selection.d0,
        "selected_layer": selected_layer_index,
        "selected_C": selected_layer_c,
        "train_positive_rate": float(np.mean(y_values[train_index])),
        "dev_positive_rate": float(np.mean(y_values[dev_index])),
        "test_positive_rate": float(np.mean(y_values[test_index])),
        "scaler_mean": scaler_mean.tolist(),
        "scaler_std": scaler_std.tolist(),
        "weights_standardized": weights_standardized.tolist(),
        "weights_model_space": weights_model_space.tolist(),
        "weights_model_space_unit_norm": weights_model_space_unit_norm.tolist(),
        "bias_standardized": bias_standardized,
        "bias_model_space": bias_model_space,
        "test_auroc": final_probe_eval["auroc"],
        "test_auprc": final_probe_eval["auprc"],
        "delta_auroc_vs_m": baseline_delta_vs_m,
        "delta_auroc_vs_o": baseline_delta_vs_o,
        "delta_auroc_vs_m_o": baseline_delta_vs_m_o,
        "label_shuffle_control": shuffle_control,
        "output_margin_control": output_margin_control,
        "commitment_separation_by_cell": separation_rows,
        "calibration_rows": calibration_rows,
        "compatibility_fingerprint": sha256_json(
            {
                "run": config["run"],
                "state_manifest": state_manifest,
                "selected_layer": selected_layer_index,
                "selected_C": selected_layer_c,
                "d0": d0_selection.d0,
            }
        ),
    }
    write_json(ROOT / config["outputs"]["selected_probe_json"], selected_probe_payload)
    write_jsonl(ROOT / config["outputs"]["score_rows_jsonl"], score_rows)

    save_probe_layer_plot(
        layer_metric_rows,
        ROOT / config["outputs"]["layer_plot_png"],
        "M4 dev AUROC by layer",
    )
    save_commitment_calibration_plot(
        calibration_rows,
        ROOT / config["outputs"]["calibration_plot_png"],
        "M4 commitment score calibration",
    )

    print(
        json.dumps(
            sanitize(
                {
                    "config": str(config_path),
                    "run_id": config["run"]["run_id"],
                    "d0": d0_selection.d0,
                    "selected_layer": selected_layer_index,
                    "selected_C": selected_layer_c,
                    "test_auroc": final_probe_eval["auroc"],
                    "delta_auroc_vs_o": baseline_delta_vs_o,
                    "delta_auroc_vs_m_o": baseline_delta_vs_m_o,
                    "output_margin_control_scheme": output_margin_control["chosen_scheme"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
