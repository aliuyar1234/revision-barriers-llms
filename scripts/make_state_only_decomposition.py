from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.barriers import DOSES, normalize_barrier_numeric
from src.analysis.bootstrap import confidence_interval


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: sanitize(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [sanitize(inner) for inner in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: sanitize(row.get(name)) for name in fieldnames})


def mean_flipability(barrier_value: Any) -> float:
    barrier = normalize_barrier_numeric(barrier_value)
    return sum(1.0 if barrier <= dose else 0.0 for dose in DOSES) / len(DOSES)


@dataclass
class IsotonicModel:
    x_min: float
    x_max: float
    thresholds: list[float]
    values: list[float]

    def predict(self, x: float) -> float:
        clipped = min(max(x, self.x_min), self.x_max)
        for threshold, value in zip(self.thresholds, self.values, strict=True):
            if clipped <= threshold:
                return value
        return self.values[-1]


def fit_isotonic_decreasing(xs: list[float], ys: list[float]) -> IsotonicModel:
    if not xs:
        raise ValueError("Cannot fit isotonic regression with no data.")
    ordered = sorted(zip(xs, ys), key=lambda item: item[0])
    blocks = [{"sum": y, "count": 1, "start": x, "end": x} for x, y in ordered]

    index = 0
    while index < len(blocks) - 1:
        mean_here = blocks[index]["sum"] / blocks[index]["count"]
        mean_next = blocks[index + 1]["sum"] / blocks[index + 1]["count"]
        if mean_here < mean_next:
            merged = {
                "sum": blocks[index]["sum"] + blocks[index + 1]["sum"],
                "count": blocks[index]["count"] + blocks[index + 1]["count"],
                "start": blocks[index]["start"],
                "end": blocks[index + 1]["end"],
            }
            blocks[index : index + 2] = [merged]
            index = max(index - 1, 0)
        else:
            index += 1

    thresholds = [block["end"] for block in blocks]
    values = [block["sum"] / block["count"] for block in blocks]
    return IsotonicModel(
        x_min=ordered[0][0],
        x_max=ordered[-1][0],
        thresholds=thresholds,
        values=values,
    )


def bootstrap_three_deltas(rows: list[dict[str, Any]], *, n_resamples: int, seed: int) -> dict[str, Any]:
    if not rows:
        blank = {"mean": float("nan"), "ci": {"low": float("nan"), "high": float("nan")}, "n_pairs": 0}
        return {"actual": blank, "predicted": blank, "residual": blank}

    def summarize(sample_rows: list[dict[str, Any]], key: str) -> float:
        return sum(row[key] for row in sample_rows) / len(sample_rows)

    rng = random.Random(seed)
    estimates = {"actual": [], "predicted": [], "residual": []}
    for _ in range(n_resamples):
        sample = [rng.choice(rows) for _ in range(len(rows))]
        estimates["actual"].append(summarize(sample, "actual_delta"))
        estimates["predicted"].append(summarize(sample, "predicted_delta"))
        estimates["residual"].append(summarize(sample, "residual_delta"))

    summary: dict[str, Any] = {}
    for label, key in [("actual", "actual_delta"), ("predicted", "predicted_delta"), ("residual", "residual_delta")]:
        summary[label] = {
            "mean": summarize(rows, key),
            "ci": confidence_interval(estimates[label]),
            "n_pairs": len(rows),
        }
    return summary


def build_examples(row_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in row_rows:
        examples.append(
            {
                "example_id": row["example_id"],
                "pair_id": row["pair_id"],
                "split": row["split"],
                "history_type": row["history_type"],
                "m": row["m"],
                "k": row["k"],
                "valid_main": bool(row["valid_main"]),
                "o_prefix": float(row["o"]),
                "mean_flipability": mean_flipability(row["B_raw_numeric"]),
            }
        )
    return examples


def build_test_pairs(
    pair_rows: list[dict[str, Any]],
    predicted_examples: dict[str, dict[str, Any]],
    *,
    eval_split: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in pair_rows:
        if row["split"] != eval_split or not row["valid_pair_main"]:
            continue
        fresh = predicted_examples[row["fresh_row_id"]]
        committed = predicted_examples[row["committed_row_id"]]
        output.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "m": row["m"],
                "k": row["k"],
                "in_S_lead_pair": bool(row["in_S_lead_pair"]),
                "actual_delta": fresh["mean_flipability"] - committed["mean_flipability"],
                "predicted_delta": fresh["predicted_mean_flipability"] - committed["predicted_mean_flipability"],
                "residual_delta": (fresh["mean_flipability"] - committed["mean_flipability"])
                - (fresh["predicted_mean_flipability"] - committed["predicted_mean_flipability"]),
            }
        )
    return output


def summarize_subset(rows: list[dict[str, Any]], *, n_resamples: int, seed: int) -> dict[str, Any]:
    boot = bootstrap_three_deltas(rows, n_resamples=n_resamples, seed=seed)
    return {
        "pair_count": len(rows),
        "actual": boot["actual"],
        "predicted": boot["predicted"],
        "residual": boot["residual"],
    }


def save_plot(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    panel_a = [row for row in summary_rows if row["subset"] in {"overall", "S_lead"}]
    x_positions = list(range(len(panel_a)))
    width = 0.22
    offsets = {"actual": -width, "predicted": 0.0, "residual": width}
    colors = {"actual": "tab:blue", "predicted": "tab:orange", "residual": "tab:green"}

    for key in ["actual", "predicted", "residual"]:
        means = [row[f"{key}_mean"] for row in panel_a]
        lowers = [row[f"{key}_mean"] - row[f"{key}_ci_low"] for row in panel_a]
        uppers = [row[f"{key}_ci_high"] - row[f"{key}_mean"] for row in panel_a]
        shifted = [x + offsets[key] for x in x_positions]
        axes[0].bar(shifted, means, width=width, color=colors[key], alpha=0.85, label=key)
        axes[0].errorbar(shifted, means, yerr=[lowers, uppers], fmt="none", ecolor="black", capsize=4)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x_positions, [row["subset"] for row in panel_a])
    axes[0].set_ylabel("paired mean flipability gap")
    axes[0].set_title("Actual vs state-only predicted paired gap")
    axes[0].legend()

    panel_b = [row for row in summary_rows if row["subset"] in {"k=0", "k>0", "k=1", "k=2"}]
    x_positions_b = list(range(len(panel_b)))
    means_b = [row["residual_mean"] for row in panel_b]
    lowers_b = [row["residual_mean"] - row["residual_ci_low"] for row in panel_b]
    uppers_b = [row["residual_ci_high"] - row["residual_mean"] for row in panel_b]
    axes[1].bar(x_positions_b, means_b, color="tab:green", alpha=0.85)
    axes[1].errorbar(x_positions_b, means_b, yerr=[lowers_b, uppers_b], fmt="none", ecolor="black", capsize=4)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x_positions_b, [row["subset"] for row in panel_b])
    axes[1].set_ylabel("residual paired gap")
    axes[1].set_title("Residual history effect by subset")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    config_path = Path(sys.argv[sys.argv.index("--config") + 1]) if "--config" in sys.argv else None
    if config_path is None:
        raise SystemExit("Usage: python scripts/make_state_only_decomposition.py --config <path>")
    if not config_path.is_absolute():
        candidate = ROOT / config_path
        if candidate.exists():
            config_path = candidate
    config = load_yaml(config_path)

    row_rows = read_jsonl(ROOT / config["inputs"]["row_jsonl"])
    pair_rows = read_jsonl(ROOT / config["inputs"]["pair_jsonl"])
    run_id = config["run"]["run_id"]
    fit_split = str(config["run"]["fit_split"])
    eval_split = str(config["run"]["eval_split"])
    n_resamples = int(config["run"]["bootstrap_resamples"])
    seed = int(config["run"]["bootstrap_seed"])

    examples = build_examples(row_rows)
    train_examples = [row for row in examples if row["split"] == fit_split and row["valid_main"]]
    models = {}
    train_ranges = {}
    for m in sorted({row["m"] for row in train_examples}):
        subset = [row for row in train_examples if row["m"] == m]
        xs = [row["o_prefix"] for row in subset]
        ys = [row["mean_flipability"] for row in subset]
        models[m] = fit_isotonic_decreasing(xs, ys)
        train_ranges[m] = {"min_o_prefix": min(xs), "max_o_prefix": max(xs), "n_examples": len(subset)}

    predicted_examples: dict[str, dict[str, Any]] = {}
    for row in examples:
        if row["valid_main"]:
            model = models[row["m"]]
            row = {**row, "predicted_mean_flipability": model.predict(row["o_prefix"])}
        predicted_examples[row["example_id"]] = row

    test_pairs = build_test_pairs(pair_rows, predicted_examples, eval_split=eval_split)

    subsets = {
        "overall": test_pairs,
        "S_lead": [row for row in test_pairs if row["in_S_lead_pair"]],
        "k=0": [row for row in test_pairs if row["k"] == 0],
        "k>0": [row for row in test_pairs if row["k"] > 0],
        "k=1": [row for row in test_pairs if row["k"] == 1],
        "k=2": [row for row in test_pairs if row["k"] == 2],
    }

    summary = {
        "run_id": run_id,
        "source_artifacts": {
            "row_jsonl": config["inputs"]["row_jsonl"],
            "pair_jsonl": config["inputs"]["pair_jsonl"],
        },
        "target_definition": "Reuses the M7 mean flipability semantics: per-example mean over doses of 1[B_raw <= d].",
        "fit_split": fit_split,
        "eval_split": eval_split,
        "feature_set": ["m", "o_prefix"],
        "model_class": "isotonic regression of mean flipability on o_prefix, fitted separately by m with monotone decreasing constraint",
        "train_ranges": train_ranges,
        "subsets": {},
    }

    table_rows = []
    ordered_subsets = ["overall", "S_lead", "k=0", "k>0", "k=1", "k=2"]
    for index, name in enumerate(ordered_subsets):
        subset_summary = summarize_subset(subsets[name], n_resamples=n_resamples, seed=seed + index)
        summary["subsets"][name] = subset_summary
        table_rows.append(
            {
                "subset": name,
                "pair_count": subset_summary["pair_count"],
                "actual_mean": subset_summary["actual"]["mean"],
                "actual_ci_low": subset_summary["actual"]["ci"]["low"],
                "actual_ci_high": subset_summary["actual"]["ci"]["high"],
                "predicted_mean": subset_summary["predicted"]["mean"],
                "predicted_ci_low": subset_summary["predicted"]["ci"]["low"],
                "predicted_ci_high": subset_summary["predicted"]["ci"]["high"],
                "residual_mean": subset_summary["residual"]["mean"],
                "residual_ci_low": subset_summary["residual"]["ci"]["low"],
                "residual_ci_high": subset_summary["residual"]["ci"]["high"],
            }
        )

    write_json(ROOT / config["outputs"]["summary_json"], summary)
    write_csv(
        ROOT / config["outputs"]["summary_table_csv"],
        table_rows,
        [
            "subset",
            "pair_count",
            "actual_mean",
            "actual_ci_low",
            "actual_ci_high",
            "predicted_mean",
            "predicted_ci_low",
            "predicted_ci_high",
            "residual_mean",
            "residual_ci_low",
            "residual_ci_high",
        ],
    )
    save_plot(table_rows, ROOT / config["outputs"]["summary_plot_png"])
    print(json.dumps(sanitize(summary), indent=2))


if __name__ == "__main__":
    main()
