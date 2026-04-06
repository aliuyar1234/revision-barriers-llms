from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.barriers import build_valid_main, clip_barrier, compute_barrier
from src.analysis.intervention_eval import (
    assess_c4_result,
    bootstrap_delta_b,
    build_s_move_manifest,
    create_random_control_directions,
    passes_side_effect_thresholds,
    select_alpha,
    summarize_random_control_ratios,
    summarize_side_effects,
)
from src.data.mhst_prompts import (
    MHST_CANONICAL_ANSWERS,
    render_mhst_full_prompt,
    render_mhst_intervention_prefix,
    render_mhst_prefix_prompt,
)
from src.data.mhst_worlds import OPTION_LABELS
from src.models.hooks import ResidualIntervention, load_qwen_residual_intervention_scorer


class PauseRequested(RuntimeError):
    pass


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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_with_tempfile(path: Path, text_writer: Any, *, mode: str = "w", newline: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open(mode, encoding="utf-8", newline=newline) as handle:
        text_writer(handle)
    os.replace(temp_path, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def _writer(handle: Any) -> None:
        json.dump(sanitize(payload), handle, indent=2)

    write_with_tempfile(path, _writer)


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize(row), ensure_ascii=False) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def write_text(path: Path, text: str) -> None:
    def _writer(handle: Any) -> None:
        handle.write(text)

    write_with_tempfile(path, _writer)


def append_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def emit_progress(log_path: Path, message: str) -> None:
    stamped = f"[{utc_now_iso()}] {message}"
    print(stamped, flush=True)
    append_log_line(log_path, stamped)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(sanitize(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_hashes() -> dict[str, str]:
    relevant = [
        "scripts/run_intervention.py",
        "src/models/hooks.py",
        "src/analysis/intervention_eval.py",
        "src/data/mhst_prompts.py",
        "src/analysis/barriers.py",
        "src/models/option_scoring.py",
    ]
    return {path: sha256_file(ROOT / path) for path in relevant}


def alpha_tag(alpha: float) -> str:
    return f"{alpha:+.2f}".replace(".", "p")


def build_artifacts(outputs: dict[str, str]) -> dict[str, Path]:
    return {name: ROOT / path for name, path in outputs.items()}


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {row["result_id"]: row for row in rows}


def ensure_stop_not_requested(stop_flag_path: Path) -> None:
    if stop_flag_path.exists():
        raise PauseRequested(
            f"Stop flag detected at {stop_flag_path}; remove it and rerun the same command to resume."
        )


def build_runtime_state(
    *,
    run_id: str,
    status: str,
    phase: str,
    compatibility_fingerprint: dict[str, Any],
    completed_rows: int,
    pause_reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "phase": phase,
        "updated_at": utc_now_iso(),
        "completed_rows": completed_rows,
        "pause_reason": pause_reason,
        "error": error,
        "compatibility_fingerprint": compatibility_fingerprint,
    }


def ensure_json_matches(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = read_json(path)
        if existing != sanitize(payload):
            raise ValueError(f"Existing frozen artifact disagrees with deterministic rebuild: {path}")
        return existing
    write_json(path, payload)
    return sanitize(payload)


def ensure_random_directions(
    *,
    array_path: Path,
    manifest_path: Path,
    seed: int,
    n_directions: int,
    dimension: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected = create_random_control_directions(seed=seed, n_directions=n_directions, dimension=dimension)
    if array_path.exists():
        existing = np.load(array_path)
        if existing.shape != expected.shape or not np.allclose(existing, expected, atol=1e-7):
            raise ValueError("Existing frozen random-control directions do not match the deterministic seed-0 build.")
    else:
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(array_path, expected)

    manifest = {
        "seed": seed,
        "n_directions": n_directions,
        "dimension": dimension,
        "array_path": str(array_path.relative_to(ROOT)),
        "sha256": sha256_file(array_path),
    }
    ensure_json_matches(manifest_path, manifest)
    return expected, manifest


def option_scores_to_probability_map(score_map: dict[str, float]) -> dict[str, float]:
    score_array = np.asarray([score_map[label] for label in OPTION_LABELS], dtype=np.float64)
    score_array = score_array - np.max(score_array)
    probs = np.exp(score_array)
    probs = probs / np.sum(probs)
    return {label: float(prob) for label, prob in zip(OPTION_LABELS, probs.tolist(), strict=True)}


def entropy_from_probability_map(probability_map: dict[str, float]) -> float:
    probs = np.asarray(list(probability_map.values()), dtype=np.float64)
    return float(-np.sum(probs * np.log(probs)))


def torch_from_numpy(vector: np.ndarray) -> Any:
    import torch

    return torch.tensor(vector, dtype=torch.float32)


def score_behavior_example(
    scorer: Any,
    record: dict[str, Any],
    *,
    layer_index: int,
    t_star_token_index: int,
    direction: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    intervention = None
    if alpha != 0.0:
        intervention = ResidualIntervention(
            layer_index=layer_index,
            token_index=t_star_token_index,
            direction=torch_from_numpy(direction),
            alpha=alpha,
        )

    option_scores_by_dose: dict[str, dict[str, float]] = {}
    for dose in range(7):
        prompt = render_mhst_full_prompt(record, dose)
        scores = scorer.score_options(prompt, MHST_CANONICAL_ANSWERS, intervention=intervention)
        option_scores_by_dose[str(dose)] = {
            option_label: float(score) for option_label, score in zip(OPTION_LABELS, scores, strict=True)
        }

    barrier_numeric = compute_barrier(option_scores_by_dose, record["incumbent_option"], record["challenger_option"])
    barrier_raw = "inf" if math.isinf(barrier_numeric) else int(barrier_numeric)
    distractor_options = [record["suspect_to_option"][suspect_id] for suspect_id in record["distractors"]]
    valid_main, distractor_top_any, invalid_doses = build_valid_main(option_scores_by_dose, barrier_numeric, distractor_options)
    dose_zero_scores = option_scores_by_dose["0"]
    probability_map = option_scores_to_probability_map(dose_zero_scores)
    predicted_label = max(probability_map, key=probability_map.get)

    return {
        "option_scores_by_dose": option_scores_by_dose,
        "B_raw": barrier_raw,
        "B_raw_numeric": barrier_numeric,
        "B_cap": clip_barrier(barrier_numeric),
        "valid_main_alpha": valid_main,
        "distractor_top_any_alpha": distractor_top_any,
        "invalid_doses_alpha": invalid_doses,
        "label": predicted_label,
        "entropy": entropy_from_probability_map(probability_map),
    }


def score_nonconflict_example(
    scorer: Any,
    record: dict[str, Any],
    *,
    layer_index: int,
    direction: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    t_star_token_index = scorer.prefix_token_index(render_mhst_intervention_prefix(record))
    if t_star_token_index is None:
        raise ValueError(f"Could not compute t* for sanity example {record['example_id']}.")
    intervention = None
    if alpha != 0.0:
        intervention = ResidualIntervention(
            layer_index=layer_index,
            token_index=t_star_token_index,
            direction=torch_from_numpy(direction),
            alpha=alpha,
        )
    prompt = render_mhst_prefix_prompt(record)
    scored = scorer.score_options_with_metrics(prompt, MHST_CANONICAL_ANSWERS, intervention=intervention)
    label = scored["predicted_label"]
    return {
        "t_star_token_index": t_star_token_index,
        "label": label,
        "entropy": scored["entropy"],
        "is_correct": label == record["expected_option"],
    }


def evaluate_rows(
    *,
    scorer: Any,
    existing_rows: dict[str, dict[str, Any]],
    rows_path: Path,
    run_state_path: Path,
    heartbeat_path: Path,
    log_path: Path,
    stop_flag_path: Path,
    run_id: str,
    compatibility_fingerprint: dict[str, Any],
    examples: list[dict[str, Any]],
    evaluation_group: str,
    direction_tag: str,
    alpha: float,
    layer_index: int,
    direction: np.ndarray,
    pair_records_by_example_id: dict[str, dict[str, Any]] | None = None,
    frozen_baseline_by_example_id: dict[str, dict[str, Any]] | None = None,
    sanity_records_by_example_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for example in examples:
        ensure_stop_not_requested(stop_flag_path)
        result_id = f"{evaluation_group}__{direction_tag}__{alpha_tag(alpha)}__{example['example_id']}"
        existing = existing_rows.get(result_id)
        if existing is not None:
            out_rows.append(existing)
            continue

        if evaluation_group == "nonconflict":
            record = sanity_records_by_example_id[example["example_id"]]
            scored = score_nonconflict_example(
                scorer,
                record,
                layer_index=layer_index,
                direction=direction,
                alpha=alpha,
            )
            row = {
                "result_id": result_id,
                "run_id": run_id,
                "evaluation_group": evaluation_group,
                "example_id": record["example_id"],
                "pair_id": record["pair_id"],
                "split": record["split"],
                "history_type": record["history_type"],
                "in_S_move": False,
                "layer": layer_index,
                "t_star_token_index": scored["t_star_token_index"],
                "direction_tag": direction_tag,
                "alpha": alpha,
                "B_raw": None,
                "B_cap": None,
                "delta_B_cap_vs_zero": None,
                "label": scored["label"],
                "entropy": scored["entropy"],
                "expected_label": record["expected_option"],
                "is_correct": scored["is_correct"],
            }
        else:
            record = pair_records_by_example_id[example["example_id"]]
            scored = score_behavior_example(
                scorer,
                record,
                layer_index=layer_index,
                t_star_token_index=example["t_star_token_index"],
                direction=direction,
                alpha=alpha,
            )
            baseline = frozen_baseline_by_example_id[record["example_id"]]
            row = {
                "result_id": result_id,
                "run_id": run_id,
                "evaluation_group": evaluation_group,
                "example_id": record["example_id"],
                "pair_id": record["pair_id"],
                "split": record["split"],
                "history_type": record["history_type"],
                "m": record["m"],
                "k": record["k"],
                "in_S_move": True,
                "layer": layer_index,
                "t_star_token_index": example["t_star_token_index"],
                "direction_tag": direction_tag,
                "alpha": alpha,
                "B_raw": scored["B_raw"],
                "B_cap": scored["B_cap"],
                "delta_B_cap_vs_zero": scored["B_cap"] - int(baseline["B_cap"]),
                "label": scored["label"],
                "entropy": scored["entropy"],
                "valid_main_alpha": scored["valid_main_alpha"],
                "invalid_doses_alpha": scored["invalid_doses_alpha"],
                "baseline_B_raw": baseline["B_raw"],
                "baseline_B_cap": baseline["B_cap"],
                "baseline_valid_main": baseline["valid_main"],
                "qc_matches_frozen": (
                    alpha != 0.0
                    or (
                        scored["B_raw"] == baseline["B_raw"]
                        and scored["B_cap"] == baseline["B_cap"]
                        and scored["valid_main_alpha"] == baseline["valid_main"]
                    )
                ),
            }

        append_jsonl_row(rows_path, row)
        existing_rows[result_id] = row
        out_rows.append(row)

        state = build_runtime_state(
            run_id=run_id,
            status="running",
            phase=evaluation_group,
            compatibility_fingerprint=compatibility_fingerprint,
            completed_rows=len(existing_rows),
        )
        write_json(run_state_path, state)
        write_json(
            heartbeat_path,
            {
                "updated_at": utc_now_iso(),
                "phase": evaluation_group,
                "completed_rows": len(existing_rows),
                "last_result_id": result_id,
            },
        )
    emit_progress(log_path, f"Completed {evaluation_group} {direction_tag} alpha={alpha:+.2f} on {len(out_rows)} rows.")
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M5 intervention workflow.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)
    artifacts = build_artifacts(config["outputs"])

    write_text(artifacts["config_snapshot_yaml"], config_path.read_text(encoding="utf-8"))

    pair_records = read_jsonl(ROOT / config["inputs"]["pair_jsonl"])
    pair_records_by_example_id = {row["example_id"]: row for row in pair_records}
    sanity_records = read_jsonl(ROOT / config["inputs"]["sanity_jsonl"])
    sanity_records_by_example_id = {row["example_id"]: row for row in sanity_records}
    frozen_behavior_rows = read_jsonl(ROOT / config["inputs"]["behavior_rows_jsonl"])
    frozen_pairs = read_jsonl(ROOT / config["inputs"]["behavior_pairs_jsonl"])
    state_rows = read_jsonl(ROOT / config["inputs"]["state_rows_jsonl"])
    selected_probe = read_json(ROOT / config["inputs"]["selected_probe_json"])

    s_move_dev_manifest = build_s_move_manifest(frozen_behavior_rows, frozen_pairs, state_rows, split="dev")
    s_move_test_manifest = build_s_move_manifest(frozen_behavior_rows, frozen_pairs, state_rows, split="test")
    s_move_dev_manifest = ensure_json_matches(artifacts["s_move_dev_json"], s_move_dev_manifest)
    s_move_test_manifest = ensure_json_matches(artifacts["s_move_test_json"], s_move_test_manifest)

    random_dirs, random_manifest = ensure_random_directions(
        array_path=artifacts["random_directions_npy"],
        manifest_path=artifacts["random_directions_json"],
        seed=int(config["run"].get("random_control_seed", 0)),
        n_directions=int(config["run"].get("random_control_count", 5)),
        dimension=len(selected_probe["weights_model_space_unit_norm"]),
    )

    scorer = load_qwen_residual_intervention_scorer(config["model"])
    compatibility_fingerprint = {
        "selected_probe_sha256": sha256_file(ROOT / config["inputs"]["selected_probe_json"]),
        "s_move_dev_sha256": sha256_file(artifacts["s_move_dev_json"]),
        "s_move_test_sha256": sha256_file(artifacts["s_move_test_json"]),
        "random_control_sha256": random_manifest["sha256"],
        "model_revision": scorer.model_revision,
        "tokenizer_revision": scorer.tokenizer_revision,
        "alpha_grid": config["run"]["alpha_grid"],
        "control_set_sha256": sha256_file(ROOT / config["inputs"]["sanity_jsonl"]),
        "scoring_prompt_config_sha256": sha256_json(
            {
                "model": config["model"],
                "inputs": config["inputs"],
                "run": config["run"],
            }
        ),
        "source_file_sha256": source_hashes(),
    }

    existing_state = read_json(artifacts["run_state_json"]) if artifacts["run_state_json"].exists() else None
    if existing_state is not None and existing_state.get("compatibility_fingerprint") != compatibility_fingerprint:
        raise ValueError("M5 resume refused because the stored compatibility fingerprint does not match the current inputs.")

    existing_rows = load_existing_rows(artifacts["rows_jsonl"])
    frozen_baseline_by_example_id = {row["example_id"]: row for row in frozen_behavior_rows}
    run_id = config["run"]["run_id"]
    layer_index = int(selected_probe["selected_layer"])
    commit_direction = np.asarray(selected_probe["weights_model_space_unit_norm"], dtype=np.float32)
    alpha_grid = [float(value) for value in config["run"]["alpha_grid"]]
    control_examples = [{"example_id": row["example_id"]} for row in sanity_records]

    try:
        ensure_stop_not_requested(artifacts["stop_flag_path"])
        write_json(
            artifacts["run_state_json"],
            build_runtime_state(
                run_id=run_id,
                status="running",
                phase="starting",
                compatibility_fingerprint=compatibility_fingerprint,
                completed_rows=len(existing_rows),
            ),
        )

        alpha0_qc_examples = s_move_dev_manifest["examples"] + s_move_test_manifest["examples"]
        alpha0_qc_rows = evaluate_rows(
            scorer=scorer,
            existing_rows=existing_rows,
            rows_path=artifacts["rows_jsonl"],
            run_state_path=artifacts["run_state_json"],
            heartbeat_path=artifacts["heartbeat_json"],
            log_path=artifacts["progress_log_txt"],
            stop_flag_path=artifacts["stop_flag_path"],
            run_id=run_id,
            compatibility_fingerprint=compatibility_fingerprint,
            examples=alpha0_qc_examples,
            evaluation_group="alpha0_qc",
            direction_tag="commit",
            alpha=0.0,
            layer_index=layer_index,
            direction=commit_direction,
            pair_records_by_example_id=pair_records_by_example_id,
            frozen_baseline_by_example_id=frozen_baseline_by_example_id,
        )
        mismatches = [row["example_id"] for row in alpha0_qc_rows if not row["qc_matches_frozen"]]
        if mismatches:
            raise ValueError(
                "Alpha=0 QC disagreed with frozen baseline artifacts for these examples: "
                + ", ".join(mismatches[:5])
            )

        control_alpha0_rows = evaluate_rows(
            scorer=scorer,
            existing_rows=existing_rows,
            rows_path=artifacts["rows_jsonl"],
            run_state_path=artifacts["run_state_json"],
            heartbeat_path=artifacts["heartbeat_json"],
            log_path=artifacts["progress_log_txt"],
            stop_flag_path=artifacts["stop_flag_path"],
            run_id=run_id,
            compatibility_fingerprint=compatibility_fingerprint,
            examples=control_examples,
            evaluation_group="nonconflict",
            direction_tag="commit",
            alpha=0.0,
            layer_index=layer_index,
            direction=commit_direction,
            sanity_records_by_example_id=sanity_records_by_example_id,
        )

        candidate_rows: list[dict[str, Any]] = []
        for a in alpha_grid:
            dev_neg_rows = evaluate_rows(
                scorer=scorer,
                existing_rows=existing_rows,
                rows_path=artifacts["rows_jsonl"],
                run_state_path=artifacts["run_state_json"],
                heartbeat_path=artifacts["heartbeat_json"],
                log_path=artifacts["progress_log_txt"],
                stop_flag_path=artifacts["stop_flag_path"],
                run_id=run_id,
                compatibility_fingerprint=compatibility_fingerprint,
                examples=s_move_dev_manifest["examples"],
                evaluation_group="s_move_dev",
                direction_tag="commit",
                alpha=-a,
                layer_index=layer_index,
                direction=commit_direction,
                pair_records_by_example_id=pair_records_by_example_id,
                frozen_baseline_by_example_id=frozen_baseline_by_example_id,
            )
            dev_pos_rows = evaluate_rows(
                scorer=scorer,
                existing_rows=existing_rows,
                rows_path=artifacts["rows_jsonl"],
                run_state_path=artifacts["run_state_json"],
                heartbeat_path=artifacts["heartbeat_json"],
                log_path=artifacts["progress_log_txt"],
                stop_flag_path=artifacts["stop_flag_path"],
                run_id=run_id,
                compatibility_fingerprint=compatibility_fingerprint,
                examples=s_move_dev_manifest["examples"],
                evaluation_group="s_move_dev",
                direction_tag="commit",
                alpha=+a,
                layer_index=layer_index,
                direction=commit_direction,
                pair_records_by_example_id=pair_records_by_example_id,
                frozen_baseline_by_example_id=frozen_baseline_by_example_id,
            )
            control_neg_rows = evaluate_rows(
                scorer=scorer,
                existing_rows=existing_rows,
                rows_path=artifacts["rows_jsonl"],
                run_state_path=artifacts["run_state_json"],
                heartbeat_path=artifacts["heartbeat_json"],
                log_path=artifacts["progress_log_txt"],
                stop_flag_path=artifacts["stop_flag_path"],
                run_id=run_id,
                compatibility_fingerprint=compatibility_fingerprint,
                examples=control_examples,
                evaluation_group="nonconflict",
                direction_tag="commit",
                alpha=-a,
                layer_index=layer_index,
                direction=commit_direction,
                sanity_records_by_example_id=sanity_records_by_example_id,
            )
            control_pos_rows = evaluate_rows(
                scorer=scorer,
                existing_rows=existing_rows,
                rows_path=artifacts["rows_jsonl"],
                run_state_path=artifacts["run_state_json"],
                heartbeat_path=artifacts["heartbeat_json"],
                log_path=artifacts["progress_log_txt"],
                stop_flag_path=artifacts["stop_flag_path"],
                run_id=run_id,
                compatibility_fingerprint=compatibility_fingerprint,
                examples=control_examples,
                evaluation_group="nonconflict",
                direction_tag="commit",
                alpha=+a,
                layer_index=layer_index,
                direction=commit_direction,
                sanity_records_by_example_id=sanity_records_by_example_id,
            )

            neg_mean = sum(float(row["delta_B_cap_vs_zero"]) for row in dev_neg_rows) / len(dev_neg_rows)
            pos_mean = sum(float(row["delta_B_cap_vs_zero"]) for row in dev_pos_rows) / len(dev_pos_rows)
            neg_sidefx = summarize_side_effects(control_alpha0_rows, control_neg_rows)
            pos_sidefx = summarize_side_effects(control_alpha0_rows, control_pos_rows)
            direction_ok = neg_mean < 0 and pos_mean > 0
            sidefx_ok = passes_side_effect_thresholds(neg_sidefx) and passes_side_effect_thresholds(pos_sidefx)
            candidate_rows.append(
                {
                    "alpha": a,
                    "neg_delta": neg_mean,
                    "pos_delta": pos_mean,
                    "neg_side_effect_summary": neg_sidefx,
                    "pos_side_effect_summary": pos_sidefx,
                    "direction_ok": direction_ok,
                    "side_effect_ok": sidefx_ok,
                    "qualifies": direction_ok and sidefx_ok,
                    "total_abs_shift": abs(neg_mean) + abs(pos_mean),
                }
            )

        alpha_selection = select_alpha(candidate_rows)
        selected_alpha = alpha_selection.selected_alpha
        underpowered = s_move_dev_manifest["count"] < 25 or s_move_test_manifest["count"] < 25
        commit_neg_summary = None
        commit_pos_summary = None
        random_ratio_summary = None
        chosen_side_effect_neg = None
        chosen_side_effect_pos = None
        random_rows_by_direction: dict[str, dict[str, list[dict[str, Any]]]] = {}

        if selected_alpha is not None and not underpowered:
            commit_neg_rows = evaluate_rows(
                scorer=scorer,
                existing_rows=existing_rows,
                rows_path=artifacts["rows_jsonl"],
                run_state_path=artifacts["run_state_json"],
                heartbeat_path=artifacts["heartbeat_json"],
                log_path=artifacts["progress_log_txt"],
                stop_flag_path=artifacts["stop_flag_path"],
                run_id=run_id,
                compatibility_fingerprint=compatibility_fingerprint,
                examples=s_move_test_manifest["examples"],
                evaluation_group="s_move_test",
                direction_tag="commit",
                alpha=-selected_alpha,
                layer_index=layer_index,
                direction=commit_direction,
                pair_records_by_example_id=pair_records_by_example_id,
                frozen_baseline_by_example_id=frozen_baseline_by_example_id,
            )
            commit_pos_rows = evaluate_rows(
                scorer=scorer,
                existing_rows=existing_rows,
                rows_path=artifacts["rows_jsonl"],
                run_state_path=artifacts["run_state_json"],
                heartbeat_path=artifacts["heartbeat_json"],
                log_path=artifacts["progress_log_txt"],
                stop_flag_path=artifacts["stop_flag_path"],
                run_id=run_id,
                compatibility_fingerprint=compatibility_fingerprint,
                examples=s_move_test_manifest["examples"],
                evaluation_group="s_move_test",
                direction_tag="commit",
                alpha=+selected_alpha,
                layer_index=layer_index,
                direction=commit_direction,
                pair_records_by_example_id=pair_records_by_example_id,
                frozen_baseline_by_example_id=frozen_baseline_by_example_id,
            )
            commit_neg_summary = bootstrap_delta_b(
                commit_neg_rows,
                n_resamples=int(config["run"].get("bootstrap_resamples", 1000)),
                seed=int(config["run"].get("bootstrap_seed", 0)),
            )
            commit_pos_summary = bootstrap_delta_b(
                commit_pos_rows,
                n_resamples=int(config["run"].get("bootstrap_resamples", 1000)),
                seed=int(config["run"].get("bootstrap_seed", 0)) + 1,
            )

            for direction_index, vector in enumerate(random_dirs):
                direction_tag = f"rand_{direction_index:02d}"
                neg_rows = evaluate_rows(
                    scorer=scorer,
                    existing_rows=existing_rows,
                    rows_path=artifacts["rows_jsonl"],
                    run_state_path=artifacts["run_state_json"],
                    heartbeat_path=artifacts["heartbeat_json"],
                    log_path=artifacts["progress_log_txt"],
                    stop_flag_path=artifacts["stop_flag_path"],
                    run_id=run_id,
                    compatibility_fingerprint=compatibility_fingerprint,
                    examples=s_move_test_manifest["examples"],
                    evaluation_group="s_move_test",
                    direction_tag=direction_tag,
                    alpha=-selected_alpha,
                    layer_index=layer_index,
                    direction=vector,
                    pair_records_by_example_id=pair_records_by_example_id,
                    frozen_baseline_by_example_id=frozen_baseline_by_example_id,
                )
                pos_rows = evaluate_rows(
                    scorer=scorer,
                    existing_rows=existing_rows,
                    rows_path=artifacts["rows_jsonl"],
                    run_state_path=artifacts["run_state_json"],
                    heartbeat_path=artifacts["heartbeat_json"],
                    log_path=artifacts["progress_log_txt"],
                    stop_flag_path=artifacts["stop_flag_path"],
                    run_id=run_id,
                    compatibility_fingerprint=compatibility_fingerprint,
                    examples=s_move_test_manifest["examples"],
                    evaluation_group="s_move_test",
                    direction_tag=direction_tag,
                    alpha=+selected_alpha,
                    layer_index=layer_index,
                    direction=vector,
                    pair_records_by_example_id=pair_records_by_example_id,
                    frozen_baseline_by_example_id=frozen_baseline_by_example_id,
                )
                random_rows_by_direction[direction_tag] = {"neg": neg_rows, "pos": pos_rows}

            random_ratio_summary = summarize_random_control_ratios(
                random_rows_by_direction,
                commit_neg_rows,
                commit_pos_rows,
            )
            chosen_candidate = next(row for row in candidate_rows if row["alpha"] == selected_alpha)
            chosen_side_effect_neg = chosen_candidate["neg_side_effect_summary"]
            chosen_side_effect_pos = chosen_candidate["pos_side_effect_summary"]

        c4_assessment = assess_c4_result(
            alpha_selected=selected_alpha,
            commit_neg_summary=commit_neg_summary,
            commit_pos_summary=commit_pos_summary,
            random_ratio_summary=random_ratio_summary,
            side_effect_neg=chosen_side_effect_neg,
            side_effect_pos=chosen_side_effect_pos,
            underpowered=underpowered,
        )

        summary_payload = {
            "run_id": run_id,
            "config": str(config_path),
            "selected_layer": layer_index,
            "selected_alpha": selected_alpha,
            "direction_source": config["inputs"]["selected_probe_json"],
            "S_move_dev_count": s_move_dev_manifest["count"],
            "S_move_test_count": s_move_test_manifest["count"],
            "delta_B_move_neg_alpha": None if commit_neg_summary is None else commit_neg_summary["mean"],
            "delta_B_move_pos_alpha": None if commit_pos_summary is None else commit_pos_summary["mean"],
            "delta_B_move_neg_alpha_ci": None if commit_neg_summary is None else commit_neg_summary["ci"],
            "delta_B_move_pos_alpha_ci": None if commit_pos_summary is None else commit_pos_summary["ci"],
            "random_control_summary": random_ratio_summary,
            "side_effect_summary": {
                "neg_alpha": chosen_side_effect_neg,
                "pos_alpha": chosen_side_effect_pos,
            },
            "underpowered": underpowered,
            "c4_assessment": c4_assessment,
            "compatibility_fingerprint": compatibility_fingerprint,
        }
        controls_payload = {
            "run_id": run_id,
            "alpha_selection_candidates": candidate_rows,
            "selected_alpha": selected_alpha,
            "qualified_candidates": alpha_selection.qualified_candidates,
            "random_control_count": len(random_dirs),
            "random_control_artifact": str(artifacts["random_directions_npy"].relative_to(ROOT)),
            "max_abs_random_shift_ratio_neg_alpha": None
            if random_ratio_summary is None
            else random_ratio_summary["max_abs_random_shift_ratio_neg_alpha"],
            "max_abs_random_shift_ratio_pos_alpha": None
            if random_ratio_summary is None
            else random_ratio_summary["max_abs_random_shift_ratio_pos_alpha"],
            "non_conflict_accuracy_drop_neg_alpha": None
            if chosen_side_effect_neg is None
            else chosen_side_effect_neg["accuracy_drop"],
            "non_conflict_accuracy_drop_pos_alpha": None
            if chosen_side_effect_pos is None
            else chosen_side_effect_pos["accuracy_drop"],
            "non_conflict_entropy_change_neg_alpha": None
            if chosen_side_effect_neg is None
            else chosen_side_effect_neg["entropy_change"],
            "non_conflict_entropy_change_pos_alpha": None
            if chosen_side_effect_pos is None
            else chosen_side_effect_pos["entropy_change"],
            "max_label_frequency_drift_neg_alpha": None
            if chosen_side_effect_neg is None
            else chosen_side_effect_neg["max_label_frequency_drift"],
            "max_label_frequency_drift_pos_alpha": None
            if chosen_side_effect_pos is None
            else chosen_side_effect_pos["max_label_frequency_drift"],
            "underpowered": underpowered,
            "c4_assessment": c4_assessment,
        }
        write_json(artifacts["summary_json"], summary_payload)
        write_json(artifacts["controls_json"], controls_payload)
        write_json(
            artifacts["run_state_json"],
            build_runtime_state(
                run_id=run_id,
                status="completed",
                phase="completed",
                compatibility_fingerprint=compatibility_fingerprint,
                completed_rows=len(existing_rows),
            ),
        )
        emit_progress(artifacts["progress_log_txt"], f"M5 intervention run completed with assessment={c4_assessment}.")
        print(json.dumps(sanitize(summary_payload), indent=2))
    except PauseRequested as pause:
        write_json(
            artifacts["run_state_json"],
            build_runtime_state(
                run_id=run_id,
                status="paused",
                phase="paused",
                compatibility_fingerprint=compatibility_fingerprint,
                completed_rows=len(existing_rows),
                pause_reason=str(pause),
            ),
        )
        emit_progress(artifacts["progress_log_txt"], str(pause))
        print(json.dumps({"status": "paused", "reason": str(pause)}, indent=2))
    except Exception as error:
        write_json(
            artifacts["run_state_json"],
            build_runtime_state(
                run_id=run_id,
                status="failed",
                phase="failed",
                compatibility_fingerprint=compatibility_fingerprint,
                completed_rows=len(existing_rows),
                error=f"{type(error).__name__}: {error}",
            ),
        )
        emit_progress(artifacts["progress_log_txt"], f"M5 intervention run failed: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
