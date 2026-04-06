from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.barriers import (
    build_inspection_buckets,
    build_pair_summaries,
    build_valid_main,
    clip_barrier,
    compute_barrier,
    invalid_main_report,
    reversibility_profile,
    summarize_by_cell,
    summarize_by_k,
    summarize_deltas,
)
from src.analysis.bootstrap import bootstrap_pair_delta, bootstrap_profile_difference
from src.analysis.output_margin import build_output_margin_control
from src.analysis.plots import save_cell_delta_heatmap, save_output_margin_plot, save_reversibility_plot
from src.analysis.qc import build_oracle_scores, determine_prefix_leader, evaluate_sanity_accuracy
from src.data.mhst_prompts import (
    MHST_CANONICAL_ANSWERS,
    render_mhst_full_prompt,
    render_mhst_full_prompt_prefix,
    render_mhst_prefix_prompt,
)
from src.data.mhst_worlds import OPTION_LABELS
from src.data.rcicl_prompts import (
    RCICL_CANONICAL_ANSWERS,
    render_rcicl_full_prompt,
    render_rcicl_full_prompt_prefix,
    render_rcicl_prefix_prompt,
)
from src.models.load_model import load_option_scorer

M3_RESUME_FINGERPRINT_FILES = [
    "scripts/run_behavior.py",
    "src/analysis/barriers.py",
    "src/analysis/bootstrap.py",
    "src/analysis/output_margin.py",
    "src/analysis/qc.py",
    "src/data/mhst_prompts.py",
    "src/data/rcicl_prompts.py",
    "src/data/rcicl_worlds.py",
    "src/models/load_model.py",
    "src/models/option_scoring.py",
]

M3_FINALIZE_COMPATIBILITY_KEYS = [
    "science_config_sha256",
    "pair_input_sha256",
    "sanity_input_sha256",
    "selected_example_ids_sha256",
    "selected_row_count",
]


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def _writer(handle: Any) -> None:
        json.dump(sanitize_for_json(payload), handle, indent=2)

    write_with_tempfile(path, _writer)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    def _writer(handle: Any) -> None:
        for row in rows:
            handle.write(json.dumps(sanitize_for_json(row), ensure_ascii=False) + "\n")

    write_with_tempfile(path, _writer)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    def _writer(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: sanitize_for_json(row.get(name)) for name in fieldnames})

    write_with_tempfile(path, _writer, newline="")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_with_tempfile(path: Path, writer: Any, *, newline: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline=newline) as handle:
        writer(handle)
    os.replace(temp_path, path)


def write_text(path: Path, text: str) -> None:
    def _writer(handle: Any) -> None:
        handle.write(text)

    write_with_tempfile(path, _writer)


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_for_json(row), ensure_ascii=False) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def append_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(sanitize_for_json(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_path_string(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_sidecar_path(base_path: Path, suffix: str, extension: str | None = None) -> Path:
    resolved_extension = base_path.suffix if extension is None else extension
    return base_path.with_name(f"{base_path.stem}{suffix}{resolved_extension}")


def build_m3_artifact_paths(outputs: dict[str, str]) -> dict[str, Path]:
    scored_rows_path = ROOT / outputs["scored_rows_jsonl"]
    return {
        "scored_rows": scored_rows_path,
        "scored_rows_partial": ROOT / outputs.get(
            "scored_rows_partial_jsonl",
            relative_path_string(build_sidecar_path(scored_rows_path, "__partial")),
        ),
        "run_state": ROOT / outputs.get(
            "run_state_json",
            relative_path_string(build_sidecar_path(scored_rows_path, "__run_state", ".json")),
        ),
        "heartbeat": ROOT / outputs.get(
            "heartbeat_json",
            relative_path_string(build_sidecar_path(scored_rows_path, "__heartbeat", ".json")),
        ),
        "config_snapshot": ROOT / outputs.get(
            "config_snapshot_yaml",
            relative_path_string(build_sidecar_path(scored_rows_path, "__config", ".yaml")),
        ),
        "progress_log": ROOT / outputs.get(
            "progress_log_txt",
            relative_path_string(build_sidecar_path(scored_rows_path, "__progress", ".log")),
        ),
        "stop_flag": ROOT / outputs.get(
            "stop_flag_path",
            relative_path_string(build_sidecar_path(scored_rows_path, "__STOP", ".flag")),
        ),
    }


def build_model_runtime_metadata(scorer: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "backend": model_config["backend"],
        "configured_model_name_or_path": model_config.get("model_name_or_path"),
        "configured_device": model_config.get("device"),
        "configured_dtype": model_config.get("dtype"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "scorer_class": type(scorer).__name__,
    }
    if hasattr(scorer, "device"):
        metadata["resolved_device"] = getattr(scorer, "device")
    if hasattr(scorer, "model"):
        model = getattr(scorer, "model")
        metadata["model_class"] = model.__class__.__name__
        metadata["resolved_model_name_or_path"] = getattr(getattr(model, "config", None), "_name_or_path", None)
        first_parameter = next(model.parameters(), None)
        metadata["resolved_model_dtype"] = None if first_parameter is None else str(first_parameter.dtype)
    if hasattr(scorer, "tokenizer"):
        tokenizer = getattr(scorer, "tokenizer")
        metadata["tokenizer_class"] = tokenizer.__class__.__name__
        metadata["resolved_tokenizer_name_or_path"] = getattr(tokenizer, "name_or_path", None)
    return metadata


def read_jsonl_example_ids(path: Path) -> list[str]:
    example_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            example_ids.append(payload["example_id"])
    return example_ids


def build_resume_signature(config: dict[str, Any], pair_rows: list[dict[str, Any]], scorer: Any) -> dict[str, Any]:
    source_digests = {
        path: sha256_file(ROOT / path)
        for path in M3_RESUME_FINGERPRINT_FILES
    }
    science_config = {
        "model": config["model"],
        "inputs": config["inputs"],
        "run": config["run"],
    }
    return {
        "science_config_sha256": sha256_json(science_config),
        "science_config": sanitize_for_json(science_config),
        "pair_input_sha256": sha256_file(ROOT / config["inputs"]["pair_jsonl"]),
        "sanity_input_sha256": sha256_file(ROOT / config["inputs"]["sanity_jsonl"]),
        "selected_example_ids_sha256": sha256_json([row["example_id"] for row in pair_rows]),
        "selected_row_count": len(pair_rows),
        "source_file_sha256": source_digests,
        "source_file_bundle_sha256": sha256_json(source_digests),
        "model_runtime": build_model_runtime_metadata(scorer, config["model"]),
    }


def finalize_compatible_signature(saved_signature: dict[str, Any], current_signature: dict[str, Any]) -> bool:
    return all(saved_signature.get(key) == current_signature.get(key) for key in M3_FINALIZE_COMPATIBILITY_KEYS)


def build_progress_snapshot(
    *,
    state: dict[str, Any],
    total_rows: int,
    completed_rows: int,
    session_start_time: float,
    session_start_completed_rows: int,
    current_row: dict[str, Any] | None,
) -> dict[str, Any]:
    elapsed_seconds = max(0.0, time.time() - session_start_time)
    session_completed_rows = max(0, completed_rows - session_start_completed_rows)
    rows_per_minute = None
    eta_seconds = None
    if elapsed_seconds > 0 and session_completed_rows > 0:
        rows_per_minute = session_completed_rows * 60.0 / elapsed_seconds
        if rows_per_minute > 0:
            eta_seconds = max(0.0, (total_rows - completed_rows) / rows_per_minute * 60.0)

    return {
        "run_id": state["run_id"],
        "status": state["status"],
        "phase": state["phase"],
        "updated_at": utc_now_iso(),
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "remaining_rows": total_rows - completed_rows,
        "percent_complete": None if total_rows == 0 else round(100.0 * completed_rows / total_rows, 2),
        "session_elapsed_seconds": round(elapsed_seconds, 3),
        "session_completed_rows": session_completed_rows,
        "rows_per_minute": None if rows_per_minute is None else round(rows_per_minute, 3),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds, 3),
        "last_completed_example_id": state.get("last_completed_example_id"),
        "pause_reason": state.get("pause_reason"),
        "current_row": None
        if current_row is None
        else {
            "example_id": current_row["example_id"],
            "pair_id": current_row["pair_id"],
            "split": current_row["split"],
            "m": current_row["m"],
            "k": current_row["k"],
            "history_type": current_row["history_type"],
        },
    }


def format_progress_line(heartbeat: dict[str, Any]) -> str:
    current_row = heartbeat["current_row"]
    current_cell = "idle"
    if current_row is not None:
        current_cell = (
            f"example_id={current_row['example_id']} split={current_row['split']} "
            f"m={current_row['m']} k={current_row['k']} history={current_row['history_type']}"
        )
    eta_text = "unknown"
    if heartbeat["eta_seconds"] is not None:
        eta_text = f"{heartbeat['eta_seconds'] / 60.0:.1f}m"
    rpm_text = "n/a"
    if heartbeat["rows_per_minute"] is not None:
        rpm_text = f"{heartbeat['rows_per_minute']:.2f}"
    return (
        f"phase={heartbeat['phase']} status={heartbeat['status']} "
        f"progress={heartbeat['completed_rows']}/{heartbeat['total_rows']} "
        f"({heartbeat['percent_complete']}%) rows_per_min={rpm_text} eta={eta_text} "
        f"{current_cell}"
    )


def emit_progress(log_path: Path, message: str) -> None:
    stamped = f"[{utc_now_iso()}] {message}"
    print(stamped, flush=True)
    append_log_line(log_path, stamped)


def oracle_metadata_for_counts(record: dict[str, Any], support_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "oracle_scores": build_oracle_scores(
            OPTION_LABELS,
            support_counts,
            record["option_to_suspect"],
        )
    }


def task_family(record: dict[str, Any]) -> str:
    return str(record.get("task_family", "mhst"))


def canonical_answers_for_record(record: dict[str, Any]) -> list[str]:
    if task_family(record) == "rcicl":
        return RCICL_CANONICAL_ANSWERS
    return MHST_CANONICAL_ANSWERS


def prefix_prompt_for_record(record: dict[str, Any]) -> str:
    if task_family(record) == "rcicl":
        return render_rcicl_prefix_prompt(record)
    return render_mhst_prefix_prompt(record)


def full_prompt_prefix_for_record(record: dict[str, Any]) -> str:
    if task_family(record) == "rcicl":
        return render_rcicl_full_prompt_prefix(record)
    return render_mhst_full_prompt_prefix(record)


def full_prompt_for_record(record: dict[str, Any], dose: int) -> str:
    if task_family(record) == "rcicl":
        return render_rcicl_full_prompt(record, dose)
    return render_mhst_full_prompt(record, dose)


def oracle_metadata_for_record(record: dict[str, Any], *, dose: int | None = None) -> dict[str, Any] | None:
    if task_family(record) == "mhst":
        support_counts = record["prefix_counts"] if dose is None else record["support_counts_by_dose"][str(dose)]
        return oracle_metadata_for_counts(record, support_counts)
    if isinstance(record.get("_oracle_scores"), dict):
        key = "prefix" if dose is None else str(dose)
        if key in record["_oracle_scores"]:
            return {"oracle_scores": record["_oracle_scores"][key]}
    return None


def generic_prefix_score_map(scorer: Any, record: dict[str, Any]) -> tuple[str, dict[str, float], str, bool]:
    prompt = prefix_prompt_for_record(record)
    answer_scores = scorer.score_options(
        prompt,
        canonical_answers_for_record(record),
        metadata=oracle_metadata_for_record(record),
    )
    answer_labels = [answer.strip() for answer in canonical_answers_for_record(record)]
    score_map = dict(zip(answer_labels, answer_scores, strict=True))
    leader, has_top_tie = determine_prefix_leader(score_map)
    return prompt, score_map, leader, has_top_tie


def score_prefix_only(scorer: Any, record: dict[str, Any]) -> dict[str, Any]:
    prompt, score_map, leader, has_top_tie = generic_prefix_score_map(scorer, record)
    return {
        "prefix_prompt": prompt,
        "prefix_end_token_idx": scorer.prefix_token_index(full_prompt_prefix_for_record(record)),
        "prefix_option_scores": score_map,
        "prefix_leader": leader,
        "prefix_has_top_tie": has_top_tie,
        "o": score_map[record["incumbent_option"]] - score_map[record["challenger_option"]],
    }


def score_sanity_record(scorer: Any, record: dict[str, Any]) -> dict[str, Any]:
    prefix = score_prefix_only(scorer, record)
    return {
        "example_id": record["example_id"],
        "pair_id": record["pair_id"],
        "split": record["split"],
        "task_variant": record["task_variant"],
        "history_type": record["history_type"],
        "m": record["m"],
        "k": record["k"],
        "incumbent_option": record["incumbent_option"],
        "challenger_option": record["challenger_option"],
        **prefix,
        "expected_option": record["expected_option"],
        "predicted_option": prefix["prefix_leader"],
        "is_correct": prefix["prefix_leader"] == record["expected_option"],
    }


def score_behavior_record(scorer: Any, record: dict[str, Any]) -> dict[str, Any]:
    prefix = score_prefix_only(scorer, record)

    option_scores_by_dose: dict[str, dict[str, float]] = {}
    full_prompts: dict[str, str] = {}
    answer_labels = [answer.strip() for answer in canonical_answers_for_record(record)]
    for dose in range(7):
        prompt = full_prompt_for_record(record, dose)
        full_prompts[str(dose)] = prompt
        answer_scores = scorer.score_options(
            prompt,
            canonical_answers_for_record(record),
            metadata=oracle_metadata_for_record(record, dose=dose),
        )
        option_scores_by_dose[str(dose)] = dict(zip(answer_labels, answer_scores, strict=True))

    distractor_options = [record["suspect_to_option"][suspect_id] for suspect_id in record.get("distractors", [])]
    barrier_numeric = compute_barrier(option_scores_by_dose, record["incumbent_option"], record["challenger_option"])
    barrier_raw = "inf" if math.isinf(barrier_numeric) else int(barrier_numeric)
    valid_main, distractor_top_any, invalid_doses = build_valid_main(option_scores_by_dose, barrier_numeric, distractor_options)

    return {
        "example_id": record["example_id"],
        "pair_id": record["pair_id"],
        "world_id": record["world_id"],
        "split": record["split"],
        "task_variant": record["task_variant"],
        "history_type": record["history_type"],
        "m": record["m"],
        "k": record["k"],
        "incumbent": record["incumbent"],
        "challenger": record["challenger"],
        "distractors": record.get("distractors", []),
        "incumbent_option": record["incumbent_option"],
        "challenger_option": record["challenger_option"],
        "distractor_options": distractor_options,
        "late_distractor_cycle": record.get("late_distractor_cycle", []),
        **prefix,
        "full_prompts": full_prompts,
        "option_scores_by_dose": option_scores_by_dose,
        "B_raw": barrier_raw,
        "B_raw_numeric": barrier_numeric,
        "B_cap": clip_barrier(barrier_numeric),
        "valid_main": bool(record.get("valid_main_locked", False) or valid_main),
        "distractor_top_any": distractor_top_any,
        "invalid_doses": invalid_doses,
    }


def filter_pair_rows(rows: list[dict[str, Any]], run_config: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_splits = set(run_config.get("include_splits", []))
    allowed_margins = set(run_config.get("include_margins", []))
    allowed_history_depths = set(run_config.get("include_history_depths", []))

    filtered = rows
    if allowed_splits:
        filtered = [row for row in filtered if row["split"] in allowed_splits]
    if allowed_margins:
        filtered = [row for row in filtered if row["m"] in allowed_margins]
    if allowed_history_depths:
        filtered = [row for row in filtered if row["k"] in allowed_history_depths]
    return filtered


def reviewed_pairs_with_bucket_annotations(buckets: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    bucket_counts = {bucket_name: len(rows) for bucket_name, rows in buckets.items()}
    reviewed_rows: dict[str, dict[str, Any]] = {}
    reviewed_counts: dict[str, int] = {}

    for bucket_name, rows in buckets.items():
        chosen = rows if len(rows) <= 10 else rows[:10]
        reviewed_counts[bucket_name] = len(chosen)
        for row in chosen:
            existing = reviewed_rows.get(row["pair_id"])
            if existing is None:
                reviewed_rows[row["pair_id"]] = {
                    **row,
                    "inspection_buckets": [bucket_name],
                }
            else:
                existing["inspection_buckets"].append(bucket_name)

    return list(reviewed_rows.values()), bucket_counts, reviewed_counts


def classify_reviewed_pair(row: dict[str, Any]) -> tuple[str, str]:
    buckets = set(row["inspection_buckets"])
    if "A_k0_non_null" in buckets:
        return (
            "implementation_bug",
            "k=0 should be null because fresh and committed are construction-identical here; any non-null gap is a bug signal.",
        )
    if "C_invalid_main" in buckets and row["distractor_top_any"]:
        return (
            "expected_model_behavior",
            "A distractor entered the topset before a clean challenger flip; keep excluded from main analysis and track the rate.",
        )
    if "D_not_in_S_lead" in buckets and not row["valid_pair_main"]:
        return (
            "unclear_followup",
            "Pair misses the strong-leader subset and also fails main validity; inspect coverage and rates before deciding on repair.",
        )
    if "D_not_in_S_lead" in buckets:
        return (
            "expected_model_behavior",
            "Prefix-only leader or tie behavior excludes this pair from S_lead without implying a construction bug by itself.",
        )
    if "E_saturation_inf" in buckets:
        return (
            "expected_model_behavior",
            "At least one member never flips within the tested dose set; keep as saturation evidence rather than auto-repairing it away.",
        )
    if "B_k2_non_positive" in buckets:
        return (
            "expected_model_behavior",
            "This k=2 pair is weak or wrong-sign evidence for the main effect, but no construction bug is visible from the saved fields alone.",
        )
    return ("unclear_followup", "Reviewed pair did not match a more specific heuristic disposition.")


def summarize_manual_inspection(reviewed_rows: list[dict[str, Any]], bucket_counts: dict[str, int], reviewed_counts: dict[str, int]) -> dict[str, Any]:
    pattern_lines: dict[str, str] = {}
    for bucket_name in bucket_counts:
        rows = [row for row in reviewed_rows if bucket_name in row["inspection_buckets"]]
        disposition_counts: dict[str, int] = {}
        for row in rows:
            disposition_counts[row["inspection_disposition"]] = disposition_counts.get(row["inspection_disposition"], 0) + 1
        if not rows:
            pattern_lines[bucket_name] = "No reviewed pairs in this bucket."
        else:
            pattern_lines[bucket_name] = ", ".join(
                f"{label}={count}" for label, count in sorted(disposition_counts.items())
            )

    action_decision = "no repair needed"
    if bucket_counts.get("A_k0_non_null", 0) > 0:
        action_decision = "repair needed before test"
    elif any(row["inspection_disposition"] == "implementation_bug" for row in reviewed_rows):
        action_decision = "repair needed before test"

    return {
        "bucket_counts": bucket_counts,
        "reviewed_counts": reviewed_counts,
        "observed_patterns": pattern_lines,
        "action_decision": action_decision,
    }


def build_inspection_rows(pair_summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets = build_inspection_buckets(pair_summaries)
    reviewed_pairs, bucket_counts, reviewed_counts = reviewed_pairs_with_bucket_annotations(buckets)
    inspection_rows: list[dict[str, Any]] = []
    for row in reviewed_pairs:
        disposition, note = classify_reviewed_pair(row)
        inspection_rows.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "m": row["m"],
                "k": row["k"],
                "fresh_row_id": row["fresh_row_id"],
                "committed_row_id": row["committed_row_id"],
                "incumbent": row["incumbent"],
                "challenger": row["challenger"],
                "lead_prefix_fresh": row["lead_prefix_fresh"],
                "lead_prefix_committed": row["lead_prefix_committed"],
                "o_prefix_fresh": row["o_prefix_fresh"],
                "o_prefix_committed": row["o_prefix_committed"],
                "B_raw_fresh": row["B_raw_fresh"],
                "B_raw_committed": row["B_raw_committed"],
                "B_cap_fresh": row["B_cap_fresh"],
                "B_cap_committed": row["B_cap_committed"],
                "deltaB_pair_cap": row["deltaB_pair_cap"],
                "valid_main_fresh": row["valid_main_fresh"],
                "valid_main_committed": row["valid_main_committed"],
                "valid_pair_main": row["valid_pair_main"],
                "in_S_lead_pair": row["in_S_lead_pair"],
                "distractor_top_any": row["distractor_top_any"],
                "option_scores_by_dose_fresh": row["option_scores_by_dose_fresh"],
                "option_scores_by_dose_committed": row["option_scores_by_dose_committed"],
                "rendered_fresh_prompt": row["full_prompt_dose0_fresh"],
                "rendered_committed_prompt": row["full_prompt_dose0_committed"],
                "inspection_buckets": row["inspection_buckets"],
                "inspection_disposition": disposition,
                "inspection_note": note,
            }
        )
    return inspection_rows, summarize_manual_inspection(inspection_rows, bucket_counts, reviewed_counts)


def subset_summary(scored_rows: list[dict[str, Any]], pair_summaries: list[dict[str, Any]], bootstrap_seed: int) -> dict[str, Any]:
    valid_pair_ids = {summary["pair_id"] for summary in pair_summaries if summary["valid_pair_main"]}
    paired_valid_rows = [row for row in scored_rows if row["pair_id"] in valid_pair_ids]
    valid_S_lead_pairs = [summary for summary in pair_summaries if summary["valid_pair_main"] and summary["in_S_lead_pair"]]
    valid_S_lead_pair_ids = {summary["pair_id"] for summary in valid_S_lead_pairs}
    valid_S_lead_rows = [row for row in scored_rows if row["pair_id"] in valid_S_lead_pair_ids]

    return {
        "pair_delta_summary": summarize_deltas([summary for summary in pair_summaries if summary["valid_pair_main"]]),
        "pair_delta_bootstrap": bootstrap_pair_delta(pair_summaries, seed=bootstrap_seed),
        "fresh_profile": reversibility_profile([row for row in paired_valid_rows if row["history_type"] == "fresh"]),
        "committed_profile": reversibility_profile([row for row in paired_valid_rows if row["history_type"] == "committed"]),
        "profile_bootstrap": bootstrap_profile_difference(pair_summaries, seed=bootstrap_seed),
        "pair_count": len(pair_summaries),
        "valid_pair_count": sum(1 for summary in pair_summaries if summary["valid_pair_main"]),
        "S_lead_count": sum(1 for summary in pair_summaries if summary["in_S_lead_pair"]),
        "S_lead_valid_pair_count": len(valid_S_lead_pairs),
        "invalid_pair_count": sum(1 for summary in pair_summaries if not summary["valid_pair_main"]),
        "S_lead_delta_summary": summarize_deltas(valid_S_lead_pairs),
        "S_lead_delta_bootstrap": bootstrap_pair_delta(valid_S_lead_pairs, seed=bootstrap_seed),
        "S_lead_fresh_profile": reversibility_profile([row for row in valid_S_lead_rows if row["history_type"] == "fresh"]),
        "S_lead_committed_profile": reversibility_profile([row for row in valid_S_lead_rows if row["history_type"] == "committed"]),
    }


def add_bootstrap_to_by_k(by_k: dict[str, Any], pair_summaries: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    enriched: dict[str, Any] = {}
    for key, summary in by_k.items():
        k = int(key)
        pair_subset = [row for row in pair_summaries if row["k"] == k]
        enriched[key] = {
            **summary,
            "delta_bootstrap": bootstrap_pair_delta(pair_subset, seed=seed + k),
        }
    return enriched


def add_bootstrap_to_cells(cell_rows: list[dict[str, Any]], pair_summaries: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in cell_rows:
        pair_subset = [summary for summary in pair_summaries if summary["m"] == row["m"] and summary["k"] == row["k"]]
        enriched.append(
            {
                **row,
                "delta_bootstrap": bootstrap_pair_delta(pair_subset, seed=seed + row["m"] * 10 + row["k"]),
            }
        )
    return enriched


def build_summary_table_rows(model_name: str, split: str, cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in cell_rows:
        rows.append(
            {
                "model": model_name,
                "split": split,
                "m": row["m"],
                "k": row["k"],
                "pair_count": row["pair_count"],
                "valid_pair_count": row["valid_pair_count"],
                "S_lead_count": row["S_lead_count"],
                "mean_deltaB_pair_cap": row["delta_summary"]["mean_deltaB_pair_cap"],
                "delta_ci_low": row["delta_bootstrap"]["ci"]["low"],
                "delta_ci_high": row["delta_bootstrap"]["ci"]["high"],
            }
        )
    return rows


def get_sanity_thresholds(run_config: dict[str, Any]) -> tuple[float, float | None]:
    overall = float(run_config.get("sanity_overall_threshold", 0.90))
    per_label = run_config.get("sanity_per_label_threshold", 0.85)
    return overall, None if per_label is None else float(per_label)


def run_prefix_sanity(config_path: Path, config: dict[str, Any], scorer: Any) -> dict[str, Any]:
    pair_rows = read_jsonl(ROOT / config["inputs"]["pair_jsonl"])
    sanity_rows = read_jsonl(ROOT / config["inputs"]["sanity_jsonl"])

    pair_limit = config["run"].get("pair_prefix_limit")
    if pair_limit is not None:
        pair_rows = pair_rows[: int(pair_limit)]

    scored_pairs = [score_behavior_record(scorer, row) for row in pair_rows]
    scored_sanity = [score_sanity_record(scorer, row) for row in sanity_rows]
    overall_threshold, per_label_threshold = get_sanity_thresholds(config["run"])
    sanity_summary = evaluate_sanity_accuracy(
        scored_sanity,
        overall_threshold=overall_threshold,
        per_label_threshold=per_label_threshold,
    )
    sanity_summary["config"] = str(config_path)
    sanity_summary["backend"] = config["model"]["backend"]

    write_jsonl(ROOT / config["outputs"]["pair_prefix_scores_jsonl"], scored_pairs)
    write_jsonl(ROOT / config["outputs"]["sanity_scores_jsonl"], scored_sanity)
    write_json(ROOT / config["outputs"]["sanity_summary_json"], sanity_summary)

    return {
        "config": str(config_path),
        "backend": config["model"]["backend"],
        "pair_rows_scored": len(scored_pairs),
        "sanity_rows_scored": len(scored_sanity),
        "sanity_overall_accuracy": sanity_summary["overall_accuracy"],
        "sanity_passes_threshold": sanity_summary["passes_threshold"],
    }


def run_m2_pilot(config_path: Path, config: dict[str, Any], scorer: Any) -> dict[str, Any]:
    pair_rows = read_jsonl(ROOT / config["inputs"]["pair_jsonl"])
    pair_rows = filter_pair_rows(pair_rows, config["run"])
    sanity_rows = read_jsonl(ROOT / config["inputs"]["sanity_jsonl"])

    scored_rows = [score_behavior_record(scorer, row) for row in pair_rows]
    pair_summaries = build_pair_summaries(scored_rows)
    overall_summary = subset_summary(scored_rows, pair_summaries, bootstrap_seed=int(config["run"].get("bootstrap_seed", 0)))
    summary_by_k = add_bootstrap_to_by_k(
        summarize_by_k(scored_rows, pair_summaries),
        pair_summaries,
        seed=int(config["run"].get("bootstrap_seed", 0)),
    )
    inspection_rows, manual_inspection_summary = build_inspection_rows(pair_summaries)

    scored_sanity = [score_sanity_record(scorer, row) for row in sanity_rows]
    overall_threshold, per_label_threshold = get_sanity_thresholds(config["run"])
    sanity_summary = evaluate_sanity_accuracy(
        scored_sanity,
        overall_threshold=overall_threshold,
        per_label_threshold=per_label_threshold,
    )
    sanity_summary["config"] = str(config_path)
    sanity_summary["backend"] = config["model"]["backend"]

    write_jsonl(ROOT / config["outputs"]["scored_rows_jsonl"], scored_rows)
    write_jsonl(ROOT / config["outputs"]["pair_summary_jsonl"], pair_summaries)
    write_jsonl(ROOT / config["outputs"]["inspection_pairs_jsonl"], inspection_rows)
    write_json(
        ROOT / config["outputs"]["summary_json"],
        {
            "config": str(config_path),
            "backend": config["model"]["backend"],
            "overall": overall_summary,
            "by_k": summary_by_k,
            "manual_inspection": manual_inspection_summary,
        },
    )
    write_json(ROOT / config["outputs"]["sanity_summary_json"], sanity_summary)
    save_reversibility_plot(overall_summary, ROOT / config["outputs"]["overall_plot_png"], "M2 pilot reversibility (paired-valid)")
    for k, summary in summary_by_k.items():
        save_reversibility_plot(summary, ROOT / config["outputs"]["plot_png_by_k"].format(k=k), f"M2 pilot reversibility (k={k})")

    return {
        "config": str(config_path),
        "backend": config["model"]["backend"],
        "pair_rows_scored": len(scored_rows),
        "pair_count": len(pair_summaries),
        "valid_pair_count": overall_summary["valid_pair_count"],
        "S_lead_count": overall_summary["S_lead_count"],
        "mean_deltaB_pair_cap": overall_summary["pair_delta_summary"]["mean_deltaB_pair_cap"],
        "sanity_overall_accuracy": sanity_summary["overall_accuracy"],
        "sanity_passes_threshold": sanity_summary["passes_threshold"],
        "manual_inspection_action": manual_inspection_summary["action_decision"],
    }


def build_run_status_payload(
    *,
    status: str,
    config_path: Path,
    config: dict[str, Any],
    pair_rows_total: int,
    pair_rows_completed: int,
    artifacts: dict[str, Path],
    heartbeat: dict[str, Any],
    pause_reason: str | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "config": str(config_path),
        "backend": config["model"]["backend"],
        "pair_rows_total": pair_rows_total,
        "pair_rows_completed": pair_rows_completed,
        "run_state_path": relative_path_string(artifacts["run_state"]),
        "heartbeat_path": relative_path_string(artifacts["heartbeat"]),
        "partial_rows_path": relative_path_string(artifacts["scored_rows_partial"]),
        "progress_log_path": relative_path_string(artifacts["progress_log"]),
        "stop_flag_path": relative_path_string(artifacts["stop_flag"]),
        "heartbeat": heartbeat,
    }
    if pause_reason is not None:
        payload["pause_reason"] = pause_reason
    if result is not None:
        payload.update(result)
    return payload


def finalize_m3_outputs(
    config_path: Path,
    config: dict[str, Any],
    scorer: Any,
    scored_rows: list[dict[str, Any]],
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    sanity_rows = read_jsonl(ROOT / config["inputs"]["sanity_jsonl"])
    bootstrap_seed = int(config["run"].get("bootstrap_seed", 0))
    pair_summaries = build_pair_summaries(scored_rows)
    split_names = sorted({row["split"] for row in scored_rows})

    summaries_by_split: dict[str, Any] = {}
    cell_summary_rows: list[dict[str, Any]] = []
    for split in split_names:
        split_rows = [row for row in scored_rows if row["split"] == split]
        split_pairs = [row for row in pair_summaries if row["split"] == split]
        overall = subset_summary(split_rows, split_pairs, bootstrap_seed=bootstrap_seed + len(split))
        by_k = add_bootstrap_to_by_k(summarize_by_k(split_rows, split_pairs), split_pairs, seed=bootstrap_seed + len(split) * 10)
        by_cell = add_bootstrap_to_cells(summarize_by_cell(split_rows, split_pairs), split_pairs, seed=bootstrap_seed + len(split) * 100)
        invalid_report = invalid_main_report(split_rows, split_pairs)
        summaries_by_split[split] = {
            "overall": overall,
            "by_k": by_k,
            "by_cell": by_cell,
            "invalid_main": invalid_report,
        }
        cell_summary_rows.extend(build_summary_table_rows(config["run"]["model_label"], split, by_cell))

    output_margin_summary = None
    if config["run"].get("output_margin_control", False):
        dev_rows = [row for row in scored_rows if row["split"] == "dev"]
        test_rows = [row for row in scored_rows if row["split"] == "test"]
        dev_pairs = [row for row in pair_summaries if row["split"] == "dev" and row["valid_pair_main"]]
        test_pairs = [row for row in pair_summaries if row["split"] == "test" and row["valid_pair_main"]]
        dev_valid_pair_ids = {row["pair_id"] for row in dev_pairs}
        test_valid_pair_ids = {row["pair_id"] for row in test_pairs}
        output_margin_summary = build_output_margin_control(
            [row for row in dev_rows if row["pair_id"] in dev_valid_pair_ids],
            [row for row in test_rows if row["pair_id"] in test_valid_pair_ids],
            min_per_history_type=int(config["run"].get("output_margin_min_per_history_type", 25)),
        )

    scored_sanity = [score_sanity_record(scorer, row) for row in sanity_rows]
    overall_threshold, per_label_threshold = get_sanity_thresholds(config["run"])
    sanity_summary = evaluate_sanity_accuracy(
        scored_sanity,
        overall_threshold=overall_threshold,
        per_label_threshold=per_label_threshold,
    )
    sanity_summary["config"] = str(config_path)
    sanity_summary["backend"] = config["model"]["backend"]

    write_jsonl(artifacts["scored_rows"], scored_rows)
    write_jsonl(ROOT / config["outputs"]["pair_summary_jsonl"], pair_summaries)
    headline_split = str(config["run"].get("headline_split", "test" if "test" in summaries_by_split else split_names[0]))
    headline_overall = summaries_by_split.get(headline_split, {}).get("overall", {})
    filtered_input_rows = filter_pair_rows(read_jsonl(ROOT / config["inputs"]["pair_jsonl"]), config["run"])
    summary_payload = {
        "run_id": artifacts["scored_rows"].stem,
        "config": str(config_path),
        "backend": config["model"]["backend"],
        "headline_split": headline_split,
        "pair_count": headline_overall.get("pair_count"),
        "valid_pair_count": headline_overall.get("valid_pair_count"),
        "mean_deltaB_pair_cap": headline_overall.get("pair_delta_summary", {}).get("mean_deltaB_pair_cap"),
        "reversibility_profile_fresh": headline_overall.get("fresh_profile"),
        "reversibility_profile_committed": headline_overall.get("committed_profile"),
        "no_conflict_accuracy": sanity_summary["overall_accuracy"],
        "compatibility_fingerprint": sha256_json(build_resume_signature(config, filtered_input_rows, scorer)),
        "by_split": summaries_by_split,
        "output_margin_control": output_margin_summary,
    }
    write_json(ROOT / config["outputs"]["summary_json"], summary_payload)
    write_json(ROOT / config["outputs"]["sanity_summary_json"], sanity_summary)
    write_csv(
        ROOT / config["outputs"]["summary_table_csv"],
        cell_summary_rows,
        [
            "model",
            "split",
            "m",
            "k",
            "pair_count",
            "valid_pair_count",
            "S_lead_count",
            "mean_deltaB_pair_cap",
            "delta_ci_low",
            "delta_ci_high",
        ],
    )

    if "test" in summaries_by_split:
        save_reversibility_plot(
            summaries_by_split["test"]["overall"],
            ROOT / config["outputs"]["overall_plot_png"],
            f"{config['run']['model_label']} test reversibility (paired-valid)",
        )
        for k, summary in summaries_by_split["test"]["by_k"].items():
            save_reversibility_plot(
                summary,
                ROOT / config["outputs"]["plot_png_by_k"].format(k=k),
                f"{config['run']['model_label']} test reversibility (k={k})",
            )
        save_cell_delta_heatmap(
            summaries_by_split["test"]["by_cell"],
            ROOT / config["outputs"]["cell_heatmap_png"],
            f"{config['run']['model_label']} test deltaB by cell",
        )
        if output_margin_summary and output_margin_summary["chosen_scheme"] != "underpowered":
            save_output_margin_plot(
                output_margin_summary,
                ROOT / config["outputs"]["output_margin_plot_png"],
                f"{config['run']['model_label']} output-margin control",
            )

    result = {
        "config": str(config_path),
        "backend": config["model"]["backend"],
        "status": "completed",
        "pair_rows_scored": len(scored_rows),
        "pair_count": len(pair_summaries),
        "headline_split": headline_split,
        "headline_valid_pair_count": headline_overall.get("valid_pair_count"),
        "headline_mean_deltaB_pair_cap": headline_overall.get("pair_delta_summary", {}).get("mean_deltaB_pair_cap"),
        "test_valid_pair_count": summaries_by_split.get("test", {}).get("overall", {}).get("valid_pair_count"),
        "test_mean_deltaB_pair_cap": summaries_by_split.get("test", {}).get("overall", {}).get("pair_delta_summary", {}).get("mean_deltaB_pair_cap"),
        "test_S_lead_valid_pair_count": summaries_by_split.get("test", {}).get("overall", {}).get("S_lead_valid_pair_count"),
        "output_margin_scheme": None if output_margin_summary is None else output_margin_summary["chosen_scheme"],
        "output_margin_satisfies_control": None if output_margin_summary is None else output_margin_summary["satisfies_mandatory_control"],
        "sanity_overall_accuracy": sanity_summary["overall_accuracy"],
        "sanity_passes_threshold": sanity_summary["passes_threshold"],
        "run_state_path": relative_path_string(artifacts["run_state"]),
        "heartbeat_path": relative_path_string(artifacts["heartbeat"]),
        "partial_rows_path": relative_path_string(artifacts["scored_rows_partial"]),
        "progress_log_path": relative_path_string(artifacts["progress_log"]),
    }
    return result


def run_m3_behavior(
    config_path: Path,
    config: dict[str, Any],
    scorer: Any,
    *,
    resume: bool = True,
    finalize_only: bool = False,
) -> dict[str, Any]:
    pair_rows = filter_pair_rows(read_jsonl(ROOT / config["inputs"]["pair_jsonl"]), config["run"])
    artifacts = build_m3_artifact_paths(config["outputs"])
    execution_config = config.get("execution", {})
    progress_every_rows = max(1, int(execution_config.get("progress_every_rows", 10)))
    heartbeat_seconds = max(1.0, float(execution_config.get("heartbeat_seconds", 30)))
    allow_resume = resume and bool(execution_config.get("allow_resume", True))

    write_text(artifacts["config_snapshot"], config_path.read_text(encoding="utf-8"))
    resume_signature = build_resume_signature(config, pair_rows, scorer)
    expected_example_ids = {row["example_id"] for row in pair_rows}

    existing_example_ids: list[str] = []
    if artifacts["scored_rows_partial"].exists():
        existing_example_ids = read_jsonl_example_ids(artifacts["scored_rows_partial"])
    duplicate_example_ids = sorted(
        [example_id for example_id, count in Counter(existing_example_ids).items() if count > 1]
    )
    if duplicate_example_ids:
        raise ValueError(f"Partial checkpoint contains duplicate example_ids: {duplicate_example_ids[:5]}")
    unknown_example_ids = sorted(set(existing_example_ids) - expected_example_ids)
    if unknown_example_ids:
        raise ValueError(f"Partial checkpoint contains unknown example_ids: {unknown_example_ids[:5]}")
    if existing_example_ids and not allow_resume and not finalize_only:
        raise ValueError(
            "Partial checkpoint rows already exist for this run. Resume is disabled; rerun with --resume or clean the checkpoint files first."
        )

    completed_example_ids = set(existing_example_ids)
    completed_rows = len(completed_example_ids)
    total_rows = len(pair_rows)
    if completed_rows > total_rows:
        raise ValueError("Checkpoint row count exceeds the selected row count for this run.")

    existing_state = read_json(artifacts["run_state"]) if artifacts["run_state"].exists() else None
    finalize_with_updated_sources = False
    if existing_state is not None:
        saved_signature = existing_state.get("resume_signature", {})
        current_signature = sanitize_for_json(resume_signature)
        if saved_signature != current_signature:
            if completed_rows == total_rows and finalize_compatible_signature(saved_signature, current_signature):
                finalize_with_updated_sources = True
            else:
                raise ValueError(
                    "Resume refused because the stored run signature does not match the current code/config/input/model signature."
                )
        run_id = existing_state["run_id"]
        created_at = existing_state.get("created_at", utc_now_iso())
    else:
        run_id = artifacts["scored_rows"].stem
        created_at = utc_now_iso()

    state: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "phase": "scoring",
        "created_at": created_at,
        "updated_at": utc_now_iso(),
        "completed_at": None,
        "pause_reason": None,
        "error": None,
        "config_path": str(config_path),
        "config_snapshot_path": relative_path_string(artifacts["config_snapshot"]),
        "execution_config": sanitize_for_json(execution_config),
        "resume_signature": sanitize_for_json(resume_signature),
        "finalize_with_updated_sources": finalize_with_updated_sources,
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "last_completed_example_id": existing_example_ids[-1] if existing_example_ids else None,
        "outputs": {
            "scored_rows_partial_jsonl": relative_path_string(artifacts["scored_rows_partial"]),
            "scored_rows_jsonl": relative_path_string(artifacts["scored_rows"]),
            "run_state_json": relative_path_string(artifacts["run_state"]),
            "heartbeat_json": relative_path_string(artifacts["heartbeat"]),
            "progress_log_txt": relative_path_string(artifacts["progress_log"]),
            "stop_flag_path": relative_path_string(artifacts["stop_flag"]),
            "summary_json": config["outputs"]["summary_json"],
            "pair_summary_jsonl": config["outputs"]["pair_summary_jsonl"],
            "sanity_summary_json": config["outputs"]["sanity_summary_json"],
            "summary_table_csv": config["outputs"]["summary_table_csv"],
        },
    }

    session_start_time = time.time()
    session_start_completed_rows = completed_rows
    last_progress_emit = 0.0

    if artifacts["stop_flag"].exists() and not finalize_only and completed_rows < total_rows:
        state["status"] = "paused"
        state["phase"] = "paused"
        state["pause_reason"] = (
            f"Stop flag present at {relative_path_string(artifacts['stop_flag'])}; remove it and rerun the same command to resume."
        )
        state["updated_at"] = utc_now_iso()
        heartbeat = build_progress_snapshot(
            state=state,
            total_rows=total_rows,
            completed_rows=completed_rows,
            session_start_time=session_start_time,
            session_start_completed_rows=session_start_completed_rows,
            current_row=None,
        )
        write_json(artifacts["run_state"], state)
        write_json(artifacts["heartbeat"], heartbeat)
        emit_progress(artifacts["progress_log"], state["pause_reason"])
        return build_run_status_payload(
            status="paused",
            config_path=config_path,
            config=config,
            pair_rows_total=total_rows,
            pair_rows_completed=completed_rows,
            artifacts=artifacts,
            heartbeat=heartbeat,
            pause_reason=state["pause_reason"],
        )

    try:
        if not finalize_only and completed_rows < total_rows:
            emit_progress(
                artifacts["progress_log"],
                f"Starting/resuming behavior scoring with checkpointing at {relative_path_string(artifacts['scored_rows_partial'])}. "
                f"Completed rows already present: {completed_rows}/{total_rows}.",
            )
            remaining_rows = [row for row in pair_rows if row["example_id"] not in completed_example_ids]
            for row in remaining_rows:
                state["status"] = "running"
                state["phase"] = "scoring"
                state["pause_reason"] = None
                state["updated_at"] = utc_now_iso()
                heartbeat = build_progress_snapshot(
                    state=state,
                    total_rows=total_rows,
                    completed_rows=completed_rows,
                    session_start_time=session_start_time,
                    session_start_completed_rows=session_start_completed_rows,
                    current_row=row,
                )
                write_json(artifacts["run_state"], state)
                write_json(artifacts["heartbeat"], heartbeat)

                now = time.time()
                if (
                    completed_rows == session_start_completed_rows
                    or (completed_rows + 1) % progress_every_rows == 0
                    or now - last_progress_emit >= heartbeat_seconds
                ):
                    emit_progress(artifacts["progress_log"], f"Scoring checkpoint: {format_progress_line(heartbeat)}")
                    last_progress_emit = now

                try:
                    scored_row = score_behavior_record(scorer, row)
                except KeyboardInterrupt:
                    state["status"] = "paused"
                    state["phase"] = "paused"
                    state["pause_reason"] = (
                        "Keyboard interrupt received during row scoring; the current row was not checkpointed and will be retried on resume."
                    )
                    state["updated_at"] = utc_now_iso()
                    heartbeat = build_progress_snapshot(
                        state=state,
                        total_rows=total_rows,
                        completed_rows=completed_rows,
                        session_start_time=session_start_time,
                        session_start_completed_rows=session_start_completed_rows,
                        current_row=row,
                    )
                    write_json(artifacts["run_state"], state)
                    write_json(artifacts["heartbeat"], heartbeat)
                    emit_progress(artifacts["progress_log"], state["pause_reason"])
                    return build_run_status_payload(
                        status="paused",
                        config_path=config_path,
                        config=config,
                        pair_rows_total=total_rows,
                        pair_rows_completed=completed_rows,
                        artifacts=artifacts,
                        heartbeat=heartbeat,
                        pause_reason=state["pause_reason"],
                    )

                append_jsonl_row(artifacts["scored_rows_partial"], scored_row)
                completed_example_ids.add(scored_row["example_id"])
                completed_rows = len(completed_example_ids)
                state["completed_rows"] = completed_rows
                state["last_completed_example_id"] = scored_row["example_id"]
                state["updated_at"] = utc_now_iso()

                heartbeat = build_progress_snapshot(
                    state=state,
                    total_rows=total_rows,
                    completed_rows=completed_rows,
                    session_start_time=session_start_time,
                    session_start_completed_rows=session_start_completed_rows,
                    current_row=row,
                )
                write_json(artifacts["run_state"], state)
                write_json(artifacts["heartbeat"], heartbeat)

                now = time.time()
                if (
                    completed_rows == total_rows
                    or completed_rows % progress_every_rows == 0
                    or now - last_progress_emit >= heartbeat_seconds
                ):
                    emit_progress(artifacts["progress_log"], f"Completed checkpoint: {format_progress_line(heartbeat)}")
                    last_progress_emit = now

                if artifacts["stop_flag"].exists():
                    state["status"] = "paused"
                    state["phase"] = "paused"
                    state["pause_reason"] = (
                        f"Stop flag detected at {relative_path_string(artifacts['stop_flag'])}; "
                        "the latest completed row was checkpointed successfully."
                    )
                    state["updated_at"] = utc_now_iso()
                    heartbeat = build_progress_snapshot(
                        state=state,
                        total_rows=total_rows,
                        completed_rows=completed_rows,
                        session_start_time=session_start_time,
                        session_start_completed_rows=session_start_completed_rows,
                        current_row=None,
                    )
                    write_json(artifacts["run_state"], state)
                    write_json(artifacts["heartbeat"], heartbeat)
                    emit_progress(artifacts["progress_log"], state["pause_reason"])
                    return build_run_status_payload(
                        status="paused",
                        config_path=config_path,
                        config=config,
                        pair_rows_total=total_rows,
                        pair_rows_completed=completed_rows,
                        artifacts=artifacts,
                        heartbeat=heartbeat,
                        pause_reason=state["pause_reason"],
                    )
        elif finalize_only:
            emit_progress(
                artifacts["progress_log"],
                "Finalize-only mode requested; rebuilding final outputs from the existing partial checkpoint.",
            )
            if finalize_with_updated_sources:
                emit_progress(
                    artifacts["progress_log"],
                    "Finalize-only run is reusing a completed checkpoint under updated analysis code because the scientific run identity still matches.",
                )

        if completed_rows != total_rows:
            raise ValueError(
                f"Cannot finalize behavior outputs because only {completed_rows}/{total_rows} rows have been checkpointed."
            )

        state["status"] = "running"
        state["phase"] = "finalizing"
        state["pause_reason"] = None
        state["updated_at"] = utc_now_iso()
        heartbeat = build_progress_snapshot(
            state=state,
            total_rows=total_rows,
            completed_rows=completed_rows,
            session_start_time=session_start_time,
            session_start_completed_rows=session_start_completed_rows,
            current_row=None,
        )
        write_json(artifacts["run_state"], state)
        write_json(artifacts["heartbeat"], heartbeat)
        emit_progress(artifacts["progress_log"], "All selected rows are checkpointed; finalizing summaries, tables, and plots.")
        if finalize_with_updated_sources and not finalize_only:
            emit_progress(
                artifacts["progress_log"],
                "Completed checkpoint is being finalized under updated analysis code after a scoring-complete retry.",
            )

        scored_rows = read_jsonl(artifacts["scored_rows_partial"])
        result = finalize_m3_outputs(config_path, config, scorer, scored_rows, artifacts)

        state["status"] = "completed"
        state["phase"] = "completed"
        state["completed_at"] = utc_now_iso()
        state["updated_at"] = state["completed_at"]
        state["completed_rows"] = total_rows
        state["pause_reason"] = None
        state["error"] = None
        heartbeat = build_progress_snapshot(
            state=state,
            total_rows=total_rows,
            completed_rows=completed_rows,
            session_start_time=session_start_time,
            session_start_completed_rows=session_start_completed_rows,
            current_row=None,
        )
        write_json(artifacts["run_state"], state)
        write_json(artifacts["heartbeat"], heartbeat)
        emit_progress(artifacts["progress_log"], f"Behavior run completed successfully. {format_progress_line(heartbeat)}")
        return result
    except Exception as error:
        state["status"] = "failed"
        state["phase"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        state["updated_at"] = utc_now_iso()
        heartbeat = build_progress_snapshot(
            state=state,
            total_rows=total_rows,
            completed_rows=completed_rows,
            session_start_time=session_start_time,
            session_start_completed_rows=session_start_completed_rows,
            current_row=None,
        )
        write_json(artifacts["run_state"], state)
        write_json(artifacts["heartbeat"], heartbeat)
        emit_progress(artifacts["progress_log"], f"Behavior run failed: {state['error']}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behavior workflows.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--backend-override", default=None, help="Optional scorer backend override.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from a compatible partial behavior checkpoint when one exists.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Skip behavior scoring and rebuild final artifacts from a completed partial checkpoint.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = ROOT / config_path
        if candidate.exists():
            config_path = candidate
    config = load_yaml(config_path)
    if args.backend_override is not None:
        config["model"]["backend"] = args.backend_override

    scorer = load_option_scorer(config["model"])
    mode = config["run"].get("mode", "prefix_sanity")
    if mode == "prefix_sanity":
        result = run_prefix_sanity(config_path, config, scorer)
    elif mode == "m2_pilot":
        result = run_m2_pilot(config_path, config, scorer)
    elif mode == "m3_full":
        result = run_m3_behavior(config_path, config, scorer, resume=args.resume, finalize_only=args.finalize_only)
    else:
        raise ValueError(f"Unsupported run mode: {mode}")

    print(json.dumps(sanitize_for_json(result), indent=2))


if __name__ == "__main__":
    main()
