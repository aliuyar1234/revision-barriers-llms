from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from src.data.mhst_worlds import OPTION_LABELS

DOSES = list(range(7))


def normalize_barrier_numeric(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "inf":
            return math.inf
        if lowered == "-inf":
            return -math.inf
        return float(lowered)
    return float(value)


def compute_barrier(option_scores_by_dose: dict[str, dict[str, float]], incumbent_option: str, challenger_option: str) -> float:
    for dose in DOSES:
        score_map = option_scores_by_dose[str(dose)]
        if score_map[challenger_option] >= score_map[incumbent_option]:
            return float(dose)
    return math.inf


def clip_barrier(barrier_raw: float) -> int:
    return 7 if math.isinf(barrier_raw) else int(barrier_raw)


def topset(score_map: dict[str, float], tolerance: float = 1e-8) -> list[str]:
    top_score = max(score_map.values())
    mhst_order = {label: index for index, label in enumerate(OPTION_LABELS)}
    return sorted(
        [label for label, score in score_map.items() if math.isclose(score, top_score, abs_tol=tolerance)],
        key=lambda label: (0, mhst_order[label]) if label in mhst_order else (1, label),
    )


def build_valid_main(
    option_scores_by_dose: dict[str, dict[str, float]],
    barrier_raw: float,
    distractor_options: list[str],
) -> tuple[bool, bool, list[int]]:
    if math.isinf(barrier_raw):
        doses_to_check = DOSES
    else:
        doses_to_check = list(range(int(barrier_raw) + 1))

    invalid_doses: list[int] = []
    distractor_top_any = False
    for dose in doses_to_check:
        current_topset = topset(option_scores_by_dose[str(dose)])
        if any(option in current_topset for option in distractor_options):
            invalid_doses.append(dose)
            distractor_top_any = True

    return len(invalid_doses) == 0, distractor_top_any, invalid_doses


def reversibility_profile(rows: list[dict[str, Any]]) -> dict[str, float]:
    profile: dict[str, float] = {}
    if not rows:
        return {str(dose): float("nan") for dose in DOSES}

    for dose in DOSES:
        flips = sum(1 for row in rows if normalize_barrier_numeric(row["B_raw_numeric"]) <= dose)
        profile[str(dose)] = flips / len(rows)
    return profile


def summarize_deltas(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not pair_rows:
        return {
            "pair_count": 0,
            "mean_deltaB_pair_cap": float("nan"),
            "positive_count": 0,
            "zero_count": 0,
            "negative_count": 0,
        }

    deltas = [row["deltaB_pair_cap"] for row in pair_rows]
    return {
        "pair_count": len(pair_rows),
        "mean_deltaB_pair_cap": sum(deltas) / len(deltas),
        "positive_count": sum(1 for delta in deltas if delta > 0),
        "zero_count": sum(1 for delta in deltas if delta == 0),
        "negative_count": sum(1 for delta in deltas if delta < 0),
    }


def build_pair_summaries(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scored_rows:
        grouped[row["pair_id"]][row["history_type"]] = row

    pair_summaries: list[dict[str, Any]] = []
    for pair_id, pair_rows in sorted(grouped.items()):
        if set(pair_rows) != {"fresh", "committed"}:
            raise ValueError(f"Pair {pair_id} is missing fresh/committed rows.")

        fresh = pair_rows["fresh"]
        committed = pair_rows["committed"]
        valid_pair_main = bool(fresh["valid_main"] and committed["valid_main"])
        in_S_lead_pair = bool(
            fresh["prefix_leader"] == fresh["incumbent_option"]
            and committed["prefix_leader"] == committed["incumbent_option"]
            and not fresh["prefix_has_top_tie"]
            and not committed["prefix_has_top_tie"]
        )
        pair_summary = {
            "pair_id": pair_id,
            "split": fresh["split"],
            "m": fresh["m"],
            "k": fresh["k"],
            "fresh_row_id": fresh["example_id"],
            "committed_row_id": committed["example_id"],
            "incumbent": fresh["incumbent"],
            "challenger": fresh["challenger"],
            "lead_prefix_fresh": fresh["prefix_leader"],
            "lead_prefix_committed": committed["prefix_leader"],
            "o_prefix_fresh": fresh["o"],
            "o_prefix_committed": committed["o"],
            "B_raw_fresh": fresh["B_raw"],
            "B_raw_committed": committed["B_raw"],
            "B_raw_fresh_numeric": fresh["B_raw_numeric"],
            "B_raw_committed_numeric": committed["B_raw_numeric"],
            "B_cap_fresh": fresh["B_cap"],
            "B_cap_committed": committed["B_cap"],
            "deltaB_pair_cap": committed["B_cap"] - fresh["B_cap"],
            "valid_main_fresh": fresh["valid_main"],
            "valid_main_committed": committed["valid_main"],
            "valid_pair_main": valid_pair_main,
            "in_S_lead_pair": in_S_lead_pair,
            "distractor_top_any": bool(fresh["distractor_top_any"] or committed["distractor_top_any"]),
            "prefix_prompt_fresh": fresh["prefix_prompt"],
            "prefix_prompt_committed": committed["prefix_prompt"],
            "full_prompt_dose0_fresh": fresh["full_prompts"]["0"],
            "full_prompt_dose0_committed": committed["full_prompts"]["0"],
            "option_scores_by_dose_fresh": fresh["option_scores_by_dose"],
            "option_scores_by_dose_committed": committed["option_scores_by_dose"],
            "invalid_doses_fresh": fresh["invalid_doses"],
            "invalid_doses_committed": committed["invalid_doses"],
        }
        pair_summaries.append(pair_summary)
    return pair_summaries


def filter_rows_for_pair_valid(rows: list[dict[str, Any]], pair_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_pair_ids = {summary["pair_id"] for summary in pair_summaries if summary["valid_pair_main"]}
    return [row for row in rows if row["pair_id"] in valid_pair_ids]


def summarize_by_k(
    scored_rows: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in sorted({summary["k"] for summary in pair_summaries}):
        pair_subset = [summary for summary in pair_summaries if summary["k"] == k]
        row_subset = [row for row in scored_rows if row["k"] == k and row["pair_id"] in {summary["pair_id"] for summary in pair_subset if summary["valid_pair_main"]}]
        out[str(k)] = {
            "delta_summary": summarize_deltas([summary for summary in pair_subset if summary["valid_pair_main"]]),
            "fresh_profile": reversibility_profile([row for row in row_subset if row["history_type"] == "fresh"]),
            "committed_profile": reversibility_profile([row for row in row_subset if row["history_type"] == "committed"]),
            "pair_count": len(pair_subset),
            "valid_pair_count": sum(1 for summary in pair_subset if summary["valid_pair_main"]),
            "S_lead_count": sum(1 for summary in pair_subset if summary["in_S_lead_pair"]),
        }
    return out


def summarize_by_cell(
    scored_rows: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    all_cells = sorted({(summary["m"], summary["k"]) for summary in pair_summaries})
    for m, k in all_cells:
        pair_subset = [summary for summary in pair_summaries if summary["m"] == m and summary["k"] == k]
        valid_pair_ids = {summary["pair_id"] for summary in pair_subset if summary["valid_pair_main"]}
        row_subset = [row for row in scored_rows if row["pair_id"] in valid_pair_ids and row["m"] == m and row["k"] == k]
        cell_rows.append(
            {
                "m": m,
                "k": k,
                "pair_count": len(pair_subset),
                "valid_pair_count": sum(1 for summary in pair_subset if summary["valid_pair_main"]),
                "S_lead_count": sum(1 for summary in pair_subset if summary["in_S_lead_pair"]),
                "delta_summary": summarize_deltas([summary for summary in pair_subset if summary["valid_pair_main"]]),
                "fresh_profile": reversibility_profile([row for row in row_subset if row["history_type"] == "fresh"]),
                "committed_profile": reversibility_profile([row for row in row_subset if row["history_type"] == "committed"]),
            }
        )
    return cell_rows


def invalid_main_report(scored_rows: list[dict[str, Any]], pair_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    row_report: dict[str, dict[str, float]] = {}
    for history_type in sorted({row["history_type"] for row in scored_rows}):
        rows = [row for row in scored_rows if row["history_type"] == history_type]
        invalid_count = sum(1 for row in rows if not row["valid_main"])
        row_report[history_type] = {
            "count": len(rows),
            "invalid_count": invalid_count,
            "invalid_rate": invalid_count / len(rows) if rows else float("nan"),
        }

    pair_invalid_count = sum(1 for summary in pair_summaries if not summary["valid_pair_main"])
    return {
        "rows_by_history_type": row_report,
        "pair_count": len(pair_summaries),
        "invalid_pair_count": pair_invalid_count,
        "invalid_pair_rate": pair_invalid_count / len(pair_summaries) if pair_summaries else float("nan"),
    }


def build_inspection_buckets(pair_summaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bucket_a = [
        summary
        for summary in pair_summaries
        if summary["k"] == 0
        and (
            summary["deltaB_pair_cap"] != 0
            or not math.isclose(summary["o_prefix_fresh"], summary["o_prefix_committed"], abs_tol=1e-8)
            or summary["prefix_prompt_fresh"] != summary["prefix_prompt_committed"]
        )
    ]
    bucket_b = sorted(
        [summary for summary in pair_summaries if summary["k"] == 2 and summary["deltaB_pair_cap"] <= 0],
        key=lambda summary: (summary["deltaB_pair_cap"], summary["pair_id"]),
    )
    bucket_c = [
        summary
        for summary in pair_summaries
        if not summary["valid_main_fresh"] or not summary["valid_main_committed"] or not summary["valid_pair_main"]
    ]
    bucket_d = [summary for summary in pair_summaries if not summary["in_S_lead_pair"]]
    bucket_e = [
        summary
        for summary in pair_summaries
        if summary["B_raw_fresh"] == "inf" or summary["B_raw_committed"] == "inf"
    ]
    return {
        "A_k0_non_null": bucket_a,
        "B_k2_non_positive": bucket_b,
        "C_invalid_main": bucket_c,
        "D_not_in_S_lead": bucket_d,
        "E_saturation_inf": bucket_e,
    }
