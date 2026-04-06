from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.analysis.bootstrap import confidence_interval


@dataclass(frozen=True)
class AlphaSelectionResult:
    selected_alpha: float | None
    qualified_candidates: list[dict[str, Any]]


def build_s_move_manifest(
    behavior_rows: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    pair_in_s_lead = {row["pair_id"]: bool(row["in_S_lead_pair"]) for row in pair_summaries}
    t_star_by_example_id = {row["example_id"]: row["t_star_token_index"] for row in state_rows}

    examples: list[dict[str, Any]] = []
    for row in behavior_rows:
        if row["split"] != split:
            continue
        if row["history_type"] != "committed":
            continue
        if not row["valid_main"]:
            continue
        if not pair_in_s_lead.get(row["pair_id"], False):
            continue
        if row["B_raw"] not in {1, 2, 3, 4, 5, 6}:
            continue

        examples.append(
            {
                "example_id": row["example_id"],
                "pair_id": row["pair_id"],
                "split": row["split"],
                "history_type": row["history_type"],
                "m": row["m"],
                "k": row["k"],
                "valid_main": row["valid_main"],
                "B_raw": row["B_raw"],
                "B_cap": row["B_cap"],
                "in_S_lead_pair": pair_in_s_lead[row["pair_id"]],
                "t_star_token_index": t_star_by_example_id[row["example_id"]],
            }
        )

    examples.sort(key=lambda row: row["example_id"])
    return {
        "split": split,
        "count": len(examples),
        "example_ids": [row["example_id"] for row in examples],
        "examples": examples,
    }


def create_random_control_directions(*, seed: int, n_directions: int, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n_directions, dimension))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


def mean_delta_b(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return sum(float(row["delta_B_cap_vs_zero"]) for row in rows) / len(rows)


def bootstrap_delta_b(rows: list[dict[str, Any]], *, n_resamples: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"mean": float("nan"), "ci": {"low": float("nan"), "high": float("nan")}, "n_examples": 0}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(rows) for _ in range(len(rows))]
        estimates.append(mean_delta_b(sample))
    return {
        "mean": mean_delta_b(rows),
        "ci": confidence_interval(estimates),
        "n_examples": len(rows),
    }


def passes_side_effect_thresholds(side_effect_summary: dict[str, Any]) -> bool:
    return (
        side_effect_summary["accuracy_drop"] <= 0.05
        and side_effect_summary["entropy_change"] <= 0.15
        and side_effect_summary["max_label_frequency_drift"] <= 0.10
    )


def summarize_side_effects(alpha_zero_rows: list[dict[str, Any]], alpha_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else float("nan")

    def _accuracy(rows: list[dict[str, Any]]) -> float:
        return sum(1 for row in rows if row["is_correct"]) / len(rows) if rows else float("nan")

    def _label_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
        counts = Counter(row["label"] for row in rows)
        labels = sorted({*counts.keys(), *(row["label"] for row in alpha_zero_rows), *(row["label"] for row in rows)})
        return {label: counts[label] / len(rows) for label in labels} if rows else {}

    baseline_accuracy = _accuracy(alpha_zero_rows)
    current_accuracy = _accuracy(alpha_rows)
    baseline_entropy = _mean(alpha_zero_rows, "entropy")
    current_entropy = _mean(alpha_rows, "entropy")
    baseline_distribution = _label_distribution(alpha_zero_rows)
    current_distribution = _label_distribution(alpha_rows)
    all_labels = sorted(set(baseline_distribution) | set(current_distribution))
    max_label_frequency_drift = max(
        (abs(current_distribution.get(label, 0.0) - baseline_distribution.get(label, 0.0)) for label in all_labels),
        default=0.0,
    )

    return {
        "baseline_accuracy": baseline_accuracy,
        "current_accuracy": current_accuracy,
        "accuracy_drop": abs(current_accuracy - baseline_accuracy),
        "baseline_entropy": baseline_entropy,
        "current_entropy": current_entropy,
        "entropy_change": abs(current_entropy - baseline_entropy),
        "baseline_label_distribution": baseline_distribution,
        "current_label_distribution": current_distribution,
        "max_label_frequency_drift": max_label_frequency_drift,
    }


def select_alpha(candidate_rows: list[dict[str, Any]]) -> AlphaSelectionResult:
    qualified = [row for row in candidate_rows if row["qualifies"]]
    if not qualified:
        return AlphaSelectionResult(selected_alpha=None, qualified_candidates=[])

    best_total_shift = max(row["total_abs_shift"] for row in qualified)
    near_best = [
        row for row in qualified if row["total_abs_shift"] >= 0.9 * best_total_shift
    ]
    near_best.sort(key=lambda row: row["alpha"])
    selected = near_best[0]
    return AlphaSelectionResult(selected_alpha=float(selected["alpha"]), qualified_candidates=qualified)


def summarize_random_control_ratios(
    random_rows_by_direction: dict[str, dict[str, list[dict[str, Any]]]],
    commit_neg_rows: list[dict[str, Any]],
    commit_pos_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    commit_neg_abs = abs(mean_delta_b(commit_neg_rows))
    commit_pos_abs = abs(mean_delta_b(commit_pos_rows))

    neg_ratios: dict[str, float] = {}
    pos_ratios: dict[str, float] = {}
    for direction_tag, direction_rows in random_rows_by_direction.items():
        neg_mean = abs(mean_delta_b(direction_rows.get("neg", [])))
        pos_mean = abs(mean_delta_b(direction_rows.get("pos", [])))
        neg_ratios[direction_tag] = math.inf if commit_neg_abs == 0 else neg_mean / commit_neg_abs
        pos_ratios[direction_tag] = math.inf if commit_pos_abs == 0 else pos_mean / commit_pos_abs

    return {
        "neg_ratios": neg_ratios,
        "pos_ratios": pos_ratios,
        "max_abs_random_shift_ratio_neg_alpha": max(neg_ratios.values(), default=float("nan")),
        "max_abs_random_shift_ratio_pos_alpha": max(pos_ratios.values(), default=float("nan")),
    }


def assess_c4_result(
    *,
    alpha_selected: float | None,
    commit_neg_summary: dict[str, Any] | None,
    commit_pos_summary: dict[str, Any] | None,
    random_ratio_summary: dict[str, Any] | None,
    side_effect_neg: dict[str, Any] | None,
    side_effect_pos: dict[str, Any] | None,
    underpowered: bool,
) -> str:
    if underpowered:
        return "underpowered"
    if alpha_selected is None:
        return "unsupported"
    if commit_neg_summary is None or commit_pos_summary is None or random_ratio_summary is None:
        return "unsupported"

    neg_ok = commit_neg_summary["mean"] < 0 and commit_neg_summary["ci"]["high"] < 0
    pos_ok = commit_pos_summary["mean"] > 0 and commit_pos_summary["ci"]["low"] > 0
    ratio_neg = random_ratio_summary["max_abs_random_shift_ratio_neg_alpha"]
    ratio_pos = random_ratio_summary["max_abs_random_shift_ratio_pos_alpha"]
    sidefx_ok = (
        side_effect_neg is not None
        and side_effect_pos is not None
        and passes_side_effect_thresholds(side_effect_neg)
        and passes_side_effect_thresholds(side_effect_pos)
    )

    if neg_ok and pos_ok and ratio_neg <= 0.5 and ratio_pos <= 0.5 and sidefx_ok:
        return "supported"
    if ratio_neg > 0.75 or ratio_pos > 0.75:
        return "unsupported"
    if not neg_ok and not pos_ok:
        return "unsupported"
    return "weakened"
