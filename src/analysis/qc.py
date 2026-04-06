from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def summarize_pair_invariants(pair_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    violations: dict[str, int] = defaultdict(int)
    total_pairs = len(pair_summaries)
    for summary in pair_summaries:
        for check_name, passed in summary["pair_checks"].items():
            if not passed:
                violations[check_name] += 1

    return {
        "total_pairs": total_pairs,
        "all_passed": all(count == 0 for count in violations.values()),
        "violations": dict(sorted(violations.items())),
    }


def determine_prefix_leader(scores: dict[str, float], tolerance: float = 1e-8) -> tuple[str, bool]:
    top_score = max(scores.values())
    tied = [label for label, score in scores.items() if math.isclose(score, top_score, abs_tol=tolerance)]
    return tied[0], len(tied) > 1


def evaluate_sanity_accuracy(
    scored_rows: list[dict[str, Any]],
    *,
    overall_threshold: float = 0.90,
    per_label_threshold: float | None = 0.85,
) -> dict[str, Any]:
    by_expected: dict[str, list[bool]] = defaultdict(list)
    total_correct = 0
    for row in scored_rows:
        is_correct = bool(row["is_correct"])
        total_correct += int(is_correct)
        by_expected[row["expected_option"]].append(is_correct)

    overall_accuracy = total_correct / max(len(scored_rows), 1)
    per_label_accuracy = {
        label: sum(flags) / len(flags)
        for label, flags in sorted(by_expected.items())
    }
    return {
        "num_examples": len(scored_rows),
        "overall_accuracy": overall_accuracy,
        "per_label_accuracy": per_label_accuracy,
        "passes_threshold": (
            overall_accuracy >= overall_threshold
            and (
                per_label_threshold is None
                or all(accuracy >= per_label_threshold for accuracy in per_label_accuracy.values())
            )
        ),
        "overall_threshold": overall_threshold,
        "per_label_threshold": per_label_threshold,
    }


def build_oracle_scores(
    visible_option_labels: list[str],
    support_counts: dict[str, int],
    option_to_suspect: dict[str, str],
) -> list[float]:
    return [float(support_counts[option_to_suspect[label]]) for label in visible_option_labels]


def count_rows_by_split(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["split"] for row in rows)
    return dict(sorted(counts.items()))
