from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from src.analysis.barriers import DOSES, normalize_barrier_numeric


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def confidence_interval(values: list[float], alpha: float = 0.05) -> dict[str, float]:
    if not values:
        return {"low": float("nan"), "high": float("nan")}
    ordered = sorted(values)
    return {
        "low": _quantile(ordered, alpha / 2),
        "high": _quantile(ordered, 1 - alpha / 2),
    }


def bootstrap_pair_delta(pair_summaries: list[dict[str, Any]], n_resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    valid_pairs = [summary for summary in pair_summaries if summary["valid_pair_main"]]
    if not valid_pairs:
        return {"mean": float("nan"), "ci": {"low": float("nan"), "high": float("nan")}, "n_pairs": 0}

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(valid_pairs) for _ in range(len(valid_pairs))]
        estimates.append(sum(row["deltaB_pair_cap"] for row in sample) / len(sample))

    return {
        "mean": sum(row["deltaB_pair_cap"] for row in valid_pairs) / len(valid_pairs),
        "ci": confidence_interval(estimates),
        "n_pairs": len(valid_pairs),
    }


def bootstrap_profile_difference(
    pair_summaries: list[dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    valid_pairs = [summary for summary in pair_summaries if summary["valid_pair_main"]]
    if not valid_pairs:
        return {
            "fresh": {str(dose): {"mean": float("nan"), "ci": {"low": float("nan"), "high": float("nan")}} for dose in DOSES},
            "committed": {str(dose): {"mean": float("nan"), "ci": {"low": float("nan"), "high": float("nan")}} for dose in DOSES},
            "difference": {str(dose): {"mean": float("nan"), "ci": {"low": float("nan"), "high": float("nan")}} for dose in DOSES},
            "n_pairs": 0,
        }

    rng = random.Random(seed)
    boot_fresh: dict[str, list[float]] = defaultdict(list)
    boot_committed: dict[str, list[float]] = defaultdict(list)
    boot_diff: dict[str, list[float]] = defaultdict(list)

    def profile_value(rows: list[dict[str, Any]], dose: int, key: str) -> float:
        threshold = float(dose)
        return sum(1 for row in rows if normalize_barrier_numeric(row[key]) <= threshold) / len(rows)

    for _ in range(n_resamples):
        sample = [rng.choice(valid_pairs) for _ in range(len(valid_pairs))]
        fresh_rows = sample
        committed_rows = sample
        for dose in DOSES:
            fresh_rate = profile_value(fresh_rows, dose, "B_raw_fresh_numeric")
            committed_rate = profile_value(committed_rows, dose, "B_raw_committed_numeric")
            boot_fresh[str(dose)].append(fresh_rate)
            boot_committed[str(dose)].append(committed_rate)
            boot_diff[str(dose)].append(committed_rate - fresh_rate)

    out = {"fresh": {}, "committed": {}, "difference": {}, "n_pairs": len(valid_pairs)}
    for dose in DOSES:
        dose_key = str(dose)
        fresh_mean = profile_value(valid_pairs, dose, "B_raw_fresh_numeric")
        committed_mean = profile_value(valid_pairs, dose, "B_raw_committed_numeric")
        diff_mean = committed_mean - fresh_mean
        out["fresh"][dose_key] = {"mean": fresh_mean, "ci": confidence_interval(boot_fresh[dose_key])}
        out["committed"][dose_key] = {"mean": committed_mean, "ci": confidence_interval(boot_committed[dose_key])}
        out["difference"][dose_key] = {"mean": diff_mean, "ci": confidence_interval(boot_diff[dose_key])}
    return out
