from __future__ import annotations

import bisect
from collections import Counter
from typing import Any

import numpy as np


def compute_frozen_edges(values: list[float], num_bins: int) -> list[float]:
    if not values:
        return []
    quantiles = np.linspace(0, 1, num_bins + 1)[1:-1]
    edges = [float(np.quantile(values, quantile)) for quantile in quantiles]
    deduped: list[float] = []
    for edge in edges:
        if not deduped or edge > deduped[-1]:
            deduped.append(edge)
    return deduped


def assign_bin(value: float, edges: list[float]) -> int:
    return bisect.bisect_right(edges, value)


def build_output_margin_control(
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    min_per_history_type: int = 25,
) -> dict[str, Any]:
    dev_values = [row["o"] for row in dev_rows]
    quintile_edges = compute_frozen_edges(dev_values, 5)
    tertile_edges = compute_frozen_edges(dev_values, 3)

    def build_bins(edges: list[float]) -> list[dict[str, Any]]:
        bins: dict[int, dict[str, Any]] = {}
        for row in test_rows:
            bin_index = assign_bin(row["o"], edges)
            current = bins.setdefault(
                bin_index,
                {
                    "bin_index": bin_index,
                    "rows": [],
                    "history_counts": Counter(),
                },
            )
            current["rows"].append(row)
            current["history_counts"][row["history_type"]] += 1

        out: list[dict[str, Any]] = []
        for bin_index, payload in sorted(bins.items()):
            fresh_rows = [row for row in payload["rows"] if row["history_type"] == "fresh"]
            committed_rows = [row for row in payload["rows"] if row["history_type"] == "committed"]
            fresh_mean = sum(row["B_cap"] for row in fresh_rows) / len(fresh_rows) if fresh_rows else None
            committed_mean = sum(row["B_cap"] for row in committed_rows) / len(committed_rows) if committed_rows else None
            out.append(
                {
                    "bin_index": bin_index,
                    "fresh_count": len(fresh_rows),
                    "committed_count": len(committed_rows),
                    "eligible": len(fresh_rows) >= min_per_history_type and len(committed_rows) >= min_per_history_type,
                    "fresh_mean_B_cap": fresh_mean,
                    "committed_mean_B_cap": committed_mean,
                    "delta_mean_B_cap": None if fresh_mean is None or committed_mean is None else committed_mean - fresh_mean,
                    "fresh_rows": fresh_rows,
                    "committed_rows": committed_rows,
                }
            )
        return out

    quintile_bins = build_bins(quintile_edges)
    tertile_bins = build_bins(tertile_edges)
    eligible_quintiles = [row for row in quintile_bins if row["eligible"]]
    eligible_tertiles = [row for row in tertile_bins if row["eligible"]]

    if len(eligible_quintiles) >= 3:
        chosen_scheme = "quintile"
        chosen_bins = eligible_quintiles
        chosen_edges = quintile_edges
    elif len(eligible_tertiles) >= 2:
        chosen_scheme = "tertile"
        chosen_bins = eligible_tertiles
        chosen_edges = tertile_edges
    else:
        chosen_scheme = "underpowered"
        chosen_bins = eligible_tertiles
        chosen_edges = tertile_edges

    central_bins = []
    if chosen_scheme == "quintile":
        central_indices = {1, 2, 3}
        central_bins = [row for row in chosen_bins if row["bin_index"] in central_indices]
    elif chosen_scheme == "tertile":
        central_bins = [row for row in chosen_bins if row["bin_index"] == 1]

    pooled_rows = [row for row in test_rows if any(row["o"] == candidate["o"] for candidate in [])]
    del pooled_rows

    pooled_delta = None
    if chosen_bins:
        pooled_fresh = [row for payload in chosen_bins for row in payload["fresh_rows"]]
        pooled_committed = [row for payload in chosen_bins for row in payload["committed_rows"]]
        if pooled_fresh and pooled_committed:
            pooled_delta = (
                sum(row["B_cap"] for row in pooled_committed) / len(pooled_committed)
                - sum(row["B_cap"] for row in pooled_fresh) / len(pooled_fresh)
            )

    return {
        "dev_edges": {
            "quintile": quintile_edges,
            "tertile": tertile_edges,
        },
        "quintile_bins": [
            {key: value for key, value in row.items() if key not in {"fresh_rows", "committed_rows"}}
            for row in quintile_bins
        ],
        "tertile_bins": [
            {key: value for key, value in row.items() if key not in {"fresh_rows", "committed_rows"}}
            for row in tertile_bins
        ],
        "chosen_scheme": chosen_scheme,
        "chosen_edges": chosen_edges,
        "chosen_bins": [
            {key: value for key, value in row.items() if key not in {"fresh_rows", "committed_rows"}}
            for row in chosen_bins
        ],
        "central_bin_indices": [row["bin_index"] for row in central_bins],
        "central_bins_same_sign": all(
            row["delta_mean_B_cap"] is not None and row["delta_mean_B_cap"] > 0 for row in central_bins
        )
        if central_bins
        else None,
        "pooled_valid_bin_delta_mean_B_cap": pooled_delta,
        "pooled_valid_bins_same_sign": None if pooled_delta is None else pooled_delta > 0,
        "satisfies_mandatory_control": chosen_scheme != "underpowered",
    }
