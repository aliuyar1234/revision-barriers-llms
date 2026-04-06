from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.analysis.output_margin import assign_bin, compute_frozen_edges
from src.analysis.barriers import normalize_barrier_numeric


@dataclass
class D0Selection:
    d0: int
    dev_positive_rate: float
    candidate_rates: dict[int, float]
    used_fallback: bool


def choose_d0(dev_rows: list[dict[str, Any]]) -> D0Selection:
    candidate_rates: dict[int, float] = {}

    def positive_rate(candidate_d0: int) -> float:
        labels = [build_revisability_label(row["B_raw"], candidate_d0) for row in dev_rows]
        return sum(labels) / len(labels) if labels else float("nan")

    default_d0 = 3
    default_rate = positive_rate(default_d0)
    candidate_rates[default_d0] = default_rate
    if 0.25 <= default_rate <= 0.75:
        return D0Selection(
            d0=default_d0,
            dev_positive_rate=default_rate,
            candidate_rates=candidate_rates,
            used_fallback=False,
        )

    fallback_candidates = [2, 4, 1, 5]
    for candidate_d0 in fallback_candidates:
        candidate_rates[candidate_d0] = positive_rate(candidate_d0)

    in_range = [
        candidate
        for candidate in fallback_candidates
        if 0.25 <= candidate_rates[candidate] <= 0.75
    ]
    candidate_pool = in_range if in_range else fallback_candidates
    selected_d0 = min(
        candidate_pool,
        key=lambda candidate: (
            abs(candidate_rates[candidate] - 0.5),
            abs(candidate - default_d0),
            candidate,
        ),
    )
    return D0Selection(
        d0=selected_d0,
        dev_positive_rate=candidate_rates[selected_d0],
        candidate_rates=candidate_rates,
        used_fallback=True,
    )


def build_revisability_label(barrier_raw: Any, d0: int) -> int:
    return int(normalize_barrier_numeric(barrier_raw) > float(d0))


def add_revisability_labels(rows: list[dict[str, Any]], d0: int) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for row in rows:
        labeled.append({**row, "z": build_revisability_label(row["B_raw"], d0)})
    return labeled


def commitment_separation_by_cell(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    min_examples_per_history_type: int,
) -> list[dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    all_cells = sorted({(row["m"], row["k"]) for row in rows})
    for m_value, k_value in all_cells:
        cell_subset = [row for row in rows if row["m"] == m_value and row["k"] == k_value]
        fresh_rows = [row for row in cell_subset if row["history_type"] == "fresh"]
        committed_rows = [row for row in cell_subset if row["history_type"] == "committed"]
        fresh_mean = float(np.mean([row[score_key] for row in fresh_rows])) if fresh_rows else float("nan")
        committed_mean = float(np.mean([row[score_key] for row in committed_rows])) if committed_rows else float("nan")
        cell_rows.append(
            {
                "m": m_value,
                "k": k_value,
                "fresh_count": len(fresh_rows),
                "committed_count": len(committed_rows),
                "eligible": len(fresh_rows) >= min_examples_per_history_type and len(committed_rows) >= min_examples_per_history_type,
                "fresh_mean_score": fresh_mean,
                "committed_mean_score": committed_mean,
                "delta_mean_score": committed_mean - fresh_mean if fresh_rows and committed_rows else float("nan"),
            }
        )
    return cell_rows


def calibration_by_decile(rows: list[dict[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row[score_key])
    row_count = len(ordered)
    deciles: list[dict[str, Any]] = []
    for decile_index in range(10):
        start = int(math.floor(decile_index * row_count / 10))
        end = int(math.floor((decile_index + 1) * row_count / 10))
        if start == end:
            continue
        decile_rows = ordered[start:end]
        deciles.append(
            {
                "decile": decile_index,
                "count": len(decile_rows),
                "mean_score": float(np.mean([row[score_key] for row in decile_rows])),
                "positive_rate": float(np.mean([row["z"] for row in decile_rows])),
                "mean_B_cap": float(np.mean([row["B_cap"] for row in decile_rows])),
            }
        )
    return deciles


def build_latent_output_margin_control(
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    probe_score_key: str,
    combined_baseline_score_key: str,
    min_examples_per_history_type: int = 25,
) -> dict[str, Any]:
    dev_o_values = [row["o"] for row in dev_rows]
    quintile_edges = compute_frozen_edges(dev_o_values, 5)
    tertile_edges = compute_frozen_edges(dev_o_values, 3)

    def auroc_or_nan(rows_subset: list[dict[str, Any]], score_key: str) -> float:
        labels = np.asarray([row["z"] for row in rows_subset], dtype=np.int64)
        if len(np.unique(labels)) < 2:
            return float("nan")
        scores = np.asarray([row[score_key] for row in rows_subset], dtype=np.float64)
        from src.analysis.baselines import safe_auroc

        return safe_auroc(labels, scores)

    def build_bins(edges: list[float]) -> list[dict[str, Any]]:
        bins: dict[int, list[dict[str, Any]]] = {}
        for row in test_rows:
            bin_index = assign_bin(float(row["o"]), edges)
            bins.setdefault(bin_index, []).append(row)

        out: list[dict[str, Any]] = []
        for bin_index, bin_rows in sorted(bins.items()):
            fresh_count = sum(1 for row in bin_rows if row["history_type"] == "fresh")
            committed_count = sum(1 for row in bin_rows if row["history_type"] == "committed")
            probe_auroc = auroc_or_nan(bin_rows, probe_score_key)
            combined_auroc = auroc_or_nan(bin_rows, combined_baseline_score_key)
            out.append(
                {
                    "bin_index": bin_index,
                    "count": len(bin_rows),
                    "fresh_count": fresh_count,
                    "committed_count": committed_count,
                    "eligible": fresh_count >= min_examples_per_history_type and committed_count >= min_examples_per_history_type,
                    "probe_auroc": probe_auroc,
                    "combined_auroc": combined_auroc,
                    "delta_auroc_probe_minus_combined": probe_auroc - combined_auroc
                    if not math.isnan(probe_auroc) and not math.isnan(combined_auroc)
                    else float("nan"),
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
    elif len(eligible_tertiles) >= 2:
        chosen_scheme = "tertile"
        chosen_bins = eligible_tertiles
    else:
        chosen_scheme = "underpowered"
        chosen_bins = eligible_tertiles

    if chosen_scheme == "quintile":
        central_bin_indices = [row["bin_index"] for row in chosen_bins if row["bin_index"] in {1, 2, 3}]
    elif chosen_scheme == "tertile":
        central_bin_indices = [row["bin_index"] for row in chosen_bins if row["bin_index"] == 1]
    else:
        central_bin_indices = []

    central_bins = [row for row in chosen_bins if row["bin_index"] in set(central_bin_indices)]
    pooled_chosen_rows = [row for row in test_rows if assign_bin(float(row["o"]), quintile_edges if chosen_scheme == "quintile" else tertile_edges) in {bin_row["bin_index"] for bin_row in chosen_bins}]

    from src.analysis.baselines import safe_auroc

    pooled_probe_auroc = float("nan")
    pooled_combined_auroc = float("nan")
    pooled_delta = float("nan")
    if pooled_chosen_rows:
        pooled_labels = np.asarray([row["z"] for row in pooled_chosen_rows], dtype=np.int64)
        if len(np.unique(pooled_labels)) >= 2:
            pooled_probe_scores = np.asarray([row[probe_score_key] for row in pooled_chosen_rows], dtype=np.float64)
            pooled_combined_scores = np.asarray([row[combined_baseline_score_key] for row in pooled_chosen_rows], dtype=np.float64)
            pooled_probe_auroc = safe_auroc(pooled_labels, pooled_probe_scores)
            pooled_combined_auroc = safe_auroc(pooled_labels, pooled_combined_scores)
            pooled_delta = pooled_probe_auroc - pooled_combined_auroc

    return {
        "dev_edges": {
            "quintile": quintile_edges,
            "tertile": tertile_edges,
        },
        "quintile_bins": quintile_bins,
        "tertile_bins": tertile_bins,
        "chosen_scheme": chosen_scheme,
        "chosen_bins": chosen_bins,
        "central_bin_indices": central_bin_indices,
        "central_bins_positive_delta": all(row["delta_auroc_probe_minus_combined"] > 0 for row in central_bins) if central_bins else None,
        "pooled_probe_auroc": pooled_probe_auroc,
        "pooled_combined_auroc": pooled_combined_auroc,
        "pooled_delta_auroc_probe_minus_combined": pooled_delta,
        "pooled_positive_delta": None if math.isnan(pooled_delta) else pooled_delta > 0,
        "satisfies_mandatory_control": chosen_scheme != "underpowered",
    }
