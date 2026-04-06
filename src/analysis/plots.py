from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_reversibility_plot(summary: dict[str, Any], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fresh = summary["fresh_profile"]
    committed = summary["committed_profile"]
    doses = [int(dose) for dose in fresh]

    plt.figure(figsize=(6, 4))
    plt.plot(doses, [fresh[str(dose)] for dose in doses], marker="o", label="fresh")
    plt.plot(doses, [committed[str(dose)] for dose in doses], marker="o", label="committed")
    plt.xlabel("dose")
    plt.ylabel("R(d)")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_cell_delta_heatmap(cell_rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ms = sorted({row["m"] for row in cell_rows})
    ks = sorted({row["k"] for row in cell_rows})
    matrix = np.full((len(ms), len(ks)), np.nan)
    for row in cell_rows:
        matrix[ms.index(row["m"]), ks.index(row["k"])] = row["delta_summary"]["mean_deltaB_pair_cap"]

    plt.figure(figsize=(5, 4))
    image = plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(image, label="mean deltaB_pair_cap")
    plt.xticks(range(len(ks)), [f"k={k}" for k in ks])
    plt.yticks(range(len(ms)), [f"m={m}" for m in ms])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_output_margin_plot(control_summary: dict[str, Any], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bins = control_summary["chosen_bins"]
    if not bins:
        return

    x = [row["bin_index"] for row in bins]
    fresh = [row["fresh_mean_B_cap"] for row in bins]
    committed = [row["committed_mean_B_cap"] for row in bins]

    plt.figure(figsize=(6, 4))
    plt.plot(x, fresh, marker="o", label="fresh")
    plt.plot(x, committed, marker="o", label="committed")
    plt.xlabel(f"{control_summary['chosen_scheme']} bin")
    plt.ylabel("mean B_cap")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_probe_layer_plot(layer_metrics: list[dict[str, Any]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not layer_metrics:
        return

    ordered = sorted(layer_metrics, key=lambda row: row["layer"])
    layers = [row["layer"] for row in ordered]
    aurocs = [row["dev_auroc"] for row in ordered]

    plt.figure(figsize=(6, 4))
    plt.plot(layers, aurocs, marker="o")
    plt.xlabel("layer")
    plt.ylabel("dev AUROC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_commitment_calibration_plot(calibration_rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not calibration_rows:
        return

    deciles = [row["decile"] for row in calibration_rows]
    positive_rates = [row["positive_rate"] for row in calibration_rows]
    mean_b_caps = [row["mean_B_cap"] for row in calibration_rows]

    figure, left_axis = plt.subplots(figsize=(6, 4))
    left_axis.plot(deciles, positive_rates, marker="o", label="P(B_raw > d0)")
    left_axis.set_xlabel("score decile")
    left_axis.set_ylabel("positive rate")
    left_axis.set_ylim(-0.02, 1.02)

    right_axis = left_axis.twinx()
    right_axis.plot(deciles, mean_b_caps, marker="s", color="tab:orange", label="mean B_cap")
    right_axis.set_ylabel("mean B_cap")

    left_handles, left_labels = left_axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(figure)
