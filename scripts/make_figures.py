from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def copy_figure(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def save_matched_history_schematic(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    boxes = {
        "fresh": (0.08, 0.58, 0.34, 0.26),
        "committed": (0.08, 0.16, 0.34, 0.26),
        "shared": (0.58, 0.30, 0.30, 0.32),
    }
    for x, y, w, h in boxes.values():
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5, color="black"))

    ax.text(0.25, 0.79, "Fresh history", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(
        0.25,
        0.69,
        "same designated incumbent/challenger pair\nsame designated evidence margin m\nchallenger evidence appears earlier",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(0.25, 0.37, "Committed history", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(
        0.25,
        0.27,
        "same designated incumbent/challenger pair\nsame designated evidence margin m\nincumbent evidence appears earlier",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(0.73, 0.56, "Shared late-evidence schedule", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(
        0.73,
        0.44,
        "fixed 6-slot late packet\nsame dose family for both prefixes\nmeasure how much challenger evidence is\nneeded to flip the incumbent",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.annotate("", xy=(0.58, 0.46), xytext=(0.42, 0.68), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", xy=(0.58, 0.46), xytext=(0.42, 0.29), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.text(0.73, 0.17, "Revision barrier = additional late challenger evidence\nrequired to reverse the incumbent", ha="center", va="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_motivating_example_panel(pair_rows: list[dict[str, Any]], output_path: Path) -> None:
    valid = [row for row in pair_rows if row["split"] == "test" and row["valid_pair_main"] and row["in_S_lead_pair"] and row["deltaB_pair_cap"] > 0]
    valid.sort(key=lambda row: (-row["deltaB_pair_cap"], row["pair_id"]))
    chosen = valid[0]

    fresh_prompt = chosen["prefix_prompt_fresh"].split("Final answer:")[0].strip()
    committed_prompt = chosen["prefix_prompt_committed"].split("Final answer:")[0].strip()

    def shorten(text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= 14:
            return text
        return "\n".join(lines[:14]) + "\n..."

    fresh_text = shorten(fresh_prompt)
    committed_text = shorten(committed_prompt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for axis in axes:
        axis.axis("off")

    axes[0].set_title(
        f"Fresh history\nleader={chosen['lead_prefix_fresh']}  B_raw={chosen['B_raw_fresh']}",
        fontsize=12,
        weight="bold",
    )
    axes[0].text(0.01, 0.99, fresh_text, va="top", ha="left", fontsize=9, family="monospace")

    axes[1].set_title(
        f"Committed history\nleader={chosen['lead_prefix_committed']}  B_raw={chosen['B_raw_committed']}",
        fontsize=12,
        weight="bold",
    )
    axes[1].text(0.01, 0.99, committed_text, va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle(
        f"Motivating matched-history pair: {chosen['pair_id']}  |  m={chosen['m']}  k={chosen['k']}  deltaB_pair_cap={chosen['deltaB_pair_cap']}",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_history_tax_distribution(pair_rows: list[dict[str, Any]], output_path: Path) -> None:
    deltas = [row["deltaB_pair_cap"] for row in pair_rows if row["split"] == "test" and row["valid_pair_main"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.hist(deltas, bins=range(min(deltas), max(deltas) + 2), color="tab:blue", alpha=0.85, align="left", rwidth=0.85)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("deltaB_pair_cap (committed minus fresh)")
    axis.set_ylabel("valid test pair count")
    axis.set_title("Held-out MHST history-tax distribution")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_scope_bar_chart(scope_rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["label"] for row in scope_rows]
    values = [row["mean"] for row in scope_rows]
    lowers = [row["mean"] - row["ci_low"] for row in scope_rows]
    uppers = [row["ci_high"] - row["mean"] for row in scope_rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(range(len(scope_rows)), values, color=["tab:blue", "tab:orange", "tab:green"], alpha=0.85)
    axis.errorbar(range(len(scope_rows)), values, yerr=[lowers, uppers], fmt="none", ecolor="black", capsize=4)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(scope_rows)), labels)
    axis.set_ylabel("mean deltaB_pair_cap")
    axis.set_title("Primary and auxiliary behavioral scope summaries")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    output_dir = ROOT / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_matched_history_schematic(output_dir / "figure_1_matched_history_schematic.png")

    copy_figure(
        ROOT / "outputs/behavior/m3_qwen/E2__qwen25_7b__mhst__reversibility_overall_test__seed0.png",
        output_dir / "figure_2_mhst_overall_reversibility.png",
    )
    copy_figure(
        ROOT / "outputs/behavior/m3_qwen/E2__qwen25_7b__mhst__cell_heatmap_test__seed0.png",
        output_dir / "figure_3_mhst_cell_heatmap.png",
    )
    copy_figure(
        ROOT / "outputs/behavior/m3_qwen/E2__qwen25_7b__mhst__output_margin_control_test__seed0.png",
        output_dir / "figure_6_output_margin_control.png",
    )
    copy_figure(
        ROOT / "outputs/behavior/m7_qwen/E2__qwen25_7b__mhst__companion_mean_flip_plot__seed0.png",
        output_dir / "figure_5_companion_mean_flipability.png",
    )
    copy_figure(
        ROOT / "outputs/behavior/m3_llama/E3__llama31_8b__mhst__reversibility_overall_test__seed0.png",
        output_dir / "figure_9_llama_scope_curve.png",
    )
    copy_figure(
        ROOT / "outputs/behavior/m6_qwen_rcicl/E6__qwen25_7b__rcicl__reversibility_overall_test__seed0.png",
        output_dir / "figure_9_rcicl_scope_curve.png",
    )

    pair_rows = read_jsonl(ROOT / "outputs/behavior/m3_qwen/E2__qwen25_7b__mhst__pairs__seed0.jsonl")
    save_history_tax_distribution(pair_rows, output_dir / "figure_4_history_tax_distribution.png")
    save_motivating_example_panel(pair_rows, output_dir / "figure_1b_motivating_example_pair.png")

    scope_rows = [
        {
            "label": "MHST/Qwen",
            "mean": 2.7784,
            "ci_low": 2.4108,
            "ci_high": 3.1578,
        },
        {
            "label": "MHST/Llama",
            "mean": 0.9230769230769231,
            "ci_low": 0.5897435897435898,
            "ci_high": 1.2826923076923071,
        },
        {
            "label": "RC-ICL/Qwen",
            "mean": 0.5244444444444445,
            "ci_low": 0.20433333333333334,
            "ci_high": 0.840111111111111,
        },
    ]
    save_scope_bar_chart(scope_rows, output_dir / "figure_9_scope_summary_bar_chart.png")


if __name__ == "__main__":
    main()
