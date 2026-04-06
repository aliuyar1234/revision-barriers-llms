from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.barriers import DOSES, normalize_barrier_numeric
from src.analysis.bootstrap import confidence_interval


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: sanitize_for_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(inner) for inner in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: sanitize_for_json(row.get(name)) for name in fieldnames})


def mean_flipability(barrier_value: Any) -> float:
    barrier = normalize_barrier_numeric(barrier_value)
    return sum(1.0 if barrier <= dose else 0.0 for dose in DOSES) / len(DOSES)


def build_companion_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    companion_rows: list[dict[str, Any]] = []
    for row in pair_rows:
        fresh_mean_flipability = mean_flipability(row["B_raw_fresh_numeric"])
        committed_mean_flipability = mean_flipability(row["B_raw_committed_numeric"])
        companion_rows.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "m": row["m"],
                "k": row["k"],
                "valid_pair_main": bool(row["valid_pair_main"]),
                "in_S_lead_pair": bool(row["in_S_lead_pair"]),
                "fresh_mean_flipability": fresh_mean_flipability,
                "committed_mean_flipability": committed_mean_flipability,
                "deltaMeanFlip_pair": fresh_mean_flipability - committed_mean_flipability,
            }
        )
    return companion_rows


def bootstrap_mean_delta(rows: list[dict[str, Any]], *, n_resamples: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"mean": float("nan"), "ci": {"low": float("nan"), "high": float("nan")}, "n_pairs": 0}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(rows) for _ in range(len(rows))]
        estimates.append(sum(row["deltaMeanFlip_pair"] for row in sample) / len(sample))
    mean_value = sum(row["deltaMeanFlip_pair"] for row in rows) / len(rows)
    return {
        "mean": mean_value,
        "ci": confidence_interval(estimates),
        "n_pairs": len(rows),
    }


def summarize_subset(rows: list[dict[str, Any]], *, n_resamples: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {
            "pair_count": 0,
            "mean_deltaMeanFlip_pair": float("nan"),
            "positive_count": 0,
            "zero_count": 0,
            "negative_count": 0,
            "fresh_mean_flipability": float("nan"),
            "committed_mean_flipability": float("nan"),
            "delta_bootstrap": bootstrap_mean_delta(rows, n_resamples=n_resamples, seed=seed),
        }
    return {
        "pair_count": len(rows),
        "mean_deltaMeanFlip_pair": sum(row["deltaMeanFlip_pair"] for row in rows) / len(rows),
        "positive_count": sum(1 for row in rows if row["deltaMeanFlip_pair"] > 0),
        "zero_count": sum(1 for row in rows if row["deltaMeanFlip_pair"] == 0),
        "negative_count": sum(1 for row in rows if row["deltaMeanFlip_pair"] < 0),
        "fresh_mean_flipability": sum(row["fresh_mean_flipability"] for row in rows) / len(rows),
        "committed_mean_flipability": sum(row["committed_mean_flipability"] for row in rows) / len(rows),
        "delta_bootstrap": bootstrap_mean_delta(rows, n_resamples=n_resamples, seed=seed),
    }


def save_subset_plot(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    subset_labels = [row["subset"] for row in rows]
    means = [row["mean_deltaMeanFlip_pair"] for row in rows]
    lowers = [row["mean_deltaMeanFlip_pair"] - row["ci_low"] for row in rows]
    uppers = [row["ci_high"] - row["mean_deltaMeanFlip_pair"] for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(range(len(rows)), means, color="tab:blue", alpha=0.8)
    axis.errorbar(range(len(rows)), means, yerr=[lowers, uppers], fmt="none", ecolor="black", capsize=4)
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(range(len(rows)), subset_labels)
    axis.set_ylabel("mean fresh minus committed flipability")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one bounded companion behavioral summary from saved pair artifacts.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = ROOT / config_path
        if candidate.exists():
            config_path = candidate
    config = load_yaml(config_path)

    pair_rows = read_jsonl(ROOT / config["inputs"]["pair_summary_jsonl"])
    split = str(config["run"].get("split", "test"))
    n_resamples = int(config["run"].get("bootstrap_resamples", 1000))
    seed = int(config["run"].get("bootstrap_seed", 0))
    model_label = str(config["run"]["model_label"])

    companion_rows = [row for row in build_companion_rows(pair_rows) if row["split"] == split and row["valid_pair_main"]]
    overall_rows = companion_rows
    s_lead_rows = [row for row in companion_rows if row["in_S_lead_pair"]]
    by_k_rows = {k: [row for row in companion_rows if row["k"] == k] for k in sorted({row["k"] for row in companion_rows})}

    summary = {
        "config": str(config_path),
        "model_label": model_label,
        "split": split,
        "overall": summarize_subset(overall_rows, n_resamples=n_resamples, seed=seed),
        "S_lead": summarize_subset(s_lead_rows, n_resamples=n_resamples, seed=seed + 1),
        "by_k": {
            str(k): summarize_subset(rows, n_resamples=n_resamples, seed=seed + 10 + k)
            for k, rows in by_k_rows.items()
        },
    }

    table_rows = []
    subset_specs = [
        ("overall", summary["overall"]),
        ("S_lead", summary["S_lead"]),
    ] + [(f"k={k}", summary["by_k"][str(k)]) for k in sorted(by_k_rows)]
    for subset_name, subset_summary in subset_specs:
        table_rows.append(
            {
                "subset": subset_name,
                "pair_count": subset_summary["pair_count"],
                "mean_deltaMeanFlip_pair": subset_summary["mean_deltaMeanFlip_pair"],
                "ci_low": subset_summary["delta_bootstrap"]["ci"]["low"],
                "ci_high": subset_summary["delta_bootstrap"]["ci"]["high"],
                "positive_count": subset_summary["positive_count"],
                "zero_count": subset_summary["zero_count"],
                "negative_count": subset_summary["negative_count"],
                "fresh_mean_flipability": subset_summary["fresh_mean_flipability"],
                "committed_mean_flipability": subset_summary["committed_mean_flipability"],
            }
        )

    write_json(ROOT / config["outputs"]["summary_json"], summary)
    write_csv(
        ROOT / config["outputs"]["summary_table_csv"],
        table_rows,
        [
            "subset",
            "pair_count",
            "mean_deltaMeanFlip_pair",
            "ci_low",
            "ci_high",
            "positive_count",
            "zero_count",
            "negative_count",
            "fresh_mean_flipability",
            "committed_mean_flipability",
        ],
    )
    save_subset_plot(
        table_rows,
        ROOT / config["outputs"]["summary_plot_png"],
        f"{model_label} {split} companion summary (mean flipability)",
    )

    print(json.dumps(sanitize_for_json(summary), indent=2))


if __name__ == "__main__":
    main()
