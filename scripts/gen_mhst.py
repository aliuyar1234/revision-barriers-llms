from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.qc import count_rows_by_split, summarize_pair_invariants
from src.data.mhst_prompts import render_mhst_prompt_samples
from src.data.mhst_worlds import generate_mhst_pairs, generate_mhst_sanity


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MHST v1 worlds and sanity prompts.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)

    pair_rows, pair_summaries = generate_mhst_pairs(config)
    sanity_rows = generate_mhst_sanity(config)

    pair_path = ROOT / config["outputs"]["pair_jsonl"]
    sanity_path = ROOT / config["outputs"]["sanity_jsonl"]
    invariant_path = ROOT / config["outputs"]["invariant_report_json"]
    sample_prompts_path = ROOT / config["outputs"]["sample_prompts_json"]

    write_jsonl(pair_path, pair_rows)
    write_jsonl(sanity_path, sanity_rows)

    invariant_summary = summarize_pair_invariants(pair_summaries)
    invariant_summary["pair_rows_by_split"] = count_rows_by_split(pair_rows)
    invariant_summary["sanity_rows_by_split"] = count_rows_by_split(sanity_rows)
    write_json(invariant_path, invariant_summary)

    sample_count = int(config["outputs"].get("sample_prompt_count", 3))
    samples = []
    for record in pair_rows[:sample_count]:
        samples.append(
            {
                "example_id": record["example_id"],
                "pair_id": record["pair_id"],
                "history_type": record["history_type"],
                "prompts": render_mhst_prompt_samples(record),
            }
        )
    for record in sanity_rows[:1]:
        samples.append(
            {
                "example_id": record["example_id"],
                "history_type": record["history_type"],
                "prompts": {
                    "prefix_prompt": render_mhst_prompt_samples(record)["prefix_prompt"],
                },
            }
        )
    write_json(sample_prompts_path, {"samples": samples})

    summary = {
        "config": str(config_path),
        "pair_rows": len(pair_rows),
        "pair_count": len(pair_summaries),
        "sanity_rows": len(sanity_rows),
        "pair_output": str(pair_path),
        "sanity_output": str(sanity_path),
        "invariant_output": str(invariant_path),
        "sample_prompts_output": str(sample_prompts_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
