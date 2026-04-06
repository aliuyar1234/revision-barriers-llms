from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mhst_prompts import render_mhst_intervention_prefix
from src.models.hidden_states import load_hidden_state_extractor


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(sanitize(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def filter_rows(rows: list[dict[str, Any]], run_config: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_splits = set(run_config.get("include_splits", []))
    allowed_margins = set(run_config.get("include_margins", []))
    allowed_history_depths = set(run_config.get("include_history_depths", []))

    filtered = rows
    if allowed_splits:
        filtered = [row for row in filtered if row["split"] in allowed_splits]
    if allowed_margins:
        filtered = [row for row in filtered if row["m"] in allowed_margins]
    if allowed_history_depths:
        filtered = [row for row in filtered if row["k"] in allowed_history_depths]

    max_examples = run_config.get("max_examples")
    if max_examples is not None:
        filtered = filtered[: int(max_examples)]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract prefix-boundary hidden states for MHST examples.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)

    pair_rows = filter_rows(read_jsonl(ROOT / config["inputs"]["pair_jsonl"]), config["run"])
    behavior_rows = read_jsonl(ROOT / config["inputs"]["behavior_rows_jsonl"])
    behavior_by_example_id = {row["example_id"]: row for row in behavior_rows}

    missing_behavior = [row["example_id"] for row in pair_rows if row["example_id"] not in behavior_by_example_id]
    if missing_behavior:
        raise ValueError(
            f"Missing behavior rows for {len(missing_behavior)} selected examples; the state extractor requires behavior metadata coverage."
        )

    extractor = load_hidden_state_extractor(config["model"])
    storage_dtype = np.dtype(config["run"].get("storage_dtype", "float16"))
    batch_size = int(config["run"].get("batch_size", 4))

    state_array_path = ROOT / config["outputs"]["state_array_npy"]
    state_array_path.parent.mkdir(parents=True, exist_ok=True)
    state_array = np.lib.format.open_memmap(
        state_array_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(len(pair_rows), extractor.num_layers, extractor.hidden_size),
    )

    metadata_rows: list[dict[str, Any]] = []
    for start_index in range(0, len(pair_rows), batch_size):
        batch_records = pair_rows[start_index : start_index + batch_size]
        prompts = [render_mhst_intervention_prefix(record) for record in batch_records]
        states_cpu, token_indices = extractor.extract_batch(prompts)
        state_array[start_index : start_index + len(batch_records)] = states_cpu.numpy().astype(storage_dtype)

        for offset, record in enumerate(batch_records):
            behavior_row = behavior_by_example_id[record["example_id"]]
            metadata_rows.append(
                {
                    "example_id": record["example_id"],
                    "pair_id": record["pair_id"],
                    "split": record["split"],
                    "history_type": record["history_type"],
                    "m": record["m"],
                    "k": record["k"],
                    "valid_main": behavior_row["valid_main"],
                    "B_raw": behavior_row["B_raw"],
                    "B_cap": behavior_row["B_cap"],
                    "prefix_leader": behavior_row["prefix_leader"],
                    "o": behavior_row["o"],
                    "t_star_token_index": token_indices[offset],
                    "prompt_token_count": token_indices[offset] + 1,
                }
            )
    state_array.flush()

    write_jsonl(ROOT / config["outputs"]["state_rows_jsonl"], metadata_rows)

    manifest = {
        "run_id": config["run"]["run_id"],
        "config": str(config_path),
        "model_name": config["model"]["model_name_or_path"],
        "model_revision": config["model"].get("model_name_or_path"),
        "tokenizer_revision": config["model"].get("model_name_or_path"),
        "seed": int(config["run"].get("seed", 0)),
        "n_examples": len(metadata_rows),
        "n_layers": extractor.num_layers,
        "d_model": extractor.hidden_size,
        "state_dtype": str(storage_dtype),
        "state_array_npy": config["outputs"]["state_array_npy"],
        "state_rows_jsonl": config["outputs"]["state_rows_jsonl"],
        "compatibility_fingerprint": sha256_json(
            {
                "model": config["model"],
                "inputs": config["inputs"],
                "run": config["run"],
                "selected_example_ids": [row["example_id"] for row in pair_rows],
            }
        ),
    }
    write_json(ROOT / config["outputs"]["state_manifest_json"], manifest)

    print(
        json.dumps(
            {
                "config": str(config_path),
                "run_id": config["run"]["run_id"],
                "n_examples": len(metadata_rows),
                "n_layers": extractor.num_layers,
                "d_model": extractor.hidden_size,
                "state_array_npy": config["outputs"]["state_array_npy"],
                "state_rows_jsonl": config["outputs"]["state_rows_jsonl"],
                "state_manifest_json": config["outputs"]["state_manifest_json"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
