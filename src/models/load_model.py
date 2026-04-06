from __future__ import annotations

from typing import Any

from src.models.option_scoring import HFCausalScorer, OptionScorer, OracleScorer


def load_option_scorer(model_config: dict[str, Any]) -> OptionScorer:
    backend = model_config["backend"]
    if backend == "hf_causal_lm":
        return HFCausalScorer(
            model_name_or_path=model_config["model_name_or_path"],
            device=model_config.get("device", "auto"),
            dtype=model_config.get("dtype", "auto"),
            local_files_only=bool(model_config.get("local_files_only", True)),
        )
    if backend == "oracle":
        return OracleScorer()
    raise ValueError(f"Unsupported scoring backend: {backend}")
