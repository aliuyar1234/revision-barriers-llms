from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.analysis.qc import determine_prefix_leader


@dataclass(frozen=True)
class ResidualIntervention:
    layer_index: int
    token_index: int
    direction: torch.Tensor
    alpha: float


@dataclass
class QwenResidualInterventionScorer:
    model_name_or_path: str
    device: str = "cpu"
    dtype: str = "auto"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        torch_dtype = None
        if self.dtype != "auto":
            torch_dtype = getattr(torch, self.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            local_files_only=self.local_files_only,
        )
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.bos_token_id = self.tokenizer.bos_token_id
        self.model_revision = getattr(getattr(self.model, "config", None), "_name_or_path", None)
        self.tokenizer_revision = getattr(self.tokenizer, "name_or_path", None)

        if "qwen" not in str(getattr(self.model.config, "model_type", "")).lower():
            raise ValueError(
                f"QwenResidualInterventionScorer expects a Qwen-family model, got model_type={self.model.config.model_type!r}."
            )

        try:
            self.layers = self.model.model.layers
        except AttributeError as error:
            raise ValueError("Qwen model does not expose model.layers as expected for the narrow v1 hook path.") from error

    def _encode_prompt(self, prompt: str) -> list[int]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.bos_token_id is None:
            return prompt_ids
        return [self.bos_token_id, *prompt_ids]

    def prefix_token_index(self, prompt: str) -> int | None:
        prompt_ids = self._encode_prompt(prompt)
        if not prompt_ids:
            return None
        return len(prompt_ids) - 1

    @contextmanager
    def _install_intervention(self, intervention: ResidualIntervention | None) -> Iterator[None]:
        if intervention is None or intervention.alpha == 0.0:
            yield
            return

        if intervention.layer_index < 0 or intervention.layer_index >= len(self.layers):
            raise ValueError(f"Invalid layer_index for intervention: {intervention.layer_index}")

        direction = intervention.direction.to(device=self.device)

        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden_states = output[0] if isinstance(output, tuple) else output
            if intervention.token_index < 0 or intervention.token_index >= hidden_states.shape[1]:
                raise ValueError(
                    f"Intervention token_index {intervention.token_index} is out of range for sequence length {hidden_states.shape[1]}."
                )
            updated = hidden_states.clone()
            updated[:, intervention.token_index, :] = updated[:, intervention.token_index, :] + (
                intervention.alpha * direction.to(dtype=updated.dtype)
            )
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated

        handle = self.layers[intervention.layer_index].register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    def score_options(
        self,
        prompt: str,
        answer_texts: list[str],
        *,
        intervention: ResidualIntervention | None = None,
    ) -> list[float]:
        prompt_ids = self._encode_prompt(prompt)
        scores: list[float] = []

        for answer_text in answer_texts:
            answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            if not answer_ids:
                raise ValueError("Canonical answer tokenization produced an empty answer sequence.")

            input_ids = prompt_ids + answer_ids
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            with self._install_intervention(intervention):
                with torch.no_grad():
                    logits = self.model(input_ids=input_tensor, use_cache=False).logits[0]

            total = 0.0
            for offset, token_id in enumerate(answer_ids):
                position = len(prompt_ids) + offset
                token_logprobs = torch.log_softmax(logits[position - 1], dim=-1)
                total += float(token_logprobs[token_id].item())
            scores.append(total)
        return scores

    def score_options_with_metrics(
        self,
        prompt: str,
        answer_texts: list[str],
        *,
        intervention: ResidualIntervention | None = None,
    ) -> dict[str, Any]:
        scores = self.score_options(prompt, answer_texts, intervention=intervention)
        score_tensor = torch.tensor(scores, dtype=torch.float64)
        probs = torch.softmax(score_tensor, dim=0)
        entropy = float(-(probs * torch.log(probs)).sum().item())
        labels = [answer_text.strip() for answer_text in answer_texts]
        score_map = {label: float(score) for label, score in zip(labels, scores, strict=True)}
        predicted_label, has_top_tie = determine_prefix_leader(score_map)
        return {
            "scores": scores,
            "score_map": score_map,
            "probabilities": {label: float(prob) for label, prob in zip(labels, probs.tolist(), strict=True)},
            "entropy": entropy,
            "predicted_label": predicted_label,
            "has_top_tie": has_top_tie,
        }


def load_qwen_residual_intervention_scorer(model_config: dict[str, Any]) -> QwenResidualInterventionScorer:
    if model_config["backend"] != "hf_causal_lm":
        raise ValueError(
            f"Residual intervention scoring only supports hf_causal_lm backends, got: {model_config['backend']}"
        )
    return QwenResidualInterventionScorer(
        model_name_or_path=model_config["model_name_or_path"],
        device=model_config.get("device", "auto"),
        dtype=model_config.get("dtype", "auto"),
        local_files_only=bool(model_config.get("local_files_only", True)),
    )
