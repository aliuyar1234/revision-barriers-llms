from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class OptionScorer(Protocol):
    def score_options(
        self,
        prompt: str,
        answer_texts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> list[float]:
        ...

    def prefix_token_index(self, prompt: str) -> int | None:
        ...


@dataclass
class HFCausalScorer:
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

    def score_options(
        self,
        prompt: str,
        answer_texts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> list[float]:
        del metadata
        prompt_ids = self._encode_prompt(prompt)
        scores: list[float] = []
        for answer_text in answer_texts:
            answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            if not answer_ids:
                raise ValueError("Canonical answer tokenization produced an empty answer sequence.")

            input_ids = prompt_ids + answer_ids
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_tensor).logits[0]

            total = 0.0
            for offset, token_id in enumerate(answer_ids):
                position = len(prompt_ids) + offset
                token_logprobs = torch.log_softmax(logits[position - 1], dim=-1)
                total += float(token_logprobs[token_id].item())
            scores.append(total)
        return scores


class OracleScorer:
    def score_options(
        self,
        prompt: str,
        answer_texts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> list[float]:
        del prompt, answer_texts
        if metadata is None or "oracle_scores" not in metadata:
            raise ValueError("Oracle scoring requires oracle_scores metadata.")
        return [float(score) for score in metadata["oracle_scores"]]

    def prefix_token_index(self, prompt: str) -> int | None:
        del prompt
        return None
