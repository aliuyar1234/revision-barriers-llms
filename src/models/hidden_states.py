from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFHiddenStateExtractor:
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
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.bos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must provide a pad, eos, or bos token id for batched hidden-state extraction.")

        self.num_layers = int(self.model.config.num_hidden_layers)
        self.hidden_size = int(self.model.config.hidden_size)

    def _encode_prompt(self, prompt: str) -> list[int]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.bos_token_id is None:
            return prompt_ids
        return [self.bos_token_id, *prompt_ids]

    def prefix_boundary_index(self, prompt: str) -> int:
        prompt_ids = self._encode_prompt(prompt)
        if not prompt_ids:
            raise ValueError("Prompt produced no tokens.")
        return len(prompt_ids) - 1

    def extract_batch(self, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
        if not prompts:
            raise ValueError("At least one prompt is required for hidden-state extraction.")

        encoded = [self._encode_prompt(prompt) for prompt in prompts]
        token_indices = [len(prompt_ids) - 1 for prompt_ids in encoded]
        max_len = max(len(prompt_ids) for prompt_ids in encoded)
        batch_size = len(encoded)

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for row_index, prompt_ids in enumerate(encoded):
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            input_ids[row_index, : len(prompt_ids)] = prompt_tensor
            attention_mask[row_index, : len(prompt_ids)] = 1

        gather_index = torch.tensor(token_indices, dtype=torch.long, device=self.device)
        batch_index = torch.arange(batch_size, device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        post_block_states = []
        for layer_hidden_state in outputs.hidden_states[1:]:
            post_block_states.append(layer_hidden_state[batch_index, gather_index, :])
        stacked = torch.stack(post_block_states, dim=1)
        return stacked.detach().cpu(), token_indices


def load_hidden_state_extractor(model_config: dict[str, Any]) -> HFHiddenStateExtractor:
    backend = model_config["backend"]
    if backend != "hf_causal_lm":
        raise ValueError(f"Hidden-state extraction only supports hf_causal_lm backends, got: {backend}")
    return HFHiddenStateExtractor(
        model_name_or_path=model_config["model_name_or_path"],
        device=model_config.get("device", "auto"),
        dtype=model_config.get("dtype", "auto"),
        local_files_only=bool(model_config.get("local_files_only", True)),
    )
