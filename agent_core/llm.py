"""Shared language model: Qwen3-1.7B with thinking mode, loaded once and used by every agent."""

from __future__ import annotations

import os
import threading

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS = 1536


class LLM:
    """Wraps tokenizer and model. Loads lazily, so importing this module costs nothing."""

    def __init__(self, model_id: str | None = None, max_new_tokens: int = MAX_NEW_TOKENS):
        self.model_id = model_id or os.environ.get("ARCA_MODEL", DEFAULT_MODEL)
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()  # one generation at a time on the shared GPU

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
        self._model = model.to(device).eval()

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        """One greedy generation. ``messages`` is a list of {"role", "content"} dicts."""
        import torch

        with self._lock:
            self._load()
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            new_tokens = output[0, inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def count_tokens(self, text: str) -> int:
        """Exact count if the tokenizer is loaded, otherwise a rough estimate."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return max(1, len(text) // 4)