from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from app.config.settings import settings


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str


class InferenceManager:
    def __init__(self) -> None:
        self._cache: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def _select_device(self) -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, cache_key: str, base_model: str, adapter_path: str | None = None) -> LoadedModel:
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install ML stack: pip install torch transformers peft") from exc

        local_only = not settings.allow_remote_models
        base_model_path = base_model
        if local_only:
            from pathlib import Path
            from app.utils.paths import safe_join

            root = settings.model_root.resolve()
            candidate = Path(base_model)
            if candidate.is_absolute():
                resolved = candidate.resolve()
                if resolved == root or root in resolved.parents:
                    base_model_path = str(resolved)
            else:
                base_model_path = str(safe_join(root, base_model))
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=local_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = self._select_device()
        model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=local_only)

        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.to(device)
        model.eval()

        loaded = LoadedModel(model=model, tokenizer=tokenizer, device=device)
        with self._lock:
            self._cache[cache_key] = loaded
        return loaded

    def generate(
        self,
        cache_key: str,
        base_model: str,
        adapter_path: str | None,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        loaded = self.load(cache_key, base_model, adapter_path)
        tokenizer = loaded.tokenizer
        model = loaded.model
        device = loaded.device

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = temperature > 0
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output[0][input_len:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).lstrip()


_manager = InferenceManager()


def get_inference_manager() -> InferenceManager:
    return _manager


__all__ = ["InferenceManager", "get_inference_manager"]
