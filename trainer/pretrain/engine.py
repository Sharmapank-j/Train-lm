"""From-scratch LLM pre-training engine.

Initialises a new model from an ``ArchitectureConfig``, trains it with
next-token prediction (causal language modelling) on a plain-text or JSONL
corpus, and saves checkpoints + metrics.

Design decisions
----------------
* Works fully offline after initial import — no internet calls.
* Defaults to CPU, uses GPU / MPS when ``torch.cuda.is_available()`` /
  ``torch.backends.mps.is_available()``.
* Uses HuggingFace ``Trainer`` for gradient accumulation, mixed-precision,
  checkpointing, and metrics out of the box.
* Users can also call ``run_pretrain_loop()`` directly as a background job.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("train_lm.pretrain")


# ---------------------------------------------------------------------------
# Pre-training config
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    # Identity
    run_name: str = "pretrain-run"
    notes: str = ""

    # Architecture (can be a preset name or a full dict)
    architecture: dict[str, Any] = field(default_factory=dict)

    # Tokenizer
    tokenizer_path: str = ""          # path to trained tokenizer dir
    tokenizer_preset: str = ""        # use a preset vocab size if no trained tokenizer

    # Data
    corpus_path: str = ""             # path to plain-text or JSONL corpus
    max_seq_length: int = 2048

    # Training hyper-parameters
    learning_rate: float = 3e-4
    batch_size: int = 4               # per-device batch size
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 200
    max_steps: int = -1               # -1 → use num_train_epochs
    num_train_epochs: int = 1
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # Checkpointing
    output_dir: str = "checkpoints/pretrain"
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 500
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _build_dataset(corpus_path: str, tokenizer, max_seq_length: int):
    """Return a tokenised HuggingFace dataset for causal LM pre-training."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install 'datasets': pip install datasets") from exc

    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    if path.suffix == ".jsonl":
        raw = load_dataset("json", data_files=str(path), split="train")
        # Expect a "text" field; fall back to joining all string values
        if "text" not in raw.column_names:
            def _join(ex):
                return {"text": " ".join(str(v) for v in ex.values() if isinstance(v, str))}
            raw = raw.map(_join)
    else:
        raw = load_dataset("text", data_files=str(path), split="train")

    def _tokenize(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = raw.map(_tokenize, batched=True, remove_columns=raw.column_names)
    return tokenized


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def run_pretrain(
    cfg: PretrainConfig,
    progress_callback: Callable[[float, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train an LLM from scratch.  Returns a metadata dict.

    This is designed to be called inside a background job coroutine.
    ``progress_callback(fraction, metrics)`` is invoked after each logging step
    if provided.
    """
    # -----------------------------------------------------------------
    # Lazy imports — keep cold-start fast for non-ML API requests
    # -----------------------------------------------------------------
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            TrainerCallback,
            TrainingArguments,
            Trainer,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError("Install ML stack: pip install torch transformers") from exc

    from trainer.architecture.config import ArchitectureConfig

    set_seed(cfg.seed)

    # -----------------------------------------------------------------
    # Resolve architecture
    # -----------------------------------------------------------------
    if isinstance(cfg.architecture, str):
        from trainer.architecture.config import get_preset
        arch = get_preset(cfg.architecture)
    elif isinstance(cfg.architecture, dict) and cfg.architecture:
        arch = ArchitectureConfig(**cfg.architecture)
    else:
        raise ValueError("PretrainConfig.architecture must be a preset name or an ArchitectureConfig dict")

    logger.info("pretrain.start", extra={
        "run_name": cfg.run_name,
        "arch": arch.name,
        "param_est": arch.parameter_estimate,
    })

    # -----------------------------------------------------------------
    # Device selection
    # -----------------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("pretrain.device", extra={"device": device})

    # -----------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------
    if cfg.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    else:
        # Fallback: use a small pre-existing tokenizer for quick experiments
        from transformers import LlamaTokenizerFast
        logger.warning("pretrain.no_tokenizer — using fallback; train a custom tokenizer first")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        # Override vocab size on the architecture
        arch = ArchitectureConfig(**{**arch.model_dump(), "vocab_size": tokenizer.vocab_size})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------
    # Model (random initialisation)
    # -----------------------------------------------------------------
    hf_cfg = arch.to_hf_config()
    model = AutoModelForCausalLM.from_config(hf_cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("pretrain.model_init", extra={"total_params": total_params})

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    train_dataset = _build_dataset(cfg.corpus_path, tokenizer, cfg.max_seq_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # -----------------------------------------------------------------
    # Training arguments
    # -----------------------------------------------------------------
    output_dir = Path(cfg.output_dir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=cfg.run_name,
        num_train_epochs=cfg.num_train_epochs if cfg.max_steps == -1 else 1,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16 and device == "cuda",
        bf16=cfg.bf16 and device == "cuda",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="no",
        seed=cfg.seed,
        report_to=[],  # No external telemetry — offline-first
        save_total_limit=3,
        load_best_model_at_end=False,
        dataloader_num_workers=0,  # Safe default for all platforms incl. Termux
        no_cuda=(device == "cpu"),
        use_mps_device=(device == "mps"),
    )

    # -----------------------------------------------------------------
    # Progress callback
    # -----------------------------------------------------------------
    total_steps: list[int] = [max(1, len(train_dataset) // max(1, cfg.batch_size * cfg.gradient_accumulation_steps))]
    metrics_history: list[dict[str, Any]] = []

    class _ProgressCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            metrics_history.append({**logs, "step": state.global_step})
            if progress_callback:
                fraction = min(1.0, state.global_step / total_steps[0])
                progress_callback(fraction, {**logs})

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[_ProgressCB()],
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    logger.info("pretrain.done", extra={"run_name": cfg.run_name, "output_dir": str(output_dir)})

    return {
        "run_name": cfg.run_name,
        "output_dir": str(output_dir),
        "final_model_path": str(output_dir / "final"),
        "total_params": total_params,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_loss": train_result.metrics.get("train_loss", None),
        "metrics_history": metrics_history,
        "architecture": arch.to_dict(),
    }


__all__ = ["PretrainConfig", "run_pretrain"]
