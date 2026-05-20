"""LoRA / QLoRA fine-tuning engine."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("train_lm.finetune")


@dataclass
class FinetuneConfig:
    run_name: str = "finetune-run"
    base_model: str = ""
    dataset_path: str = ""
    output_dir: str = "checkpoints/finetune"
    method: str = "lora"  # lora or qlora
    quantization_type: str = "4bit"  # none, 4bit, 8bit (qlora only)
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    allow_remote_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _prepare_text(example: dict[str, Any]) -> dict[str, str]:
    if isinstance(example.get("messages"), list):
        lines: list[str] = []
        for msg in example["messages"]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                lines.append(f"{role.title()}: {content}")
        return {"text": "\n".join(lines)}
    if "instruction" in example and "output" in example:
        return {"text": f"Instruction: {example.get('instruction','')}\nResponse: {example.get('output','')}"}
    if "prompt" in example and "completion" in example:
        return {"text": f"{example.get('prompt','')}{example.get('completion','')}"}
    if "text" in example:
        return {"text": str(example.get("text", ""))}
    joined = " ".join(str(v) for v in example.values() if isinstance(v, str))
    return {"text": joined}


def _build_dataset(path: str, tokenizer, max_seq_length: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install 'datasets': pip install datasets") from exc

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = dataset.map(_prepare_text, remove_columns=dataset.column_names)

    def _tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def run_finetune(
    cfg: FinetuneConfig,
    progress_callback: Callable[[float, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("Install ML stack: pip install torch transformers peft bitsandbytes") from exc

    set_seed(cfg.seed)

    quantization_config = None
    if cfg.method == "qlora":
        if cfg.quantization_type == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif cfg.quantization_type == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("QLoRA requires quantization_type to be 4bit or 8bit")

    local_only = not cfg.allow_remote_model
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=local_only,
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if cfg.method == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = _build_dataset(cfg.dataset_path, tokenizer, cfg.max_seq_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(cfg.output_dir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=cfg.run_name,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        lr_scheduler_type=cfg.scheduler,
        optim=cfg.optimizer,
        report_to=[],
        save_total_limit=2,
        fp16=cfg.mixed_precision,
        bf16=False,
        seed=cfg.seed,
    )

    total_steps = max(1, len(dataset) // max(1, cfg.batch_size * cfg.gradient_accumulation_steps))
    metrics_history: list[dict[str, Any]] = []

    class _ProgressCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            metrics_history.append({**logs, "step": state.global_step})
            if progress_callback:
                fraction = min(1.0, state.global_step / total_steps)
                progress_callback(fraction, logs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[_ProgressCB()],
    )

    train_result = trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    return {
        "run_name": cfg.run_name,
        "output_dir": str(output_dir),
        "adapter_path": str(adapter_dir),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_loss": train_result.metrics.get("train_loss"),
        "metrics_history": metrics_history,
    }


__all__ = ["FinetuneConfig", "run_finetune"]
