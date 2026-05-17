"""Transformer architecture configuration for from-scratch LLM creation.

Supports Llama-style (RoPE, SwiGLU, GQA), GPT-2-style, and custom configs.
Architecture definitions map directly onto HuggingFace ``LlamaConfig`` /
``MistralConfig`` / ``GPT2Config`` so they can be instantiated with
``AutoModelForCausalLM.from_config(arch.to_hf_config())``.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Architecture types
# ---------------------------------------------------------------------------

ArchType = Literal["llama", "mistral", "gpt2", "custom"]


class ArchitectureConfig(BaseModel):
    """Full transformer architecture specification for from-scratch pre-training."""

    # Identity
    arch_type: ArchType = "llama"
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable model name")
    description: str = ""

    # Vocabulary
    vocab_size: int = Field(32000, ge=256, le=256_000, description="Vocabulary size")

    # Dimensions
    hidden_size: int = Field(2048, ge=64, le=65536, description="Model hidden dimension")
    intermediate_size: int = Field(5504, ge=64, le=262144, description="FFN intermediate dimension")
    num_hidden_layers: int = Field(22, ge=1, le=256, description="Number of transformer layers")
    num_attention_heads: int = Field(32, ge=1, le=256, description="Number of attention heads")
    num_key_value_heads: int = Field(4, ge=1, le=256,
                                     description="GQA key/value heads (set equal to num_attention_heads to disable GQA)")

    # Context
    max_position_embeddings: int = Field(4096, ge=64, le=1_048_576, description="Maximum context length")
    rope_theta: float = Field(10000.0, gt=0, description="RoPE base frequency")

    # Norms / activations
    rms_norm_eps: float = Field(1e-5, gt=0, description="RMSNorm epsilon")
    hidden_act: str = Field("silu", description="Activation function (silu, gelu, relu, gelu_new)")

    # Regularisation
    attention_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    hidden_dropout: float = Field(0.0, ge=0.0, lt=1.0)

    # Architecture flags
    tie_word_embeddings: bool = False
    use_cache: bool = True

    # Extra pass-through kwargs for HuggingFace configs
    extra_hf_kwargs: dict[str, Any] = Field(default_factory=dict, description="Additional HuggingFace config kwargs")

    @model_validator(mode="after")
    def check_head_divisibility(self) -> "ArchitectureConfig":
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot exceed num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def parameter_estimate(self) -> int:
        """Rough parameter count estimate (excludes embeddings for simplicity)."""
        attn = 4 * self.hidden_size * self.hidden_size  # Q K V O projections (approx)
        ffn = 3 * self.hidden_size * self.intermediate_size  # gate, up, down
        layer = attn + ffn
        return int(self.num_hidden_layers * layer + self.vocab_size * self.hidden_size * (1 + (not self.tie_word_embeddings)))

    def to_hf_config(self):
        """Convert to a HuggingFace config object (lazy import — no torch required at parse time)."""
        if self.arch_type in ("llama", "mistral"):
            from transformers import LlamaConfig
            return LlamaConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                max_position_embeddings=self.max_position_embeddings,
                rope_theta=self.rope_theta,
                rms_norm_eps=self.rms_norm_eps,
                hidden_act=self.hidden_act,
                attention_dropout=self.attention_dropout,
                tie_word_embeddings=self.tie_word_embeddings,
                use_cache=self.use_cache,
                **self.extra_hf_kwargs,
            )
        if self.arch_type == "gpt2":
            from transformers import GPT2Config
            n_inner = self.intermediate_size or 4 * self.hidden_size
            return GPT2Config(
                vocab_size=self.vocab_size,
                n_embd=self.hidden_size,
                n_layer=self.num_hidden_layers,
                n_head=self.num_attention_heads,
                n_inner=n_inner,
                n_positions=self.max_position_embeddings,
                activation_function=self.hidden_act,
                attn_pdrop=self.attention_dropout,
                resid_pdrop=self.hidden_dropout,
                tie_word_embeddings=self.tie_word_embeddings,
                **self.extra_hf_kwargs,
            )
        raise ValueError(f"Unsupported arch_type for HuggingFace conversion: {self.arch_type}")

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d["parameter_estimate"] = self.parameter_estimate
        d["head_dim"] = self.head_dim
        return d


# ---------------------------------------------------------------------------
# Built-in size presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    # ~15 M params — suitable for CI tests and dev on any laptop
    "TinyLM-15M": dict(
        arch_type="llama",
        name="TinyLM-15M",
        description="15M-parameter model — fast iteration, CI, Termux/mobile",
        vocab_size=8192,
        hidden_size=256,
        intermediate_size=688,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    ),
    # ~125 M params — GPT-2 scale, workable on CPU
    "SmallLM-125M": dict(
        arch_type="llama",
        name="SmallLM-125M",
        description="125M-parameter model — comparable to GPT-2 small, runs on CPU",
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    ),
    # ~360 M params — edge GPU / good CPU
    "MediumLM-360M": dict(
        arch_type="llama",
        name="MediumLM-360M",
        description="360M-parameter model — GPU-recommended, capable on good CPU",
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
    ),
    # ~1.1 B params — 1B-class model, requires GPU or quantised CPU run
    "LargeLM-1B": dict(
        arch_type="llama",
        name="LargeLM-1B",
        description="1.1B-parameter model — comparable to TinyLlama, GPU recommended",
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        max_position_embeddings=4096,
    ),
    # ~3 B params — multi-GPU or high-end single GPU
    "XLargeLM-3B": dict(
        arch_type="llama",
        name="XLargeLM-3B",
        description="3B-parameter model — multi-GPU or high-VRAM single GPU",
        vocab_size=32000,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=24,
        num_key_value_heads=8,
        max_position_embeddings=4096,
    ),
}


def get_preset(name: str) -> ArchitectureConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return ArchitectureConfig(**PRESETS[name])


__all__ = ["ArchitectureConfig", "ArchType", "PRESETS", "get_preset"]
