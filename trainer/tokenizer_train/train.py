"""Train a custom BPE, Unigram, or WordPiece tokenizer from a text corpus.

Uses the HuggingFace ``tokenizers`` library (Rust-backed, no Python GIL).
The trained tokenizer is saved in the standard HuggingFace format so it can
be loaded with ``transformers.AutoTokenizer.from_pretrained(path)``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal

logger = logging.getLogger("train_lm.tokenizer_train")

TokenizerAlgorithm = Literal["bpe", "unigram", "wordpiece"]

_DEFAULT_SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TokenizerTrainConfig:
    algorithm: TokenizerAlgorithm = "bpe"
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: list[str] = field(default_factory=lambda: list(_DEFAULT_SPECIAL_TOKENS))
    add_prefix_space: bool = True
    lowercase: bool = False
    byte_level: bool = True  # BPE only — use byte-level pre-tokeniser (like GPT-2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "add_prefix_space": self.add_prefix_space,
            "lowercase": self.lowercase,
            "byte_level": self.byte_level,
        }


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _iter_lines(corpus_path: Path, batch_size: int = 1000) -> Generator[list[str], None, None]:
    """Yield lines from a text file in batches."""
    batch: list[str] = []
    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(
    corpus_path: Path,
    output_dir: Path,
    cfg: TokenizerTrainConfig | None = None,
) -> dict[str, Any]:
    """Train a tokenizer from *corpus_path* and save it to *output_dir*.

    Returns a metadata dict with vocab_size, algorithm, output_dir, etc.

    Raises ``ImportError`` if the ``tokenizers`` package is not installed.
    """
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers, processors
    except ImportError as exc:
        raise ImportError(
            "The 'tokenizers' package is required for tokenizer training. "
            "Install it with: pip install tokenizers"
        ) from exc

    if cfg is None:
        cfg = TokenizerTrainConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("tokenizer.train.start", extra={"algorithm": cfg.algorithm, "vocab_size": cfg.vocab_size})

    # ----------------------------------------------------------------
    # Build tokenizer by algorithm
    # ----------------------------------------------------------------
    if cfg.algorithm == "bpe":
        if cfg.byte_level:
            tok = Tokenizer(models.BPE(unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=cfg.add_prefix_space)
            tok.decoder = decoders.ByteLevel()
            tok.post_processor = processors.ByteLevel(trim_offsets=True)
        else:
            tok = Tokenizer(models.BPE(unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=cfg.vocab_size,
            min_frequency=cfg.min_frequency,
            special_tokens=cfg.special_tokens,
            show_progress=False,
        )

    elif cfg.algorithm == "unigram":
        tok = Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.Metaspace(add_prefix_space=cfg.add_prefix_space)
        trainer = trainers.UnigramTrainer(
            vocab_size=cfg.vocab_size,
            special_tokens=cfg.special_tokens,
            unk_token="<unk>",
        )

    elif cfg.algorithm == "wordpiece":
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        if cfg.lowercase:
            tok.normalizer = normalizers.BertNormalizer(lowercase=True)
        tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tok.decoder = decoders.WordPiece()
        trainer = trainers.WordPieceTrainer(
            vocab_size=cfg.vocab_size,
            min_frequency=cfg.min_frequency,
            special_tokens=cfg.special_tokens,
        )

    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm!r}")

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    def _batch_generator():
        for batch in _iter_lines(corpus_path):
            yield batch

    tok.train_from_iterator(_batch_generator(), trainer=trainer)
    actual_vocab_size = tok.get_vocab_size()

    # ----------------------------------------------------------------
    # Save in HuggingFace fast-tokenizer format
    # ----------------------------------------------------------------
    tokenizer_path = output_dir / "tokenizer.json"
    tok.save(str(tokenizer_path))

    # Write a minimal tokenizer_config.json so AutoTokenizer can load it
    import json
    tok_config = {
        "model_type": "llama",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=2), encoding="utf-8")

    logger.info("tokenizer.train.done", extra={"vocab_size": actual_vocab_size, "output_dir": str(output_dir)})

    return {
        "algorithm": cfg.algorithm,
        "vocab_size": actual_vocab_size,
        "output_dir": str(output_dir),
        "tokenizer_json": str(tokenizer_path),
        "special_tokens": cfg.special_tokens,
    }


__all__ = ["TokenizerTrainConfig", "TokenizerAlgorithm", "train_tokenizer"]
