"""Preprocessing pipeline: clean, deduplicate, filter JSONL datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.preprocessing.validator import _rough_token_estimate


@dataclass
class PreprocessConfig:
    strip_whitespace: bool = True
    deduplicate: bool = True
    max_tokens: int = 4096
    min_tokens: int = 4
    normalize_roles: bool = True


@dataclass
class PreprocessResult:
    input_count: int = 0
    output_count: int = 0
    removed_duplicates: int = 0
    removed_too_long: int = 0
    removed_too_short: int = 0
    removed_malformed: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "removed_duplicates": self.removed_duplicates,
            "removed_too_long": self.removed_too_long,
            "removed_too_short": self.removed_too_short,
            "removed_malformed": self.removed_malformed,
            "warnings": self.warnings,
        }


_ROLE_ALIASES: dict[str, str] = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}


def _normalize_record(record: dict[str, Any], cfg: PreprocessConfig) -> dict[str, Any]:
    """Return a cleaned record."""
    # Alpaca-style
    if "instruction" in record and cfg.strip_whitespace:
        for key in ("instruction", "input", "output"):
            if key in record and isinstance(record[key], str):
                record[key] = record[key].strip()

    # Messages style (OpenAI / ShareGPT)
    for key in ("messages", "conversations"):
        if key in record and isinstance(record[key], list):
            cleaned = []
            for msg in record[key]:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role") or msg.get("from") or ""
                content = msg.get("content") or msg.get("value") or ""
                if cfg.strip_whitespace:
                    content = content.strip()
                if cfg.normalize_roles:
                    role = _ROLE_ALIASES.get(role.lower(), role)
                cleaned.append({"role": role, "content": content})
            record[key] = cleaned

    return record


def preprocess_jsonl(content: bytes, cfg: PreprocessConfig | None = None) -> tuple[bytes, PreprocessResult]:
    """Clean and filter a JSONL dataset.  Returns (processed_bytes, result)."""
    if cfg is None:
        cfg = PreprocessConfig()

    result = PreprocessResult()
    seen: set[str] = set()
    out_lines: list[str] = []

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        result.removed_malformed += 1
        result.warnings.append("File is not valid UTF-8 — cannot preprocess")
        return b"", result

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        result.input_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            result.removed_malformed += 1
            continue
        if not isinstance(record, dict):
            result.removed_malformed += 1
            continue

        record = _normalize_record(record, cfg)
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
        tok_est = _rough_token_estimate(serialized)

        if tok_est > cfg.max_tokens:
            result.removed_too_long += 1
            continue
        if tok_est < cfg.min_tokens:
            result.removed_too_short += 1
            continue
        if cfg.deduplicate:
            if serialized in seen:
                result.removed_duplicates += 1
                continue
            seen.add(serialized)

        out_lines.append(json.dumps(record, ensure_ascii=False))
        result.output_count += 1

    return "\n".join(out_lines).encode("utf-8"), result


__all__ = ["PreprocessConfig", "PreprocessResult", "preprocess_jsonl"]
