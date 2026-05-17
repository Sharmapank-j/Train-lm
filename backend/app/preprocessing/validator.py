"""Dataset format validation and JSONL/Alpaca/ShareGPT parsing."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_FORMATS = {"jsonl", "alpaca", "sharegpt", "openai", "plain"}


@dataclass
class ValidationReport:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    duplicates: int = 0
    empty: int = 0
    estimated_tokens: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "duplicates": self.duplicates,
            "empty": self.empty,
            "estimated_tokens": self.estimated_tokens,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _rough_token_estimate(text: str) -> int:
    """~4 chars per token heuristic."""
    return max(1, len(text) // 4)


def validate_jsonl(content: bytes) -> ValidationReport:
    report = ValidationReport()
    seen: set[str] = set()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        report.errors.append(f"File is not valid UTF-8: {exc}")
        return report

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        report.total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            report.invalid += 1
            report.warnings.append(f"Line {lineno}: malformed JSON")
            continue

        if not isinstance(record, dict):
            report.invalid += 1
            report.warnings.append(f"Line {lineno}: expected JSON object, got {type(record).__name__}")
            continue

        record_str = json.dumps(record, sort_keys=True)
        if record_str in seen:
            report.duplicates += 1
            report.warnings.append(f"Line {lineno}: duplicate record")
            continue
        seen.add(record_str)

        # Check for common formats
        if not _has_content(record):
            report.empty += 1
            report.invalid += 1
            report.warnings.append(f"Line {lineno}: record appears empty or missing required fields")
            continue

        report.valid += 1
        report.estimated_tokens += _rough_token_estimate(record_str)

    if report.total == 0:
        report.errors.append("Dataset contains no records")
    elif report.valid == 0:
        report.errors.append(
            f"Dataset has no valid records ({report.invalid} invalid, "
            f"{report.duplicates} duplicates, {report.empty} empty)"
        )
    return report


def _has_content(record: dict[str, Any]) -> bool:
    """Check if a record has meaningful content in any supported schema."""
    # Alpaca / instruction-response
    if "instruction" in record:
        return bool(str(record.get("instruction", "")).strip())
    # OpenAI / ShareGPT messages
    if "messages" in record:
        msgs = record.get("messages", [])
        return isinstance(msgs, list) and len(msgs) > 0
    if "conversations" in record:
        convs = record.get("conversations", [])
        return isinstance(convs, list) and len(convs) > 0
    # Plain text
    if "text" in record:
        return bool(str(record.get("text", "")).strip())
    # Prompt/completion
    if "prompt" in record or "completion" in record:
        return True
    return False


def checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


__all__ = ["validate_jsonl", "ValidationReport", "checksum", "SUPPORTED_FORMATS"]
