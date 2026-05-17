from __future__ import annotations

from datetime import datetime, UTC
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def success_response(message: str, data: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": True,
        "message": message,
        "data": data or {},
        "metadata": metadata or {},
        "request_id": str(uuid4()),
        "timestamp": now_iso(),
    }


def error_response(error_code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": False,
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "request_id": str(uuid4()),
        "timestamp": now_iso(),
    }
