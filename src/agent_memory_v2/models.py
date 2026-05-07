"""Core dataclasses: Message, MemoryRecord, RecallResult, and SCHEMA_VERSION constant."""

from __future__ import annotations

__all__ = ["SCHEMA_VERSION", "Message", "MemoryRecord", "RecallResult", "utc_now_iso"]

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

SCHEMA_VERSION: int = 1


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    timestamp: str = field(default_factory=utc_now_iso)
    message_id: str = field(default_factory=lambda: str(uuid4()))
    conversation_id: str = "default"
    turn_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    role: str
    text: str
    summary: str
    timestamp: str
    conversation_id: str
    turn_id: str
    message_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RecallResult:
    record: MemoryRecord
    score: float
