from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
