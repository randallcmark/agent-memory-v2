from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent_memory_v2.models import MemoryRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_sort_key(record: MemoryRecord) -> tuple[str, str]:
    return record.timestamp, record.memory_id


def build_profile(records: list[MemoryRecord]) -> dict:
    preferences: dict[str, dict] = {}
    facts: dict[str, dict] = {}
    tasks: dict[str, dict] = {}

    for record in sorted(records, key=_record_sort_key):
        metadata = record.metadata or {}
        profile_key = metadata.get("profile_key")
        if not profile_key:
            continue

        entry = {
            "value": metadata.get("extracted_value"),
            "summary": record.summary,
            "timestamp": record.timestamp,
            "memory_class": metadata.get("memory_class", record.role),
            "source_message_id": metadata.get("source_memory_id", record.message_id),
        }

        if record.role == "preference":
            preferences[profile_key] = entry
        elif record.role == "fact":
            facts[profile_key] = entry
        elif record.role == "task":
            tasks[profile_key] = entry

    return {
        "updated_at": _utc_now_iso(),
        "counts": {
            "preferences": len(preferences),
            "facts": len(facts),
            "tasks": len(tasks),
        },
        "preferences": preferences,
        "facts": facts,
        "tasks": tasks,
    }


class UserProfileStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return build_profile([])
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return build_profile([])
        return data

    def save(self, profile: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)

    def reset(self) -> None:
        self.save(build_profile([]))

    def rebuild_from_records(self, records: list[MemoryRecord]) -> dict:
        profile = build_profile(records)
        self.save(profile)
        return profile
