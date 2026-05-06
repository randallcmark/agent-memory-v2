from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent_memory_v2.models import MemoryRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_sort_key(record: MemoryRecord) -> tuple[str, str]:
    return record.timestamp, record.memory_id


def _key_mode(profile_key: str) -> str:
    try:
        from agent_memory_v2.taxonomy import get_taxonomy
        tk = get_taxonomy().get(profile_key)
        if tk is not None:
            return tk.mode
    except Exception:
        pass
    return "scalar"


def _upsert(bucket: dict[str, dict], key: str, new_entry: dict, entry_mode: str) -> None:
    if entry_mode == "additive":
        existing = bucket.get(key)
        new_val = new_entry.get("value")
        if existing is None:
            all_vals = [new_val] if new_val is not None else []
            bucket[key] = {**new_entry, "all_values": all_vals}
        else:
            all_vals = list(existing.get("all_values") or [])
            if new_val is not None and new_val not in all_vals:
                all_vals.append(new_val)
            bucket[key] = {**new_entry, "all_values": all_vals}
    else:
        bucket[key] = new_entry


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

        mode = _key_mode(str(profile_key))

        if record.role == "preference":
            _upsert(preferences, str(profile_key), entry, mode)
        elif record.role == "fact":
            _upsert(facts, str(profile_key), entry, mode)
        elif record.role == "task":
            tasks[str(profile_key)] = entry

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

    def update_from_record(self, record: MemoryRecord) -> dict:
        """Apply a single new sidecar record to the existing profile (O(1) hot path)."""
        metadata = record.metadata or {}
        profile_key = metadata.get("profile_key")
        if not profile_key:
            return self.load()

        profile = self.load()
        entry = {
            "value": metadata.get("extracted_value"),
            "summary": record.summary,
            "timestamp": record.timestamp,
            "memory_class": metadata.get("memory_class", record.role),
            "source_message_id": metadata.get("source_memory_id", record.message_id),
        }
        mode = _key_mode(str(profile_key))

        section_name = {
            "preference": "preferences",
            "fact": "facts",
            "task": "tasks",
        }.get(record.role)

        if section_name is None:
            return profile

        section = profile.setdefault(section_name, {})
        is_new = str(profile_key) not in section

        if record.role == "task":
            section[str(profile_key)] = entry
        else:
            _upsert(section, str(profile_key), entry, mode)

        if is_new:
            counts = profile.setdefault("counts", {})
            counts[section_name] = len(section)

        profile["updated_at"] = _utc_now_iso()
        self.save(profile)
        return profile
