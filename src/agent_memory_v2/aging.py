from __future__ import annotations

from datetime import datetime, timezone

from agent_memory_v2.classifier import detect_task_resolution
from agent_memory_v2.config import AppConfig
from agent_memory_v2.models import MemoryRecord


def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def age_days(ts: str | None, *, now_dt: datetime | None = None) -> float | None:
    parsed = parse_timestamp(ts)
    if parsed is None:
        return None
    now_dt = now_dt or datetime.now(timezone.utc)
    delta = now_dt - parsed
    return max(delta.total_seconds(), 0.0) / 86400.0


def age_bucket(days: float | None) -> str:
    if days is None:
        return "unknown"
    if days < 1:
        return "0-1d"
    if days < 7:
        return "1-7d"
    if days < 30:
        return "7-30d"
    if days < 90:
        return "30-90d"
    return "90d+"


def effective_age_days(
    record: MemoryRecord,
    *,
    now_dt: datetime | None = None,
) -> float | None:
    now_dt = now_dt or datetime.now(timezone.utc)
    created = parse_timestamp(record.timestamp)
    last_recalled = parse_timestamp(record.metadata.get("last_recalled_at"))
    most_recent = max(
        (dt for dt in (created, last_recalled) if dt is not None),
        default=None,
    )
    if most_recent is None:
        return None
    return max((now_dt - most_recent).total_seconds(), 0.0) / 86400.0


def age_penalty(
    *,
    memory_class: str,
    durable: bool,
    age_days_value: float | None,
    config: AppConfig,
) -> float:
    if age_days_value is None:
        return 0.0

    aging = config.aging
    decay = aging.get("decay", {}) or {}
    if not aging.get("enabled", True):
        return 0.0

    if not durable and memory_class in {"turn", "message"}:
        start_days = float(decay.get("ephemeral_start_days", 1))
        full_days = float(decay.get("ephemeral_full_days", 14))
        max_penalty = float(decay.get("ephemeral_max_penalty", 0.08))
    elif memory_class == "task":
        start_days = float(decay.get("task_start_days", 30))
        full_days = float(decay.get("task_full_days", 90))
        max_penalty = float(decay.get("task_max_penalty", 0.03))
    else:
        start_days = float(decay.get("durable_start_days", 180))
        full_days = float(decay.get("durable_full_days", 365))
        max_penalty = float(decay.get("durable_max_penalty", 0.01))

    if age_days_value <= start_days:
        return 0.0
    if full_days <= start_days:
        return -max_penalty
    progress = min((age_days_value - start_days) / (full_days - start_days), 1.0)
    return -(max_penalty * progress)


def prune_dry_run_decision(
    record: MemoryRecord,
    *,
    config: AppConfig,
    now_dt: datetime | None = None,
) -> dict:
    age = age_days(record.timestamp, now_dt=now_dt)
    memory_class = record.metadata.get("memory_class", record.role)
    durable = bool(record.metadata.get("durable", False))
    prune_cfg = config.aging.get("prune", {}) or {}
    ephemeral_ttl = float(prune_cfg.get("ephemeral_turn_ttl_days", 14))
    task_ttl = float(prune_cfg.get("task_ttl_days", 90))
    min_recall_count = int(prune_cfg.get("min_recall_count_to_keep", 3))
    recall_count = int(record.metadata.get("recall_count", 0))

    decision = "keep"
    reason = "retained"
    if age is None:
        decision = "review"
        reason = "missing_timestamp"
    elif recall_count >= min_recall_count:
        decision = "keep"
        reason = "actively_recalled"
    elif not durable and memory_class in {"turn", "message"} and age > ephemeral_ttl:
        decision = "prune"
        reason = "stale_ephemeral"
    elif memory_class == "task" and age > task_ttl:
        decision = "review"
        reason = "stale_task"
    elif durable and record.metadata.get("profile_key"):
        decision = "keep"
        reason = "profile_backed_durable"

    return {
        "memory_id": record.memory_id,
        "message_id": record.message_id,
        "memory_class": memory_class,
        "durable": durable,
        "age_days": round(age, 3) if age is not None else None,
        "age_bucket": age_bucket(age),
        "decision": decision,
        "reason": reason,
    }


def normalize_task_value(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def resolved_task_values(records: list[MemoryRecord]) -> dict[str, str]:
    values: dict[str, str] = {}
    for record in sorted(records, key=lambda item: (item.timestamp, item.memory_id)):
        source_text = record.summary or record.text
        resolved = detect_task_resolution(source_text)
        if resolved:
            values[normalize_task_value(resolved)] = record.timestamp
    return values
