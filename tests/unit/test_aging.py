from datetime import UTC, datetime
from pathlib import Path

from agent_memory_v2.aging import (
    age_bucket,
    age_days,
    age_penalty,
    effective_age_days,
    prune_dry_run_decision,
)
from agent_memory_v2.config import AppConfig
from agent_memory_v2.models import MemoryRecord


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root_dir=tmp_path,
        settings_path=tmp_path / "settings.yaml",
        raw={
            "llm": {"host": "http://localhost:11434", "model": "llama3:8b", "timeout_seconds": 10},
            "embeddings": {"provider": "hash", "model": "unused", "dimensions": 3},
            "memory": {
                "embedding_dim": 3,
                "top_k": 3,
                "similarity_threshold": 0.2,
                "index_path": "data/memory/test.index",
                "metadata_path": "data/memory/test.json",
                "interaction_log_path": "data/logs/interactions.jsonl",
            },
            "aging": {
                "enabled": True,
                "decay": {
                    "ephemeral_start_days": 1,
                    "ephemeral_full_days": 14,
                    "ephemeral_max_penalty": 0.08,
                    "task_start_days": 30,
                    "task_full_days": 90,
                    "task_max_penalty": 0.03,
                    "durable_start_days": 180,
                    "durable_full_days": 365,
                    "durable_max_penalty": 0.01,
                },
                "prune": {
                    "ephemeral_turn_ttl_days": 14,
                    "task_ttl_days": 90,
                },
            },
            "prompting": {"memory_heading": "Relevant memory", "input_heading": "Current user input"},
        },
    )


def test_age_days_and_bucket():
    now_dt = datetime(2026, 3, 29, 12, 0, tzinfo=UTC)
    days = age_days("2026-03-28T12:00:00+00:00", now_dt=now_dt)
    assert days == 1.0
    assert age_bucket(days) == "1-7d"


def test_age_penalty_hits_ephemeral_more_than_durable(tmp_path: Path):
    config = make_config(tmp_path)
    ephemeral = age_penalty(memory_class="turn", durable=False, age_days_value=14.0, config=config)
    durable = age_penalty(memory_class="preference", durable=True, age_days_value=14.0, config=config)
    assert ephemeral < durable
    assert ephemeral < 0.0
    assert durable == 0.0


def _make_record(
    memory_id: str = "m1",
    timestamp: str = "2026-03-01T12:00:00+00:00",
    metadata: dict | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        role="turn",
        text="hello",
        summary="hello",
        timestamp=timestamp,
        conversation_id="default",
        turn_id="t1",
        message_id=memory_id,
        metadata=metadata or {"memory_class": "turn", "durable": False},
    )


_NOW = datetime(2026, 3, 29, 12, 0, tzinfo=UTC)


def test_prune_dry_run_marks_stale_ephemeral_turn(tmp_path: Path):
    config = make_config(tmp_path)
    record = _make_record()
    decision = prune_dry_run_decision(record, config=config, now_dt=_NOW)
    assert decision["decision"] == "prune"
    assert decision["reason"] == "stale_ephemeral"


# ---------------------------------------------------------------------------
# effective_age_days
# ---------------------------------------------------------------------------


def test_effective_age_days_uses_timestamp_when_never_recalled():
    record = _make_record(timestamp="2026-03-22T12:00:00+00:00")
    age = effective_age_days(record, now_dt=_NOW)
    assert age is not None
    assert abs(age - 7.0) < 0.01


def test_effective_age_days_uses_last_recalled_when_more_recent():
    record = _make_record(
        timestamp="2026-03-01T12:00:00+00:00",
        metadata={"last_recalled_at": "2026-03-28T12:00:00+00:00"},
    )
    age = effective_age_days(record, now_dt=_NOW)
    assert age is not None
    assert abs(age - 1.0) < 0.01


def test_effective_age_days_uses_timestamp_when_older_than_last_recalled():
    # last_recalled_at older than timestamp should not happen normally,
    # but effective_age_days should still use the most recent (= timestamp here)
    record = _make_record(
        timestamp="2026-03-28T12:00:00+00:00",
        metadata={"last_recalled_at": "2026-03-01T12:00:00+00:00"},
    )
    age = effective_age_days(record, now_dt=_NOW)
    assert age is not None
    assert abs(age - 1.0) < 0.01


def test_effective_age_days_returns_none_for_missing_timestamp():
    record = MemoryRecord(
        memory_id="m1",
        role="turn",
        text="x",
        summary="x",
        timestamp="",
        conversation_id="default",
        turn_id="t1",
        message_id="m1",
        metadata={},
    )
    assert effective_age_days(record, now_dt=_NOW) is None


# ---------------------------------------------------------------------------
# prune_dry_run_decision — recall_count guard
# ---------------------------------------------------------------------------


def test_prune_keeps_actively_recalled_record_despite_age(tmp_path: Path):
    config = make_config(tmp_path)
    record = _make_record(
        metadata={"memory_class": "turn", "durable": False, "recall_count": 3},
    )
    decision = prune_dry_run_decision(record, config=config, now_dt=_NOW)
    assert decision["decision"] == "keep"
    assert decision["reason"] == "actively_recalled"


def test_prune_uses_default_min_recall_count_of_three(tmp_path: Path):
    config = make_config(tmp_path)
    below_threshold = _make_record(
        metadata={"memory_class": "turn", "durable": False, "recall_count": 2},
    )
    at_threshold = _make_record(
        metadata={"memory_class": "turn", "durable": False, "recall_count": 3},
    )
    below_decision = prune_dry_run_decision(below_threshold, config=config, now_dt=_NOW)
    at_decision = prune_dry_run_decision(at_threshold, config=config, now_dt=_NOW)

    assert below_decision["decision"] == "prune"
    assert at_decision["decision"] == "keep"
    assert at_decision["reason"] == "actively_recalled"


def test_prune_respects_configurable_min_recall_count(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["aging"]["prune"]["min_recall_count_to_keep"] = 5

    record_count_4 = _make_record(
        metadata={"memory_class": "turn", "durable": False, "recall_count": 4},
    )
    record_count_5 = _make_record(
        metadata={"memory_class": "turn", "durable": False, "recall_count": 5},
    )

    assert prune_dry_run_decision(record_count_4, config=config, now_dt=_NOW)["decision"] == "prune"
    assert prune_dry_run_decision(record_count_5, config=config, now_dt=_NOW)["decision"] == "keep"


def test_prune_recall_count_guard_checked_before_age(tmp_path: Path):
    config = make_config(tmp_path)
    # Even a very fresh record with recall_count >= 3 should report actively_recalled,
    # not the fallback retained reason
    fresh_often_recalled = _make_record(
        timestamp="2026-03-29T11:59:00+00:00",
        metadata={"memory_class": "turn", "durable": False, "recall_count": 10},
    )
    decision = prune_dry_run_decision(fresh_often_recalled, config=config, now_dt=_NOW)
    assert decision["reason"] == "actively_recalled"

