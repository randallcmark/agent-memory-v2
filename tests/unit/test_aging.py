from datetime import datetime, timezone
from pathlib import Path

from agent_memory_v2.aging import age_bucket, age_days, age_penalty, prune_dry_run_decision
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
    now_dt = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
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


def test_prune_dry_run_marks_stale_ephemeral_turn(tmp_path: Path):
    config = make_config(tmp_path)
    record = MemoryRecord(
        memory_id="m1",
        role="turn",
        text="hello",
        summary="hello",
        timestamp="2026-03-01T12:00:00+00:00",
        conversation_id="default",
        turn_id="t1",
        message_id="m1",
        metadata={"memory_class": "turn", "durable": False},
    )
    decision = prune_dry_run_decision(
        record,
        config=config,
        now_dt=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    )
    assert decision["decision"] == "prune"
    assert decision["reason"] == "stale_ephemeral"

