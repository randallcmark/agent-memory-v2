from pathlib import Path

from agent_memory_v2.config import AppConfig
from agent_memory_v2.maintenance import default_state, evaluate_due, mark_interaction


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
            "aging": {"enabled": True, "prune": {"ephemeral_turn_ttl_days": 14, "task_ttl_days": 90}},
            "maintenance": {
                "enabled": True,
                "state_path": "data/maintenance/state.json",
                "lock_path": "data/maintenance/lock",
                "min_interval_minutes": 30,
                "max_new_records": 2,
                "run_profile_rebuild": True,
                "run_prune": True,
            },
            "prompting": {"memory_heading": "Relevant memory", "input_heading": "Current user input"},
        },
    )


def test_default_state_shape():
    state = default_state()
    assert state["maintenance_due"] is False
    assert state["new_records_since_run"] == 0


def test_mark_interaction_sets_due_after_threshold(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr("agent_memory_v2.maintenance.prune_dry_run", lambda config, limit=1: {"summary": {"prune": 0}})
    monkeypatch.setattr(
        "agent_memory_v2.maintenance.prune_sidecar_dry_run",
        lambda config, limit=1: {"summary": {"prune": 0}},
    )
    first = mark_interaction(config)
    second = mark_interaction(config)
    assert first["maintenance_due"] in {False, True}
    assert second["maintenance_due"] is True
    assert "new_record_threshold" in second["due_reasons"]


def test_evaluate_due_includes_prune_candidates(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    state = default_state()
    monkeypatch.setattr("agent_memory_v2.maintenance.prune_dry_run", lambda config, limit=1: {"summary": {"prune": 1}})
    monkeypatch.setattr(
        "agent_memory_v2.maintenance.prune_sidecar_dry_run",
        lambda config, limit=1: {"summary": {"prune": 0}},
    )
    result = evaluate_due(config, state)
    assert result["maintenance_due"] is True
    assert "prune_candidates" in result["due_reasons"]


def test_evaluate_due_includes_sidecar_candidates(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    state = default_state()
    monkeypatch.setattr("agent_memory_v2.maintenance.prune_dry_run", lambda config, limit=1: {"summary": {"prune": 0}})
    monkeypatch.setattr(
        "agent_memory_v2.maintenance.prune_sidecar_dry_run",
        lambda config, limit=1: {"summary": {"prune": 1}},
    )
    result = evaluate_due(config, state)
    assert result["maintenance_due"] is True
    assert "sidecar_prune_candidates" in result["due_reasons"]
