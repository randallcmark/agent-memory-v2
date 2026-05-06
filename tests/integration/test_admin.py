from pathlib import Path

import numpy as np

from agent_memory_v2.admin import (
    get_aging_report,
    get_profile,
    get_store_stats,
    list_memories,
    list_sidecar_memories,
    prune_dry_run,
    prune_sidecar_dry_run,
    prune_sidecar_store,
    prune_store,
    rebuild_profile,
    rebuild_store,
    reset_store,
)
from agent_memory_v2.config import AppConfig
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline


class StubEncoder:
    def encode(self, text: str) -> np.ndarray:
        if "milk" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype="float32")
        return np.array([0.0, 1.0, 0.0], dtype="float32")


class StubOllama:
    def generate(self, prompt: str) -> str:
        return "stub-response"


class StubExtractorClient:
    def __init__(self, profile) -> None:
        self.profile = profile

    def generate(self, prompt: str) -> str:
        return (
            '{"memory_class":"fact","durable":true,"profile_key":"identity.location",'
            '"extracted_value":"Edinburgh in the UK","confidence":0.9,'
            '"supersedes_profile_key":"identity.location"}'
        )


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root_dir=tmp_path,
        settings_path=tmp_path / "settings.yaml",
        raw={
            "llm": {
                "host": "http://127.0.0.1:11434",
                "model": "llama3:8b",
                "temperature": 0.2,
                "max_tokens": 100,
                "timeout_seconds": 10,
                "preflight": {"enabled": False},
            },
            "embeddings": {
                "provider": "hash",
                "model": "unused",
                "dimensions": 3,
            },
            "memory": {
                "embedding_dim": 3,
                "top_k": 3,
                "similarity_threshold": 0.2,
                "index_path": "data/memory/test.index",
                "metadata_path": "data/memory/test.json",
                "interaction_log_path": "data/logs/interactions.jsonl",
            },
            "prompting": {
                "memory_heading": "Relevant memory",
                "input_heading": "Current user input",
            },
        },
    )


def test_admin_stats_list_reset_and_rebuild(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    user_message = Message(role="user", text="remember milk", turn_id="turn-1")
    agent_message = Message(role="agent", text="I will remember milk", turn_id="turn-1")
    pipeline.ingest_turn(user_message, agent_message)

    stats = get_store_stats(config)
    assert stats["count"] == 1
    assert stats["class_counts"]["turn"] == 1
    assert stats["durable_counts"]["ephemeral"] == 1

    items = list_memories(config, limit=5)
    assert len(items) == 1
    assert items[0]["role"] == "turn"
    assert items[0]["memory_class"] == "turn"
    assert items[0]["durable"] is False

    reset_result = reset_store(config)
    assert reset_result["reset"] is True
    assert get_store_stats(config)["count"] == 0

    monkeypatch.setattr("agent_memory_v2.admin.build_encoder", lambda **kwargs: StubEncoder())
    rebuild_result = rebuild_store(config)
    assert rebuild_result["rebuilt"] == 0

    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(user_message, agent_message)
    monkeypatch.setattr("agent_memory_v2.admin.build_encoder", lambda **kwargs: StubEncoder())
    rebuild_result = rebuild_store(config)
    assert rebuild_result["rebuilt"] == 1
    assert get_store_stats(config)["count"] == 1


def test_rebuild_store_passes_ollama_embedding_settings(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    config.raw["embeddings"] = {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "host": "http://localhost:11434",
        "timeout_seconds": 60,
        "dimensions": 768,
    }
    config.raw["memory"]["embedding_dim"] = 3

    captured = {}

    def fake_build_encoder(**kwargs):
        captured.update(kwargs)
        return StubEncoder()

    monkeypatch.setattr("agent_memory_v2.admin.build_encoder", fake_build_encoder)
    rebuild_result = rebuild_store(config)

    assert rebuild_result["rebuilt"] == 0
    assert captured["provider"] == "ollama"
    assert captured["host"] == "http://localhost:11434"
    assert captured["timeout_seconds"] == 60
    assert captured["dimensions"] == 768


def test_rebuild_store_reclassifies_historical_logs(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    interaction_log = config.resolve_path(config.memory["interaction_log_path"])
    interaction_log.parent.mkdir(parents=True, exist_ok=True)
    interaction_log.write_text(
        '{"role":"turn","text":"User: Please remember that I prefer oat milk.\\nAgent: noted","summary":"Please remember that I prefer oat milk.","timestamp":"2026-03-29T09:00:00+00:00","message_id":"m1","turn_id":"t1","conversation_id":"default","metadata":{"kind":"turn_memory","memory_class":"turn"}}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr("agent_memory_v2.admin.build_encoder", lambda **kwargs: StubEncoder())
    rebuild_result = rebuild_store(config)
    items = list_memories(config, limit=5)

    assert rebuild_result["rebuilt"] == 1
    assert items[0]["memory_class"] == "preference"
    assert items[0]["extracted_value"] == "oat milk"
    assert items[0]["metadata"]["classification_confidence"] >= 0.9
    assert items[0]["metadata"]["profile_key"] == "preference.general"


def test_rebuild_store_runs_hybrid_extraction_for_semantic_candidates(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    config.raw["embeddings"] = {"provider": "hash", "model": "hash", "dimensions": 128}
    config.raw["memory"]["embedding_dim"] = 128
    config.raw["semantic_router"] = {"enabled": True, "threshold": 0.72, "debug_metadata": True}
    config.raw["structured_extractor"] = {
        "enabled": True,
        "admission_threshold": 0.75,
        "max_value_chars": 160,
        "allowed_profile_keys": ["identity.location"],
    }
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact", "task"],
    }
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    interaction_log = config.resolve_path(config.memory["interaction_log_path"])
    interaction_log.parent.mkdir(parents=True, exist_ok=True)
    interaction_log.write_text(
        '{"role":"turn","text":"User: I am based in Edinburgh in the UK.\\nAgent: noted",'
        '"summary":"I am based in Edinburgh in the UK.",'
        '"timestamp":"2026-03-29T09:00:00+00:00","message_id":"m1",'
        '"turn_id":"t1","conversation_id":"default",'
        '"metadata":{"kind":"turn_memory","memory_class":"turn"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("agent_memory_v2.admin.OllamaClient", StubExtractorClient)

    rebuild_result = rebuild_store(config)
    items = list_memories(config, limit=5)
    sidecar_items = list_sidecar_memories(config, limit=5)
    profile = get_profile(config)

    assert rebuild_result["rebuilt"] == 1
    assert rebuild_result["sidecar_rebuilt"] == 1
    assert items[0]["metadata"]["classification_source"] == "structured_extractor"
    assert items[0]["memory_class"] == "fact"
    assert items[0]["extracted_value"] == "Edinburgh in the UK"
    assert items[0]["metadata"]["rule_classification"]["memory_class"] == "turn"
    assert len(sidecar_items) == 1
    assert sidecar_items[0]["text"] == "Edinburgh in the UK"
    assert profile["facts"]["identity.location"]["value"] == "Edinburgh in the UK"


def test_list_memories_supports_filters(tmp_path: Path):
    config = make_config(tmp_path)
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(
            role="user",
            text="I prefer oat milk.",
            turn_id="turn-1",
            timestamp="2026-03-29T08:00:00+00:00",
        ),
        Message(
            role="agent",
            text="Noted.",
            turn_id="turn-1",
            timestamp="2026-03-29T08:00:01+00:00",
        ),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="Remind me to call the dentist.",
            turn_id="turn-2",
            timestamp="2026-03-30T08:00:00+00:00",
        ),
        Message(
            role="agent",
            text="Noted.",
            turn_id="turn-2",
            timestamp="2026-03-30T08:00:01+00:00",
        ),
    )

    preference_items = list_memories(config, memory_class="preference")
    keyword_items = list_memories(config, query="dentist")
    dated_items = list_memories(
        config,
        date_from="2026-03-30T00:00:00+00:00",
        date_to="2026-03-30T23:59:59+00:00",
    )

    assert len(preference_items) == 1
    assert preference_items[0]["extracted_value"] == "oat milk"
    assert preference_items[0]["durable"] is True
    assert len(keyword_items) == 1
    assert keyword_items[0]["memory_class"] == "task"
    assert len(dated_items) == 1
    assert dated_items[0]["extracted_value"] == "call the dentist"


def test_sidecar_stats_and_listing(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact"],
    }
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(
            role="user",
            text="I prefer oat milk.",
            turn_id="turn-1",
            timestamp="2026-03-29T08:00:00+00:00",
        ),
        Message(
            role="agent",
            text="Noted.",
            turn_id="turn-1",
            timestamp="2026-03-29T08:00:01+00:00",
        ),
    )

    stats = get_store_stats(config)
    sidecar_items = list_sidecar_memories(config, limit=5)

    assert stats["sidecar"]["enabled"] is True
    assert stats["sidecar"]["count"] == 1
    assert len(sidecar_items) == 1
    assert sidecar_items[0]["memory_class"] == "preference"
    assert sidecar_items[0]["text"] == "oat milk"
    profile = get_profile(config)
    assert profile["preferences"]["preference.general"]["value"] == "oat milk"


def test_rebuild_profile_from_sidecar(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact"],
    }
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(role="user", text="I prefer oat milk.", turn_id="turn-1"),
        Message(role="agent", text="Noted.", turn_id="turn-1"),
    )

    result = rebuild_profile(config)

    assert result["rebuilt_profile"] is True
    assert result["counts"]["preferences"] == 1


def test_prune_dry_run_marks_expired_and_resolved_tasks(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["aging"] = {
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
            "task_ttl_days": 30,
            "resolved_task_ttl_days": 0,
            "dry_run_limit": 50,
        },
    }
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(
            role="user",
            text="Remind me to call the dentist.",
            turn_id="task-old",
            timestamp="2026-02-01T08:00:00+00:00",
        ),
        Message(role="agent", text="Noted.", turn_id="task-old", timestamp="2026-02-01T08:00:01+00:00"),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="Remind me to renew my passport.",
            turn_id="task-resolve",
            timestamp="2026-03-20T08:00:00+00:00",
        ),
        Message(role="agent", text="Noted.", turn_id="task-resolve", timestamp="2026-03-20T08:00:01+00:00"),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="I completed renew my passport.",
            turn_id="task-resolution",
            timestamp="2026-03-29T08:00:00+00:00",
        ),
        Message(role="agent", text="Good.", turn_id="task-resolution", timestamp="2026-03-29T08:00:01+00:00"),
    )

    dry_run = prune_dry_run(config)
    reasons = {item["reason"] for item in dry_run["candidates"]}
    assert "expired_task" in reasons
    assert "resolved_task" in reasons


def test_aging_report_and_prune_dry_run(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["aging"] = {
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
            "dry_run_limit": 50,
            "archive_path": "data/archive/pruned.jsonl",
        },
    }
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(
            role="user",
            text="hello",
            turn_id="old-turn",
            timestamp="2026-03-01T08:00:00+00:00",
        ),
        Message(
            role="agent",
            text="hi",
            turn_id="old-turn",
            timestamp="2026-03-01T08:00:01+00:00",
        ),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="I prefer oat milk.",
            turn_id="new-turn",
            timestamp="2026-03-29T08:00:00+00:00",
        ),
        Message(
            role="agent",
            text="Noted.",
            turn_id="new-turn",
            timestamp="2026-03-29T08:00:01+00:00",
        ),
    )

    report = get_aging_report(config)
    dry_run = prune_dry_run(config)

    assert report["count"] == 2
    assert "bucket_counts" in report
    assert dry_run["count"] == 2
    assert dry_run["summary"]["keep"] >= 1
    assert any(item["reason"] == "stale_ephemeral" for item in dry_run["candidates"])


def test_prune_store_removes_stale_ephemeral_turns(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    config.raw["aging"] = {
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
            "dry_run_limit": 50,
            "archive_path": "data/archive/pruned.jsonl",
        },
    }
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(
            role="user",
            text="hello",
            turn_id="old-turn",
            timestamp="2026-03-01T08:00:00+00:00",
        ),
        Message(
            role="agent",
            text="hi",
            turn_id="old-turn",
            timestamp="2026-03-01T08:00:01+00:00",
        ),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="I prefer oat milk.",
            turn_id="new-turn",
            timestamp="2026-03-29T08:00:00+00:00",
        ),
        Message(
            role="agent",
            text="Noted.",
            turn_id="new-turn",
            timestamp="2026-03-29T08:00:01+00:00",
        ),
    )

    monkeypatch.setattr("agent_memory_v2.admin.build_encoder", lambda **kwargs: StubEncoder())
    result = prune_store(config)
    items = list_memories(config, limit=10)
    archive_path = config.resolve_path("data/archive/pruned.jsonl")

    assert result["pruned"] == 1
    assert result["kept"] == 1
    assert result["archive_path"] == str(archive_path)
    assert archive_path.exists()
    assert "hello" in archive_path.read_text(encoding="utf-8")
    assert len(items) == 1
    assert items[0]["memory_class"] == "preference"


def test_prune_store_is_not_capped_by_dry_run_limit(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    config.raw["aging"] = {
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
            "dry_run_limit": 2,
            "archive_path": "data/archive/pruned.jsonl",
        },
    }
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    for index in range(4):
        pipeline.ingest_turn(
            Message(
                role="user",
                text=f"hello {index}",
                turn_id=f"old-turn-{index}",
                timestamp="2026-03-01T08:00:00+00:00",
            ),
            Message(
                role="agent",
                text="hi",
                turn_id=f"old-turn-{index}",
                timestamp="2026-03-01T08:00:01+00:00",
            ),
        )

    dry_run = prune_dry_run(config)
    monkeypatch.setattr("agent_memory_v2.admin.build_encoder", lambda **kwargs: StubEncoder())
    result = prune_store(config)
    items = list_memories(config, limit=10)

    assert dry_run["count"] == 4
    assert len(dry_run["candidates"]) == 2
    assert result["pruned"] == 4
    assert result["kept"] == 0
    assert items == []


def test_sidecar_prune_keeps_latest_profile_key_value(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact"],
    }
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    config.raw["aging"] = {
        "sidecar": {
            "enabled": True,
            "archive_path": "data/archive/pruned_sidecar.jsonl",
        }
    }
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(role="user", text="I prefer tea.", turn_id="pref-1", timestamp="2026-03-20T08:00:00+00:00"),
        Message(role="agent", text="Noted.", turn_id="pref-1", timestamp="2026-03-20T08:00:01+00:00"),
    )
    pipeline.ingest_turn(
        Message(role="user", text="I prefer coffee.", turn_id="pref-2", timestamp="2026-03-29T08:00:00+00:00"),
        Message(role="agent", text="Noted.", turn_id="pref-2", timestamp="2026-03-29T08:00:01+00:00"),
    )

    dry_run = prune_sidecar_dry_run(config)
    monkeypatch.setattr("agent_memory_v2.admin.build_encoder", lambda **kwargs: StubEncoder())
    result = prune_sidecar_store(config)
    sidecar_items = list_sidecar_memories(config, limit=10)
    profile = get_profile(config)

    assert dry_run["summary"]["prune"] == 1
    assert result["pruned"] == 1
    assert len(sidecar_items) == 1
    assert sidecar_items[0]["text"] == "coffee"
    assert profile["preferences"]["preference.general"]["value"] == "coffee"


def test_rebuild_store_uses_canonical_sidecar_embedding_for_recall(tmp_path: Path, monkeypatch):
    class CanonicalSidecarEncoder:
        def encode(self, text: str) -> np.ndarray:
            lowered = text.lower()
            if "edinburgh" in lowered:
                return np.array([1.0, 0.0, 0.0], dtype="float32")
            return np.array([0.0, 1.0, 0.0], dtype="float32")

    config = make_config(tmp_path)
    config.raw["semantic_router"] = {"enabled": True, "threshold": 0.72, "debug_metadata": True}
    config.raw["structured_extractor"] = {
        "enabled": True,
        "admission_threshold": 0.75,
        "max_value_chars": 160,
        "allowed_profile_keys": ["identity.location"],
    }
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact", "task"],
    }
    interaction_log = config.resolve_path(config.memory["interaction_log_path"])
    interaction_log.parent.mkdir(parents=True, exist_ok=True)
    interaction_log.write_text(
        '{"role":"turn","text":"User: I am based in the UK.\\nAgent: noted",'
        '"summary":"I am based in the UK.",'
        '"timestamp":"2026-03-29T09:00:00+00:00","message_id":"m1",'
        '"turn_id":"t1","conversation_id":"default",'
        '"metadata":{"kind":"turn_memory","memory_class":"turn"}}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr("agent_memory_v2.admin.build_encoder", lambda **kwargs: CanonicalSidecarEncoder())
    monkeypatch.setattr("agent_memory_v2.admin.OllamaClient", StubExtractorClient)

    rebuild_store(config)

    pipeline = MemoryPipeline(
        config,
        encoder=CanonicalSidecarEncoder(),
        ollama=StubOllama(),
    )
    recalled = pipeline.recall(Message(role="user", text="Edinburgh in the UK", message_id="new-id"))

    assert recalled[0]["store_kind"] == "sidecar_memory"
    assert recalled[0]["text"] == "Edinburgh in the UK"
    assert recalled[0]["memory_class"] == "fact"
