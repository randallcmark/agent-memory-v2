from pathlib import Path

import numpy as np

from agent_memory_v2.config import AppConfig
from agent_memory_v2.maintenance import maintenance_status
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline


class StubEncoder:
    def encode(self, text: str) -> np.ndarray:
        if "oat milk" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype="float32")
        return np.array([0.0, 1.0, 0.0], dtype="float32")


class StubOllama:
    def generate(self, prompt: str) -> str:
        return "stub-response"


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


def test_turn_memory_persists_across_pipeline_restart(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/facts.index",
        "metadata_path": "data/sidecar/facts.json",
        "store_classes": ["preference", "fact"],
    }
    config.raw["maintenance"] = {
        "enabled": True,
        "state_path": "data/maintenance/state.json",
        "lock_path": "data/maintenance/lock",
        "min_interval_minutes": 30,
        "max_new_records": 1,
        "run_profile_rebuild": True,
        "run_prune": True,
    }
    first = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    user_message = Message(role="user", text="Please remember I prefer oat milk.", turn_id="t1")
    agent_message = Message(role="agent", text="I will remember that.", turn_id="t1")
    first.ingest_turn(user_message, agent_message)

    second = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    recalled = second.recall(Message(role="user", text="What do I prefer? oat milk?", message_id="new"))

    assert len(recalled) == 1
    assert recalled[0]["memory_class"] == "preference"
    assert recalled[0]["store_kind"] == "sidecar_memory"
    assert recalled[0]["text"] == "oat milk"
    status = maintenance_status(config)
    assert status["new_records_since_run"] >= 1


def test_named_user_uses_segregated_storage(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_V2_USER", "mark")
    config = make_config(tmp_path)
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(role="user", text="Please remember I prefer oat milk.", turn_id="t-user"),
        Message(role="agent", text="Noted.", turn_id="t-user"),
    )

    assert "/data/users/mark/memory/test.index" in str(config.resolve_path(config.memory["index_path"]))
    assert config.resolve_path(config.memory["metadata_path"]).exists()
