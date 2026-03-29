from pathlib import Path

import numpy as np

from agent_memory_v2.config import AppConfig
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline
from agent_memory_v2.sanitise_cli import sanitise_repo
from agent_memory_v2.seed_cli import main as seed_main


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
            "embeddings": {"provider": "hash", "model": "unused", "dimensions": 3},
            "memory": {
                "embedding_dim": 3,
                "top_k": 3,
                "similarity_threshold": 0.2,
                "index_path": "data/memory/test.index",
                "metadata_path": "data/memory/test.json",
                "interaction_log_path": "data/logs/interactions.jsonl",
            },
            "sidecar": {
                "enabled": True,
                "top_k": 2,
                "similarity_threshold": 0.2,
                "index_path": "data/sidecar/facts.index",
                "metadata_path": "data/sidecar/facts.json",
                "store_classes": ["preference", "fact"],
            },
            "profile": {"enabled": True, "path": "data/profile/user_profile.json", "inject": True},
            "prompting": {"memory_heading": "Relevant memory", "input_heading": "Current user input"},
        },
    )


def test_sanitise_repo_removes_runtime_state(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr("agent_memory_v2.sanitise_cli.load_config", lambda: config)
    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(role="user", text="I prefer oat milk.", turn_id="t1"),
        Message(role="agent", text="Noted.", turn_id="t1"),
    )

    result = sanitise_repo(apply_changes=True)
    assert result["ok"] is True
    assert not config.resolve_path("data").exists()


def test_seed_cli_loads_generic_records(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    seed_path = tmp_path / "seeds.jsonl"
    seed_path.write_text(
        '{"user_text":"I prefer oat milk.","agent_reply":"Noted."}\n'
        '{"user_text":"My name is Alex.","agent_reply":"Noted."}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("agent_memory_v2.seed_cli.load_config", lambda: config)
    monkeypatch.setattr(
        "sys.argv",
        ["seed_cli", "--seed-file", str(seed_path), "--conversation-id", "seed-test"],
    )

    exit_code = seed_main()
    assert exit_code == 0
    assert config.resolve_path(config.memory["metadata_path"]).exists()
