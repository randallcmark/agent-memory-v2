import json
import zipfile
from pathlib import Path

from agent_memory_v2.config import AppConfig
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline
from agent_memory_v2.state_cli import export_state, import_state


class StubEncoder:
    def encode(self, text: str):
        import numpy as np

        if "milk" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype="float32")
        return np.array([0.0, 1.0, 0.0], dtype="float32")


class StubOllama:
    def generate(self, prompt: str) -> str:
        return "stub-response"


def make_config(tmp_path: Path) -> AppConfig:
    settings_path = tmp_path / "config" / "settings.yaml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("{}", encoding="utf-8")
    return AppConfig(
        root_dir=tmp_path,
        settings_path=settings_path,
        raw={
            "app": {"name": "agent_memory_v2"},
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
            "profile": {
                "enabled": True,
                "path": "data/profile/user_profile.json",
                "inject": True,
            },
            "maintenance": {
                "enabled": True,
                "state_path": "data/maintenance/state.json",
                "lock_path": "data/maintenance/lock",
                "min_interval_minutes": 30,
                "max_new_records": 10,
                "run_profile_rebuild": True,
                "run_prune": True,
            },
            "prompting": {
                "memory_heading": "Relevant memory",
                "input_heading": "Current user input",
            },
        },
    )


def test_export_and_import_state(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr("agent_memory_v2.state_cli.load_config", lambda: config)

    pipeline = MemoryPipeline(config, encoder=StubEncoder(), ollama=StubOllama())
    pipeline.ingest_turn(
        Message(role="user", text="I prefer oat milk.", turn_id="t1"),
        Message(role="agent", text="Noted.", turn_id="t1"),
    )

    archive_path = tmp_path / "backup" / "state.zip"
    exported = export_state(archive_path)
    assert exported["ok"] is True
    assert archive_path.exists()

    with zipfile.ZipFile(archive_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert "index_path" in manifest["files"]
        assert "metadata_path" in manifest["files"]
        assert "interaction_log_path" in manifest["files"]
        assert "sidecar_index_path" in manifest["files"]
        assert "sidecar_metadata_path" in manifest["files"]
        assert "profile_path" in manifest["files"]
        assert "maintenance_state_path" in manifest["files"]

    metadata_path = config.resolve_path(config.memory["metadata_path"])
    metadata_path.write_text("[]", encoding="utf-8")

    imported = import_state(archive_path)
    assert imported["ok"] is True
    restored = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(restored) == 1
    assert restored[0]["summary"] == "I prefer oat milk."
