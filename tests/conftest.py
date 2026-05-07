"""Shared test helpers for the agent_memory_v2 test suite.

Helpers defined here are importable by any test file:
    from conftest import make_store, make_record, make_config, StubEncoder, StubOllama
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agent_memory_v2.config import AppConfig
from agent_memory_v2.models import MemoryRecord
from agent_memory_v2.store import MemoryStore

# ---------------------------------------------------------------------------
# Core object factories
# ---------------------------------------------------------------------------

def make_record(memory_id: str = "m1", metadata: dict | None = None) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        role="user",
        text="remember milk",
        summary="remember milk",
        timestamp="2026-01-01T00:00:00+00:00",
        conversation_id="default",
        turn_id="t1",
        message_id=memory_id,
        metadata=metadata or {},
    )


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        index_path=tmp_path / "memory.index",
        metadata_path=tmp_path / "memory.json",
        embedding_dim=3,
    )


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root_dir=tmp_path,
        settings_path=tmp_path / "settings.yaml",
        raw={
            "llm": {
                "host": "http://localhost:11434",
                "model": "llama3:8b",
                "temperature": 0.2,
                "max_tokens": 100,
                "timeout_seconds": 10,
            },
            "embeddings": {"model": "all-MiniLM-L6-v2"},
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


# ---------------------------------------------------------------------------
# Stub collaborators used across pipeline and integration tests
# ---------------------------------------------------------------------------

class StubEncoder:
    """Deterministic encoder: milk-related text maps to [1,0,0], everything else [0,1,0]."""

    def encode(self, text: str) -> np.ndarray:
        if "milk" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype="float32")
        return np.array([0.0, 1.0, 0.0], dtype="float32")


class StubOllama:
    """Always returns a fixed stub response."""

    def generate(self, prompt: str) -> str:
        return "stub-response"


class StubExtractionOllama:
    """Returns a pre-configured response string for structured extraction tests."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


class QueueExtractionOllama:
    """Pops from a queue of responses; raises on exhaustion."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def generate(self, prompt: str) -> str:
        del prompt
        if not self.responses:
            raise AssertionError("No queued extraction response")
        return self.responses.pop(0)
