from pathlib import Path

import numpy as np

from agent_memory_v2.models import MemoryRecord
from agent_memory_v2.store import MemoryStore


def test_store_add_and_search(tmp_path: Path):
    store = MemoryStore(
        index_path=tmp_path / "memory.index",
        metadata_path=tmp_path / "memory.json",
        embedding_dim=3,
    )
    record = MemoryRecord(
        memory_id="m1",
        role="user",
        text="remember milk",
        summary="remember milk",
        timestamp="2026-01-01T00:00:00+00:00",
        conversation_id="default",
        turn_id="t1",
        message_id="m1",
        metadata={},
    )
    store.add(record, np.array([1.0, 0.0, 0.0], dtype="float32"))
    results = store.search(
        np.array([1.0, 0.0, 0.0], dtype="float32"),
        top_k=3,
        similarity_threshold=0.1,
    )
    assert len(results) == 1
    assert results[0].record.text == "remember milk"
