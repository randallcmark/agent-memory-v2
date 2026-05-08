from pathlib import Path

import numpy as np
from conftest import make_record, make_store


def test_store_add_and_search(tmp_path: Path):
    store = make_store(tmp_path)
    store.add(make_record(), np.array([1.0, 0.0, 0.0], dtype="float32"))
    results = store.search(
        np.array([1.0, 0.0, 0.0], dtype="float32"),
        top_k=3,
        similarity_threshold=0.1,
    )
    assert len(results) == 1
    assert results[0].record.text == "remember milk"


# ---------------------------------------------------------------------------
# update_record_metadata
# ---------------------------------------------------------------------------


def test_update_record_metadata_returns_true_on_success(tmp_path: Path):
    store = make_store(tmp_path)
    store.add(make_record("m1"), np.array([1.0, 0.0, 0.0], dtype="float32"))

    updated = store.update_record_metadata("m1", {"recall_count": 1})

    assert updated is True


def test_update_record_metadata_returns_false_for_unknown_id(tmp_path: Path):
    store = make_store(tmp_path)
    store.add(make_record("m1"), np.array([1.0, 0.0, 0.0], dtype="float32"))

    updated = store.update_record_metadata("does-not-exist", {"recall_count": 1})

    assert updated is False


def test_update_record_metadata_merges_with_existing(tmp_path: Path):
    store = make_store(tmp_path)
    store.add(
        make_record("m1", metadata={"kind": "turn_memory"}),
        np.array([1.0, 0.0, 0.0], dtype="float32"),
    )

    store.update_record_metadata(
        "m1", {"recall_count": 3, "last_recalled_at": "2026-05-01T00:00:00+00:00"}
    )

    record = store.records[0]
    assert record.metadata["kind"] == "turn_memory"
    assert record.metadata["recall_count"] == 3
    assert record.metadata["last_recalled_at"] == "2026-05-01T00:00:00+00:00"


def test_update_record_metadata_persists_across_reload(tmp_path: Path):
    store = make_store(tmp_path)
    store.add(make_record("m1"), np.array([1.0, 0.0, 0.0], dtype="float32"))
    store.update_record_metadata("m1", {"recall_count": 5})

    reloaded = make_store(tmp_path)

    assert reloaded.records[0].metadata["recall_count"] == 5


def test_update_record_metadata_increments_correctly(tmp_path: Path):
    store = make_store(tmp_path)
    store.add(
        make_record("m1", metadata={"recall_count": 2}), np.array([1.0, 0.0, 0.0], dtype="float32")
    )

    current = int(store.records[0].metadata.get("recall_count", 0))
    store.update_record_metadata("m1", {"recall_count": current + 1})

    assert store.records[0].metadata["recall_count"] == 3
