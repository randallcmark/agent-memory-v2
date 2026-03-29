from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from agent_memory_v2.models import MemoryRecord, RecallResult


class MemoryStore:
    def __init__(self, index_path: Path, metadata_path: Path, embedding_dim: int) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.records: list[MemoryRecord] = []
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            self.records = [MemoryRecord(**item) for item in raw]

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump([record.__dict__ for record in self.records], handle, indent=2)

    def add(self, record: MemoryRecord, vector: np.ndarray) -> None:
        row = np.asarray(vector, dtype="float32").reshape(1, -1)
        if row.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {row.shape[1]}, expected {self.embedding_dim}"
            )
        self.index.add(row)
        self.records.append(record)
        self._save()

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.records = []
        self._save()

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        similarity_threshold: float,
        exclude_message_id: str | None = None,
    ) -> list[RecallResult]:
        if not self.records or self.index.ntotal == 0:
            return []
        row = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(row, top_k)
        results: list[RecallResult] = []
        for score, idx in zip(scores[0], ids[0], strict=False):
            if idx < 0:
                continue
            record = self.records[int(idx)]
            if exclude_message_id and record.message_id == exclude_message_id:
                continue
            if float(score) < similarity_threshold:
                continue
            results.append(RecallResult(record=record, score=float(score)))
        return results
