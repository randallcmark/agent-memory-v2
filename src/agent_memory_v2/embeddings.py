from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from agent_memory_v2.ollama import OllamaClient, OllamaProfile


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingEncoder:
    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError


class HashEmbeddingEncoder(EmbeddingEncoder):
    def __init__(self, dimensions: int) -> None:
        self.dimensions = int(dimensions)

    def encode(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("encode(text) requires a non-empty string")
        vector = np.zeros(self.dimensions, dtype="float32")
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:8], "big") % self.dimensions
            vector[slot] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("Failed to produce a non-zero embedding")
        return vector / norm


class SentenceTransformerEncoder(EmbeddingEncoder):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = _get_model(model_name)

    def encode(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("encode(text) requires a non-empty string")
        vector = self.model.encode(text, normalize_embeddings=True)
        array = np.asarray(vector, dtype="float32")
        if array.ndim != 1:
            raise ValueError(f"Expected 1D embedding vector, got shape {array.shape}")
        return array


class OllamaEmbeddingEncoder(EmbeddingEncoder):
    def __init__(self, *, host: str, model_name: str, timeout_seconds: int) -> None:
        self.client = OllamaClient(
            OllamaProfile(
                host=host,
                model=model_name,
                temperature=0.0,
                max_tokens=1,
                timeout_seconds=timeout_seconds,
            )
        )

    def encode(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("encode(text) requires a non-empty string")
        vector = self.client.embed(text)
        array = np.asarray(vector, dtype="float32")
        if array.ndim != 1:
            raise ValueError(f"Expected 1D embedding vector, got shape {array.shape}")
        norm = np.linalg.norm(array)
        if norm == 0.0:
            raise ValueError("Ollama embedding vector had zero norm")
        return array / norm


def build_encoder(
    *,
    provider: str,
    model_name: str,
    dimensions: int,
    host: str | None = None,
    timeout_seconds: int = 60,
) -> EmbeddingEncoder:
    if provider == "hash":
        return HashEmbeddingEncoder(dimensions)
    if provider == "ollama":
        if not host:
            raise ValueError("Ollama embeddings require a host")
        return OllamaEmbeddingEncoder(
            host=host,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
        )
    if provider == "sentence-transformers":
        return SentenceTransformerEncoder(model_name)
    raise ValueError(f"Unsupported embedding provider: {provider}")
