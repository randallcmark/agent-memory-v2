"""Custom exception hierarchy for agent_memory_v2."""

from __future__ import annotations

__all__ = [
    "AgentMemoryError",
    "MemoryStoreError",
    "EmbeddingError",
    "ConfigError",
    "TaxonomyError",
]


class AgentMemoryError(Exception):
    """Base exception for all agent_memory_v2 errors."""


class MemoryStoreError(AgentMemoryError):
    """Raised when a MemoryStore operation fails (index corruption, dimension mismatch, etc.)."""


class EmbeddingError(AgentMemoryError):
    """Raised when an embedding operation fails (bad input, encoder error, zero-norm vector)."""


class ConfigError(AgentMemoryError):
    """Raised when configuration is invalid or missing required values."""


class TaxonomyError(AgentMemoryError):
    """Raised when the taxonomy cannot be loaded or is structurally invalid."""
