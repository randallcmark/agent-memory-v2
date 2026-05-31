"""Experiment harness: agent-memory-v2 as a controller wrapped around Claude.

This package wires the memory pipeline around a Claude (Anthropic API) generation
call so we can run controlled memory experiments and journal every turn.

Arms implemented here:
- ``A`` — curated controller: recall + neutral memory injection + ingest around each turn.
- ``C`` — control: no memory at all.

The agentic / MCP arm (``B``) is deferred; see
``docs/exec-plans/active/memory-controller.md``.
"""

from __future__ import annotations

__all__ = [
    "experiment_config",
    "MemoryController",
    "Session",
    "TurnResult",
    "Journal",
    "build_manifest",
]

from agent_memory_v2.experiment.config import experiment_config
from agent_memory_v2.experiment.controller import (
    MemoryController,
    Session,
    TurnResult,
)
from agent_memory_v2.experiment.journal import Journal, build_manifest
