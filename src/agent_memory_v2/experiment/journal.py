"""Per-turn JSONL journaling and per-run manifest for the experiment harness.

The deliverable of this harness is data, not verdicts: every turn is appended as
one JSON object to ``journal.jsonl`` and each run gets a ``manifest.json`` capturing
provenance (model, embeddings, snapshot, git sha, exact injection framing). Scenario
ground-truth labels travel verbatim in the journal for Mark's later analysis; the
harness performs no scoring.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_memory_v2.config import AppConfig
from agent_memory_v2.eval_history import git_metadata
from agent_memory_v2.experiment.controller import (
    _HELPER_PREAMBLE,
    BASE_SYSTEM,
    TurnResult,
)


def _settings_hash(config: AppConfig) -> str:
    blob = json.dumps(config.raw, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_manifest(
    *,
    exp_id: str,
    arm: str,
    scenario: str,
    lifecycle: str,
    iteration: int,
    config: AppConfig,
    generator_name: str,
    model: str,
    temperature: float,
    snapshot: str | None,
    snapshot_fingerprint: dict[str, Any] | None,
    repo_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "exp_id": exp_id,
        "arm": arm,
        "scenario": scenario,
        "lifecycle": lifecycle,
        "iteration": iteration,
        "created_at": datetime.now(UTC).isoformat(),
        "generator": generator_name,
        "model": model,
        "temperature": temperature,
        "embeddings": {
            "provider": config.embeddings.get("provider"),
            "model": config.embeddings.get("model"),
            "dimensions": config.embedding_dim,
        },
        "settings_hash": _settings_hash(config),
        "snapshot": snapshot,
        "snapshot_fingerprint": snapshot_fingerprint,
        "git": git_metadata(repo_root),
        # §5 audit trail: the exact framing used for injected memory, so the
        # "helper, not personality" claim is verifiable after the fact.
        "base_system_prompt": BASE_SYSTEM,
        "injection_framing": _HELPER_PREAMBLE,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


class Journal:
    """Append-only JSONL writer for turns, plus a one-shot manifest writer."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.run_dir / "journal.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    def record_turn(
        self,
        *,
        manifest: dict[str, Any],
        session_id: str,
        phase: str | None,
        result: TurnResult,
        scenario_meta: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "exp_id": manifest["exp_id"],
            "arm": manifest["arm"],
            "scenario": manifest["scenario"],
            "lifecycle": manifest["lifecycle"],
            "iteration": manifest["iteration"],
            "session_id": session_id,
            "phase": phase,
            "turn_index": result.turn_index,
            "ts": datetime.now(UTC).isoformat(),
            "user_text": result.user_text,
            "messages_sent": result.messages_sent,
            "recalled_items": result.recalled_items,
            "injected_context": result.injected_context,
            "system_prompt": result.system_prompt,
            "final_prompt": {
                "system": result.system_prompt,
                "messages": result.messages_sent,
            },
            "response": result.response,
            "persisted_memory": result.persisted_memory,
            "usage": result.usage,
            "duration_ms": result.duration_ms,
            "memory_state_after": result.memory_state_after,
            "scenario_meta": scenario_meta or {},
        }
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def turn_to_dict(result: TurnResult) -> dict[str, Any]:
        return asdict(result)
