"""Per-run isolated config for the experiment harness.

Unlike ``agent_eval.isolated_agent_config`` (which force-sets ``hash`` embeddings),
this keeps the real embedding provider from ``settings.yaml`` — by decision the
experiment uses ``nomic-embed-text`` via Ollama for recall. Isolation is achieved
purely by relocating ``root_dir`` so every ``data/...`` path resolves under a
disposable per-run directory; the live store is never touched.
"""

from __future__ import annotations

import copy
from pathlib import Path

from agent_memory_v2.config import AppConfig


def experiment_config(base: AppConfig, root_dir: Path) -> AppConfig:
    """Return a copy of ``base`` whose state lives entirely under ``root_dir``.

    All store/sidecar/profile/log paths are resolved relative to ``root_dir`` by
    ``AppConfig.resolve_path``, so simply changing ``root_dir`` isolates the run.
    Embedding provider/model/dimensions are left untouched (Ollama/nomic by default).
    Maintenance is disabled so background compaction can't perturb a run.
    """
    raw = copy.deepcopy(base.raw)
    raw.setdefault("maintenance", {})["enabled"] = False
    # Preflight pings a local generation model we no longer use (llama3 retired).
    if "llm" in raw and isinstance(raw["llm"], dict):
        raw["llm"].setdefault("preflight", {})["enabled"] = False
    return AppConfig(
        root_dir=Path(root_dir).resolve(),
        settings_path=base.settings_path,
        raw=raw,
    )
