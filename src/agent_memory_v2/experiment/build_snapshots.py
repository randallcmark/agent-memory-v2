"""Build the snapshot libraries used by the ``seeded`` and ``specific`` lifecycles.

- specific_<scenario>: hand-authored exact target records written via the pipeline's
  storage path, with no distractors — the cleanest recall condition.
- seed_<persona>: replay a persona's turns through the Arm A controller, then snapshot.

Personas are intentionally minimal (decision #1: keep Claude as Claude); a persona
file is a list of plain user turns under ``evals/experiment_personas/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_memory_v2.config import AppConfig
from agent_memory_v2.experiment.config import experiment_config
from agent_memory_v2.experiment.controller import MemoryController, Session
from agent_memory_v2.experiment.generators import FakeGenerator
from agent_memory_v2.experiment.scenarios import Scenario
from agent_memory_v2.experiment.snapshots import snapshot_save
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline


def build_specific_snapshot(
    *, base_config: AppConfig, scenario: Scenario, work_root: Path, library: Path
) -> dict[str, Any]:
    """Write exactly the scenario's ground-truth memories, then snapshot."""
    config = experiment_config(base_config, work_root)
    pipeline = MemoryPipeline(config)
    for gt in scenario.ground_truth_memories:
        key = str(gt.get("key") or "")
        value = str(gt.get("value") or "")
        memory_class = str(gt.get("memory_class") or _class_from_key(key))
        text = str(gt.get("text") or value)
        message = Message(role="agent", text=text, conversation_id=f"specific:{scenario.name}")
        pipeline._store_memory(  # noqa: SLF001 - intentional direct seed write
            role=memory_class,
            text=text,
            summary=text,
            timestamp=message.timestamp,
            conversation_id=message.conversation_id,
            turn_id=message.turn_id,
            message_id=message.message_id,
            metadata={
                "kind": "sidecar_memory",
                "memory_class": memory_class,
                "extracted_value": value,
                "classification_confidence": 1.0,
                "durable": True,
                "durability_reason": "experiment_seed",
                "profile_key": key,
                "classification_source": "experiment_seed",
            },
        )
    dest = library / f"specific_{scenario.name}"
    return snapshot_save(work_root, dest)


def build_seed_snapshot(
    *, base_config: AppConfig, persona_name: str, turns: list[str], work_root: Path, library: Path
) -> dict[str, Any]:
    """Replay persona turns through the Arm A controller, then snapshot."""
    controller = MemoryController(
        base_config=base_config, root_dir=work_root, generator=FakeGenerator(), arm="A"
    )
    session = Session(session_id=f"seed:{persona_name}")
    for turn in turns:
        controller.run_turn(session, turn)
    dest = library / f"seed_{persona_name}"
    return snapshot_save(work_root, dest)


def load_persona(path: Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("turns") or []
    return [str(t) for t in data]


def _class_from_key(key: str) -> str:
    prefix = key.split(".", 1)[0] if key else ""
    if prefix == "preference":
        return "preference"
    if prefix == "task":
        return "task"
    return "fact"
