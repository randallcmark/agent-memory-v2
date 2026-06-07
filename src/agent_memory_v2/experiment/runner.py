"""Run a scenario through an arm under a given lifecycle, journaling every turn.

Ties together: scenario phases/sessions, the MemoryController (Arm A/C), snapshot
lifecycles, and the Journal. No scoring — probe expectations are attached to the
relevant journal records as labels for external analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_memory_v2.config import AppConfig
from agent_memory_v2.experiment.controller import MemoryController, Session
from agent_memory_v2.experiment.generators import build_generator
from agent_memory_v2.experiment.journal import Journal, build_manifest
from agent_memory_v2.experiment.scenarios import Scenario
from agent_memory_v2.experiment.snapshots import (
    snapshot_fingerprint,
    snapshot_load,
    snapshot_save,
)

LIFECYCLES = ("cold_start", "seeded", "specific", "compiled_resumed")


@dataclass
class RunSpec:
    exp_id: str
    arm: str
    lifecycle: str
    iteration: int
    generator_name: str
    model: str | None
    temperature: float
    run_dir: Path
    snapshot_library: Path


def _resolve_input_snapshot(scenario: Scenario, lifecycle: str, library: Path) -> Path | None:
    """Snapshot to preload before the run, or None for cold_start."""
    if lifecycle == "cold_start":
        return None
    if lifecycle == "seeded":
        name = scenario.seed or f"seed_{scenario.name}"
        return library / name
    if lifecycle == "specific":
        return library / f"specific_{scenario.name}"
    # compiled_resumed preloads nothing; it builds its own mid-run snapshot.
    return None


def run_scenario(
    *,
    base_config: AppConfig,
    scenario: Scenario,
    spec: RunSpec,
) -> dict[str, Any]:
    """Execute one (arm × scenario × lifecycle × iteration) cell."""
    run_root = spec.run_dir / "state"
    run_root.mkdir(parents=True, exist_ok=True)

    input_snapshot = _resolve_input_snapshot(scenario, spec.lifecycle, spec.snapshot_library)
    snapshot_fp: dict[str, Any] | None = None
    if input_snapshot is not None:
        load_info = snapshot_load(input_snapshot, run_root)
        snapshot_fp = {k: load_info[k] for k in ("file_count", "total_bytes", "fingerprint")}

    generator = build_generator(spec.generator_name, model=spec.model, temperature=spec.temperature)
    controller = MemoryController(
        base_config=base_config, root_dir=run_root, generator=generator, arm=spec.arm
    )

    journal = Journal(spec.run_dir)
    manifest = build_manifest(
        exp_id=spec.exp_id,
        arm=spec.arm,
        scenario=scenario.name,
        lifecycle=spec.lifecycle,
        iteration=spec.iteration,
        config=controller.config,
        generator_name=generator.name,
        model=getattr(generator, "model", spec.model or ""),
        temperature=spec.temperature,
        snapshot=(str(input_snapshot) if input_snapshot else None),
        snapshot_fingerprint=snapshot_fp,
        repo_root=base_config.root_dir,
        extra={"scenario_description": scenario.description, "seed": scenario.seed},
    )
    journal.write_manifest(manifest)

    turns_recorded = 0
    is_resume = spec.lifecycle == "compiled_resumed"
    mid_snapshot: Path | None = None

    for phase_index, phase in enumerate(scenario.phases):
        # compiled_resumed: between phases, snapshot then rebuild controller on a
        # fresh root with a fresh session (empty in-context history) and reload.
        if is_resume and phase_index > 0 and mid_snapshot is not None:
            run_root = spec.run_dir / f"state_phase{phase_index}"
            run_root.mkdir(parents=True, exist_ok=True)
            snapshot_load(mid_snapshot, run_root)
            controller = MemoryController(
                base_config=base_config, root_dir=run_root, generator=generator, arm=spec.arm
            )

        session = Session(session_id=f"{spec.exp_id}:{phase.session}:{spec.iteration}")
        for turn_index, user_text in enumerate(phase.turns):
            result = controller.run_turn(session, user_text)
            probe = scenario.probe_for(phase.session, turn_index)
            scenario_meta: dict[str, Any] = {
                "phase_session": phase.session,
                "turn_in_phase": turn_index,
                "ground_truth_memories": scenario.ground_truth_memories,
            }
            if probe is not None:
                scenario_meta["probe"] = {
                    "expected_contains": probe.expected_contains,
                    "expected_not_contains": probe.expected_not_contains,
                }
            journal.record_turn(
                manifest=manifest,
                session_id=session.session_id,
                phase=phase.session,
                result=result,
                scenario_meta=scenario_meta,
            )
            turns_recorded += 1

        if phase.snapshot_after or (is_resume and phase_index == 0):
            mid_snapshot = spec.run_dir / f"_mid_snapshot_after_{phase.session}"
            snapshot_save(run_root, mid_snapshot)

    # Persist the final memory state as a snapshot for inspection / reuse.
    final_snapshot = spec.run_dir / "final_memory_snapshot"
    snapshot_save(run_root, final_snapshot)

    return {
        "exp_id": spec.exp_id,
        "arm": spec.arm,
        "scenario": scenario.name,
        "lifecycle": spec.lifecycle,
        "iteration": spec.iteration,
        "turns_recorded": turns_recorded,
        "run_dir": str(spec.run_dir),
        "final_snapshot": str(final_snapshot),
        "final_fingerprint": snapshot_fingerprint(final_snapshot),
        "manifest_path": str(journal.manifest_path),
        "journal_path": str(journal.journal_path),
    }
