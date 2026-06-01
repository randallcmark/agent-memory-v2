"""CLI for the experiment harness: run cells, run the matrix, build snapshots.

Commands:
  run        one (arm × scenario × lifecycle) cell, N iterations
  matrix     the full grid across arms × scenarios × lifecycles × iterations
  build-snapshots   build seeded/specific snapshot libraries
  list       list available experiment scenarios

The harness produces transcripts + structured per-turn JSONL only; analysis is
external (no scoring, no judge).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_memory_v2.config import load_config
from agent_memory_v2.experiment.build_snapshots import (
    build_seed_snapshot,
    build_specific_snapshot,
    load_persona,
)
from agent_memory_v2.experiment.runner import LIFECYCLES, RunSpec, run_scenario
from agent_memory_v2.experiment.scenarios import (
    Scenario,
    find_scenario,
    load_scenarios,
)

DEFAULT_SCENARIO_DIR = "evals/experiment_scenarios"
DEFAULT_PERSONA_DIR = "evals/experiment_personas"
DEFAULT_OUTPUT = "artifacts/experiments"
DEFAULT_SNAPSHOT_LIB = "experiments/snapshots"


def _applicable_lifecycles(scenario: Scenario, requested: list[str]) -> list[str]:
    out = []
    for lc in requested:
        # compiled_resumed needs >=2 phases.
        if lc == "compiled_resumed" and len(scenario.phases) < 2:
            continue
        # seeded needs a declared persona seed (otherwise no snapshot to load).
        if lc == "seeded" and not scenario.seed:
            continue
        # specific needs ground-truth memories to have been snapshotted.
        if lc == "specific" and not scenario.ground_truth_memories:
            continue
        out.append(lc)
    return out


def _run_cell(
    *,
    base_config,
    scenario: Scenario,
    arm: str,
    lifecycle: str,
    iterations: int,
    provider: str,
    model: str | None,
    temperature: float,
    exp_id: str,
    output_root: Path,
    snapshot_library: Path,
) -> list[dict[str, Any]]:
    results = []
    for iteration in range(iterations):
        run_dir = output_root / exp_id / arm / scenario.name / lifecycle / f"iter_{iteration}"
        spec = RunSpec(
            exp_id=exp_id,
            arm=arm,
            lifecycle=lifecycle,
            iteration=iteration,
            generator_name=provider,
            model=model,
            temperature=temperature,
            run_dir=run_dir,
            snapshot_library=snapshot_library,
        )
        results.append(run_scenario(base_config=base_config, scenario=scenario, spec=spec))
    return results


def _cmd_run(args: argparse.Namespace) -> int:
    base_config = load_config()
    scenario_dir = base_config.root_dir / args.scenario_dir
    scenario = find_scenario(scenario_dir, args.scenario)
    output_root = base_config.root_dir / args.output
    snapshot_library = base_config.root_dir / args.snapshot_lib
    exp_id = args.exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    arms = args.arms.split(",")
    lifecycles = _applicable_lifecycles(scenario, args.lifecycles.split(","))

    all_results = []
    for arm in arms:
        for lifecycle in lifecycles:
            all_results.extend(
                _run_cell(
                    base_config=base_config,
                    scenario=scenario,
                    arm=arm,
                    lifecycle=lifecycle,
                    iterations=args.iterations,
                    provider=args.provider,
                    model=args.model,
                    temperature=args.temperature,
                    exp_id=exp_id,
                    output_root=output_root,
                    snapshot_library=snapshot_library,
                )
            )
    print(
        json.dumps({"exp_id": exp_id, "cells": len(all_results), "results": all_results}, indent=2)
    )
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    base_config = load_config()
    scenario_dir = base_config.root_dir / args.scenario_dir
    scenarios = load_scenarios(scenario_dir)
    output_root = base_config.root_dir / args.output
    snapshot_library = base_config.root_dir / args.snapshot_lib
    exp_id = args.exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    arms = args.arms.split(",")
    requested_lifecycles = args.lifecycles.split(",")

    all_results = []
    for scenario in scenarios:
        lifecycles = _applicable_lifecycles(scenario, requested_lifecycles)
        for arm in arms:
            for lifecycle in lifecycles:
                all_results.extend(
                    _run_cell(
                        base_config=base_config,
                        scenario=scenario,
                        arm=arm,
                        lifecycle=lifecycle,
                        iterations=args.iterations,
                        provider=args.provider,
                        model=args.model,
                        temperature=args.temperature,
                        exp_id=exp_id,
                        output_root=output_root,
                        snapshot_library=snapshot_library,
                    )
                )
    print(
        json.dumps({"exp_id": exp_id, "cells": len(all_results), "results": all_results}, indent=2)
    )
    return 0


def _cmd_build_snapshots(args: argparse.Namespace) -> int:
    import tempfile

    base_config = load_config()
    scenario_dir = base_config.root_dir / args.scenario_dir
    persona_dir = base_config.root_dir / args.persona_dir
    snapshot_library = base_config.root_dir / args.snapshot_lib
    built = []

    # specific snapshots: one per scenario that declares ground-truth memories.
    for scenario in load_scenarios(scenario_dir):
        if not scenario.ground_truth_memories:
            continue
        with tempfile.TemporaryDirectory(prefix="amv2-exp-specific-") as td:
            info = build_specific_snapshot(
                base_config=base_config,
                scenario=scenario,
                work_root=Path(td),
                library=snapshot_library,
            )
        built.append({"kind": "specific", "scenario": scenario.name, **info})

    # seed snapshots: one per persona file.
    if persona_dir.exists():
        for persona_path in sorted(persona_dir.glob("*.json")):
            persona_name = persona_path.stem
            turns = load_persona(persona_path)
            with tempfile.TemporaryDirectory(prefix="amv2-exp-seed-") as td:
                info = build_seed_snapshot(
                    base_config=base_config,
                    persona_name=persona_name,
                    turns=turns,
                    work_root=Path(td),
                    library=snapshot_library,
                )
            built.append({"kind": "seed", "persona": persona_name, **info})

    print(json.dumps({"snapshot_library": str(snapshot_library), "built": built}, indent=2))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    base_config = load_config()
    scenario_dir = base_config.root_dir / args.scenario_dir
    scenarios = load_scenarios(scenario_dir)
    out = [
        {
            "name": s.name,
            "description": s.description,
            "phases": len(s.phases),
            "probes": len(s.probes),
            "seed": s.seed,
            "ground_truth_memories": len(s.ground_truth_memories),
        }
        for s in scenarios
    ]
    print(json.dumps({"scenario_dir": str(scenario_dir), "scenarios": out}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="agent-memory-v2 experiment harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--arms", default="A,C", help="comma-separated arms (A,C)")
        p.add_argument(
            "--lifecycles",
            default=",".join(LIFECYCLES),
            help="comma-separated lifecycles",
        )
        p.add_argument("--iterations", type=int, default=1)
        p.add_argument("--provider", default="fake", choices=["fake", "anthropic", "claude"])
        p.add_argument("--model", default=None)
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--exp-id", default=None)
        p.add_argument("--scenario-dir", default=DEFAULT_SCENARIO_DIR)
        p.add_argument("--persona-dir", default=DEFAULT_PERSONA_DIR)
        p.add_argument("--output", default=DEFAULT_OUTPUT)
        p.add_argument("--snapshot-lib", default=DEFAULT_SNAPSHOT_LIB)

    run_cmd = sub.add_parser("run", help="run one scenario across arms/lifecycles")
    run_cmd.add_argument("--scenario", required=True)
    add_common(run_cmd)
    run_cmd.set_defaults(func=_cmd_run)

    matrix_cmd = sub.add_parser("matrix", help="run all scenarios across arms/lifecycles")
    add_common(matrix_cmd)
    matrix_cmd.set_defaults(func=_cmd_matrix)

    build_cmd = sub.add_parser("build-snapshots", help="build seeded/specific snapshot libraries")
    build_cmd.add_argument("--scenario-dir", default=DEFAULT_SCENARIO_DIR)
    build_cmd.add_argument("--persona-dir", default=DEFAULT_PERSONA_DIR)
    build_cmd.add_argument("--snapshot-lib", default=DEFAULT_SNAPSHOT_LIB)
    build_cmd.set_defaults(func=_cmd_build_snapshots)

    list_cmd = sub.add_parser("list", help="list experiment scenarios")
    list_cmd.add_argument("--scenario-dir", default=DEFAULT_SCENARIO_DIR)
    list_cmd.set_defaults(func=_cmd_list)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
