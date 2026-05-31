"""Experiment scenario format and loader.

Distinct from ``evals/scenarios.json`` (the agent-eval format). An experiment
scenario describes a multi-phase, multi-session interaction plus probe turns whose
expectations are carried verbatim into the journal as labels for external analysis.

Schema (JSON file under ``evals/experiment_scenarios/``):

    {
      "name": "single_fact_recall",
      "description": "...",
      "seed": "seed_persona" | null,
      "phases": [
        {"session": "s1", "turns": ["...", "..."], "snapshot_after": false}
      ],
      "probes": [
        {"phase": "s1", "turn": 2,
         "expected_contains": ["..."], "expected_not_contains": ["..."]}
      ],
      "ground_truth_memories": [{"key": "identity.name", "value": "Mark"}]
    }

``turn`` in a probe is the zero-based index of the turn *within its phase*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Phase:
    session: str
    turns: list[str]
    snapshot_after: bool = False


@dataclass(frozen=True)
class Probe:
    phase: str
    turn: int
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    phases: list[Phase]
    probes: list[Probe]
    seed: str | None = None
    ground_truth_memories: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def probe_for(self, phase: str, turn: int) -> Probe | None:
        for probe in self.probes:
            if probe.phase == phase and probe.turn == turn:
                return probe
        return None


def _parse_scenario(data: dict[str, Any]) -> Scenario:
    phases = [
        Phase(
            session=str(p.get("session") or f"s{i+1}"),
            turns=[str(t) for t in (p.get("turns") or [])],
            snapshot_after=bool(p.get("snapshot_after", False)),
        )
        for i, p in enumerate(data.get("phases") or [])
    ]
    probes = [
        Probe(
            phase=str(pr.get("phase") or ""),
            turn=int(pr.get("turn", 0)),
            expected_contains=[str(x) for x in (pr.get("expected_contains") or [])],
            expected_not_contains=[str(x) for x in (pr.get("expected_not_contains") or [])],
        )
        for pr in (data.get("probes") or [])
    ]
    return Scenario(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        phases=phases,
        probes=probes,
        seed=(str(data["seed"]) if data.get("seed") else None),
        ground_truth_memories=list(data.get("ground_truth_memories") or []),
        raw=data,
    )


def load_scenario_file(path: Path) -> Scenario:
    with Path(path).open("r", encoding="utf-8") as handle:
        return _parse_scenario(json.load(handle))


def load_scenarios(directory: Path) -> list[Scenario]:
    directory = Path(directory)
    scenarios: list[Scenario] = []
    for path in sorted(directory.glob("*.json")):
        scenarios.append(load_scenario_file(path))
    return scenarios


def find_scenario(directory: Path, name: str) -> Scenario:
    for scenario in load_scenarios(directory):
        if scenario.name == name:
            return scenario
    raise RuntimeError(f"Unknown experiment scenario: {name}")
