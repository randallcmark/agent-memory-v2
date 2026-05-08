from __future__ import annotations

import json
from pathlib import Path

from agent_memory_v2.scenario_cli import compare_runs, load_scenarios


def test_load_scenarios_returns_named_entries(tmp_path: Path) -> None:
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(
        json.dumps({"scenarios": [{"name": "example", "query": "Hello", "setup_turns": []}]}),
        encoding="utf-8",
    )
    scenarios = load_scenarios(dataset)
    assert scenarios[0]["name"] == "example"


def test_compare_runs_reports_prompt_and_response_differences() -> None:
    comparison = compare_runs(
        {
            "scenario": {"name": "a"},
            "artifact_dir": "/tmp/a",
            "query": {"text": "Q"},
            "prompt": "prompt a",
            "response": "response a",
            "merged_recall": {"merged": [{"text": "A"}]},
        },
        {
            "scenario": {"name": "b"},
            "artifact_dir": "/tmp/b",
            "query": {"text": "Q"},
            "prompt": "prompt b",
            "response": "response b",
            "merged_recall": {"merged": [{"text": "B"}]},
        },
    )
    assert comparison["query_same"] is True
    assert comparison["response_same"] is False
    assert comparison["prompt_same"] is False
