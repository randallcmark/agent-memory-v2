"""CLI for running agent tool-loop eval scenarios against fake, OpenAI, or Claude providers."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_memory_v2.agent_eval import FakeAgentProvider, run_agent_scenario
from agent_memory_v2.claude_provider import ClaudeProvider
from agent_memory_v2.config import AppConfig, load_config
from agent_memory_v2.eval_history import (
    compact_summary,
    compare_summaries,
    find_previous_comparable,
    load_history,
    write_history,
)
from agent_memory_v2.openai_provider import OpenAIProvider
from agent_memory_v2.scenario_cli import _load_scenarios


def _find_scenario(path: Path, name: str) -> dict[str, Any]:
    for scenario in _load_scenarios(path):
        if scenario.get("name") == name:
            return scenario
    raise RuntimeError(f"Unknown scenario: {name}")


def _artifact_dir(base_dir: Path, provider: str, scenario_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / stamp / provider / scenario_name


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_provider(name: str, scenario: dict[str, Any], *, fake_mode: str = "normal"):
    if name == "fake":
        return FakeAgentProvider(scenario, mode=fake_mode)
    if name == "openai":
        return OpenAIProvider()
    if name in {"anthropic", "claude"}:
        return ClaudeProvider(provider_name=name)
    raise RuntimeError(f"Unsupported provider: {name}")


def _run_one(
    *,
    base_config: AppConfig,
    scenario: dict[str, Any],
    provider_name: str,
    fake_mode: str,
    artifact_base: Path,
    save_all: bool,
    max_tool_calls: int,
) -> dict[str, Any]:
    provider = _build_provider(provider_name, scenario, fake_mode=fake_mode)
    with tempfile.TemporaryDirectory(prefix=f"agent-memory-v2-agent-eval-{provider_name}-") as temp_dir:
        result = run_agent_scenario(
            base_config=base_config,
            scenario=scenario,
            provider=provider,
            temp_root=Path(temp_dir),
            max_tool_calls=max_tool_calls,
        )
    if save_all or not result["passed"]:
        artifact_dir = _artifact_dir(artifact_base, provider_name, scenario["name"])
        _write_artifact(artifact_dir, result)
        result["artifact_path"] = str(artifact_dir)
    return result


def _stage_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed_cases = sum(1 for item in results if item["passed"])
    return {
        "stage": "agent_eval",
        "passed": passed_cases == total,
        "passed_cases": passed_cases,
        "failed_cases": total - passed_cases,
        "total_cases": total,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent tool-loop evaluation for agent_memory_v2.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument("--scenario", required=True)
    run_cmd.add_argument("--provider", choices=["fake", "openai", "anthropic", "claude"], default="fake")
    run_cmd.add_argument("--dataset", default="evals/scenarios.json")
    run_cmd.add_argument("--artifact-dir", default="artifacts/agent_eval")
    run_cmd.add_argument("--save-all", action="store_true")
    run_cmd.add_argument("--record-history", action="store_true")
    run_cmd.add_argument("--history-dir", default="artifacts/eval_history/agent_eval")
    run_cmd.add_argument("--max-tool-calls", type=int, default=5)
    run_cmd.add_argument("--fake-mode", choices=["normal", "invalid", "evolve", "loop"], default="normal")

    run_all_cmd = subparsers.add_parser("run-all")
    run_all_cmd.add_argument("--provider", choices=["fake", "openai", "anthropic", "claude"], default="fake")
    run_all_cmd.add_argument("--dataset", default="evals/scenarios.json")
    run_all_cmd.add_argument("--artifact-dir", default="artifacts/agent_eval")
    run_all_cmd.add_argument("--save-all", action="store_true")
    run_all_cmd.add_argument("--record-history", action="store_true")
    run_all_cmd.add_argument("--history-dir", default="artifacts/eval_history/agent_eval")
    run_all_cmd.add_argument("--max-tool-calls", type=int, default=5)
    run_all_cmd.add_argument("--fake-mode", choices=["normal", "invalid", "evolve", "loop"], default="normal")

    history_cmd = subparsers.add_parser("history")
    history_cmd.add_argument("--history-dir", default="artifacts/eval_history/agent_eval")

    compare_cmd = subparsers.add_parser("compare")
    compare_cmd.add_argument("--history-dir", default="artifacts/eval_history/agent_eval")

    args = parser.parse_args()
    base_config = load_config()
    history_root = base_config.root_dir / getattr(args, "history_dir", "artifacts/eval_history/agent_eval")

    if args.command == "history":
        print(json.dumps({"history": load_history(history_root)}, indent=2))
        raise SystemExit(0)

    if args.command == "compare":
        history = load_history(history_root)
        current, previous = find_previous_comparable(history)
        result = (
            {"comparison": None, "reason": "need_at_least_two_history_entries"}
            if current is None or previous is None
            else {"comparison": compare_summaries(current, previous)}
        )
        print(json.dumps(result, indent=2))
        raise SystemExit(0)

    dataset_path = base_config.root_dir / args.dataset
    artifact_base = base_config.root_dir / args.artifact_dir

    if args.command == "run":
        scenarios = [_find_scenario(dataset_path, args.scenario)]
    else:
        scenarios = _load_scenarios(dataset_path)

    results = [
        _run_one(
            base_config=base_config,
            scenario=scenario,
            provider_name=args.provider,
            fake_mode=args.fake_mode,
            artifact_base=artifact_base,
            save_all=args.save_all,
            max_tool_calls=args.max_tool_calls,
        )
        for scenario in scenarios
    ]
    summary = _stage_summary(results)
    result: dict[str, Any] = {
        "passed": summary["passed"],
        "stages": [summary],
    }
    if args.record_history:
        compact = compact_summary(
            eval_kind="agent_eval",
            stage_name=args.command,
            dataset_path=args.dataset,
            result=result,
            root_dir=base_config.root_dir,
            runtime={
                "provider": args.provider,
                "fake_mode": args.fake_mode,
                "max_tool_calls": args.max_tool_calls,
            },
        )
        history_path = write_history(history_root, compact)
        result["history_path"] = str(history_path)

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
