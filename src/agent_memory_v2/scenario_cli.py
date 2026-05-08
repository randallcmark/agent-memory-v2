"""Scenario CLI: runs, lists, and compares qualitative memory scenarios for manual review."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_memory_v2.config import AppConfig, load_config
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline, run_ollama_preflight
from agent_memory_v2.sentiment import detect_sentiment


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    scenarios = data.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise RuntimeError("scenario dataset must contain a 'scenarios' list")
    return scenarios


def _find_scenario(path: Path, name: str) -> dict[str, Any]:
    for scenario in load_scenarios(path):
        if scenario.get("name") == name:
            return scenario
    raise RuntimeError(f"Unknown scenario: {name}")


def _isolated_config(base: AppConfig, root_dir: Path) -> AppConfig:
    raw = copy.deepcopy(base.raw)
    raw["memory"]["index_path"] = "data/memory/memory.index"
    raw["memory"]["metadata_path"] = "data/memory/memory_metadata.json"
    raw["memory"]["interaction_log_path"] = "data/logs/interactions.jsonl"
    raw["sidecar"]["index_path"] = "data/sidecar/sidecar.index"
    raw["sidecar"]["metadata_path"] = "data/sidecar/sidecar_metadata.json"
    raw["profile"]["path"] = "data/profile/profile.json"
    raw["maintenance"]["enabled"] = False
    raw["llm"]["preflight"]["enabled"] = False
    return AppConfig(root_dir=root_dir, settings_path=base.settings_path, raw=raw)


def _artifact_dir(base_dir: Path, scenario_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{stamp}_{scenario_name}"


def _seed_turns(pipeline: MemoryPipeline, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    ingested: list[dict[str, Any]] = []
    for idx, turn in enumerate(scenario.get("setup_turns", []), start=1):
        turn_id = f"{scenario['name']}-turn-{idx}"
        user_message = Message(
            role="user",
            text=turn["user"],
            conversation_id=scenario["name"],
            turn_id=turn_id,
        )
        agent_message = Message(
            role="agent",
            text=turn["agent"],
            conversation_id=scenario["name"],
            turn_id=turn_id,
        )
        record = pipeline.ingest_turn(user_message, agent_message)
        ingested.append(
            {
                "user_message": user_message.__dict__,
                "agent_message": agent_message.__dict__,
                "stored_record": {
                    "memory_id": record.memory_id,
                    "role": record.role,
                    "summary": record.summary,
                    "metadata": record.metadata,
                },
            }
        )
    return ingested


def run_scenario(
    base_config: AppConfig,
    scenario: dict[str, Any],
    *,
    artifact_base: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="agent-memory-v2-scenario-") as temp_dir:
        config = _isolated_config(base_config, Path(temp_dir))
        pipeline = MemoryPipeline(config)
        ingested = _seed_turns(pipeline, scenario)
        query = Message(role="user", text=scenario["query"], conversation_id=scenario["name"])
        merged = pipeline.merged_recall(query)
        prompt_context = pipeline.prompt_context(merged["merged"])
        sentiment = detect_sentiment(query.text)
        prompt = pipeline.build_prompt(query, merged["merged"])
        response = pipeline.ollama.generate(prompt)
        artifact_dir = _artifact_dir(artifact_base, scenario["name"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "scenario": scenario,
            "ingested": ingested,
            "query": query.__dict__,
            "sentiment": {
                "label": sentiment.label,
                "confidence": sentiment.confidence,
                "cues": sentiment.cues,
                "guidance": sentiment.guidance,
            },
            "merged_recall": merged,
            "prompt_context": prompt_context,
            "prompt": prompt,
            "response": response,
        }
        with (artifact_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        payload["artifact_dir"] = str(artifact_dir)
        return payload


def _load_run(path_or_id: str, artifact_base: Path) -> dict[str, Any]:
    direct = Path(path_or_id)
    if direct.exists():
        target = direct
    else:
        matches = sorted(artifact_base.glob(f"*{path_or_id}*"))
        if not matches:
            raise RuntimeError(f"No scenario run found for: {path_or_id}")
        target = matches[-1]
    result_path = target / "result.json"
    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["artifact_dir"] = str(target)
    return data


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_a": a["scenario"]["name"],
        "scenario_b": b["scenario"]["name"],
        "artifact_a": a["artifact_dir"],
        "artifact_b": b["artifact_dir"],
        "query_same": a["query"]["text"] == b["query"]["text"],
        "response_same": a.get("response") == b.get("response"),
        "prompt_same": a.get("prompt") == b.get("prompt"),
        "response_a": a.get("response"),
        "response_b": b.get("response"),
        "prompt_a": a.get("prompt"),
        "prompt_b": b.get("prompt"),
        "recalled_a": a.get("merged_recall", {}).get("merged", []),
        "recalled_b": b.get("merged_recall", {}).get("merged", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualitative scenario workflow for agent_memory_v2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cmd = subparsers.add_parser("list")
    list_cmd.add_argument("--dataset", default="evals/scenarios.json")

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument("--scenario", required=True)
    run_cmd.add_argument("--dataset", default="evals/scenarios.json")
    run_cmd.add_argument("--artifact-dir", default="artifacts/scenarios")
    run_cmd.add_argument("--skip-preflight", action="store_true")

    show_cmd = subparsers.add_parser("show")
    show_cmd.add_argument("--run-id", required=True)
    show_cmd.add_argument("--artifact-dir", default="artifacts/scenarios")

    compare_cmd = subparsers.add_parser("compare")
    compare_cmd.add_argument("--run-a", required=True)
    compare_cmd.add_argument("--run-b", required=True)
    compare_cmd.add_argument("--artifact-dir", default="artifacts/scenarios")

    args = parser.parse_args()
    config = load_config()

    if args.command == "list":
        scenarios = load_scenarios(config.root_dir / args.dataset)
        print(json.dumps({"scenarios": scenarios}, indent=2))
        raise SystemExit(0)

    artifact_base = config.root_dir / getattr(args, "artifact_dir", "artifacts/scenarios")

    if args.command == "show":
        print(json.dumps(_load_run(args.run_id, artifact_base), indent=2))
        raise SystemExit(0)

    if args.command == "compare":
        run_a = _load_run(args.run_a, artifact_base)
        run_b = _load_run(args.run_b, artifact_base)
        print(json.dumps({"comparison": compare_runs(run_a, run_b)}, indent=2))
        raise SystemExit(0)

    if not args.skip_preflight:
        preflight = run_ollama_preflight(config)
        if not preflight.get("reachable") or not preflight.get("model_present"):
            print(json.dumps({"ok": False, "preflight": preflight}, indent=2))
            raise SystemExit(2)

    scenario = _find_scenario(config.root_dir / args.dataset, args.scenario)
    result = run_scenario(config, scenario, artifact_base=artifact_base)
    print(json.dumps(result, indent=2))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
