"""Live eval CLI: runs recall and memory pipeline evals against a live Ollama instance."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_memory_v2.config import AppConfig, load_config
from agent_memory_v2.eval_history import (
    compact_summary,
    compare_summaries,
    find_previous_comparable,
    load_history,
    write_history,
)
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline, run_ollama_preflight


def _load_dataset(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("live eval dataset must be a dictionary")
    return data


def _isolated_live_config(base: AppConfig, root_dir: Path) -> AppConfig:
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


def _build_pipeline(base: AppConfig, temp_root: Path) -> MemoryPipeline:
    config = _isolated_live_config(base, temp_root)
    return MemoryPipeline(config)


def _seed_turns(pipeline: MemoryPipeline, turns: list[dict[str, str]], conversation_id: str) -> None:
    for idx, turn in enumerate(turns, start=1):
        turn_id = f"{conversation_id}-turn-{idx}"
        user_message = Message(
            role="user",
            text=turn["user"],
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        agent_message = Message(
            role="agent",
            text=turn["agent"],
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        pipeline.ingest_turn(user_message, agent_message)


def _contains_all(response: str, snippets: list[str]) -> tuple[bool, list[str]]:
    lowered = response.lower()
    missing = [snippet for snippet in snippets if snippet.lower() not in lowered]
    return not missing, missing


def _contains_none(response: str, snippets: list[str]) -> tuple[bool, list[str]]:
    lowered = response.lower()
    present = [snippet for snippet in snippets if snippet.lower() in lowered]
    return not present, present


def _contains_any(response: str, snippets: list[str]) -> tuple[bool, list[str]]:
    lowered = response.lower()
    matched = [snippet for snippet in snippets if snippet.lower() in lowered]
    return bool(matched), matched


def _stage_summary(stage: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed_cases = sum(1 for item in results if item["passed"])
    return {
        "stage": stage,
        "passed": passed_cases == total,
        "passed_cases": passed_cases,
        "failed_cases": total - passed_cases,
        "total_cases": total,
        "results": results,
    }


def _artifact_dir(base_dir: Path, stage: str, case_name: str) -> Path:
    return base_dir / stage / case_name


def _write_artifacts(base_dir: Path, stage: str, case_name: str, payload: dict[str, Any]) -> None:
    target = _artifact_dir(base_dir, stage, case_name)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _run_case(
    case: dict[str, Any],
    *,
    stage: str,
    base_config: AppConfig,
    artifact_root: Path | None,
    save_all: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"agent-memory-v2-live-{stage}-") as temp_dir:
        pipeline = _build_pipeline(base_config, Path(temp_dir))
        _seed_turns(pipeline, case["turns"], case["name"])
        query = Message(role="user", text=case["query"], conversation_id=case["name"])
        recalled = pipeline.recall(query)
        prompt_context = pipeline.prompt_context(recalled)
        prompt = pipeline.build_prompt(query, recalled)
        response = pipeline.ollama.generate(prompt)

        expected_contains = case.get("expected_contains", [])
        expected_not_contains = case.get("expected_not_contains", [])
        expected_any_contains = case.get("expected_any_contains", [])

        contains_ok, missing = _contains_all(response, expected_contains)
        none_ok, forbidden = _contains_none(response, expected_not_contains)
        any_ok, matched_any = _contains_any(response, expected_any_contains) if expected_any_contains else (True, [])

        passed = contains_ok and none_ok and any_ok
        payload = {
            "name": case["name"],
            "passed": passed,
            "detail": {
                "query": case["query"],
                "expected_contains": expected_contains,
                "expected_not_contains": expected_not_contains,
                "expected_any_contains": expected_any_contains,
                "missing": missing,
                "forbidden_present": forbidden,
                "matched_any": matched_any,
                "recalled": recalled,
                "prompt_context": prompt_context,
                "prompt": prompt,
                "response": response,
            },
        }
        if artifact_root is not None and (save_all or not passed):
            _write_artifacts(artifact_root, stage, case["name"], payload)
            payload["artifact_path"] = str(_artifact_dir(artifact_root, stage, case["name"]))
        return payload


def eval_memory(
    dataset: dict[str, Any],
    *,
    base_config: AppConfig,
    artifact_root: Path | None,
    save_all: bool,
) -> dict[str, Any]:
    results = [
        _run_case(
            case,
            stage="memory",
            base_config=base_config,
            artifact_root=artifact_root,
            save_all=save_all,
        )
        for case in dataset.get("memory", [])
    ]
    return _stage_summary("memory", results)


def eval_sentiment(
    dataset: dict[str, Any],
    *,
    base_config: AppConfig,
    artifact_root: Path | None,
    save_all: bool,
) -> dict[str, Any]:
    results = [
        _run_case(
            case,
            stage="sentiment",
            base_config=base_config,
            artifact_root=artifact_root,
            save_all=save_all,
        )
        for case in dataset.get("sentiment", [])
    ]
    return _stage_summary("sentiment", results)


def eval_all(
    dataset: dict[str, Any],
    *,
    base_config: AppConfig,
    artifact_root: Path | None,
    save_all: bool,
) -> dict[str, Any]:
    stages = [
        eval_memory(dataset, base_config=base_config, artifact_root=artifact_root, save_all=save_all),
        eval_sentiment(dataset, base_config=base_config, artifact_root=artifact_root, save_all=save_all),
    ]
    return {
        "passed": all(stage["passed"] for stage in stages),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Ollama evaluation for agent_memory_v2.")
    parser.add_argument("stage", choices=["memory", "sentiment", "all", "history", "compare"])
    parser.add_argument("--dataset", default="evals/live_ollama.json")
    parser.add_argument("--artifact-dir", default="artifacts/live_eval")
    parser.add_argument("--history-dir", default="artifacts/eval_history/live")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    base_config = load_config()
    history_root = base_config.root_dir / args.history_dir

    if args.stage == "history":
        result = {"history": load_history(history_root)}
        print(json.dumps(result, indent=2))
        raise SystemExit(0)

    if args.stage == "compare":
        history = load_history(history_root)
        current, previous = find_previous_comparable(history)
        if current is None or previous is None:
            result = {"comparison": None, "reason": "need_at_least_two_history_entries"}
        else:
            result = {"comparison": compare_summaries(current, previous)}
        print(json.dumps(result, indent=2))
        raise SystemExit(0)

    if not args.skip_preflight:
        preflight = run_ollama_preflight(base_config)
        if not preflight.get("reachable") or not preflight.get("model_present"):
            print(json.dumps({"passed": False, "preflight": preflight}, indent=2))
            raise SystemExit(2)

    dataset = _load_dataset(base_config.root_dir / args.dataset)
    artifact_root = base_config.root_dir / args.artifact_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.stage == "memory":
        result = eval_memory(
            dataset,
            base_config=base_config,
            artifact_root=artifact_root,
            save_all=args.save_all,
        )
    elif args.stage == "sentiment":
        result = eval_sentiment(
            dataset,
            base_config=base_config,
            artifact_root=artifact_root,
            save_all=args.save_all,
        )
    else:
        result = eval_all(
            dataset,
            base_config=base_config,
            artifact_root=artifact_root,
            save_all=args.save_all,
        )

    if artifact_root.exists():
        result["artifact_root"] = str(artifact_root)

    if args.record_history:
        summary = compact_summary(
            eval_kind="live",
            stage_name=args.stage,
            dataset_path=args.dataset,
            result=result,
            root_dir=base_config.root_dir,
            runtime={
                "llm_model": base_config.llm.get("model"),
                "llm_host": base_config.llm.get("host"),
                "embedding_provider": base_config.embeddings.get("provider"),
                "embedding_model": base_config.embeddings.get("model"),
            },
            artifact_root=result.get("artifact_root"),
        )
        history_path = write_history(history_root, summary)
        result["history_path"] = str(history_path)

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("passed", False) else 1)


if __name__ == "__main__":
    main()
