"""Deterministic eval CLI: runs classification, semantic, sentiment, and recall eval suites."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

from agent_memory_v2.classifier import classify_text
from agent_memory_v2.config import AppConfig, load_config
from agent_memory_v2.embeddings import HashEmbeddingEncoder
from agent_memory_v2.eval_history import (
    compact_summary,
    compare_summaries,
    find_previous_comparable,
    load_history,
    write_history,
)
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline
from agent_memory_v2.semantic_router import route_semantic_candidate
from agent_memory_v2.sentiment import detect_sentiment


def _load_dataset(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("eval dataset must be a dictionary")
    return data


def _isolated_config(base: AppConfig, root_dir: Path) -> AppConfig:
    raw = copy.deepcopy(base.raw)
    raw["embeddings"]["provider"] = "hash"
    raw["embeddings"]["model"] = "hash"
    raw["memory"]["index_path"] = "data/memory/memory.index"
    raw["memory"]["metadata_path"] = "data/memory/memory_metadata.json"
    raw["memory"]["interaction_log_path"] = "data/logs/interactions.jsonl"
    raw["sidecar"]["index_path"] = "data/sidecar/sidecar.index"
    raw["sidecar"]["metadata_path"] = "data/sidecar/sidecar_metadata.json"
    raw["profile"]["path"] = "data/profile/profile.json"
    raw["maintenance"]["enabled"] = False
    return AppConfig(root_dir=root_dir, settings_path=base.settings_path, raw=raw)


def _build_pipeline(base: AppConfig, temp_root: Path) -> MemoryPipeline:
    config = _isolated_config(base, temp_root)
    return MemoryPipeline(config)


def _seed_turns(pipeline: MemoryPipeline, turns: list[dict], conversation_id: str) -> None:
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


def _pass_fail(name: str, passed: bool, detail: dict) -> dict:
    return {
        "name": name,
        "passed": passed,
        "detail": detail,
    }


def eval_classification(dataset: dict) -> dict:
    cases = dataset.get("classification", [])
    results: list[dict] = []
    for case in cases:
        expected = case["expected"]
        actual = classify_text(case["text"], default_class="turn")
        passed = True
        detail = {
            "text": case["text"],
            "actual": {
                "memory_class": actual.memory_class,
                "extracted_value": actual.extracted_value,
                "durable": actual.durable,
                "durability_reason": actual.durability_reason,
                "profile_key": actual.profile_key,
            },
            "expected": expected,
        }
        for field in (
            "memory_class",
            "extracted_value",
            "durable",
            "durability_reason",
            "profile_key",
        ):
            if field in expected and getattr(actual, field) != expected[field]:
                passed = False
        results.append(_pass_fail(case["name"], passed, detail))
    return _summary("classification", results)


def eval_sentiment(dataset: dict) -> dict:
    cases = dataset.get("sentiment", [])
    results: list[dict] = []
    for case in cases:
        expected = case["expected"]
        actual = detect_sentiment(case["text"])
        detail = {
            "text": case["text"],
            "actual": {
                "label": actual.label,
                "confidence": actual.confidence,
                "cues": actual.cues,
                "guidance": actual.guidance,
            },
            "expected": expected,
        }
        passed = actual.label == expected["label"]
        results.append(_pass_fail(case["name"], passed, detail))
    return _summary("sentiment", results)


def eval_semantic(dataset: dict) -> dict:
    cases = dataset.get("semantic", [])
    results: list[dict] = []
    encoder = HashEmbeddingEncoder(dimensions=128)
    for case in cases:
        expected = case["expected"]
        route = route_semantic_candidate(
            case["text"],
            encoder,
            threshold=float(case.get("threshold", 0.72)),
        )
        actual = route.to_metadata() if route is not None else None
        passed = actual is not None
        if actual is not None:
            for field in (
                "candidate_key",
                "candidate_class",
                "above_threshold",
                "durable_candidate",
            ):
                if field in expected and actual.get(field) != expected[field]:
                    passed = False
        results.append(
            _pass_fail(
                case["name"],
                passed,
                {
                    "text": case["text"],
                    "expected": expected,
                    "actual": actual,
                },
            )
        )
    return _summary("semantic", results)


def eval_profile(dataset: dict, base_config: AppConfig) -> dict:
    cases = dataset.get("profile", [])
    results: list[dict] = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="agent-memory-v2-eval-profile-") as temp_dir:
            pipeline = _build_pipeline(base_config, Path(temp_dir))
            _seed_turns(pipeline, case["turns"], case["name"])
            profile = pipeline.profile_store.load() if pipeline.profile_store is not None else {}
            expected = case["expected"]
            passed = True
            for section_name in ("preferences", "facts", "tasks"):
                expected_section = expected.get(section_name, {})
                actual_section = profile.get(section_name, {})
                for key, expected_value in expected_section.items():
                    actual_value = (actual_section.get(key) or {}).get("value")
                    if actual_value != expected_value:
                        passed = False
            results.append(
                _pass_fail(
                    case["name"],
                    passed,
                    {
                        "expected": expected,
                        "actual": profile,
                    },
                )
            )
    return _summary("profile", results)


def eval_recall(dataset: dict, base_config: AppConfig) -> dict:
    cases = dataset.get("recall", [])
    results: list[dict] = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="agent-memory-v2-eval-recall-") as temp_dir:
            pipeline = _build_pipeline(base_config, Path(temp_dir))
            _seed_turns(pipeline, case["turns"], case["name"])
            query = Message(role="user", text=case["query"], conversation_id=case["name"])
            recall = pipeline.merged_recall(query)
            expected = case["expected"]
            factual_text = " ".join(item["text"] for item in recall["factual"])
            top_merged_profile_key = (
                recall["merged"][0]["profile_key"] if recall["merged"] else None
            )

            passed = True
            for snippet in expected.get("factual_contains", []):
                if snippet not in factual_text:
                    passed = False
            if (
                "contextual_count_at_least" in expected
                and recall["contextual_count"] < expected["contextual_count_at_least"]
            ):
                passed = False
            if (
                "top_merged_profile_key" in expected
                and top_merged_profile_key != expected["top_merged_profile_key"]
            ):
                passed = False

            results.append(
                _pass_fail(
                    case["name"],
                    passed,
                    {
                        "query": case["query"],
                        "expected": expected,
                        "actual": recall,
                    },
                )
            )
    return _summary("recall", results)


def eval_prompt(dataset: dict, base_config: AppConfig) -> dict:
    cases = dataset.get("prompt", [])
    results: list[dict] = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="agent-memory-v2-eval-prompt-") as temp_dir:
            pipeline = _build_pipeline(base_config, Path(temp_dir))
            _seed_turns(pipeline, case["turns"], case["name"])
            query = Message(role="user", text=case["query"], conversation_id=case["name"])
            recalled = pipeline.recall(query)
            prompt_context = pipeline.prompt_context(recalled)
            prompt = pipeline.build_prompt(query, recalled)
            expected = case["expected"]
            passed = True
            for snippet in expected.get("prompt_contains", []):
                if snippet not in prompt:
                    passed = False
            if len(prompt_context["factual"]) < expected.get("factual_count_at_least", 0):
                passed = False
            if len(prompt_context["contextual"]) < expected.get("contextual_count_at_least", 0):
                passed = False
            if len(prompt_context["dropped_contextual"]) < expected.get(
                "dropped_contextual_count_at_least", 0
            ):
                passed = False
            results.append(
                _pass_fail(
                    case["name"],
                    passed,
                    {
                        "query": case["query"],
                        "expected": expected,
                        "actual": {
                            "prompt": prompt,
                            "prompt_context": prompt_context,
                        },
                    },
                )
            )
    return _summary("prompt", results)


def _summary(stage: str, results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    return {
        "stage": stage,
        "passed": passed == total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "total_cases": total,
        "results": results,
    }


def eval_all(dataset: dict, base_config: AppConfig) -> dict:
    stages = [
        eval_classification(dataset),
        eval_semantic(dataset),
        eval_sentiment(dataset),
        eval_profile(dataset, base_config),
        eval_recall(dataset, base_config),
        eval_prompt(dataset, base_config),
    ]
    return {
        "passed": all(stage["passed"] for stage in stages),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation harness for agent_memory_v2.")
    parser.add_argument(
        "stage",
        choices=[
            "classification",
            "semantic",
            "sentiment",
            "profile",
            "recall",
            "prompt",
            "all",
            "history",
            "compare",
        ],
        help="Which evaluation stage to run.",
    )
    parser.add_argument(
        "--dataset",
        default="evals/baseline.json",
        help="Path to the eval dataset JSON file.",
    )
    parser.add_argument(
        "--history-dir",
        default="artifacts/eval_history/deterministic",
        help="Directory for compact eval history summaries.",
    )
    parser.add_argument(
        "--record-history", action="store_true", help="Persist a compact history summary."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
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

    dataset = _load_dataset(base_config.root_dir / args.dataset)

    if args.stage == "classification":
        result = eval_classification(dataset)
    elif args.stage == "sentiment":
        result = eval_sentiment(dataset)
    elif args.stage == "semantic":
        result = eval_semantic(dataset)
    elif args.stage == "profile":
        result = eval_profile(dataset, base_config)
    elif args.stage == "recall":
        result = eval_recall(dataset, base_config)
    elif args.stage == "prompt":
        result = eval_prompt(dataset, base_config)
    else:
        result = eval_all(dataset, base_config)

    if args.record_history:
        summary = compact_summary(
            eval_kind="deterministic",
            stage_name=args.stage,
            dataset_path=args.dataset,
            result=result,
            root_dir=base_config.root_dir,
            runtime={
                "embedding_provider": "hash",
                "embedding_model": "hash",
            },
        )
        history_path = write_history(history_root, summary)
        result["history_path"] = str(history_path)

    print(json.dumps(result, indent=2))

    raise SystemExit(0 if result.get("passed", False) else 1)


if __name__ == "__main__":
    main()
