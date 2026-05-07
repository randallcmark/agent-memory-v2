"""CLI for classifying text as memory: shows memory class, confidence, and optional semantic routing."""

from __future__ import annotations

import argparse
import json
import sys

from agent_memory_v2.classifier import classify_text
from agent_memory_v2.cli_inputs import resolve_text_input
from agent_memory_v2.config import load_config
from agent_memory_v2.embeddings import build_encoder
from agent_memory_v2.ollama import OllamaClient, OllamaProfile
from agent_memory_v2.pipeline import should_run_semantic_router
from agent_memory_v2.semantic_router import route_semantic_candidate
from agent_memory_v2.structured_extractor import extract_structured_memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify a text snippet as memory.")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--default-class", default="turn")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Also run metadata-only semantic candidate routing for rule-classifier misses.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run the Phase 2 structured extractor when semantic routing finds a durable candidate.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        text = resolve_text_input(text=args.text, text_file=args.text_file)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    result = classify_text(text, default_class=args.default_class)
    payload = {
        "ok": True,
        "text": text,
        "memory_class": result.memory_class,
        "extracted_value": result.extracted_value,
        "confidence": result.confidence,
        "durable": result.durable,
        "durability_reason": result.durability_reason,
    }
    if args.semantic or args.extract:
        config = load_config()
        if should_run_semantic_router(result):
            router_cfg = config.semantic_router
            encoder = build_encoder(
                provider=config.embeddings.get("provider", "hash"),
                model_name=config.embeddings.get("model", ""),
                dimensions=config.embedding_dim,
                host=config.embeddings.get("host") or config.llm.get("host"),
                timeout_seconds=int(
                    config.embeddings.get("timeout_seconds", config.llm.get("timeout_seconds", 60))
                ),
            )
            route = route_semantic_candidate(
                text,
                encoder,
                threshold=float(router_cfg.get("threshold", 0.72)),
            )
            payload["semantic_candidate"] = route.to_metadata() if route is not None else None
            if args.extract:
                extractor_cfg = config.structured_extractor
                if not extractor_cfg.get("enabled", False):
                    payload["structured_extraction"] = None
                    payload["structured_extraction_skipped_reason"] = "structured_extractor_disabled"
                elif route is None or not route.above_threshold or not route.durable_candidate:
                    payload["structured_extraction"] = None
                    payload["structured_extraction_skipped_reason"] = (
                        "semantic_candidate_not_eligible"
                    )
                else:
                    llm_cfg = config.llm
                    generator = OllamaClient(
                        OllamaProfile(
                            host=llm_cfg["host"],
                            model=llm_cfg["model"],
                            temperature=0.0,
                            max_tokens=int(llm_cfg.get("max_tokens", 400)),
                            timeout_seconds=int(llm_cfg.get("timeout_seconds", 60)),
                            raw_prompt=bool(llm_cfg.get("raw_prompt", False)),
                        )
                    )
                    extraction = extract_structured_memory(
                        text,
                        route,
                        generator,
                        admission_threshold=float(extractor_cfg.get("admission_threshold", 0.75)),
                        allowed_profile_keys=set(extractor_cfg.get("allowed_profile_keys", [])),
                        max_value_chars=int(extractor_cfg.get("max_value_chars", 160)),
                    )
                    payload["structured_extraction"] = extraction.to_metadata()
        else:
            payload["semantic_candidate"] = None
            payload["semantic_skipped_reason"] = "rule_classifier_returned_durable_memory"
            if args.extract:
                payload["structured_extraction"] = None
                payload["structured_extraction_skipped_reason"] = (
                    "rule_classifier_returned_durable_memory"
                )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
