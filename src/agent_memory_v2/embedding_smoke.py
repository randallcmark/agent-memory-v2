"""Embedding smoke test: encodes a sample text and prints the resulting vector summary."""

from __future__ import annotations

import argparse
import json
import sys

from agent_memory_v2.config import load_config
from agent_memory_v2.ollama import OllamaClient, OllamaProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the configured embedding model.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--embed", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    emb_cfg = config.embeddings
    host = args.host or emb_cfg.get("host") or config.llm["host"]
    model = args.model or emb_cfg["model"]
    timeout_seconds = int(args.timeout_seconds or emb_cfg.get("timeout_seconds", 60))

    client = OllamaClient(
        OllamaProfile(
            host=host,
            model=model,
            temperature=0.0,
            max_tokens=1,
            timeout_seconds=timeout_seconds,
        )
    )
    try:
        result = client.embedding_healthcheck(run_embed=args.embed)
    except Exception as exc:
        print(json.dumps({"ok": False, "host": host, "model": model, "error": repr(exc)}, indent=2))
        return 1

    result["ok"] = bool(result["reachable"] and result["model_present"])
    if args.embed:
        result["ok"] = bool(result["ok"] and result.get("embed_ok"))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
