"""Ollama smoke test: verifies Ollama reachability and optionally runs a generate call."""

from __future__ import annotations

import argparse
import json
import sys

from agent_memory_v2.ollama import OllamaClient, OllamaProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test local Ollama connectivity.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--timeout-seconds", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--generate", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = OllamaClient(
        OllamaProfile(
            host=args.host,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
        )
    )
    try:
        result = client.healthcheck(run_generate=args.generate)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "host": args.host,
                    "model": args.model,
                    "error": repr(exc),
                },
                indent=2,
            )
        )
        return 1

    result["ok"] = bool(result["reachable"] and result["model_present"])
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
