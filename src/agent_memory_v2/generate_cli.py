from __future__ import annotations

import argparse
import json
import sys

from agent_memory_v2.cli_inputs import resolve_text_input
from agent_memory_v2.config import load_config
from agent_memory_v2.ollama import OllamaClient, OllamaProfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send an exact prompt to Ollama and return the raw response.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        prompt = resolve_text_input(text=args.prompt, text_file=args.prompt_file)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    config = load_config()
    llm_cfg = config.llm
    client = OllamaClient(
        OllamaProfile(
            host=args.host or llm_cfg["host"],
            model=args.model or llm_cfg["model"],
            temperature=float(args.temperature if args.temperature is not None else llm_cfg["temperature"]),
            max_tokens=int(args.max_tokens if args.max_tokens is not None else llm_cfg["max_tokens"]),
            timeout_seconds=int(
                args.timeout_seconds if args.timeout_seconds is not None else llm_cfg["timeout_seconds"]
            ),
        )
    )

    try:
        response = client.generate(prompt)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": repr(exc)}, indent=2))
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "prompt": prompt,
                "response": response,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
