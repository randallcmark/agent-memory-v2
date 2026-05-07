"""End-to-end smoke test: ingests a turn and recalls it to verify the full pipeline."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory_v2.config import load_config
from agent_memory_v2.models import Message
from agent_memory_v2.ollama import OllamaClient, OllamaProfile
from agent_memory_v2.pipeline import MemoryPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end memory smoke test.")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=80)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base = load_config()
    with TemporaryDirectory(prefix="agent_memory_v2_smoke_") as tmpdir:
        root = Path(tmpdir)
        raw = deepcopy(base.raw)
        raw["llm"]["host"] = args.host
        raw["llm"]["model"] = args.model
        raw["llm"]["timeout_seconds"] = args.timeout_seconds
        raw["llm"]["max_tokens"] = args.max_tokens
        raw["memory"]["index_path"] = "memory/smoke.index"
        raw["memory"]["metadata_path"] = "memory/smoke.json"
        raw["memory"]["interaction_log_path"] = "logs/smoke.jsonl"
        config = type(base)(root_dir=root, settings_path=base.settings_path, raw=raw)

        pipeline = MemoryPipeline(
            config,
            ollama=OllamaClient(
                OllamaProfile(
                    host=args.host,
                    model=args.model,
                    temperature=float(raw["llm"]["temperature"]),
                    max_tokens=args.max_tokens,
                    timeout_seconds=args.timeout_seconds,
                )
            ),
        )

        first_user = Message(role="user", text="Please remember that I prefer oat milk.")
        first_reply = pipeline.respond(first_user)
        first_agent = Message(
            role="agent",
            text=first_reply,
            conversation_id=first_user.conversation_id,
            turn_id=first_user.turn_id,
        )
        pipeline.ingest_turn(first_user, first_agent)

        second_user = Message(role="user", text="What did I say I prefer?")
        recalled = pipeline.recall(second_user)
        prompt = pipeline.build_prompt(second_user, recalled)
        second_reply = pipeline.respond(second_user)

        result = {
            "ok": bool(recalled) and bool(second_reply.strip()),
            "first_reply": first_reply,
            "recalled_count": len(recalled),
            "recalled": recalled,
            "second_prompt_preview": prompt[:500],
            "second_reply": second_reply,
        }
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
