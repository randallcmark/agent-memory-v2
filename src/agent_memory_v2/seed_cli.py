from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent_memory_v2.config import load_config
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed generic example data into agent_memory_v2.")
    parser.add_argument("--seed-file", default="seeds/generic_seed.jsonl")
    parser.add_argument("--user", default=None)
    parser.add_argument("--conversation-id", default="seed")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.user:
        os.environ["AGENT_MEMORY_V2_USER"] = args.user
    config = load_config()
    seed_path = Path(args.seed_file)
    if not seed_path.is_absolute():
        seed_path = (config.root_dir / seed_path).resolve()

    if not seed_path.exists():
        print(json.dumps({"ok": False, "error": f"Seed file not found: {seed_path}"}, indent=2))
        return 2

    pipeline = MemoryPipeline(config)
    seeded = 0
    with seed_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            user_message = Message(
                role="user",
                text=item["user_text"],
                conversation_id=args.conversation_id,
            )
            agent_message = Message(
                role="agent",
                text=item["agent_reply"],
                conversation_id=args.conversation_id,
                turn_id=user_message.turn_id,
            )
            pipeline.ingest_turn(user_message, agent_message)
            seeded += 1

    print(
        json.dumps(
            {
                "ok": True,
                "seed_file": str(seed_path),
                "seeded": seeded,
                "user": config.current_user,
                "conversation_id": args.conversation_id,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
