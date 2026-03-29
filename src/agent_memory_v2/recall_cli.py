from __future__ import annotations

import argparse
import json
import sys

from agent_memory_v2.cli_inputs import resolve_text_input
from agent_memory_v2.config import load_config
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query memory recall independently of chat.")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--message-id", default="recall-query")
    parser.add_argument("--conversation-id", default="analysis")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        text = resolve_text_input(text=args.text, text_file=args.text_file)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    config = load_config()
    pipeline = MemoryPipeline(config)
    message = Message(
        role="user",
        text=text,
        message_id=args.message_id,
        conversation_id=args.conversation_id,
    )
    merged = pipeline.merged_recall(message)
    print(
        json.dumps(
            {
                "ok": True,
                **merged,
                "recalled_count": len(merged["merged"]),
                "recalled": merged["merged"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
