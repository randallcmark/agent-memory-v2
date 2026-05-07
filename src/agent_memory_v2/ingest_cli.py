"""CLI for ingesting a user/agent turn into the memory store."""

from __future__ import annotations

import argparse
import json
import sys

from agent_memory_v2.cli_inputs import resolve_text_input
from agent_memory_v2.config import load_config
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a user message or turn without using chat.")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--reply", default=None)
    parser.add_argument("--reply-file", default=None)
    parser.add_argument("--conversation-id", default="analysis")
    parser.add_argument("--turn-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        text = resolve_text_input(text=args.text, text_file=args.text_file)
        reply = None
        if args.reply is not None or args.reply_file is not None:
            reply = resolve_text_input(text=args.reply, text_file=args.reply_file)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    config = load_config()
    pipeline = MemoryPipeline(config)

    message_kwargs = {
        "role": "user",
        "text": text,
        "conversation_id": args.conversation_id,
    }
    if args.turn_id:
        message_kwargs["turn_id"] = args.turn_id
    user_message = Message(**message_kwargs)

    if reply is None:
        record = pipeline.ingest(user_message)
    else:
        agent_message = Message(
            role="agent",
            text=reply,
            conversation_id=user_message.conversation_id,
            turn_id=user_message.turn_id,
        )
        record = pipeline.ingest_turn(user_message, agent_message)

    print(
        json.dumps(
            {
                "ok": True,
                "memory_id": record.memory_id,
                "role": record.role,
                "summary": record.summary,
                "metadata": record.metadata,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
