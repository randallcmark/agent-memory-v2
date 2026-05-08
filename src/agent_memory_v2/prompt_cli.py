"""CLI that assembles and prints the full memory-augmented prompt for a given user query."""

from __future__ import annotations

import argparse
import json
import sys

from agent_memory_v2.cli_inputs import resolve_text_input
from agent_memory_v2.config import load_config
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline
from agent_memory_v2.sentiment import detect_sentiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the prompt for a user input without generating."
    )
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--message-id", default="prompt-query")
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
    prompt_context = pipeline.prompt_context(merged["merged"])
    sentiment = detect_sentiment(text)
    prompt = pipeline.build_prompt(message, merged["merged"])
    print(
        json.dumps(
            {
                "ok": True,
                **merged,
                "sentiment": {
                    "label": sentiment.label,
                    "confidence": sentiment.confidence,
                    "cues": sentiment.cues,
                    "guidance": sentiment.guidance,
                },
                "prompt_context": prompt_context,
                "recalled_count": len(merged["merged"]),
                "recalled": merged["merged"],
                "prompt": prompt,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
