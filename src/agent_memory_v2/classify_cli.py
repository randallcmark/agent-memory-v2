from __future__ import annotations

import argparse
import json
import sys

from agent_memory_v2.classifier import classify_text
from agent_memory_v2.cli_inputs import resolve_text_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify a text snippet as memory.")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--default-class", default="turn")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        text = resolve_text_input(text=args.text, text_file=args.text_file)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    result = classify_text(text, default_class=args.default_class)
    print(
        json.dumps(
            {
                "ok": True,
                "text": text,
                "memory_class": result.memory_class,
                "extracted_value": result.extracted_value,
                "confidence": result.confidence,
                "durable": result.durable,
                "durability_reason": result.durability_reason,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
