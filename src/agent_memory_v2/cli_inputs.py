"""Shared CLI input resolver: reads text from --text arg, a file path, or stdin."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


def resolve_text_input(
    *,
    text: str | None,
    text_file: str | None,
    stdin: TextIO | None = None,
) -> str:
    if text and text_file:
        raise ValueError("Use only one of --text or --text-file")

    if text_file:
        return Path(text_file).read_text(encoding="utf-8").strip()

    if text is not None:
        return text.strip()

    stream = stdin or sys.stdin
    if not stream.isatty():
        payload = stream.read().strip()
        if payload:
            return payload

    raise ValueError("Provide input with --text, --text-file, or stdin")
