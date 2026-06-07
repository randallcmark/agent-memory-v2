"""Generation seam for the experiment harness.

A ``Generator`` turns a (system prompt, message history) into a text reply plus
token usage. ``ClaudeGenerator`` wraps the existing ``ClaudeProvider`` (Anthropic
Messages API). ``FakeGenerator`` is deterministic and offline so the controller and
journaling can be smoke-tested without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


class Generator(Protocol):
    name: str
    model: str

    def generate(self, *, system: str, messages: list[dict[str, str]]) -> GenResult: ...


class ClaudeGenerator:
    """Wrap ``ClaudeProvider`` for plain (toolless) conversational generation."""

    name = "anthropic"

    def __init__(self, *, model: str | None = None, temperature: float = 0.0) -> None:
        # Imported lazily so offline/fake runs don't require the API key at import.
        from agent_memory_v2.claude_provider import ClaudeProvider

        self._provider = ClaudeProvider(model=model, temperature=temperature)
        self.model = self._provider.model_name
        self.temperature = temperature

    def generate(self, *, system: str, messages: list[dict[str, str]]) -> GenResult:
        # ClaudeProvider.next_response forwards items carrying a "role" straight
        # through to the Messages API; tools=[] keeps it a plain text completion.
        response = self._provider.next_response(
            instructions=system,
            input_items=[dict(m) for m in messages],
            tools=[],
        )
        usage: dict[str, Any] = {}
        if isinstance(response.raw, dict):
            usage = response.raw.get("usage") or {}
        return GenResult(text=response.output_text or "", usage=usage, raw=response.raw)


class FakeGenerator:
    """Deterministic offline generator for tests and dry runs.

    Echoes the latest user turn and, if a memory block is present in the system
    prompt, notes that it saw one — enough to exercise journaling and the A/C split
    without network access.
    """

    name = "fake"
    model = "fake-generator"

    def __init__(self, *, temperature: float = 0.0) -> None:
        self.temperature = temperature

    def generate(self, *, system: str, messages: list[dict[str, str]]) -> GenResult:
        last_user = ""
        for item in reversed(messages):
            if item.get("role") == "user":
                last_user = str(item.get("content") or "")
                break
        saw_memory = MEMORY_BLOCK_HEADING in (system or "")
        prefix = "[memory-aware] " if saw_memory else ""
        text = f"{prefix}Acknowledged: {last_user}".strip()
        usage = {
            "input_tokens": len(system or "") // 4
            + sum(len(str(m.get("content", ""))) for m in messages) // 4,
            "output_tokens": len(text) // 4,
        }
        return GenResult(text=text, usage=usage, raw={"fake": True})


# Heading used by the Arm A memory block; imported here to avoid a circular import
# at module load (controller imports generators).
MEMORY_BLOCK_HEADING = "Known about the user"


def build_generator(name: str, *, model: str | None = None, temperature: float = 0.0) -> Generator:
    if name == "fake":
        return FakeGenerator(temperature=temperature)
    if name in {"anthropic", "claude"}:
        return ClaudeGenerator(model=model, temperature=temperature)
    raise RuntimeError(f"Unsupported generator: {name}")
