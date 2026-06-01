"""MemoryController: agent-memory-v2 wrapped around a Claude generation call.

Per user turn the controller:
  Arm A (curated):  recall -> neutral memory injection -> generate -> ingest -> journal
  Arm C (control):  generate -> journal   (no store, no injection, no ingest)

Design note (§5 of the exec plan — "helper, not personality"):
we deliberately do NOT call ``pipeline.build_prompt``. That method hardcodes a
persona ("You are a helpful assistant ...") and injects sentiment "Response
tuning" directives, which would re-characterise Claude. Instead we reuse only the
*retrieval* side (``recall`` + ``prompt_context``: dedupe, profile merge, char
budget) and render a neutral, clearly-labelled reference block that the assistant
may use if relevant. Claude's own system behaviour is otherwise left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_memory_v2.config import AppConfig
from agent_memory_v2.experiment.config import experiment_config
from agent_memory_v2.experiment.generators import Generator
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline, _format_profile, format_recalled_item

# Neutral, reference-framed heading. Must match generators.MEMORY_BLOCK_HEADING.
MEMORY_BLOCK_HEADING = "Known about the user"

# Base system prompt for the control arm and as the carrier for Arm A's memory
# block. Intentionally minimal so Claude stays Claude.
BASE_SYSTEM = "You are Claude, a helpful AI assistant."

_HELPER_PREAMBLE = (
    f"{MEMORY_BLOCK_HEADING} (background reference the user has shared in the past; "
    "use only if it is relevant to the current message, and do not mention this "
    "section explicitly):"
)


@dataclass
class TurnResult:
    arm: str
    turn_index: int
    user_text: str
    response: str
    system_prompt: str
    injected_context: str | None
    recalled_items: list[dict[str, Any]]
    usage: dict[str, Any]
    duration_ms: float
    memory_state_after: dict[str, Any]


@dataclass
class Session:
    """A contiguous Claude context. ``messages`` is the rolling chat history."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[dict[str, str]] = field(default_factory=list)
    turn_index: int = 0


def render_memory_block(pipeline: MemoryPipeline, recalled: list[dict[str, Any]]) -> str | None:
    """Render the neutral helper-framed memory block, or ``None`` if empty.

    Reuses ``prompt_context`` for profile-merge/dedupe/char-budget, but renders our
    own neutral framing instead of ``build_prompt``'s persona + sentiment text.
    """
    ctx = pipeline.prompt_context(recalled)
    sections: list[str] = []

    profile = ctx.get("profile") or {}
    rendered_profile = _format_profile(profile) if profile else ""
    if rendered_profile:
        sections.append(rendered_profile)

    factual = ctx.get("factual") or []
    if factual:
        lines = [format_recalled_item(item, pipeline.config) for item in factual]
        sections.append("Relevant facts:\n" + "\n".join(lines))

    contextual = ctx.get("contextual") or []
    if contextual:
        lines = [format_recalled_item(item, pipeline.config) for item in contextual]
        sections.append("Earlier context:\n" + "\n".join(lines))

    if not sections:
        return None
    return _HELPER_PREAMBLE + "\n" + "\n\n".join(sections)


class MemoryController:
    """Drives turns through a chosen arm against an isolated memory store."""

    def __init__(
        self,
        *,
        base_config: AppConfig,
        root_dir: Path,
        generator: Generator,
        arm: str,
    ) -> None:
        if arm not in {"A", "C"}:
            raise ValueError(f"Unsupported arm: {arm!r} (expected 'A' or 'C')")
        self.arm = arm
        self.generator = generator
        self.config = experiment_config(base_config, root_dir)
        # Arm C needs no store; build the pipeline only when memory is in play.
        self.pipeline: MemoryPipeline | None = MemoryPipeline(self.config) if arm == "A" else None

    # -- memory state inspection -------------------------------------------------
    def memory_state(self) -> dict[str, Any]:
        if self.pipeline is None:
            return {"main_count": 0, "sidecar_count": 0, "profile": {}}
        profile = (
            self.pipeline.profile_store.load() if self.pipeline.profile_store is not None else {}
        )
        return {
            "main_count": len(self.pipeline.store.records),
            "sidecar_count": (
                len(self.pipeline.sidecar_store.records)
                if self.pipeline.sidecar_store is not None
                else 0
            ),
            "profile": profile,
        }

    # -- the turn ----------------------------------------------------------------
    def run_turn(self, session: Session, user_text: str) -> TurnResult:
        import time

        injected_context: str | None = None
        recalled: list[dict[str, Any]] = []
        system = BASE_SYSTEM

        if self.arm == "A" and self.pipeline is not None:
            query = Message(role="user", text=user_text, conversation_id=session.session_id)
            recalled = self.pipeline.recall(query)
            injected_context = render_memory_block(self.pipeline, recalled)
            if injected_context:
                system = f"{BASE_SYSTEM}\n\n{injected_context}"

        session.messages.append({"role": "user", "content": user_text})

        start = time.perf_counter()
        gen = self.generator.generate(system=system, messages=session.messages)
        duration_ms = round((time.perf_counter() - start) * 1000, 3)

        session.messages.append({"role": "assistant", "content": gen.text})

        if self.arm == "A" and self.pipeline is not None:
            user_msg = Message(role="user", text=user_text, conversation_id=session.session_id)
            agent_msg = Message(role="agent", text=gen.text, conversation_id=session.session_id)
            self.pipeline.ingest_turn(user_msg, agent_msg)

        result = TurnResult(
            arm=self.arm,
            turn_index=session.turn_index,
            user_text=user_text,
            response=gen.text,
            system_prompt=system,
            injected_context=injected_context,
            recalled_items=recalled,
            usage=dict(gen.usage),
            duration_ms=duration_ms,
            memory_state_after=self.memory_state(),
        )
        session.turn_index += 1
        return result
