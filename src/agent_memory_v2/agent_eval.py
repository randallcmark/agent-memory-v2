"""Agent tool-loop eval harness: runs memory scenarios against a provider and scores results."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from agent_memory_v2.config import AppConfig
from agent_memory_v2.eval_history import git_metadata
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import MemoryPipeline


@dataclass(frozen=True)
class AgentToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class AgentProviderResponse:
    tool_calls: list[AgentToolCall]
    output_text: str | None = None
    raw: Any = None
    conversation_items: list[dict[str, Any]] = field(default_factory=list)


class AgentProvider(Protocol):
    provider_name: str
    model_name: str

    def next_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentProviderResponse:
        ...


def isolated_agent_config(base: AppConfig, root_dir: Path) -> AppConfig:
    raw = copy.deepcopy(base.raw)
    raw["embeddings"]["provider"] = "hash"
    raw["embeddings"]["model"] = "hash"
    raw["embeddings"]["dimensions"] = 128
    raw["memory"]["embedding_dim"] = 128
    raw["memory"]["index_path"] = "data/memory/memory.index"
    raw["memory"]["metadata_path"] = "data/memory/memory_metadata.json"
    raw["memory"]["interaction_log_path"] = "data/logs/interactions.jsonl"
    raw["sidecar"]["index_path"] = "data/sidecar/sidecar.index"
    raw["sidecar"]["metadata_path"] = "data/sidecar/sidecar_metadata.json"
    raw["profile"]["path"] = "data/profile/profile.json"
    raw["maintenance"]["enabled"] = False
    raw["llm"]["preflight"]["enabled"] = False
    raw["structured_extractor"]["enabled"] = False
    return AppConfig(root_dir=root_dir, settings_path=base.settings_path, raw=raw)


def agent_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "memory_write",
            "description": "Store a memory item from the setup turns when it is useful later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_class": {"type": "string", "enum": ["fact", "preference", "task", "context"]},
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                    "profile_key": {"type": "string"},
                    "extracted_value": {"type": "string"},
                },
                "required": ["memory_class", "text", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "memory_query",
            "description": "Retrieve memory relevant to the current user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "memory_inspect",
            "description": "Inspect current memory and profile state.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "answer",
            "description": "Provide the final answer to the user's query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                },
                "required": ["response"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "memory_evolve",
            "description": "Reserved for future AIPCS schema evolution experiments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal": {"type": "string"},
                },
                "required": ["proposal"],
                "additionalProperties": False,
            },
        },
    ]


def _instructions(scenario: dict[str, Any], max_tool_calls: int) -> str:
    setup = json.dumps(scenario.get("setup_turns", []), indent=2)
    return (
        "You are evaluating agentic memory tool use. Use only the provided tools.\n"
        "First store useful durable memories from setup turns. Then query memory if needed. "
        "End by calling the answer tool exactly once.\n"
        f"You have at most {max_tool_calls} tool calls.\n\n"
        f"Scenario: {scenario.get('name')}\n"
        f"Description: {scenario.get('description', '')}\n"
        f"Setup turns:\n{setup}\n\n"
        f"User query: {scenario.get('query', '')}\n"
    )


def _initial_input(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": (
                "Process the setup turns as prior conversation, use memory tools, "
                f"and answer this query: {scenario.get('query', '')}"
            ),
        }
    ]


def _class_default_profile_key(memory_class: str, text: str) -> str | None:
    lowered = text.lower()
    if memory_class == "preference":
        return "preference.general"
    if memory_class == "task":
        return "task.general"
    if memory_class == "fact":
        if "name" in lowered:
            return "identity.name"
        if "live" in lowered or "based" in lowered or "located" in lowered:
            return "identity.location"
    return None


def _memory_write(pipeline: MemoryPipeline, scenario_name: str, args: dict[str, Any]) -> dict[str, Any]:
    memory_class = str(args.get("memory_class") or "").strip()
    text = str(args.get("text") or "").strip()
    reason = str(args.get("reason") or "").strip()
    if memory_class not in {"fact", "preference", "task", "context"}:
        return {"ok": False, "error": "unsupported_memory_class"}
    if not text:
        return {"ok": False, "error": "missing_text"}

    durable = memory_class in {"fact", "preference", "task"}
    extracted_value = str(args.get("extracted_value") or text).strip()
    profile_key = str(args.get("profile_key") or "").strip() or _class_default_profile_key(memory_class, text)
    message = Message(role="agent", text=text, conversation_id=scenario_name)
    record = pipeline._store_memory(  # noqa: SLF001 - harness tool intentionally exercises pipeline storage.
        role=memory_class,
        text=text,
        summary=text,
        timestamp=message.timestamp,
        conversation_id=scenario_name,
        turn_id=message.turn_id,
        message_id=message.message_id,
        metadata={
            "kind": "agent_tool_memory",
            "memory_class": memory_class,
            "extracted_value": extracted_value,
            "classification_confidence": 1.0,
            "durable": durable,
            "durability_reason": "agent_tool",
            "profile_key": profile_key,
            "classification_source": "agent_tool",
            "agent_reason": reason,
        },
    )
    return {
        "ok": True,
        "memory_id": record.memory_id,
        "memory_class": memory_class,
        "durable": durable,
        "profile_key": profile_key,
        "extracted_value": extracted_value,
    }


def _memory_query(pipeline: MemoryPipeline, scenario_name: str, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "missing_query"}
    message = Message(role="user", text=query, conversation_id=scenario_name)
    return {"ok": True, **pipeline.merged_recall(message)}


def _memory_inspect(pipeline: MemoryPipeline) -> dict[str, Any]:
    profile = pipeline.profile_store.load() if pipeline.profile_store is not None else {}
    return {
        "ok": True,
        "memory_count": len(pipeline.store.records),
        "sidecar_count": len(pipeline.sidecar_store.records) if pipeline.sidecar_store is not None else 0,
        "profile": profile,
    }


def execute_tool_call(
    pipeline: MemoryPipeline,
    *,
    scenario_name: str,
    call: AgentToolCall,
) -> tuple[dict[str, Any], str | None]:
    if call.name == "memory_write":
        return _memory_write(pipeline, scenario_name, call.arguments), None
    if call.name == "memory_query":
        return _memory_query(pipeline, scenario_name, call.arguments), None
    if call.name == "memory_inspect":
        return _memory_inspect(pipeline), None
    if call.name == "memory_evolve":
        return {"ok": False, "error": "unsupported_tool", "message": "memory_evolve is reserved for AIPCS."}, None
    if call.name == "answer":
        response = str(call.arguments.get("response") or "").strip()
        return {"ok": bool(response), "response": response}, response or None
    return {"ok": False, "error": "unknown_tool", "tool": call.name}, None


def _contains_all(response: str, snippets: list[str]) -> tuple[bool, list[str]]:
    lowered = response.lower()
    missing = [snippet for snippet in snippets if snippet.lower() not in lowered]
    return not missing, missing


def _contains_none(response: str, snippets: list[str]) -> tuple[bool, list[str]]:
    lowered = response.lower()
    present = [snippet for snippet in snippets if snippet.lower() in lowered]
    return not present, present


def _score_scenario(scenario: dict[str, Any], final_answer: str | None, invalid_calls: int) -> dict[str, Any]:
    response = final_answer or ""
    expected_contains = scenario.get("expected_contains", [])
    expected_not_contains = scenario.get("expected_not_contains", [])
    contains_ok, missing = _contains_all(response, expected_contains)
    none_ok, forbidden = _contains_none(response, expected_not_contains)
    passed = bool(response.strip()) and invalid_calls == 0 and contains_ok and none_ok
    return {
        "passed": passed,
        "expected_contains": expected_contains,
        "expected_not_contains": expected_not_contains,
        "missing": missing,
        "forbidden_present": forbidden,
    }


def run_agent_scenario(
    *,
    base_config: AppConfig,
    scenario: dict[str, Any],
    provider: AgentProvider,
    temp_root: Path,
    max_tool_calls: int = 5,
) -> dict[str, Any]:
    start = time.perf_counter()
    config = isolated_agent_config(base_config, temp_root)
    pipeline = MemoryPipeline(config)
    instructions = _instructions(scenario, max_tool_calls)
    input_items = _initial_input(scenario)
    tools = agent_tool_schemas()
    trace: list[dict[str, Any]] = []
    invalid_tool_calls = 0
    final_answer: str | None = None

    for step_index in range(1, max_tool_calls + 1):
        response = provider.next_response(
            instructions=instructions,
            input_items=input_items,
            tools=tools,
        )
        if response.conversation_items:
            input_items.extend(response.conversation_items)
        step_payload = {
            "step": step_index,
            "provider_output_text": response.output_text,
            "raw": response.raw,
            "tool_calls": [],
        }
        if not response.tool_calls:
            invalid_tool_calls += 1
            step_payload["error"] = "no_tool_call"
            trace.append(step_payload)
            break
        for call in response.tool_calls:
            tool_result, answer = execute_tool_call(pipeline, scenario_name=scenario["name"], call=call)
            if not tool_result.get("ok", False) and call.name != "memory_evolve":
                invalid_tool_calls += 1
            step_payload["tool_calls"].append(
                {
                    "name": call.name,
                    "call_id": call.call_id,
                    "arguments": call.arguments,
                    "result": tool_result,
                }
            )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(tool_result),
                }
            )
            if answer is not None:
                final_answer = answer
                break
            if not tool_result.get("ok", False) and call.name != "memory_evolve":
                break
        trace.append(step_payload)
        if final_answer is not None:
            break
        if step_payload["tool_calls"]:
            last_result = step_payload["tool_calls"][-1]["result"]
            last_name = step_payload["tool_calls"][-1]["name"]
            if not last_result.get("ok", False) and last_name != "memory_evolve":
                break
        if final_answer is not None:
            break

    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
    score = _score_scenario(scenario, final_answer, invalid_tool_calls)
    return {
        "name": scenario["name"],
        "passed": score["passed"],
        "provider": provider.provider_name,
        "model": provider.model_name,
        "scenario": scenario,
        "final_answer": final_answer,
        "invalid_tool_calls": invalid_tool_calls,
        "tool_call_count": sum(len(step.get("tool_calls", [])) for step in trace),
        "elapsed_ms": elapsed_ms,
        "score": score,
        "trace": trace,
        "memory_state": _memory_inspect(pipeline),
        "git": git_metadata(base_config.root_dir),
    }


class FakeAgentProvider:
    provider_name = "fake"
    model_name = "fake-agent"

    def __init__(self, scenario: dict[str, Any] | None = None, *, mode: str = "normal") -> None:
        self.scenario = scenario or {}
        self.mode = mode
        self.index = 0
        self._calls = self._build_calls()

    def _build_calls(self) -> list[AgentToolCall]:
        if self.mode == "invalid":
            return [AgentToolCall(name="memory_write", arguments={})]
        if self.mode == "evolve":
            return [AgentToolCall(name="memory_evolve", arguments={"proposal": "add schema"})]
        if self.mode == "loop":
            return [
                AgentToolCall(name="memory_inspect", arguments={}),
                AgentToolCall(name="memory_inspect", arguments={}),
                AgentToolCall(name="memory_inspect", arguments={}),
                AgentToolCall(name="memory_inspect", arguments={}),
                AgentToolCall(name="memory_inspect", arguments={}),
                AgentToolCall(name="answer", arguments={"response": "late"}),
            ]
        calls: list[AgentToolCall] = []
        for turn in self.scenario.get("setup_turns", []):
            user_text = str(turn.get("user") or "")
            lowered = user_text.lower()
            if "prefer" in lowered or "favourite" in lowered:
                memory_class = "preference"
            elif "remind" in lowered or "need to" in lowered:
                memory_class = "task"
            elif any(marker in lowered for marker in ("name", "live", "based")):
                memory_class = "fact"
            else:
                memory_class = "context"
            calls.append(
                AgentToolCall(
                    name="memory_write",
                    arguments={
                        "memory_class": memory_class,
                        "text": user_text,
                        "reason": "setup turn may be useful for answering later",
                    },
                )
            )
        query = str(self.scenario.get("query") or "")
        calls.append(AgentToolCall(name="memory_query", arguments={"query": query}))
        calls.append(AgentToolCall(name="answer", arguments={"response": self._fake_answer()}))
        return calls

    def _fake_answer(self) -> str:
        query = str(self.scenario.get("query") or "").lower()
        setup_text = " ".join(str(turn.get("user") or "") for turn in self.scenario.get("setup_turns", []))
        if "prefer" in query and "oat milk" in setup_text.lower():
            return "You prefer oat milk."
        if "name" in query and "mark" in setup_text.lower():
            return "Your name is Mark."
        if "where" in query and "london" in setup_text.lower():
            return "You live in London."
        if "where" in query and "edinburgh" in setup_text.lower():
            return "You are based in Edinburgh in the UK."
        if "remind" in query and "passport" in setup_text.lower():
            return "You asked me to remind you to renew your passport."
        return "I used the memory tools and answered the query."

    def next_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentProviderResponse:
        del instructions, input_items, tools
        if self.index >= len(self._calls):
            return AgentProviderResponse(tool_calls=[], output_text=None, raw={"fake": "exhausted"})
        call = self._calls[self.index]
        self.index += 1
        return AgentProviderResponse(
            tool_calls=[call],
            output_text=None,
            raw={"fake_tool_call": {"name": call.name, "arguments": call.arguments}},
            conversation_items=[
                {
                    "type": "function_call",
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                    "call_id": call.call_id,
                }
            ],
        )
