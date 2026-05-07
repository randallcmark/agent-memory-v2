"""Claude (Anthropic) provider for the agent eval harness, using the Messages API."""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from agent_memory_v2.agent_eval import AgentProviderResponse, AgentToolCall
from agent_memory_v2.ollama import _with_retry


class ClaudeProvider:
    provider_name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        provider_name: str | None = None,
        timeout_seconds: int = 120,
        max_tokens: int = 4096,
    ) -> None:
        if provider_name:
            self.provider_name = provider_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Claude agent eval runs")
        self.model_name = (
            model
            or os.environ.get("AGENT_MEMORY_V2_ANTHROPIC_MODEL")
            or os.environ.get("AGENT_MEMORY_V2_CLAUDE_MODEL")
            or "claude-sonnet-4-20250514"
        )
        self.base_url = (
            base_url
            or os.environ.get("AGENT_MEMORY_V2_ANTHROPIC_BASE_URL")
            or os.environ.get("AGENT_MEMORY_V2_CLAUDE_BASE_URL")
            or "https://api.anthropic.com"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def _post_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        def _call() -> dict[str, Any]:
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("Claude returned a non-object response")
            return data

        return _with_retry(_call)

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI function-format tool schemas to Claude tool format."""
        result = []
        for tool in tools:
            converted: dict[str, Any] = {
                "name": tool["name"],
                "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            if tool.get("description"):
                converted["description"] = tool["description"]
            result.append(converted)
        return result

    @staticmethod
    def _convert_input_items(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Responses-API-format input_items to Claude messages format.

        Groups consecutive function_call items into a single assistant message and
        consecutive function_call_output items into a single user message, which is
        required by Claude's strict alternating-role constraint.
        """
        messages: list[dict[str, Any]] = []
        i = 0
        while i < len(input_items):
            item = input_items[i]
            item_type = item.get("type")

            if item_type == "function_call":
                tool_uses: list[dict[str, Any]] = []
                while i < len(input_items) and input_items[i].get("type") == "function_call":
                    fc = input_items[i]
                    raw_args = fc.get("arguments") or "{}"
                    try:
                        input_data = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    except (ValueError, TypeError):
                        input_data = {}
                    tool_uses.append(
                        {
                            "type": "tool_use",
                            "id": str(fc.get("call_id") or fc.get("id") or ""),
                            "name": str(fc.get("name") or ""),
                            "input": input_data,
                        }
                    )
                    i += 1
                messages.append({"role": "assistant", "content": tool_uses})

            elif item_type == "function_call_output":
                tool_results: list[dict[str, Any]] = []
                while i < len(input_items) and input_items[i].get("type") == "function_call_output":
                    fco = input_items[i]
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": str(fco.get("call_id") or ""),
                            "content": str(fco.get("output") or ""),
                        }
                    )
                    i += 1
                messages.append({"role": "user", "content": tool_results})

            elif "role" in item:
                messages.append({"role": item["role"], "content": item.get("content", "")})
                i += 1

            else:
                i += 1

        return messages

    @staticmethod
    def _parse_tool_calls(content: list[dict[str, Any]]) -> list[AgentToolCall]:
        calls: list[AgentToolCall] = []
        for block in content:
            if block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "").strip()
            if not name:
                continue
            input_data = block.get("input") or {}
            calls.append(
                AgentToolCall(
                    name=name,
                    arguments=dict(input_data) if isinstance(input_data, dict) else {},
                    call_id=str(block.get("id") or ""),
                )
            )
        return calls

    def next_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentProviderResponse:
        messages = self._convert_input_items(input_items)
        payload = {
            "model": self.model_name,
            "system": instructions,
            "messages": messages,
            "tools": self._convert_tools(tools),
            "max_tokens": self.max_tokens,
        }
        data = self._post_message(payload)
        content = data.get("content") or []
        if not isinstance(content, list):
            content = []
        tool_calls = self._parse_tool_calls(content)
        # Return conversation_items in Responses-API function_call format so the harness
        # can append them to input_items and we can round-trip them back to Claude format.
        conversation_items = [
            {
                "type": "function_call",
                "name": call.name,
                "arguments": json.dumps(call.arguments),
                "call_id": call.call_id,
            }
            for call in tool_calls
        ]
        text_parts = [block["text"] for block in content if block.get("type") == "text" and block.get("text")]
        output_text = " ".join(text_parts) if text_parts else None
        return AgentProviderResponse(
            tool_calls=tool_calls,
            output_text=output_text,
            raw=data,
            conversation_items=conversation_items,
        )
