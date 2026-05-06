from __future__ import annotations

import json
import os
from typing import Any

import requests

from agent_memory_v2.agent_eval import AgentProviderResponse, AgentToolCall
from agent_memory_v2.ollama import _with_retry


class OpenAIProvider:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI agent eval runs")
        self.model_name = model or os.environ.get("AGENT_MEMORY_V2_OPENAI_MODEL", "gpt-5.1")
        self.base_url = (base_url or os.environ.get("AGENT_MEMORY_V2_OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        def _call() -> dict[str, Any]:
            response = requests.post(
                f"{self.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("OpenAI returned a non-object response")
            return data

        return _with_retry(_call)

    @staticmethod
    def _parse_tool_calls(items: list[dict[str, Any]]) -> list[AgentToolCall]:
        calls: list[AgentToolCall] = []
        for item in items:
            if item.get("type") != "function_call":
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            raw_args = item.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {}
            calls.append(
                AgentToolCall(
                    name=name,
                    arguments=args,
                    call_id=str(item.get("call_id") or item.get("id") or ""),
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
        payload = {
            "model": self.model_name,
            "instructions": instructions,
            "input": input_items,
            "tools": tools,
        }
        data = self._post_response(payload)
        output = data.get("output") or []
        if not isinstance(output, list):
            output = []
        output_items = [item for item in output if isinstance(item, dict)]
        return AgentProviderResponse(
            tool_calls=self._parse_tool_calls(output_items),
            output_text=data.get("output_text"),
            raw=data,
            conversation_items=output_items,
        )
