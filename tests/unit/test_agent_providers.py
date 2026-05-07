from __future__ import annotations

from agent_memory_v2.agent_eval import agent_tool_schemas
from agent_memory_v2.claude_provider import ClaudeProvider
from agent_memory_v2.openai_provider import OpenAIProvider


def test_openai_default_model_matches_execution_plan(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_MEMORY_V2_OPENAI_MODEL", raising=False)

    provider = OpenAIProvider()

    assert provider.model_name == "gpt-5.1"


def test_claude_provider_defaults_to_anthropic_sonnet(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_MEMORY_V2_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_V2_CLAUDE_MODEL", raising=False)

    provider = ClaudeProvider()

    assert provider.provider_name == "anthropic"
    assert provider.model_name == "claude-sonnet-4-20250514"


def test_claude_provider_supports_legacy_claude_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_MEMORY_V2_ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("AGENT_MEMORY_V2_CLAUDE_MODEL", "claude-test-model")

    provider = ClaudeProvider(provider_name="claude")

    assert provider.provider_name == "claude"
    assert provider.model_name == "claude-test-model"


def test_claude_tool_schema_conversion() -> None:
    tools = ClaudeProvider._convert_tools(agent_tool_schemas())

    assert tools[0]["name"] == "memory_write"
    assert "input_schema" in tools[0]
    assert tools[0]["input_schema"]["required"] == ["memory_class", "text", "reason"]


def test_claude_tool_use_parsing() -> None:
    calls = ClaudeProvider._parse_tool_calls(
        [
            {"type": "text", "text": "thinking"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "memory_query",
                "input": {"query": "What do I prefer?"},
            },
        ]
    )

    assert len(calls) == 1
    assert calls[0].name == "memory_query"
    assert calls[0].call_id == "toolu_1"
    assert calls[0].arguments == {"query": "What do I prefer?"}
