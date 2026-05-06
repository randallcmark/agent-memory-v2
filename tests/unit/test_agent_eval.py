from __future__ import annotations

from pathlib import Path

from agent_memory_v2.agent_eval import (
    AgentToolCall,
    FakeAgentProvider,
    execute_tool_call,
    isolated_agent_config,
    run_agent_scenario,
)
from agent_memory_v2.config import load_config
from agent_memory_v2.pipeline import MemoryPipeline


def _scenario() -> dict:
    return {
        "name": "preference_recall",
        "description": "Simple durable preference recall.",
        "setup_turns": [{"user": "I prefer oat milk.", "agent": "Noted."}],
        "query": "What do I prefer?",
    }


def test_fake_provider_writes_queries_and_answers(tmp_path: Path) -> None:
    base_config = load_config()
    result = run_agent_scenario(
        base_config=base_config,
        scenario=_scenario(),
        provider=FakeAgentProvider(_scenario()),
        temp_root=tmp_path,
    )

    assert result["passed"] is True
    assert result["final_answer"] == "You prefer oat milk."
    assert result["invalid_tool_calls"] == 0
    assert [call["name"] for step in result["trace"] for call in step["tool_calls"]] == [
        "memory_write",
        "memory_query",
        "answer",
    ]
    assert result["memory_state"]["profile"]["preferences"]["preference.general"]["value"] == (
        "I prefer oat milk."
    )


def test_invalid_tool_call_is_counted(tmp_path: Path) -> None:
    result = run_agent_scenario(
        base_config=load_config(),
        scenario=_scenario(),
        provider=FakeAgentProvider(_scenario(), mode="invalid"),
        temp_root=tmp_path,
    )

    assert result["passed"] is False
    assert result["invalid_tool_calls"] == 1
    assert result["trace"][0]["tool_calls"][0]["result"]["error"] == "unsupported_memory_class"


def test_tool_call_budget_stops_loop(tmp_path: Path) -> None:
    result = run_agent_scenario(
        base_config=load_config(),
        scenario=_scenario(),
        provider=FakeAgentProvider(_scenario(), mode="loop"),
        temp_root=tmp_path,
        max_tool_calls=2,
    )

    assert result["passed"] is False
    assert result["final_answer"] is None
    assert result["tool_call_count"] == 2


def test_memory_evolve_is_reserved(tmp_path: Path) -> None:
    config = isolated_agent_config(load_config(), tmp_path)
    pipeline = MemoryPipeline(config)
    result, answer = execute_tool_call(
        pipeline,
        scenario_name="example",
        call=AgentToolCall(name="memory_evolve", arguments={"proposal": "add schema"}),
    )

    assert answer is None
    assert result["ok"] is False
    assert result["error"] == "unsupported_tool"


def test_agent_result_contains_artifact_payload_fields(tmp_path: Path) -> None:
    result = run_agent_scenario(
        base_config=load_config(),
        scenario=_scenario(),
        provider=FakeAgentProvider(_scenario()),
        temp_root=tmp_path,
    )

    assert result["provider"] == "fake"
    assert result["model"] == "fake-agent"
    assert result["elapsed_ms"] >= 0
    assert result["trace"]
    assert "git" in result
    assert "memory_state" in result
