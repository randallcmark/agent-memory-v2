from pathlib import Path

import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

from agent_memory_v2.classifier import classify_text, detect_task_resolution
from agent_memory_v2.config import AppConfig
from agent_memory_v2.embeddings import HashEmbeddingEncoder
from agent_memory_v2.models import Message
from agent_memory_v2.pipeline import (
    MemoryPipeline,
    _build_temporal_context,
    _clean_memory_text,
    _format_recalled_item,
    _parse_timestamp,
    _recency_bonus,
    _relative_time_label,
    run_ollama_preflight,
)


class StubEncoder:
    def encode(self, text: str) -> np.ndarray:
        if "milk" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype="float32")
        return np.array([0.0, 1.0, 0.0], dtype="float32")


class StubOllama:
    def generate(self, prompt: str) -> str:
        return "stub-response"


class StubExtractionOllama:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root_dir=tmp_path,
        settings_path=tmp_path / "settings.yaml",
        raw={
            "llm": {
                "host": "http://localhost:11434",
                "model": "llama3:8b",
                "temperature": 0.2,
                "max_tokens": 100,
                "timeout_seconds": 10,
            },
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "memory": {
                "embedding_dim": 3,
                "top_k": 3,
                "similarity_threshold": 0.2,
                "index_path": "data/memory/test.index",
                "metadata_path": "data/memory/test.json",
                "interaction_log_path": "data/logs/interactions.jsonl",
            },
            "prompting": {
                "memory_heading": "Relevant memory",
                "input_heading": "Current user input",
            },
        },
    )


def test_pipeline_prefers_embedding_dimension_over_memory_dimension(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["embeddings"]["dimensions"] = 5
    config.raw["memory"]["embedding_dim"] = 3

    class DimEncoder:
        def encode(self, text: str) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype="float32")

    pipeline = MemoryPipeline(
        config,
        encoder=DimEncoder(),
        ollama=StubOllama(),
    )

    stored = pipeline.ingest(Message(role="user", text="remember milk"))

    assert stored.summary == "remember milk"
    assert pipeline.store.embedding_dim == 5


def test_pipeline_ingest_and_recall(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )

    stored = pipeline.ingest(Message(role="user", text="remember milk"))
    recalled = pipeline.recall(Message(role="user", text="milk", message_id="new-id"))

    assert stored.summary == "remember milk"
    assert len(recalled) == 1
    assert recalled[0]["text"] == "remember milk"
    assert "memory_class" in recalled[0]


def test_pipeline_ingest_turn_and_recall(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    user_message = Message(role="user", text="remember milk", turn_id="turn-1")
    agent_message = Message(role="agent", text="I will remember milk", turn_id="turn-1")

    stored = pipeline.ingest_turn(user_message, agent_message)
    recalled = pipeline.recall(Message(role="user", text="milk", message_id="new-id"))

    assert stored.role == "turn"
    assert stored.metadata["kind"] == "turn_memory"
    assert recalled[0]["role"] == "turn"
    assert "Agent: I will remember milk" in recalled[0]["text"]


def test_build_prompt_includes_recalled_memory(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    prompt = pipeline.build_prompt(
        Message(role="user", text="What did I ask before?"),
        [{"role": "user", "text": "remember milk", "score": 0.99, "timestamp": "x", "message_id": "m1"}],
    )
    assert "remember milk" in prompt
    assert "What did I ask before?" in prompt
    assert "As of now:" in prompt
    assert "Absolute timestamp:" in prompt
    assert "Day frame of reference:" in prompt
    assert "Detected user sentiment:" in prompt


def test_clean_memory_text_removes_response_boilerplate():
    raw = "Your response:\n\nI've taken note of your preference."
    cleaned = _clean_memory_text(raw)
    assert "Your response:" not in cleaned
    assert cleaned == "I've taken note of your preference."


def test_format_recalled_item_keeps_score_and_cleaned_text():
    rendered = _format_recalled_item(
        {
            "role": "turn",
            "text": "User: prefer oat milk\nAgent: Your response:\n\nNoted.",
            "score": 0.42,
            "timestamp": "2026-03-29T07:00:00+00:00",
        }
        ,
        make_config(Path("/tmp")),
    )
    assert rendered.startswith("- [turn] ")
    assert "score=0.420" in rendered
    assert "Your response:" not in rendered
    assert "relative=" in rendered
    assert "at=" in rendered


def test_build_prompt_handles_no_recalled_memory(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    prompt = pipeline.build_prompt(Message(role="user", text="hello"), [])
    assert "Relevant memory" in prompt
    assert "- none" in prompt


def test_build_prompt_preserves_recall_order(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    prompt = pipeline.build_prompt(
        Message(role="user", text="What matters?"),
        [
            {
                "role": "turn",
                "text": "first memory",
                "score": 0.9,
                "timestamp": "2026-03-29T07:00:00+00:00",
                "message_id": "m1",
            },
            {
                "role": "turn",
                "text": "second memory",
                "score": 0.8,
                "timestamp": "2026-03-28T07:00:00+00:00",
                "message_id": "m2",
            },
        ],
    )
    assert prompt.index("first memory") < prompt.index("second memory")


def test_build_prompt_separates_factual_and_contextual_sections(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    prompt = pipeline.build_prompt(
        Message(role="user", text="What do I prefer?"),
        [
            {
                "role": "preference",
                "text": "oat milk",
                "score": 0.9,
                "timestamp": "2026-03-29T07:00:00+00:00",
                "message_id": "m1",
                "store_kind": "sidecar_memory",
            },
            {
                "role": "turn",
                "text": "User: I prefer oat milk.\nAgent: Noted.",
                "score": 0.8,
                "timestamp": "2026-03-29T07:05:00+00:00",
                "message_id": "m2",
                "store_kind": "turn_memory",
                "memory_class": "task",
                "durable": True,
                "durability_reason": "user_task",
            },
        ],
    )
    assert "Relevant durable facts:" in prompt
    assert "Relevant conversation context:" in prompt
    assert prompt.index("Relevant durable facts:") < prompt.index("Relevant conversation context:")


def test_build_prompt_includes_user_profile_section(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(
        config,
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.profile_store.save(
        {
            "updated_at": "2026-03-29T00:00:00+00:00",
            "counts": {"preferences": 1, "facts": 0, "tasks": 0},
            "preferences": {
                "preference.general": {
                    "value": "oat milk",
                    "timestamp": "2026-03-29T00:00:00+00:00",
                    "memory_class": "preference",
                    "source_message_id": "m1",
                }
            },
            "facts": {},
            "tasks": {},
        }
    )

    prompt = pipeline.build_prompt(Message(role="user", text="What do I prefer?"), [])
    assert "User profile:" in prompt
    assert "- preference.general: oat milk" in prompt


def test_prompt_context_dedupes_profile_backed_factual_items(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(
        config,
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.profile_store.save(
        {
            "updated_at": "2026-03-29T00:00:00+00:00",
            "counts": {"preferences": 1, "facts": 0, "tasks": 0},
            "preferences": {
                "preference.general": {
                    "value": "oat milk",
                    "timestamp": "2026-03-29T00:00:00+00:00",
                    "memory_class": "preference",
                    "source_message_id": "m1",
                }
            },
            "facts": {},
            "tasks": {},
        }
    )

    context = pipeline.prompt_context(
        [
            {
                "role": "preference",
                "text": "oat milk",
                "score": 0.9,
                "timestamp": "2026-03-29T07:00:00+00:00",
                "message_id": "m1",
                "store_kind": "sidecar_memory",
                "memory_class": "preference",
                "profile_key": "preference.general",
                "extracted_value": "oat milk",
                "durable": True,
            }
        ]
    )

    assert context["profile"]["preferences"]["preference.general"]["value"] == "oat milk"
    assert context["factual"] == []


def test_prompt_context_dedupes_superseded_profile_fact_even_when_value_differs(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(
        config,
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.profile_store.save(
        {
            "updated_at": "2026-03-29T00:00:00+00:00",
            "counts": {"preferences": 0, "facts": 1, "tasks": 0},
            "preferences": {},
            "facts": {
                "identity.location": {
                    "value": "London now",
                    "timestamp": "2026-03-29T00:00:00+00:00",
                    "memory_class": "fact",
                    "source_message_id": "m2",
                }
            },
            "tasks": {},
        }
    )

    context = pipeline.prompt_context(
        [
            {
                "role": "fact",
                "text": "Bristol",
                "score": 0.9,
                "timestamp": "2026-03-29T07:00:00+00:00",
                "message_id": "m1",
                "store_kind": "sidecar_memory",
                "memory_class": "fact",
                "profile_key": "identity.location",
                "extracted_value": "Bristol",
                "durable": True,
            }
        ]
    )

    assert context["profile"]["facts"]["identity.location"]["value"] == "London now"
    assert context["factual"] == []


def test_prompt_context_drops_nondurable_contextual_when_durable_exists(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    context = pipeline.prompt_context(
        [
            {
                "role": "turn",
                "text": "What day is it today?",
                "score": 0.8,
                "timestamp": "2026-03-29T07:00:00+00:00",
                "message_id": "m1",
                "store_kind": "turn_memory",
                "memory_class": "turn",
                "durable": False,
                "durability_reason": "question",
            },
            {
                "role": "turn",
                "text": "I prefer oat milk.",
                "score": 0.79,
                "timestamp": "2026-03-29T08:00:00+00:00",
                "message_id": "m2",
                "store_kind": "turn_memory",
                "memory_class": "preference",
                "durable": True,
                "durability_reason": "user_preference",
            },
        ]
    )

    assert len(context["contextual"]) == 1
    assert context["contextual"][0]["message_id"] == "m2"
    assert len(context["dropped_contextual"]) == 1
    assert context["dropped_contextual"][0]["message_id"] == "m1"


def test_build_prompt_omits_profile_duplicated_factual_section(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(
        config,
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.profile_store.save(
        {
            "updated_at": "2026-03-29T00:00:00+00:00",
            "counts": {"preferences": 1, "facts": 0, "tasks": 0},
            "preferences": {
                "preference.general": {
                    "value": "oat milk",
                    "timestamp": "2026-03-29T00:00:00+00:00",
                    "memory_class": "preference",
                    "source_message_id": "m1",
                }
            },
            "facts": {},
            "tasks": {},
        }
    )

    prompt = pipeline.build_prompt(
        Message(role="user", text="What do I prefer?"),
        [
            {
                "role": "preference",
                "text": "oat milk",
                "score": 0.9,
                "timestamp": "2026-03-29T07:00:00+00:00",
                "message_id": "m1",
                "store_kind": "sidecar_memory",
                "memory_class": "preference",
                "profile_key": "preference.general",
                "extracted_value": "oat milk",
                "durable": True,
            }
        ],
    )

    assert "User profile:" in prompt
    assert "- preference.general: oat milk" in prompt
    assert "Relevant durable facts:" not in prompt
    assert "Additional recalled memory:" in prompt
    assert "Relevant memory:" not in prompt


def test_prompt_context_drops_ephemeral_context_when_profile_is_sufficient(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(
        config,
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.profile_store.save(
        {
            "updated_at": "2026-03-29T00:00:00+00:00",
            "counts": {"preferences": 1, "facts": 0, "tasks": 0},
            "preferences": {
                "preference.general": {
                    "value": "oat milk",
                    "timestamp": "2026-03-29T00:00:00+00:00",
                    "memory_class": "preference",
                    "source_message_id": "m1",
                }
            },
            "facts": {},
            "tasks": {},
        }
    )

    context = pipeline.prompt_context(
        [
            {
                "role": "turn",
                "text": "What day is it today?",
                "score": 0.8,
                "timestamp": "2026-03-29T07:00:00+00:00",
                "message_id": "m1",
                "store_kind": "turn_memory",
                "memory_class": "turn",
                "durable": False,
                "durability_reason": "question",
            }
        ]
    )

    assert context["contextual"] == []
    assert len(context["dropped_contextual"]) == 1


def test_build_prompt_uses_additional_recalled_memory_heading_when_profile_exists(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(
        config,
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.profile_store.save(
        {
            "updated_at": "2026-03-29T00:00:00+00:00",
            "counts": {"preferences": 1, "facts": 0, "tasks": 0},
            "preferences": {
                "preference.general": {
                    "value": "oat milk",
                    "timestamp": "2026-03-29T00:00:00+00:00",
                    "memory_class": "preference",
                    "source_message_id": "m1",
                }
            },
            "facts": {},
            "tasks": {},
        }
    )

    prompt = pipeline.build_prompt(Message(role="user", text="What do I prefer?"), [])
    assert "Additional recalled memory:" in prompt
    assert "Relevant memory:" not in prompt


def test_build_temporal_context_includes_absolute_and_relative_dates(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["app"] = {"timezone": "Europe/London"}
    config.raw["prompting"]["inject_now"] = True
    config.raw["prompting"]["temporal_context"] = {"enabled": True}
    context = _build_temporal_context(config)
    assert "As of now:" in context
    assert "Absolute timestamp:" in context
    assert "Day frame of reference:" in context


def test_build_prompt_includes_negative_sentiment_guidance(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    prompt = pipeline.build_prompt(Message(role="user", text="I am really frustrated with this bug."), [])
    assert "Detected user sentiment: negative" in prompt
    assert "avoid sounding defensive" in prompt


def test_parse_timestamp_handles_z_suffix():
    parsed = _parse_timestamp("2026-03-29T07:00:00Z")
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_relative_time_label_yesterday():
    tz = timezone.utc
    now_dt = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
    past_dt = now_dt - timedelta(days=1)
    assert _relative_time_label(past_dt, now_dt, tz) == "yesterday"


def test_recency_bonus_prefers_fresher_memory():
    now_dt = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
    fresh = _recency_bonus("2026-03-29T10:00:00+00:00", now_dt)
    stale = _recency_bonus("2026-01-29T10:00:00+00:00", now_dt)
    assert fresh > stale


def test_classify_preference():
    result = classify_text("Please remember that I prefer oat milk.", default_class="turn")
    assert result.memory_class == "preference"
    assert result.extracted_value == "oat milk"
    assert result.durable is True
    assert result.profile_key == "preference.general"


def test_classify_like_as_preference():
    result = classify_text("I like jasmine tea.", default_class="turn")
    assert result.memory_class == "preference"
    assert result.extracted_value == "jasmine tea"
    assert result.durable is True


def test_classify_fact_work():
    result = classify_text("I work as an architect.", default_class="turn")
    assert result.memory_class == "fact"
    assert result.extracted_value == "an architect"
    assert result.durable is True
    assert result.profile_key == "identity.occupation"


def test_classify_task():
    result = classify_text("Remind me to call the dentist.", default_class="turn")
    assert result.memory_class == "task"
    assert result.extracted_value == "call the dentist"
    assert result.durable is True


def test_classify_task_dont_forget():
    result = classify_text("Don't let me forget to renew my passport.", default_class="turn")
    assert result.memory_class == "task"
    assert result.extracted_value == "renew my passport"
    assert result.durable is True


def test_detect_task_resolution():
    resolved = detect_task_resolution("I completed call the dentist.")
    assert resolved == "call the dentist"


def test_classify_question_as_ephemeral_turn():
    result = classify_text("What day is it today?", default_class="turn")
    assert result.memory_class == "turn"
    assert result.durable is False
    assert result.durability_reason in {"ephemeral_pattern", "question"}


def test_recall_prioritizes_classified_memory_when_scores_are_close(tmp_path: Path):
    class RankingEncoder:
        def encode(self, text: str) -> np.ndarray:
            lowered = text.lower()
            if "prefer" in lowered or "oat milk" in lowered:
                return np.array([1.0, 0.0, 0.0], dtype="float32")
            if "small talk" in lowered:
                return np.array([0.99, 0.01, 0.0], dtype="float32")
            return np.array([1.0, 0.0, 0.0], dtype="float32")

    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=RankingEncoder(),
        ollama=StubOllama(),
    )
    pipeline.ingest_turn(
        Message(role="user", text="I prefer oat milk.", turn_id="t-pref"),
        Message(role="agent", text="Noted.", turn_id="t-pref"),
    )
    pipeline.ingest_turn(
        Message(role="user", text="small talk", turn_id="t-turn"),
        Message(role="agent", text="small talk reply", turn_id="t-turn"),
    )

    recalled = pipeline.recall(Message(role="user", text="What do I prefer? oat milk?", message_id="new-id"))
    assert recalled[0]["memory_class"] == "preference"
    assert recalled[0]["extracted_value"] == "oat milk"


def test_recall_prefers_sidecar_record_for_durable_fact(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact"],
    }
    pipeline = MemoryPipeline(
        config,
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.ingest_turn(
        Message(role="user", text="I prefer oat milk.", turn_id="t-pref"),
        Message(role="agent", text="Noted.", turn_id="t-pref"),
    )

    recalled = pipeline.recall(
        Message(role="user", text="What do I prefer? oat milk?", message_id="new-id")
    )
    assert recalled[0]["memory_class"] == "preference"
    assert recalled[0]["store_kind"] == "sidecar_memory"
    assert recalled[0]["source_message_id"] == recalled[0]["message_id"]


def test_semantic_candidate_metadata_is_stored_without_sidecar_promotion(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["embeddings"] = {"provider": "hash", "model": "hash", "dimensions": 128}
    config.raw["memory"]["embedding_dim"] = 128
    config.raw["semantic_router"] = {"enabled": True, "threshold": 0.72, "debug_metadata": True}
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact", "task"],
    }
    pipeline = MemoryPipeline(
        config,
        encoder=HashEmbeddingEncoder(dimensions=128),
        ollama=StubOllama(),
    )

    stored = pipeline.ingest_turn(
        Message(role="user", text="I'm based in Edinburgh in the UK.", turn_id="semantic-turn"),
        Message(role="agent", text="Noted.", turn_id="semantic-turn"),
    )

    semantic_candidate = stored.metadata["semantic_candidate"]
    assert stored.metadata["memory_class"] == "turn"
    assert stored.metadata["durable"] is False
    assert semantic_candidate["candidate_key"] == "identity.location"
    assert semantic_candidate["candidate_class"] == "fact"
    assert semantic_candidate["durable_candidate"] is True
    assert semantic_candidate["above_threshold"] is True
    assert pipeline.sidecar_store is not None
    assert pipeline.sidecar_store.records == []


def test_structured_extractor_promotes_semantic_candidate_to_sidecar_and_profile(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["embeddings"] = {"provider": "hash", "model": "hash", "dimensions": 128}
    config.raw["memory"]["embedding_dim"] = 128
    config.raw["semantic_router"] = {"enabled": True, "threshold": 0.72, "debug_metadata": True}
    config.raw["structured_extractor"] = {
        "enabled": True,
        "admission_threshold": 0.75,
        "max_value_chars": 160,
        "allowed_profile_keys": ["identity.location"],
    }
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact", "task"],
    }
    config.raw["profile"] = {"enabled": True, "path": "data/profile/test.json", "inject": True}
    pipeline = MemoryPipeline(
        config,
        encoder=HashEmbeddingEncoder(dimensions=128),
        ollama=StubExtractionOllama(
            '{"memory_class":"fact","durable":true,"profile_key":"identity.location",'
            '"extracted_value":"Edinburgh","confidence":0.86,'
            '"supersedes_profile_key":"identity.location"}'
        ),
    )

    stored = pipeline.ingest_turn(
        Message(role="user", text="I'm based in Edinburgh in the UK.", turn_id="semantic-turn"),
        Message(role="agent", text="Noted.", turn_id="semantic-turn"),
    )

    assert stored.metadata["classification_source"] == "structured_extractor"
    assert stored.metadata["memory_class"] == "fact"
    assert stored.metadata["durable"] is True
    assert stored.metadata["profile_key"] == "identity.location"
    assert stored.metadata["extracted_value"] == "Edinburgh"
    assert stored.metadata["rule_classification"]["memory_class"] == "turn"
    assert stored.metadata["structured_extraction"]["accepted"] is True
    assert pipeline.sidecar_store is not None
    assert len(pipeline.sidecar_store.records) == 1
    assert pipeline.sidecar_store.records[0].text == "Edinburgh"
    assert pipeline.profile_store is not None
    profile = pipeline.profile_store.load()
    assert profile["facts"]["identity.location"]["value"] == "Edinburgh"


def test_structured_extractor_rejection_keeps_semantic_candidate_nondurable(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["embeddings"] = {"provider": "hash", "model": "hash", "dimensions": 128}
    config.raw["memory"]["embedding_dim"] = 128
    config.raw["semantic_router"] = {"enabled": True, "threshold": 0.72, "debug_metadata": True}
    config.raw["structured_extractor"] = {
        "enabled": True,
        "admission_threshold": 0.75,
        "max_value_chars": 160,
        "allowed_profile_keys": ["identity.location"],
    }
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact", "task"],
    }
    pipeline = MemoryPipeline(
        config,
        encoder=HashEmbeddingEncoder(dimensions=128),
        ollama=StubExtractionOllama(
            '{"memory_class":"fact","durable":true,"profile_key":"identity.location",'
            '"extracted_value":"Edinburgh","confidence":0.5,'
            '"supersedes_profile_key":"identity.location"}'
        ),
    )

    stored = pipeline.ingest_turn(
        Message(role="user", text="I'm based in Edinburgh in the UK.", turn_id="semantic-turn"),
        Message(role="agent", text="Noted.", turn_id="semantic-turn"),
    )

    assert stored.metadata["classification_source"] == "rule"
    assert stored.metadata["memory_class"] == "turn"
    assert stored.metadata["durable"] is False
    assert stored.metadata["structured_extraction"]["accepted"] is False
    assert stored.metadata["structured_extraction"]["rejection_reason"] == "below_confidence_threshold"
    assert pipeline.sidecar_store is not None
    assert pipeline.sidecar_store.records == []


def test_merged_recall_groups_factual_and_contextual_items(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["sidecar"] = {
        "enabled": True,
        "top_k": 2,
        "similarity_threshold": 0.2,
        "index_path": "data/sidecar/test.index",
        "metadata_path": "data/sidecar/test.json",
        "store_classes": ["preference", "fact"],
    }
    pipeline = MemoryPipeline(
        config,
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.ingest_turn(
        Message(role="user", text="I prefer oat milk.", turn_id="t-pref"),
        Message(role="agent", text="Noted.", turn_id="t-pref"),
    )
    pipeline.ingest_turn(
        Message(role="user", text="milk question", turn_id="t-turn"),
        Message(role="agent", text="milk answer", turn_id="t-turn"),
    )

    merged = pipeline.merged_recall(
        Message(role="user", text="What do I prefer? oat milk?", message_id="new-id")
    )
    assert merged["factual_count"] == 1
    assert merged["contextual_count"] >= 1
    assert merged["factual"][0]["store_kind"] == "sidecar_memory"
    assert any(item["store_kind"] == "turn_memory" for item in merged["contextual"])


def test_recall_uses_recency_when_scores_and_classes_are_close(tmp_path: Path):
    class RecencyEncoder:
        def encode(self, text: str) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0], dtype="float32")

    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=RecencyEncoder(),
        ollama=StubOllama(),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="I like tea.",
            turn_id="older-turn",
            timestamp="2026-03-20T09:00:00+00:00",
        ),
        Message(
            role="agent",
            text="Noted.",
            turn_id="older-turn",
            timestamp="2026-03-20T09:00:01+00:00",
        ),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="I like coffee.",
            turn_id="newer-turn",
            timestamp="2026-03-29T09:00:00+00:00",
        ),
        Message(
            role="agent",
            text="Noted.",
            turn_id="newer-turn",
            timestamp="2026-03-29T09:00:01+00:00",
        ),
    )

    recalled = pipeline.recall(Message(role="user", text="What drink do I like?", message_id="new-id"))
    assert recalled[0]["extracted_value"] == "coffee"
    assert recalled[0]["recency_bonus"] >= recalled[1]["recency_bonus"]


def test_recall_applies_age_penalty_to_old_ephemeral_turn(tmp_path: Path):
    class FlatEncoder:
        def encode(self, text: str) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0], dtype="float32")

    config = make_config(tmp_path)
    config.raw["aging"] = {
        "enabled": True,
        "decay": {
            "ephemeral_start_days": 1,
            "ephemeral_full_days": 14,
            "ephemeral_max_penalty": 0.08,
            "task_start_days": 30,
            "task_full_days": 90,
            "task_max_penalty": 0.03,
            "durable_start_days": 180,
            "durable_full_days": 365,
            "durable_max_penalty": 0.01,
        },
    }
    pipeline = MemoryPipeline(
        config,
        encoder=FlatEncoder(),
        ollama=StubOllama(),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="hello there",
            turn_id="old-turn",
            timestamp="2026-03-01T09:00:00+00:00",
        ),
        Message(
            role="agent",
            text="hi",
            turn_id="old-turn",
            timestamp="2026-03-01T09:00:01+00:00",
        ),
    )
    recalled = pipeline.recall(Message(role="user", text="hello", message_id="new-id"))
    assert recalled[0]["age_penalty"] < 0.0


def test_recall_prefers_durable_memory_over_ephemeral_turn(tmp_path: Path):
    class DurabilityEncoder:
        def encode(self, text: str) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0], dtype="float32")

    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=DurabilityEncoder(),
        ollama=StubOllama(),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="I prefer oat milk.",
            turn_id="durable-turn",
            timestamp="2026-03-29T08:00:00+00:00",
        ),
        Message(
            role="agent",
            text="Noted.",
            turn_id="durable-turn",
            timestamp="2026-03-29T08:00:01+00:00",
        ),
    )
    pipeline.ingest_turn(
        Message(
            role="user",
            text="What day is it today?",
            turn_id="ephemeral-turn",
            timestamp="2026-03-29T09:00:00+00:00",
        ),
        Message(
            role="agent",
            text="It's Sunday.",
            turn_id="ephemeral-turn",
            timestamp="2026-03-29T09:00:01+00:00",
        ),
    )

    recalled = pipeline.recall(Message(role="user", text="What do I prefer?", message_id="new-id"))
    assert recalled[0]["memory_class"] == "preference"
    assert recalled[0]["durable"] is True
    assert recalled[1]["durable"] is False
    assert recalled[0]["durability_bonus"] > recalled[1]["durability_bonus"]


def test_recall_excludes_current_message_id(tmp_path: Path):
    pipeline = MemoryPipeline(
        make_config(tmp_path),
        encoder=StubEncoder(),
        ollama=StubOllama(),
    )
    pipeline.ingest(Message(role="user", text="remember milk", message_id="same-id"))

    recalled = pipeline.recall(Message(role="user", text="milk", message_id="same-id"))
    assert recalled == []


def test_run_ollama_preflight_uses_config(monkeypatch, tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["llm"]["preflight"] = {
        "enabled": True,
        "generate": True,
        "timeout_seconds": 7,
        "max_tokens": 3,
    }
    captured = {}

    class StubClient:
        def __init__(self, profile):
            captured["profile"] = profile

        def healthcheck(self, *, run_generate: bool = False):
            captured["run_generate"] = run_generate
            return {"reachable": True, "model_present": True, "version": "test"}

    monkeypatch.setattr("agent_memory_v2.pipeline.OllamaClient", StubClient)
    result = run_ollama_preflight(config)

    assert result["reachable"] is True
    assert captured["profile"].max_tokens == 3
    assert captured["profile"].timeout_seconds == 7
    assert captured["run_generate"] is True


def test_run_ollama_preflight_can_be_skipped(tmp_path: Path):
    config = make_config(tmp_path)
    config.raw["llm"]["preflight"] = {"enabled": False}
    result = run_ollama_preflight(config)
    assert result == {"skipped": True}
