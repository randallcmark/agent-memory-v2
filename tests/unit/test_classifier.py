from __future__ import annotations

import pytest

from agent_memory_v2.classifier import classify_text

# ---------------------------------------------------------------------------
# "I must" negative lookahead — idioms should not become tasks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I must say I agree with you.",
        "I must admit I was wrong about that.",
        "I must confess I forgot.",
        "I must stress that this is important.",
        "I must emphasise the key points.",
        "I must emphasize the deadline.",
        "I must note the exception.",
        "I must point out the error.",
    ],
)
def test_i_must_idioms_are_not_tasks(text: str) -> None:
    result = classify_text(text)
    assert result.memory_class != "task", f"'{text}' was classified as a task but should not be"


@pytest.mark.parametrize(
    "text",
    [
        "I must finish this report by Friday.",
        "I must call the dentist tomorrow.",
        "I must renew my passport before the trip.",
        "I must complete the application form.",
    ],
)
def test_i_must_genuine_tasks_are_detected(text: str) -> None:
    result = classify_text(text)
    assert result.memory_class == "task", f"'{text}' should be classified as a task"
    assert result.durable is True


# ---------------------------------------------------------------------------
# Fact-pattern profile-key derivation — explicit mapping, not string-split
# ---------------------------------------------------------------------------


def test_fact_pattern_live_in_maps_to_location() -> None:
    result = classify_text("I live in Edinburgh.")
    assert result.memory_class == "fact"
    assert result.profile_key == "identity.location"
    assert result.extracted_value == "Edinburgh"


def test_fact_pattern_name_maps_to_identity_name() -> None:
    result = classify_text("My name is Sarah.")
    assert result.memory_class == "fact"
    assert result.profile_key == "identity.name"
    assert result.extracted_value == "Sarah"


def test_fact_pattern_work_as_maps_to_occupation() -> None:
    result = classify_text("I work as a software engineer.")
    assert result.memory_class == "fact"
    assert result.profile_key == "identity.occupation"
    assert result.extracted_value == "a software engineer"


def test_fact_pattern_work_at_maps_to_employer() -> None:
    result = classify_text("I work at the university.")
    assert result.memory_class == "fact"
    assert result.profile_key == "identity.employer"
    assert result.extracted_value == "the university"


def test_fact_pattern_allergic_to_maps_to_allergy() -> None:
    result = classify_text("I am allergic to penicillin.")
    assert result.memory_class == "fact"
    assert result.profile_key == "identity.allergy"
    assert result.extracted_value == "penicillin"


def test_fact_pattern_birthday_maps_to_birthday() -> None:
    result = classify_text("My birthday is the 12th of March.")
    assert result.memory_class == "fact"
    assert result.profile_key == "identity.birthday"


def test_fact_pattern_from_maps_to_origin() -> None:
    result = classify_text("I'm from Scotland.")
    assert result.memory_class == "fact"
    assert result.profile_key == "identity.origin"
    assert result.extracted_value == "Scotland"


def test_fact_all_patterns_are_durable() -> None:
    facts = [
        "I live in Edinburgh.",
        "My name is Alex.",
        "I work as a teacher.",
        "I work at the hospital.",
        "I am allergic to nuts.",
        "My birthday is June 5th.",
        "I'm from Wales.",
    ]
    for text in facts:
        result = classify_text(text)
        assert result.durable is True, f"'{text}' should be durable"
        assert result.memory_class == "fact", f"'{text}' should be fact"
