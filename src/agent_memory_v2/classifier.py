from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationResult:
    memory_class: str
    extracted_value: str | None = None
    confidence: float = 0.0
    durable: bool = False
    durability_reason: str | None = None
    profile_key: str | None = None


_PREFERENCE_PATTERNS = [
    re.compile(r"\bI like\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bI love\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bI prefer\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bmy preferred\s+(?P<subject>\w+)\s+is\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bmy favorite\s+(?P<subject>\w+)\s+is\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bmy favourite\s+(?P<subject>\w+)\s+is\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
]

_FACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bI'm from\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.origin"),
    (re.compile(r"\bI am from\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.origin"),
    (re.compile(r"\bmy name is\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.name"),
    (re.compile(r"\bmy birthday is\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.birthday"),
    (re.compile(r"\bI live in\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.location"),
    (re.compile(r"\bI work as\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.occupation"),
    (re.compile(r"\bI work at\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.employer"),
    (re.compile(r"\bI am allergic to\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.allergy"),
    (re.compile(r"\bremember that I am\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE), "identity.general"),
]

_TASK_PATTERNS = [
    re.compile(r"\bdon't let me forget to\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bI need to\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bplease remind me to\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bremind me to\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bI should\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(
        r"\bI must(?!\s+(?:say|admit|confess|stress|emphasise|emphasize|note|point out))\s+(?P<value>.+?)(?:[.!?]|$)",
        re.IGNORECASE,
    ),
]

_TASK_RESOLUTION_PATTERNS = [
    re.compile(r"\bI (?:already\s+)?(?:did|finished|completed|handled)\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bI (?:have\s+)?done\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bI took care of\s+(?P<value>.+?)(?:[.!?]|$)", re.IGNORECASE),
]

_EPHEMERAL_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\b[!. ]*$", re.IGNORECASE),
    re.compile(r"^\s*(bye|goodbye|see you|thanks|thank you)\b[!. ]*$", re.IGNORECASE),
    re.compile(r"\bwhat day is it\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s the date\b", re.IGNORECASE),
    re.compile(r"\bdo you know (today'?s|todays) date\b", re.IGNORECASE),
    re.compile(r"\bwhat time is it\b", re.IGNORECASE),
]


def _generic_durability(cleaned: str) -> tuple[bool, str]:
    if any(pattern.search(cleaned) for pattern in _EPHEMERAL_PATTERNS):
        return False, "ephemeral_pattern"
    if cleaned.endswith("?"):
        return False, "question"
    if len(cleaned.split()) <= 3:
        return False, "short_utterance"
    return False, "generic_turn"


def classify_text(text: str, *, default_class: str = "turn") -> ClassificationResult:
    cleaned = (text or "").strip()
    if not cleaned:
        return ClassificationResult(
            memory_class=default_class,
            extracted_value=None,
            confidence=0.0,
            durable=False,
            durability_reason="empty",
        )

    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            subject = match.groupdict().get("subject")
            profile_key = f"preference.{subject.lower()}" if subject else "preference.general"
            return ClassificationResult(
                "preference",
                match.group("value").strip(),
                0.95,
                True,
                "user_preference",
                profile_key,
            )

    for pattern, profile_key in _FACT_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return ClassificationResult(
                "fact",
                match.group("value").strip(),
                0.90,
                True,
                "user_fact",
                profile_key,
            )

    for pattern in _TASK_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return ClassificationResult(
                "task",
                match.group("value").strip(),
                0.85,
                True,
                "user_task",
                "task.general",
            )

    durable, reason = _generic_durability(cleaned)
    return ClassificationResult(
        memory_class=default_class,
        extracted_value=None,
        confidence=0.40,
        durable=durable,
        durability_reason=reason,
        profile_key=None,
    )


def detect_task_resolution(text: str) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    for pattern in _TASK_RESOLUTION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return match.group("value").strip()
    return None


def class_priority(memory_class: str) -> float:
    priorities = {
        "preference": 0.06,
        "fact": 0.05,
        "task": 0.04,
        "turn": 0.00,
        "message": 0.00,
    }
    return priorities.get(memory_class, 0.0)


def durability_bonus(durable: bool) -> float:
    return 0.03 if durable else -0.01
