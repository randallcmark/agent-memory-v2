"""Sentiment detector: classifies user text as neutral, positive, negative, distressed, or urgent."""

from __future__ import annotations

__all__ = ["SentimentResult", "detect_sentiment"]

import re
from dataclasses import dataclass

_NEGATION_WORDS = frozenset(
    [
        "not",
        "never",
        "no",
        "don't",
        "dont",
        "didn't",
        "didnt",
        "wouldn't",
        "wouldnt",
        "shouldn't",
        "shouldnt",
    ]
)


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    cues: list[str]
    guidance: str


_POSITIVE_PATTERNS = [
    re.compile(
        r"\b(thanks|thank you|great|awesome|excellent|love|perfect|amazing)\b", re.IGNORECASE
    ),
]

_NEGATIVE_PATTERNS = [
    re.compile(
        r"\b(angry|annoyed|frustrated|upset|furious|hate|terrible|awful|disappointed|sad|unhappy|miserable|confused|lost|stuck|can't figure out)\b",
        re.IGNORECASE,
    ),
]

_DISTRESS_PATTERNS = [
    re.compile(
        r"\b(anxious|overwhelmed|stressed|panic|panicking|worried|scared|struggling|can't cope|breaking down|not okay|falling apart)\b",
        re.IGNORECASE,
    ),
]

_URGENT_PATTERNS = [
    re.compile(r"\b(urgent|asap|immediately|right now|quickly)\b", re.IGNORECASE),
]


def _is_negated(text: str, match: re.Match) -> bool:
    start = match.start()
    prefix = text[:start]
    tokens = re.findall(r"\b\w+'\w+|\b\w+\b", prefix)
    nearby = [t.lower() for t in tokens[-3:]]
    return any(w in _NEGATION_WORDS for w in nearby)


def _any_unnegated(patterns: list[re.Pattern], text: str) -> bool:
    for pattern in patterns:
        for match in pattern.finditer(text):
            if not _is_negated(text, match):
                return True
    return False


def detect_sentiment(text: str) -> SentimentResult:
    cleaned = (text or "").strip()
    lowered = cleaned.lower()
    cues: list[str] = []

    if _any_unnegated(_DISTRESS_PATTERNS, cleaned):
        cues.append("distress")
    if _any_unnegated(_NEGATIVE_PATTERNS, cleaned):
        cues.append("negative")
    if _any_unnegated(_URGENT_PATTERNS, cleaned):
        cues.append("urgent")
    if _any_unnegated(_POSITIVE_PATTERNS, cleaned):
        cues.append("positive")

    if "distress" in cues:
        return SentimentResult(
            label="distressed",
            confidence=0.9,
            cues=cues,
            guidance="Respond calmly, reduce cognitive load, and be especially clear and supportive.",
        )
    if "negative" in cues:
        return SentimentResult(
            label="negative",
            confidence=0.8,
            cues=cues,
            guidance="Acknowledge frustration briefly, stay direct, and avoid sounding defensive.",
        )
    if "positive" in cues:
        return SentimentResult(
            label="positive",
            confidence=0.75,
            cues=cues,
            guidance="Match a constructive tone while staying concise and useful.",
        )
    if "urgent" in cues:
        return SentimentResult(
            label="urgent",
            confidence=0.7,
            cues=cues,
            guidance="Prioritize speed and clarity. Lead with the actionable answer first.",
        )
    if lowered.endswith("?"):
        return SentimentResult(
            label="neutral",
            confidence=0.55,
            cues=["question"],
            guidance="Answer directly and keep the tone neutral and clear.",
        )
    return SentimentResult(
        label="neutral",
        confidence=0.5,
        cues=[],
        guidance="Keep the tone neutral, direct, and proportionate to the request.",
    )
