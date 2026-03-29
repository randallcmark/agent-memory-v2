from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    cues: list[str]
    guidance: str


_POSITIVE_PATTERNS = [
    re.compile(r"\b(thanks|thank you|great|awesome|excellent|love|perfect|amazing)\b", re.IGNORECASE),
]

_NEGATIVE_PATTERNS = [
    re.compile(r"\b(angry|annoyed|frustrated|upset|furious|hate|terrible|awful|disappointed)\b", re.IGNORECASE),
]

_DISTRESS_PATTERNS = [
    re.compile(r"\b(anxious|overwhelmed|stressed|panic|panicking|worried|scared)\b", re.IGNORECASE),
]

_URGENT_PATTERNS = [
    re.compile(r"\b(urgent|asap|immediately|right now|quickly)\b", re.IGNORECASE),
]


def detect_sentiment(text: str) -> SentimentResult:
    cleaned = (text or "").strip()
    lowered = cleaned.lower()
    cues: list[str] = []

    if any(pattern.search(cleaned) for pattern in _DISTRESS_PATTERNS):
        cues.append("distress")
    if any(pattern.search(cleaned) for pattern in _NEGATIVE_PATTERNS):
        cues.append("negative")
    if any(pattern.search(cleaned) for pattern in _URGENT_PATTERNS):
        cues.append("urgent")
    if any(pattern.search(cleaned) for pattern in _POSITIVE_PATTERNS):
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
