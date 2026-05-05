from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from agent_memory_v2.classifier import ClassificationResult
from agent_memory_v2.semantic_router import SemanticRouteResult


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class StructuredExtractionResult:
    memory_class: str | None
    durable: bool
    profile_key: str | None
    extracted_value: str | None
    confidence: float
    supersedes_profile_key: str | None
    accepted: bool
    rejection_reason: str | None
    raw_response: str

    def to_metadata(self) -> dict:
        return {
            "memory_class": self.memory_class,
            "durable": self.durable,
            "profile_key": self.profile_key,
            "extracted_value": self.extracted_value,
            "confidence": self.confidence,
            "supersedes_profile_key": self.supersedes_profile_key,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "raw_response": self.raw_response,
        }

    def to_classification_result(self) -> ClassificationResult:
        if not self.accepted:
            raise ValueError("Only accepted structured extractions can become classification results")
        return ClassificationResult(
            memory_class=str(self.memory_class),
            extracted_value=self.extracted_value,
            confidence=self.confidence,
            durable=True,
            durability_reason="structured_extractor",
            profile_key=self.profile_key,
        )


def _extract_json_object(text: str) -> dict:
    cleaned = (text or "").strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found in extractor response")


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _compact_value(value: object, max_chars: int) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if len(cleaned) > max_chars:
        return None
    return cleaned


def _build_prompt(text: str, route: SemanticRouteResult, allowed_keys: set[str]) -> str:
    keys_str = "|".join(sorted(allowed_keys))
    return (
        "You extract durable memory from one user utterance.\n"
        "Return exactly one JSON object and no prose.\n\n"
        "Allowed schema:\n"
        "{\n"
        '  "memory_class": "fact|preference|task",\n'
        '  "durable": true,\n'
        f'  "profile_key": "{keys_str}",\n'
        '  "extracted_value": "short exact value from the user utterance",\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "supersedes_profile_key": "profile key or null"\n'
        "}\n\n"
        "If the utterance is not durable user memory, return durable=false with null profile_key and null extracted_value.\n"
        "Use confidence 0.80 to 0.95 for clear first-person user facts, preferences, or tasks.\n"
        "Use confidence below 0.75 when the value is ambiguous or inferred rather than stated.\n"
        "Do not invent values. Extract only what the user stated.\n\n"
        "Example:\n"
        "User utterance: I'm based in Edinburgh in the UK.\n"
        '{"memory_class":"fact","durable":true,"profile_key":"identity.location",'
        '"extracted_value":"Edinburgh in the UK","confidence":0.9,'
        '"supersedes_profile_key":"identity.location"}\n\n'
        f"Semantic candidate key: {route.candidate_key}\n"
        f"Semantic candidate class: {route.candidate_class}\n"
        f"Semantic candidate description: {route.description}\n"
        f"User utterance: {text.strip()}\n"
    )


def validate_structured_extraction(
    parsed: dict,
    *,
    route: SemanticRouteResult,
    admission_threshold: float,
    allowed_profile_keys: set[str],
    max_value_chars: int,
) -> StructuredExtractionResult:
    memory_class = parsed.get("memory_class")
    memory_class = str(memory_class).strip() if memory_class is not None else None
    profile_key = parsed.get("profile_key")
    profile_key = str(profile_key).strip() if profile_key is not None else None
    extracted_value = _compact_value(parsed.get("extracted_value"), max_value_chars)
    confidence = _coerce_confidence(parsed.get("confidence"))
    supersedes = parsed.get("supersedes_profile_key")
    supersedes_profile_key = str(supersedes).strip() if supersedes not in (None, "") else None
    durable = bool(parsed.get("durable", False))
    raw_response = json.dumps(parsed, sort_keys=True)

    rejection_reason = None
    if not durable:
        rejection_reason = "not_durable"
    elif confidence < admission_threshold:
        rejection_reason = "below_confidence_threshold"
    elif not memory_class or memory_class not in {"fact", "preference", "task"}:
        rejection_reason = "unsupported_memory_class"
    elif not profile_key or profile_key not in allowed_profile_keys:
        rejection_reason = "unsupported_profile_key"
    elif profile_key != route.candidate_key:
        rejection_reason = "profile_key_mismatch"
    elif route.candidate_class in {"fact", "preference", "task"} and memory_class != route.candidate_class:
        rejection_reason = "memory_class_mismatch"
    elif extracted_value is None:
        rejection_reason = "missing_or_oversized_value"

    return StructuredExtractionResult(
        memory_class=memory_class,
        durable=durable,
        profile_key=profile_key,
        extracted_value=extracted_value,
        confidence=confidence,
        supersedes_profile_key=supersedes_profile_key,
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
        raw_response=raw_response,
    )


def extract_structured_memory(
    text: str,
    route: SemanticRouteResult,
    generator: TextGenerator,
    *,
    admission_threshold: float,
    allowed_profile_keys: set[str],
    max_value_chars: int = 160,
) -> StructuredExtractionResult:
    raw_response = generator.generate(_build_prompt(text, route, allowed_profile_keys))
    try:
        parsed = _extract_json_object(raw_response)
    except ValueError as exc:
        return StructuredExtractionResult(
            memory_class=None,
            durable=False,
            profile_key=None,
            extracted_value=None,
            confidence=0.0,
            supersedes_profile_key=None,
            accepted=False,
            rejection_reason=str(exc),
            raw_response=raw_response,
        )

    result = validate_structured_extraction(
        parsed,
        route=route,
        admission_threshold=admission_threshold,
        allowed_profile_keys=allowed_profile_keys,
        max_value_chars=max_value_chars,
    )
    if result.accepted:
        return StructuredExtractionResult(
            memory_class=result.memory_class,
            durable=result.durable,
            profile_key=result.profile_key,
            extracted_value=result.extracted_value,
            confidence=result.confidence,
            supersedes_profile_key=result.supersedes_profile_key,
            accepted=True,
            rejection_reason=None,
            raw_response=raw_response,
        )
    return StructuredExtractionResult(
        memory_class=result.memory_class,
        durable=result.durable,
        profile_key=result.profile_key,
        extracted_value=result.extracted_value,
        confidence=result.confidence,
        supersedes_profile_key=result.supersedes_profile_key,
        accepted=False,
        rejection_reason=result.rejection_reason,
        raw_response=raw_response,
    )
