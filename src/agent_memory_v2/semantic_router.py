from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_memory_v2.embeddings import EmbeddingEncoder


@dataclass(frozen=True)
class SemanticPrototype:
    candidate_key: str
    candidate_class: str
    description: str
    durable_candidate: bool
    examples: tuple[str, ...]


@dataclass(frozen=True)
class SemanticRouteResult:
    candidate_key: str
    candidate_class: str
    description: str
    score: float
    threshold: float
    above_threshold: bool
    durable_candidate: bool
    matched_example: str

    def to_metadata(self) -> dict:
        return {
            "candidate_key": self.candidate_key,
            "candidate_class": self.candidate_class,
            "description": self.description,
            "score": self.score,
            "threshold": self.threshold,
            "above_threshold": self.above_threshold,
            "durable_candidate": self.durable_candidate,
            "matched_example": self.matched_example,
        }


PROTOTYPES: tuple[SemanticPrototype, ...] = (
    SemanticPrototype(
        candidate_key="identity.location",
        candidate_class="fact",
        description="Where the user currently lives, is located, or is based.",
        durable_candidate=True,
        examples=(
            "I live in Edinburgh.",
            "I'm based in Edinburgh in the UK.",
            "I am based in Edinburgh.",
            "My home is in London.",
            "I currently live in Bristol.",
        ),
    ),
    SemanticPrototype(
        candidate_key="identity.name",
        candidate_class="fact",
        description="The user's name or preferred form of address.",
        durable_candidate=True,
        examples=(
            "My name is Mark.",
            "I am called Sarah.",
            "Please call me Alex.",
        ),
    ),
    SemanticPrototype(
        candidate_key="identity.occupation",
        candidate_class="fact",
        description="The user's occupation, role, employer, or professional identity.",
        durable_candidate=True,
        examples=(
            "I work as a software engineer.",
            "I am a product manager.",
            "I work at a university.",
            "My job is data analyst.",
        ),
    ),
    SemanticPrototype(
        candidate_key="identity.origin",
        candidate_class="fact",
        description="Where the user is from or the conventions they identify with.",
        durable_candidate=True,
        examples=(
            "I'm from Scotland.",
            "I grew up in Wales.",
            "My conventions originate in the UK.",
        ),
    ),
    SemanticPrototype(
        candidate_key="preference.general",
        candidate_class="preference",
        description="A stated preference, liking, dislike, favorite, or preferred option.",
        durable_candidate=True,
        examples=(
            "I prefer oat milk.",
            "I like jasmine tea.",
            "My favourite colour is blue.",
            "I don't like coriander.",
        ),
    ),
    SemanticPrototype(
        candidate_key="task.general",
        candidate_class="task",
        description="A reminder, to-do, obligation, or future action the user wants retained.",
        durable_candidate=True,
        examples=(
            "Remind me to renew my passport.",
            "I need to call the dentist.",
            "Don't let me forget to submit the form.",
        ),
    ),
    SemanticPrototype(
        candidate_key="contextual.world_fact",
        candidate_class="context",
        description="A useful external or local-world fact that is not necessarily about the user profile.",
        durable_candidate=False,
        examples=(
            "The Meadows has cherry blossom trees.",
            "The spring equinox is used as a convention for the first day of spring.",
            "There are cherry trees in that park.",
        ),
    ),
    SemanticPrototype(
        candidate_key="ephemeral.question",
        candidate_class="ephemeral",
        description="A question or transient request that should usually not become durable memory.",
        durable_candidate=False,
        examples=(
            "What day is it today?",
            "Is there any significance to today?",
            "Can you help me with this?",
            "What time is it?",
        ),
    ),
)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def route_semantic_candidate(
    text: str,
    encoder: EmbeddingEncoder,
    *,
    threshold: float = 0.72,
    prototypes: tuple[SemanticPrototype, ...] = PROTOTYPES,
) -> SemanticRouteResult | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    query_vector = encoder.encode(cleaned)
    best_result: SemanticRouteResult | None = None

    for prototype in prototypes:
        for example in prototype.examples:
            score = _cosine_similarity(query_vector, encoder.encode(example))
            if best_result is None or score > best_result.score:
                best_result = SemanticRouteResult(
                    candidate_key=prototype.candidate_key,
                    candidate_class=prototype.candidate_class,
                    description=prototype.description,
                    score=score,
                    threshold=float(threshold),
                    above_threshold=score >= float(threshold),
                    durable_candidate=prototype.durable_candidate,
                    matched_example=example,
                )

    return best_result
