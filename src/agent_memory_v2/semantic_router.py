"""Embedding-based semantic router: matches user text against taxonomy prototype examples."""

from __future__ import annotations

__all__ = [
    "DEFAULT_THRESHOLD",
    "SemanticPrototype",
    "SemanticRouteResult",
    "SemanticRouter",
    "route_semantic_candidate",
    "PROTOTYPES",
]

import warnings
from dataclasses import dataclass

import numpy as np

from agent_memory_v2.embeddings import EmbeddingEncoder

DEFAULT_THRESHOLD: float = 0.72


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


def _load_prototypes() -> tuple[SemanticPrototype, ...]:
    try:
        from agent_memory_v2.taxonomy import get_taxonomy

        return get_taxonomy().to_prototypes()
    except Exception as exc:
        warnings.warn(f"Failed to load taxonomy prototypes, using fallback: {exc}", stacklevel=2)
        return _FALLBACK_PROTOTYPES


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class SemanticRouter:
    """Encodes all prototype examples once at construction; routes queries against cached vectors."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        *,
        prototypes: tuple[SemanticPrototype, ...] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.encoder = encoder
        self.prototypes = prototypes if prototypes is not None else _load_prototypes()
        self.threshold = threshold
        self._cache: list[tuple[SemanticPrototype, str, np.ndarray]] = [
            (proto, example, encoder.encode(example))
            for proto in self.prototypes
            for example in proto.examples
        ]

    def route(self, text: str) -> SemanticRouteResult | None:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        query_vector = self.encoder.encode(cleaned)
        best: SemanticRouteResult | None = None
        for proto, example, example_vector in self._cache:
            score = _cosine_similarity(query_vector, example_vector)
            if best is None or score > best.score:
                best = SemanticRouteResult(
                    candidate_key=proto.candidate_key,
                    candidate_class=proto.candidate_class,
                    description=proto.description,
                    score=score,
                    threshold=self.threshold,
                    above_threshold=score >= self.threshold,
                    durable_candidate=proto.durable_candidate,
                    matched_example=example,
                )
        return best


def route_semantic_candidate(
    text: str,
    encoder: EmbeddingEncoder,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    prototypes: tuple[SemanticPrototype, ...] | None = None,
) -> SemanticRouteResult | None:
    """Stateless routing helper; re-encodes examples on every call.

    For repeated calls prefer SemanticRouter which pre-computes and caches
    prototype vectors.
    """
    resolved = prototypes if prototypes is not None else _load_prototypes()
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    query_vector = encoder.encode(cleaned)
    best: SemanticRouteResult | None = None

    for prototype in resolved:
        for example in prototype.examples:
            score = _cosine_similarity(query_vector, encoder.encode(example))
            if best is None or score > best.score:
                best = SemanticRouteResult(
                    candidate_key=prototype.candidate_key,
                    candidate_class=prototype.candidate_class,
                    description=prototype.description,
                    score=score,
                    threshold=float(threshold),
                    above_threshold=score >= float(threshold),
                    durable_candidate=prototype.durable_candidate,
                    matched_example=example,
                )

    return best


# ---------------------------------------------------------------------------
# Fallback prototypes — used only when taxonomy.yaml cannot be loaded.
# The canonical list lives in config/taxonomy.yaml.
# ---------------------------------------------------------------------------

_FALLBACK_PROTOTYPES: tuple[SemanticPrototype, ...] = (
    SemanticPrototype(
        candidate_key="identity.location",
        candidate_class="fact",
        description="Where the user currently lives, is located, or is based.",
        durable_candidate=True,
        examples=(
            "I live in Edinburgh.",
            "I'm based in Edinburgh in the UK.",
            "I am based in Edinburgh.",
            "Actually, I am based in Glasgow now.",
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
        candidate_key="identity.dietary",
        candidate_class="fact",
        description="A dietary restriction, food preference, or eating habit the user follows.",
        durable_candidate=True,
        examples=(
            "I'm vegetarian.",
            "I don't eat meat.",
            "I'm vegan.",
            "I don't eat gluten.",
            "I'm lactose intolerant.",
            "I keep kosher.",
        ),
    ),
    SemanticPrototype(
        candidate_key="identity.health",
        candidate_class="fact",
        description="A health condition, allergy, or medical fact about the user.",
        durable_candidate=True,
        examples=(
            "I'm allergic to nuts.",
            "I have asthma.",
            "I have Type 2 diabetes.",
            "I'm allergic to penicillin.",
        ),
    ),
    SemanticPrototype(
        candidate_key="preference.communication",
        candidate_class="preference",
        description="How the user prefers to receive information or be communicated with.",
        durable_candidate=True,
        examples=(
            "I prefer short answers.",
            "Please be concise.",
            "I like bullet points.",
            "Can you keep responses brief?",
            "I prefer plain language over technical jargon.",
        ),
    ),
    SemanticPrototype(
        candidate_key="preference.schedule",
        candidate_class="preference",
        description="The user's preferred times, working patterns, or scheduling habits.",
        durable_candidate=True,
        examples=(
            "I work best in the mornings.",
            "I prefer evening meetings.",
            "Fridays are my quiet days.",
            "I usually work from home on Wednesdays.",
            "Mornings are busy for me.",
        ),
    ),
    SemanticPrototype(
        candidate_key="identity.relationship",
        candidate_class="fact",
        description="A named relationship or family fact the user has shared.",
        durable_candidate=True,
        examples=(
            "My partner's name is Sarah.",
            "I have two kids.",
            "My sister lives in Glasgow.",
            "My son just started school.",
            "I live with my partner.",
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

PROTOTYPES: tuple[SemanticPrototype, ...] = _load_prototypes()
