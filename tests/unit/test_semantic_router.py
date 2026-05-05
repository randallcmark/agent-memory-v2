import pytest

from agent_memory_v2.embeddings import HashEmbeddingEncoder
from agent_memory_v2.semantic_router import route_semantic_candidate


@pytest.fixture
def encoder() -> HashEmbeddingEncoder:
    return HashEmbeddingEncoder(dimensions=128)


def test_semantic_router_detects_location_candidate(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("I'm based in Edinburgh in the UK.", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "identity.location"
    assert result.candidate_class == "fact"
    assert result.durable_candidate is True
    assert result.above_threshold is True


def test_semantic_router_detects_location_correction_candidate(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("Actually, I am based in Glasgow now.", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "identity.location"
    assert result.candidate_class == "fact"
    assert result.durable_candidate is True
    assert result.above_threshold is True


def test_semantic_router_detects_contextual_world_fact(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("The Meadows has cherry blossom trees.", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "contextual.world_fact"
    assert result.durable_candidate is False
    assert result.above_threshold is True


def test_semantic_router_detects_ephemeral_question(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("What day is it today?", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "ephemeral.question"
    assert result.durable_candidate is False
    assert result.above_threshold is True


def test_semantic_router_reports_below_threshold(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("I'm based in Edinburgh in the UK.", encoder, threshold=1.01)

    assert result is not None
    assert result.candidate_key == "identity.location"
    assert result.above_threshold is False


def test_semantic_router_ignores_empty_text(encoder: HashEmbeddingEncoder):
    assert route_semantic_candidate("   ", encoder) is None


# ---------------------------------------------------------------------------
# New prototypes — each tested with an exact example string from its definition
# so the hash encoder gives cosine similarity 1.0 against that prototype
# ---------------------------------------------------------------------------


def test_semantic_router_detects_dietary_restriction(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("I'm vegetarian.", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "identity.dietary"
    assert result.durable_candidate is True
    assert result.above_threshold is True


def test_semantic_router_detects_health_condition(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("I'm allergic to nuts.", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "identity.health"
    assert result.durable_candidate is True
    assert result.above_threshold is True


def test_semantic_router_detects_relationship(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("My partner's name is Sarah.", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "identity.relationship"
    assert result.durable_candidate is True
    assert result.above_threshold is True


def test_semantic_router_detects_communication_preference(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("I prefer short answers.", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "preference.communication"
    assert result.durable_candidate is True
    assert result.above_threshold is True


def test_semantic_router_detects_schedule_preference(encoder: HashEmbeddingEncoder):
    result = route_semantic_candidate("I work best in the mornings.", encoder, threshold=0.72)

    assert result is not None
    assert result.candidate_key == "preference.schedule"
    assert result.durable_candidate is True
    assert result.above_threshold is True


def test_new_prototypes_are_all_durable_candidates(encoder: HashEmbeddingEncoder):
    durable_examples = [
        "I'm vegetarian.",
        "I'm allergic to nuts.",
        "My partner's name is Sarah.",
        "I prefer short answers.",
        "I work best in the mornings.",
    ]
    for text in durable_examples:
        result = route_semantic_candidate(text, encoder, threshold=0.72)
        assert result is not None and result.durable_candidate is True, (
            f"'{text}' should route to a durable candidate"
        )
