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
