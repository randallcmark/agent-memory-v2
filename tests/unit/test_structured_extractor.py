from agent_memory_v2.semantic_router import SemanticRouteResult
from agent_memory_v2.structured_extractor import (
    extract_structured_memory,
    validate_structured_extraction,
)


class StubGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def location_route() -> SemanticRouteResult:
    return SemanticRouteResult(
        candidate_key="identity.location",
        candidate_class="fact",
        description="Where the user is based.",
        score=0.9,
        threshold=0.72,
        above_threshold=True,
        durable_candidate=True,
        matched_example="I'm based in Edinburgh in the UK.",
    )


def test_structured_extractor_accepts_valid_location_json():
    generator = StubGenerator(
        """
        ```json
        {
          "memory_class": "fact",
          "durable": true,
          "profile_key": "identity.location",
          "extracted_value": "Edinburgh",
          "confidence": 0.86,
          "supersedes_profile_key": "identity.location"
        }
        ```
        """
    )

    result = extract_structured_memory(
        "I'm based in Edinburgh in the UK.",
        location_route(),
        generator,
        admission_threshold=0.75,
        allowed_profile_keys={"identity.location"},
    )

    assert result.accepted is True
    assert result.memory_class == "fact"
    assert result.profile_key == "identity.location"
    assert result.extracted_value == "Edinburgh"
    assert result.confidence == 0.86
    assert "User utterance: I'm based in Edinburgh in the UK." in generator.prompts[0]


def test_structured_extractor_rejects_low_confidence():
    result = validate_structured_extraction(
        {
            "memory_class": "fact",
            "durable": True,
            "profile_key": "identity.location",
            "extracted_value": "Edinburgh",
            "confidence": 0.5,
        },
        route=location_route(),
        admission_threshold=0.75,
        allowed_profile_keys={"identity.location"},
        max_value_chars=160,
    )

    assert result.accepted is False
    assert result.rejection_reason == "below_confidence_threshold"


def test_structured_extractor_rejects_profile_key_mismatch():
    result = validate_structured_extraction(
        {
            "memory_class": "fact",
            "durable": True,
            "profile_key": "identity.origin",
            "extracted_value": "Scotland",
            "confidence": 0.9,
        },
        route=location_route(),
        admission_threshold=0.75,
        allowed_profile_keys={"identity.location", "identity.origin"},
        max_value_chars=160,
    )

    assert result.accepted is False
    assert result.rejection_reason == "profile_key_mismatch"


def test_structured_extractor_rejects_non_json_response():
    result = extract_structured_memory(
        "I'm based in Edinburgh in the UK.",
        location_route(),
        StubGenerator("I think the answer is Edinburgh."),
        admission_threshold=0.75,
        allowed_profile_keys={"identity.location"},
    )

    assert result.accepted is False
    assert result.rejection_reason == "No JSON object found in extractor response"
