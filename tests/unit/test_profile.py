from agent_memory_v2.models import MemoryRecord
from agent_memory_v2.profile import build_profile


def make_record(*, memory_id: str, role: str, value: str, profile_key: str, timestamp: str):
    return MemoryRecord(
        memory_id=memory_id,
        role=role,
        text=value,
        summary=value,
        timestamp=timestamp,
        conversation_id="default",
        turn_id=memory_id,
        message_id=memory_id,
        metadata={
            "memory_class": role,
            "extracted_value": value,
            "profile_key": profile_key,
            "source_memory_id": memory_id,
        },
    )


def test_build_profile_uses_latest_value_for_same_key():
    older = make_record(
        memory_id="m1",
        role="preference",
        value="tea",
        profile_key="preference.general",
        timestamp="2026-03-29T09:00:00+00:00",
    )
    newer = make_record(
        memory_id="m2",
        role="preference",
        value="coffee",
        profile_key="preference.general",
        timestamp="2026-03-29T10:00:00+00:00",
    )

    profile = build_profile([older, newer])
    assert profile["preferences"]["preference.general"]["value"] == "coffee"


def test_build_profile_tracks_latest_fact_separately():
    older = make_record(
        memory_id="f1",
        role="fact",
        value="London",
        profile_key="identity.location",
        timestamp="2026-03-29T09:00:00+00:00",
    )
    newer = make_record(
        memory_id="f2",
        role="fact",
        value="Bristol",
        profile_key="identity.location",
        timestamp="2026-03-29T10:00:00+00:00",
    )

    profile = build_profile([older, newer])
    assert profile["facts"]["identity.location"]["value"] == "Bristol"


# ---------------------------------------------------------------------------
# Additive mode compaction
# ---------------------------------------------------------------------------


def test_additive_key_accumulates_all_values():
    first = make_record(
        memory_id="a1",
        role="fact",
        value="nuts",
        profile_key="identity.allergy",
        timestamp="2026-03-29T09:00:00+00:00",
    )
    second = make_record(
        memory_id="a2",
        role="fact",
        value="penicillin",
        profile_key="identity.allergy",
        timestamp="2026-03-29T10:00:00+00:00",
    )

    profile = build_profile([first, second])
    entry = profile["facts"]["identity.allergy"]
    assert entry["value"] == "penicillin"
    assert "all_values" in entry
    assert "nuts" in entry["all_values"]
    assert "penicillin" in entry["all_values"]


def test_additive_key_deduplicates_values():
    r1 = make_record(
        memory_id="d1",
        role="fact",
        value="vegetarian",
        profile_key="identity.dietary",
        timestamp="2026-03-29T09:00:00+00:00",
    )
    r2 = make_record(
        memory_id="d2",
        role="fact",
        value="vegetarian",
        profile_key="identity.dietary",
        timestamp="2026-03-29T10:00:00+00:00",
    )

    profile = build_profile([r1, r2])
    entry = profile["facts"]["identity.dietary"]
    assert entry["all_values"].count("vegetarian") == 1


def test_scalar_key_has_no_all_values():
    r = make_record(
        memory_id="s1",
        role="fact",
        value="Mark",
        profile_key="identity.name",
        timestamp="2026-03-29T09:00:00+00:00",
    )

    profile = build_profile([r])
    entry = profile["facts"]["identity.name"]
    assert entry["value"] == "Mark"
    assert "all_values" not in entry
