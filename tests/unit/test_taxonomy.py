from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory_v2.taxonomy import Taxonomy, TaxonomyKey, get_taxonomy, load_taxonomy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy()


# ---------------------------------------------------------------------------
# Load and structure
# ---------------------------------------------------------------------------


def test_taxonomy_loads_without_error(taxonomy: Taxonomy) -> None:
    assert isinstance(taxonomy, Taxonomy)


def test_taxonomy_version_is_positive_integer(taxonomy: Taxonomy) -> None:
    assert isinstance(taxonomy.version, int)
    assert taxonomy.version >= 1


def test_taxonomy_has_keys(taxonomy: Taxonomy) -> None:
    assert len(taxonomy.keys) > 0


def test_all_keys_are_taxonomy_key_instances(taxonomy: Taxonomy) -> None:
    for key in taxonomy.keys:
        assert isinstance(key, TaxonomyKey)


def test_all_keys_have_non_empty_key_field(taxonomy: Taxonomy) -> None:
    for tk in taxonomy.keys:
        assert tk.key, f"Empty key field found: {tk}"


def test_all_keys_have_dot_separated_key(taxonomy: Taxonomy) -> None:
    for tk in taxonomy.keys:
        assert "." in tk.key, f"Key '{tk.key}' is missing tier1.tier2 structure"


def test_tier1_tier2_derived_from_key(taxonomy: Taxonomy) -> None:
    for tk in taxonomy.keys:
        parts = tk.key.split(".", 1)
        assert tk.tier1 == parts[0], f"{tk.key}: tier1 mismatch"
        assert tk.tier2 == parts[1], f"{tk.key}: tier2 mismatch"


def test_all_keys_have_valid_mode(taxonomy: Taxonomy) -> None:
    valid = {"scalar", "additive", "task"}
    for tk in taxonomy.keys:
        assert tk.mode in valid, f"Key '{tk.key}' has unknown mode '{tk.mode}'"


def test_all_keys_have_valid_class(taxonomy: Taxonomy) -> None:
    valid = {"fact", "preference", "task", "context", "ephemeral"}
    for tk in taxonomy.keys:
        assert tk.cls in valid, f"Key '{tk.key}' has unknown class '{tk.cls}'"


def test_all_keys_have_description(taxonomy: Taxonomy) -> None:
    for tk in taxonomy.keys:
        assert tk.description.strip(), f"Key '{tk.key}' has no description"


def test_durable_keys_have_examples(taxonomy: Taxonomy) -> None:
    for tk in taxonomy.keys:
        if tk.durable:
            assert len(tk.examples) > 0, f"Durable key '{tk.key}' has no examples"


def test_taxonomy_version_stamped_on_every_key(taxonomy: Taxonomy) -> None:
    for tk in taxonomy.keys:
        assert tk.taxonomy_version == taxonomy.version


# ---------------------------------------------------------------------------
# Specific key presence
# ---------------------------------------------------------------------------


EXPECTED_KEYS = [
    "identity.name",
    "identity.location",
    "identity.occupation",
    "identity.employer",
    "identity.origin",
    "identity.birthday",
    "identity.dietary",
    "identity.allergy",
    "identity.health",
    "identity.relationship",
    "preference.general",
    "preference.communication",
    "preference.schedule",
    "task.general",
    "contextual.world_fact",
    "ephemeral.question",
]


@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_expected_key_present(taxonomy: Taxonomy, key: str) -> None:
    assert taxonomy.get(key) is not None, f"Expected key '{key}' not found in taxonomy"


# ---------------------------------------------------------------------------
# Mode semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected_mode",
    [
        ("identity.name", "scalar"),
        ("identity.location", "scalar"),
        ("identity.occupation", "scalar"),
        ("identity.dietary", "additive"),
        ("identity.allergy", "additive"),
        ("identity.health", "additive"),
        ("identity.relationship", "additive"),
        ("task.general", "task"),
    ],
)
def test_key_mode(taxonomy: Taxonomy, key: str, expected_mode: str) -> None:
    tk = taxonomy.get(key)
    assert tk is not None
    assert tk.mode == expected_mode, f"Key '{key}': expected mode '{expected_mode}', got '{tk.mode}'"


# ---------------------------------------------------------------------------
# durable_profile_keys
# ---------------------------------------------------------------------------


def test_durable_profile_keys_contains_expected(taxonomy: Taxonomy) -> None:
    dpk = taxonomy.durable_profile_keys()
    assert "identity.location" in dpk
    assert "task.general" in dpk


def test_durable_profile_keys_excludes_non_durable(taxonomy: Taxonomy) -> None:
    dpk = taxonomy.durable_profile_keys()
    assert "contextual.world_fact" not in dpk
    assert "ephemeral.question" not in dpk


# ---------------------------------------------------------------------------
# to_fact_patterns
# ---------------------------------------------------------------------------


def test_to_fact_patterns_returns_compiled_patterns(taxonomy: Taxonomy) -> None:
    patterns = taxonomy.to_fact_patterns()
    assert len(patterns) > 0
    for compiled, key in patterns:
        assert hasattr(compiled, "search"), "Expected compiled re.Pattern"
        assert isinstance(key, str)
        assert "." in key


def test_fact_patterns_cover_known_keys(taxonomy: Taxonomy) -> None:
    pattern_keys = {key for _, key in taxonomy.to_fact_patterns()}
    assert "identity.name" in pattern_keys
    assert "identity.location" in pattern_keys
    assert "identity.origin" in pattern_keys


def test_fact_patterns_match_expected_text(taxonomy: Taxonomy) -> None:
    patterns = {key: pat for pat, key in taxonomy.to_fact_patterns()}
    m = patterns["identity.location"].search("I live in Edinburgh.")
    assert m is not None
    assert m.group("value").strip() == "Edinburgh"

    m2 = patterns["identity.name"].search("My name is Mark.")
    assert m2 is not None
    assert m2.group("value").strip() == "Mark"


# ---------------------------------------------------------------------------
# to_prototypes
# ---------------------------------------------------------------------------


def test_to_prototypes_covers_durable_keys(taxonomy: Taxonomy) -> None:
    from agent_memory_v2.semantic_router import SemanticPrototype
    protos = taxonomy.to_prototypes()
    proto_keys = {p.candidate_key for p in protos}
    for tk in taxonomy.keys:
        if tk.durable and tk.examples:
            assert tk.key in proto_keys, f"Durable key '{tk.key}' missing from prototypes"


def test_to_prototypes_excludes_keys_without_examples(taxonomy: Taxonomy) -> None:
    from agent_memory_v2.semantic_router import SemanticPrototype
    protos = taxonomy.to_prototypes()
    proto_keys = {p.candidate_key for p in protos}
    for tk in taxonomy.keys:
        if not tk.examples:
            assert tk.key not in proto_keys, f"Key '{tk.key}' has no examples but appears in prototypes"


# ---------------------------------------------------------------------------
# get_taxonomy caching
# ---------------------------------------------------------------------------


def test_get_taxonomy_returns_same_object_on_repeated_calls() -> None:
    a = get_taxonomy()
    b = get_taxonomy()
    assert a is b, "get_taxonomy() should return the cached instance"


def test_get_taxonomy_explicit_path_bypasses_cache(tmp_path: Path) -> None:
    import shutil
    import agent_memory_v2.taxonomy as _tax_mod
    src = Path(__file__).resolve().parents[2] / "config" / "taxonomy.yaml"
    dst = tmp_path / "taxonomy.yaml"
    shutil.copy(src, dst)
    result = get_taxonomy(dst)
    assert isinstance(result, Taxonomy)
    # Explicit path call must not return the module-level cached singleton
    assert result is not _tax_mod._cached


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def test_alias_lookup_returns_canonical_key(taxonomy: Taxonomy) -> None:
    # Find a key that has aliases defined, or construct a minimal test taxonomy
    for tk in taxonomy.keys:
        if tk.aliases:
            for alias in tk.aliases:
                found = taxonomy.get(alias)
                assert found is tk, f"Alias '{alias}' did not resolve to canonical key '{tk.key}'"
            break  # one key with aliases is sufficient
