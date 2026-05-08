from __future__ import annotations

from agent_memory_v2.live_eval_cli import contains_all, contains_any, contains_none


def test_contains_all_reports_missing_items() -> None:
    ok, missing = contains_all("You said you prefer oat milk.", ["oat milk", "Mark"])
    assert ok is False
    assert missing == ["Mark"]


def test_contains_none_reports_forbidden_items() -> None:
    ok, present = contains_none("You live in Bristol.", ["London", "Bristol"])
    assert ok is False
    assert present == ["Bristol"]


def test_contains_any_matches_partial_marker() -> None:
    ok, matched = contains_any("That sounds frustrating. You prefer oat milk.", ["frustr", "sorry"])
    assert ok is True
    assert matched == ["frustr"]
