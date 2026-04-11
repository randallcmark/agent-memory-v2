from __future__ import annotations

from pathlib import Path

from agent_memory_v2.eval_history import compare_summaries, compact_summary, load_history, write_history


def test_compact_summary_includes_scores(tmp_path: Path) -> None:
    summary = compact_summary(
        eval_kind="deterministic",
        stage_name="all",
        dataset_path="evals/baseline.json",
        result={
            "passed": True,
            "stages": [
                {"stage": "classification", "passed_cases": 2, "failed_cases": 0, "total_cases": 2},
                {"stage": "prompt", "passed_cases": 1, "failed_cases": 1, "total_cases": 2},
            ],
        },
        root_dir=tmp_path,
        runtime={"embedding_provider": "hash"},
    )
    assert summary["overall_score"] == 0.75
    assert summary["stage_scores"]["classification"] == 1.0
    assert summary["stage_scores"]["prompt"] == 0.5


def test_write_and_load_history_round_trip(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    summary = {
        "recorded_at": "2026-03-29T00:00:00+00:00",
        "eval_kind": "deterministic",
        "stage_name": "all",
        "passed": True,
        "overall_score": 1.0,
        "stage_scores": {"classification": 1.0},
        "stages": [],
    }
    path = write_history(history_root, summary)
    assert path.exists()
    loaded = load_history(history_root)
    assert len(loaded) == 1
    assert loaded[0]["overall_score"] == 1.0
    assert loaded[0]["_history_path"] == str(path)


def test_compare_summaries_reports_deltas() -> None:
    current = {"overall_score": 1.0, "stage_scores": {"classification": 1.0, "prompt": 0.5}}
    previous = {"overall_score": 0.75, "stage_scores": {"classification": 0.5, "prompt": 0.5}}
    comparison = compare_summaries(current, previous)
    assert comparison["overall_delta"] == 0.25
    assert comparison["stage_deltas"]["classification"]["delta"] == 0.5
    assert comparison["stage_deltas"]["prompt"]["delta"] == 0.0
