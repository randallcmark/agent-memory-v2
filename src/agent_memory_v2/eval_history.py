"""Eval history tracker: persists scores to JSONL and computes pass-rate trends over time."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def git_metadata(root_dir: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root_dir,
            text=True,
        ).strip()
    except Exception:
        sha = None
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root_dir,
            text=True,
        ).strip()
    except Exception:
        branch = None
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root_dir,
                text=True,
            ).strip()
        )
    except Exception:
        dirty = None
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": dirty,
    }


def stage_score(stage: dict[str, Any]) -> float:
    total = int(stage.get("total_cases", 0))
    if total <= 0:
        return 0.0
    return float(stage.get("passed_cases", 0)) / float(total)


def normalize_stages(result: dict[str, Any]) -> list[dict[str, Any]]:
    stages = result.get("stages")
    if isinstance(stages, list):
        return stages
    if "stage" in result and "total_cases" in result:
        return [result]
    return []


def overall_score(result: dict[str, Any]) -> float:
    stages = normalize_stages(result)
    if not stages:
        return 0.0
    return sum(stage_score(stage) for stage in stages) / float(len(stages))


def compact_summary(
    *,
    eval_kind: str,
    stage_name: str,
    dataset_path: str,
    result: dict[str, Any],
    root_dir: Path,
    runtime: dict[str, Any] | None = None,
    artifact_root: str | None = None,
) -> dict[str, Any]:
    stages = normalize_stages(result)
    return {
        "recorded_at": utc_now_iso(),
        "eval_kind": eval_kind,
        "stage_name": stage_name,
        "dataset_path": dataset_path,
        "passed": bool(result.get("passed", False)),
        "overall_score": overall_score(result),
        "stage_scores": {
            stage.get("stage", f"stage_{idx}"): stage_score(stage) for idx, stage in enumerate(stages)
        },
        "stages": [
            {
                "stage": stage.get("stage"),
                "passed": stage.get("passed"),
                "passed_cases": stage.get("passed_cases"),
                "failed_cases": stage.get("failed_cases"),
                "total_cases": stage.get("total_cases"),
                "failed_case_names": [
                    item.get("name") for item in stage.get("results", []) if not item.get("passed", False)
                ],
            }
            for stage in stages
        ],
        "artifact_root": artifact_root,
        "runtime": runtime or {},
        **git_metadata(root_dir),
    }


def write_history(history_root: Path, summary: dict[str, Any]) -> Path:
    history_root.mkdir(parents=True, exist_ok=True)
    target = history_root / f"{utc_now_stamp()}.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return target


def load_history(history_root: Path) -> list[dict[str, Any]]:
    if not history_root.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(history_root.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data["_history_path"] = str(path)
            items.append(data)
    return items


def compare_summaries(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    current_stages = current.get("stage_scores", {})
    previous_stages = (previous or {}).get("stage_scores", {})
    return {
        "current_overall_score": current.get("overall_score", 0.0),
        "previous_overall_score": (previous or {}).get("overall_score"),
        "overall_delta": None
        if previous is None
        else current.get("overall_score", 0.0) - previous.get("overall_score", 0.0),
        "stage_deltas": {
            stage: {
                "current": score,
                "previous": previous_stages.get(stage),
                "delta": None if stage not in previous_stages else score - previous_stages.get(stage, 0.0),
            }
            for stage, score in current_stages.items()
        },
        "previous_history_path": (previous or {}).get("_history_path"),
    }


def find_previous_comparable(history: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not history:
        return None, None
    current = history[-1]
    current_scores = current.get("stage_scores") or {}
    if not current_scores:
        return current, None
    for item in reversed(history[:-1]):
        if item.get("stage_scores"):
            return current, item
    return current, None
