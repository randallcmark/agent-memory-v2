from __future__ import annotations

from agent_memory_v2.config import load_config
from agent_memory_v2.eval_cli import (
    _load_dataset,
    eval_all,
    eval_classification,
    eval_profile,
    eval_prompt,
    eval_recall,
    eval_semantic,
    eval_sentiment,
)


def test_eval_dataset_loads() -> None:
    config = load_config()
    dataset = _load_dataset(config.root_dir / "evals/baseline.json")
    assert "classification" in dataset
    assert "semantic" in dataset
    assert "recall" in dataset
    assert "prompt" in dataset


def test_eval_classification_passes_baseline() -> None:
    config = load_config()
    dataset = _load_dataset(config.root_dir / "evals/baseline.json")
    result = eval_classification(dataset)
    assert result["passed"] is True
    assert result["failed_cases"] == 0


def test_eval_all_passes_baseline() -> None:
    config = load_config()
    dataset = _load_dataset(config.root_dir / "evals/baseline.json")
    assert eval_semantic(dataset)["passed"] is True
    assert eval_sentiment(dataset)["passed"] is True
    assert eval_profile(dataset, config)["passed"] is True
    assert eval_recall(dataset, config)["passed"] is True
    assert eval_prompt(dataset, config)["passed"] is True
    assert eval_all(dataset, config)["passed"] is True
