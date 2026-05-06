from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memory_v2 import agent_eval_cli
from agent_memory_v2.config import AppConfig, load_config


def _config_root(tmp_path: Path) -> AppConfig:
    root = tmp_path / "repo"
    (root / "evals").mkdir(parents=True)
    (root / "evals/scenarios.json").write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "name": "preference_recall",
                        "description": "Simple durable preference recall.",
                        "setup_turns": [{"user": "I prefer oat milk.", "agent": "Noted."}],
                        "query": "What do I prefer?",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    base = load_config()
    return AppConfig(root_dir=root, settings_path=root / "settings.yaml", raw=base.raw)


def test_history_with_no_history_returns_empty_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(agent_eval_cli, "load_config", lambda: _config_root(tmp_path))
    monkeypatch.setattr("sys.argv", ["agent-eval", "history"])

    with pytest.raises(SystemExit) as exc:
        agent_eval_cli.main()

    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out)["history"] == []


def test_run_with_fake_provider_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(agent_eval_cli, "load_config", lambda: _config_root(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-eval",
            "run",
            "--scenario",
            "preference_recall",
            "--provider",
            "fake",
            "--save-all",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        agent_eval_cli.main()

    assert exc.value.code == 0
    output = json.loads(capsys.readouterr().out)
    result = output["stages"][0]["results"][0]
    artifact_path = Path(result["artifact_path"])
    assert artifact_path.exists()
    assert (artifact_path / "result.json").exists()


def test_compare_requires_two_history_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(agent_eval_cli, "load_config", lambda: _config_root(tmp_path))
    monkeypatch.setattr("sys.argv", ["agent-eval", "compare"])

    with pytest.raises(SystemExit) as exc:
        agent_eval_cli.main()

    assert exc.value.code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["comparison"] is None
    assert output["reason"] == "need_at_least_two_history_entries"
