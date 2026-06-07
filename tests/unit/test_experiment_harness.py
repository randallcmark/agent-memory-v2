"""CI-safe tests for the experiment harness machinery.

These exercise the controller, snapshots, scenarios, journaling, and runner with
hash embeddings + the offline FakeGenerator so they need neither Ollama nor an API
key. (The real experiment uses Ollama/nomic by decision; that path is not tested
here to keep CI deterministic, matching the repo rule that evals must not use Ollama.)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent_memory_v2.config import AppConfig, load_config
from agent_memory_v2.experiment.controller import (
    MEMORY_BLOCK_HEADING,
    MemoryController,
    Session,
)
from agent_memory_v2.experiment.generators import FakeGenerator
from agent_memory_v2.experiment.journal import Journal, build_manifest
from agent_memory_v2.experiment.runner import RunSpec, run_scenario
from agent_memory_v2.experiment.scenarios import _parse_scenario
from agent_memory_v2.experiment.snapshots import (
    snapshot_fingerprint,
    snapshot_load,
    snapshot_save,
)


@pytest.fixture
def hash_base_config() -> AppConfig:
    """A base config forced to hash embeddings for deterministic, Ollama-free tests."""
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["embeddings"]["provider"] = "hash"
    raw["embeddings"]["model"] = "hash"
    raw["embeddings"]["dimensions"] = 128
    raw["memory"]["embedding_dim"] = 128
    return AppConfig(root_dir=base.root_dir, settings_path=base.settings_path, raw=raw)


def test_arm_c_has_no_memory(hash_base_config, tmp_path):
    ctrl = MemoryController(
        base_config=hash_base_config, root_dir=tmp_path / "c", generator=FakeGenerator(), arm="C"
    )
    session = Session()
    result = ctrl.run_turn(session, "My name is Mark.")
    assert result.injected_context is None
    assert result.recalled_items == []
    assert ctrl.memory_state()["main_count"] == 0
    assert ctrl.pipeline is None


def test_arm_a_ingests_and_injects(hash_base_config, tmp_path):
    ctrl = MemoryController(
        base_config=hash_base_config, root_dir=tmp_path / "a", generator=FakeGenerator(), arm="A"
    )
    session = Session()
    ctrl.run_turn(session, "I prefer oat milk.")
    state = ctrl.memory_state()
    assert state["main_count"] >= 1
    # A later turn should surface the stored preference as a neutral injected block.
    result = ctrl.run_turn(session, "What do I prefer to drink?")
    assert result.injected_context is not None
    assert MEMORY_BLOCK_HEADING in result.injected_context
    # The block must NOT re-characterise the assistant (helper-not-personality).
    assert "You are" not in result.injected_context
    assert "Response tuning" not in result.injected_context


def test_invalid_arm_rejected(hash_base_config, tmp_path):
    with pytest.raises(ValueError):
        MemoryController(
            base_config=hash_base_config, root_dir=tmp_path, generator=FakeGenerator(), arm="B"
        )


def test_snapshot_save_load_roundtrip(hash_base_config, tmp_path):
    ctrl = MemoryController(
        base_config=hash_base_config, root_dir=tmp_path / "src", generator=FakeGenerator(), arm="A"
    )
    ctrl.run_turn(Session(), "My name is Mark.")
    snap = tmp_path / "snap"
    save_info = snapshot_save(tmp_path / "src", snap)
    assert save_info["ok"] and save_info["file_count"] >= 1

    dest_root = tmp_path / "dest"
    snapshot_load(snap, dest_root)
    # A fresh controller over the restored root should see the prior memory.
    restored = MemoryController(
        base_config=hash_base_config, root_dir=dest_root, generator=FakeGenerator(), arm="A"
    )
    assert restored.memory_state()["main_count"] >= 1
    assert snapshot_fingerprint(snap)["fingerprint"] == save_info["fingerprint"]


def test_scenario_parse_and_probe_lookup():
    scenario = _parse_scenario(
        {
            "name": "demo",
            "description": "d",
            "phases": [{"session": "s1", "turns": ["a", "b"], "snapshot_after": True}],
            "probes": [{"phase": "s1", "turn": 1, "expected_contains": ["x"]}],
            "ground_truth_memories": [{"key": "identity.name", "value": "Mark"}],
        }
    )
    assert scenario.phases[0].snapshot_after is True
    probe = scenario.probe_for("s1", 1)
    assert probe is not None and probe.expected_contains == ["x"]
    assert scenario.probe_for("s1", 0) is None


def test_manifest_records_injection_framing(hash_base_config, tmp_path):
    ctrl = MemoryController(
        base_config=hash_base_config, root_dir=tmp_path / "a", generator=FakeGenerator(), arm="A"
    )
    manifest = build_manifest(
        exp_id="t",
        arm="A",
        scenario="demo",
        lifecycle="cold_start",
        iteration=0,
        config=ctrl.config,
        generator_name="fake",
        model="fake-generator",
        temperature=0.0,
        snapshot=None,
        snapshot_fingerprint=None,
        repo_root=hash_base_config.root_dir,
    )
    # Section 5 audit trail must be present and verifiable.
    assert MEMORY_BLOCK_HEADING in manifest["injection_framing"]
    assert manifest["embeddings"]["provider"] == "hash"
    assert "base_system_prompt" in manifest


def test_journal_writes_one_record_per_turn(hash_base_config, tmp_path):
    ctrl = MemoryController(
        base_config=hash_base_config, root_dir=tmp_path / "a", generator=FakeGenerator(), arm="A"
    )
    journal = Journal(tmp_path / "run")
    manifest = build_manifest(
        exp_id="t",
        arm="A",
        scenario="demo",
        lifecycle="cold_start",
        iteration=0,
        config=ctrl.config,
        generator_name="fake",
        model="fake-generator",
        temperature=0.0,
        snapshot=None,
        snapshot_fingerprint=None,
        repo_root=hash_base_config.root_dir,
    )
    journal.write_manifest(manifest)
    session = Session()
    for text in ("My name is Mark.", "What is my name?"):
        result = ctrl.run_turn(session, text)
        journal.record_turn(
            manifest=manifest, session_id=session.session_id, phase="s1", result=result
        )
    lines = (tmp_path / "run" / "journal.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert {
        "user_text",
        "response",
        "usage",
        "memory_state_after",
        "scenario_meta",
        "messages_sent",
        "final_prompt",
        "persisted_memory",
    } <= rec.keys()
    assert rec["persisted_memory"]["metadata"]["kind"] == "turn_memory"


def test_runner_compiled_resumed_carries_memory_across_sessions(hash_base_config, tmp_path):
    scenario = _parse_scenario(
        {
            "name": "resume_demo",
            "description": "d",
            "phases": [
                {"session": "s1", "turns": ["My name is Mark."], "snapshot_after": True},
                {"session": "s2", "turns": ["What is my name?"]},
            ],
            "probes": [],
            "ground_truth_memories": [],
        }
    )
    spec = RunSpec(
        exp_id="t",
        arm="A",
        lifecycle="compiled_resumed",
        iteration=0,
        generator_name="fake",
        model=None,
        temperature=0.0,
        run_dir=tmp_path / "run",
        snapshot_library=tmp_path / "lib",
    )
    summary = run_scenario(base_config=hash_base_config, scenario=scenario, spec=spec)
    assert summary["turns_recorded"] == 2
    records = [json.loads(line) for line in Path(summary["journal_path"]).read_text().splitlines()]
    s2 = [r for r in records if r["phase"] == "s2"][0]
    # The fresh s2 session must have memory available from the resumed snapshot.
    assert s2["memory_state_after"]["main_count"] >= 1
