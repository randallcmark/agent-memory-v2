from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_memory_v2.admin import (
    prune_dry_run,
    prune_sidecar_dry_run,
    prune_sidecar_store,
    prune_store,
    rebuild_profile,
)
from agent_memory_v2.config import AppConfig, load_config


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _state_path(config: AppConfig) -> Path:
    return config.resolve_path(config.maintenance.get("state_path", "data/maintenance/state.json"))


def _lock_path(config: AppConfig) -> Path:
    return config.resolve_path(config.maintenance.get("lock_path", "data/maintenance/lock"))


def default_state() -> dict:
    return {
        "last_run_at": None,
        "last_run_status": None,
        "last_run_summary": {},
        "new_records_since_run": 0,
        "maintenance_due": False,
        "due_reasons": [],
        "last_marked_at": None,
    }


def load_state(config: AppConfig) -> dict:
    path = _state_path(config)
    if not path.exists():
        return default_state()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return default_state()
    return {**default_state(), **data}


def save_state(config: AppConfig, state: dict) -> dict:
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return state


def evaluate_due(config: AppConfig, state: dict) -> dict:
    if not config.maintenance.get("enabled", True):
        state["maintenance_due"] = False
        state["due_reasons"] = []
        return state

    reasons: list[str] = []
    now_dt = _utc_now()
    min_interval = int(config.maintenance.get("min_interval_minutes", 30))
    max_new_records = int(config.maintenance.get("max_new_records", 10))
    last_run = _parse_ts(state.get("last_run_at"))

    if last_run is None:
        reasons.append("never_run")
    elif now_dt - last_run >= timedelta(minutes=min_interval):
        reasons.append("interval_elapsed")

    if int(state.get("new_records_since_run", 0)) >= max_new_records:
        reasons.append("new_record_threshold")

    dry_run = prune_dry_run(config, limit=1)
    if dry_run.get("summary", {}).get("prune", 0) > 0:
        reasons.append("prune_candidates")

    if bool(config.aging.get("sidecar", {}).get("enabled", True)):
        sidecar_preview = prune_sidecar_dry_run(config, limit=1)
        if sidecar_preview.get("summary", {}).get("prune", 0) > 0:
            reasons.append("sidecar_prune_candidates")

    state["maintenance_due"] = bool(reasons)
    state["due_reasons"] = reasons
    state["last_marked_at"] = _utc_now_iso()
    return state


def mark_interaction(config: AppConfig, *, count: int = 1) -> dict:
    state = load_state(config)
    state["new_records_since_run"] = int(state.get("new_records_since_run", 0)) + count
    return save_state(config, evaluate_due(config, state))


@contextmanager
def maintenance_lock(config: AppConfig):
    path = _lock_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"maintenance already running: {path}") from exc
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def maintenance_status(config: AppConfig) -> dict:
    state = load_state(config)
    state = evaluate_due(config, state)
    save_state(config, state)
    state["lock_path"] = str(_lock_path(config))
    state["state_path"] = str(_state_path(config))
    state["enabled"] = bool(config.maintenance.get("enabled", True))
    return state


def run_maintenance(config: AppConfig) -> dict:
    state = load_state(config)
    state = evaluate_due(config, state)
    if not config.maintenance.get("enabled", True):
        save_state(config, state)
        return {"ok": True, "skipped": True, "reason": "disabled", "state": state}

    with maintenance_lock(config):
        actions: dict[str, dict] = {}
        if bool(config.maintenance.get("run_prune", True)):
            actions["prune"] = prune_store(config)
        if bool(config.maintenance.get("run_sidecar_prune", True)):
            actions["prune_sidecar"] = prune_sidecar_store(config)
        if bool(config.maintenance.get("run_profile_rebuild", True)):
            actions["rebuild_profile"] = rebuild_profile(config)

        state["last_run_at"] = _utc_now_iso()
        state["last_run_status"] = "ok"
        state["last_run_summary"] = actions
        state["new_records_since_run"] = 0
        state["maintenance_due"] = False
        state["due_reasons"] = []
        save_state(config, state)
        return {
            "ok": True,
            "skipped": False,
            "actions": actions,
            "state": state,
            "lock_path": str(_lock_path(config)),
            "state_path": str(_state_path(config)),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deferred maintenance runner for agent_memory_v2.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("mark")
    subparsers.add_parser("run")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()
    if args.command == "status":
        print(json.dumps(maintenance_status(config), indent=2))
        return 0
    if args.command == "mark":
        print(json.dumps(mark_interaction(config), indent=2))
        return 0
    if args.command == "run":
        try:
            print(json.dumps(run_maintenance(config), indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 2
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
