from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from agent_memory_v2.config import load_config


def _publish_safe_paths(config) -> list[Path]:
    return [
        config.resolve_path("data"),
        config.resolve_path("backups"),
        config.resolve_path(".pytest_cache"),
    ]


def sanitise_repo(*, apply_changes: bool) -> dict:
    config = load_config()
    removed: list[str] = []
    for path in _publish_safe_paths(config):
        if not path.exists():
            continue
        removed.append(str(path))
        if apply_changes:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return {
        "ok": True,
        "apply": apply_changes,
        "removed": removed,
        "note": "Runtime state removed. Re-seed with generic data before publishing if desired.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanitise the repo for publication by removing runtime state.")
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(sanitise_repo(apply_changes=args.apply), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
