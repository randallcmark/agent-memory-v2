from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from agent_memory_v2.config import load_config


def _state_paths():
    config = load_config()
    paths = {
        "index_path": config.resolve_path(config.memory["index_path"]),
        "metadata_path": config.resolve_path(config.memory["metadata_path"]),
        "interaction_log_path": config.resolve_path(config.memory["interaction_log_path"]),
        "settings_path": config.settings_path,
    }
    if config.sidecar.get("enabled", False):
        paths["sidecar_index_path"] = config.resolve_path(config.sidecar["index_path"])
        paths["sidecar_metadata_path"] = config.resolve_path(config.sidecar["metadata_path"])
    if config.profile.get("enabled", False):
        paths["profile_path"] = config.resolve_path(config.profile["path"])
    if config.maintenance.get("enabled", False):
        paths["maintenance_state_path"] = config.resolve_path(
            config.maintenance.get("state_path", "data/maintenance/state.json")
        )
    return config, paths


def export_state(output_path: Path) -> dict:
    config, paths = _state_paths()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "app": config.raw.get("app", {}).get("name", "agent_memory_v2"),
            "embedding_dim": config.embedding_dim,
            "files": {},
        }
        for key, path in paths.items():
            if path.exists():
                arcname = path.name
                zf.write(path, arcname=arcname)
                manifest["files"][key] = arcname
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return {
        "ok": True,
        "archive_path": str(output_path),
        "exported_files": sorted(manifest["files"].keys()),
    }


def import_state(input_path: Path) -> dict:
    config, paths = _state_paths()
    with zipfile.ZipFile(input_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        for key, arcname in (manifest.get("files") or {}).items():
            target = paths[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(arcname, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return {
        "ok": True,
        "archive_path": str(input_path),
        "imported_files": sorted((manifest.get("files") or {}).keys()),
        "embedding_dim": config.embedding_dim,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export or import agent memory state.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--output", required=True)

    imp = subparsers.add_parser("import")
    imp.add_argument("--input", required=True)
    imp.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "export":
        print(json.dumps(export_state(Path(args.output)), indent=2))
        return 0

    if args.command == "import":
        if not args.force:
            print(json.dumps({"ok": False, "error": "Refusing to import without --force"}, indent=2))
            return 2
        print(json.dumps(import_state(Path(args.input)), indent=2))
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
