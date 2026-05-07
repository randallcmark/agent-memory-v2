"""CLI for inspecting and patching values in config/settings.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agent_memory_v2.config import load_config


def _settings_path() -> Path:
    return load_config().settings_path


def _load_raw(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("settings.yaml must load to a dictionary")
    return data


def _save_raw(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _set_embedding_dimensions(raw: dict, dimensions: int) -> None:
    raw.setdefault("embeddings", {})
    raw.setdefault("memory", {})
    raw["embeddings"]["dimensions"] = int(dimensions)
    raw["memory"]["embedding_dim"] = int(dimensions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Config helpers for agent_memory_v2.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    use_hash = subparsers.add_parser("use-hash-embeddings")
    use_hash.add_argument("--dimensions", type=int, default=384)

    use_ollama = subparsers.add_parser("use-ollama-embeddings")
    use_ollama.add_argument("--host", default="http://localhost:11434")
    use_ollama.add_argument("--model", default="nomic-embed-text")
    use_ollama.add_argument("--timeout-seconds", type=int, default=60)
    use_ollama.add_argument("--dimensions", type=int, default=768)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = _settings_path()
    raw = _load_raw(path)

    if args.command == "use-hash-embeddings":
        raw.setdefault("embeddings", {})
        raw["embeddings"]["provider"] = "hash"
        _set_embedding_dimensions(raw, int(args.dimensions))
        _save_raw(path, raw)
        print(f"Updated embeddings provider to hash in {path}")
        return 0

    if args.command == "use-ollama-embeddings":
        raw.setdefault("embeddings", {})
        raw["embeddings"]["provider"] = "ollama"
        raw["embeddings"]["host"] = args.host
        raw["embeddings"]["model"] = args.model
        raw["embeddings"]["timeout_seconds"] = int(args.timeout_seconds)
        _set_embedding_dimensions(raw, int(args.dimensions))
        _save_raw(path, raw)
        print(f"Updated embeddings provider to ollama in {path}")
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
