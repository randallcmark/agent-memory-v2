"""Snapshot management for experiment lifecycles.

A snapshot is the full ``data/`` tree of an isolated run root (memory index +
metadata, sidecar, profile, logs). We use a plain directory copy rather than the
zip-based ``state_cli`` because the experiment already works in per-run root dirs,
so a tree copy is the simplest faithful capture and restore.

Lifecycles built on these primitives:
- cold_start: no load (fresh empty root).
- seeded / specific: ``snapshot_load`` a prebuilt library entry before the run.
- compiled_resumed: ``snapshot_save`` after phase 1, ``snapshot_load`` into a fresh
  session/root for phase 2.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _data_dir(root: Path) -> Path:
    return Path(root) / "data"


def snapshot_save(root: Path, dest: Path) -> dict[str, Any]:
    """Copy the ``data/`` tree under ``root`` into ``dest`` (overwriting)."""
    src = _data_dir(root)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "source": str(src), "snapshot": str(dest), **snapshot_fingerprint(dest)}


def snapshot_load(snapshot: Path, root: Path) -> dict[str, Any]:
    """Restore a snapshot directory into ``root/data`` (overwriting)."""
    snapshot = Path(snapshot)
    if not snapshot.exists():
        raise FileNotFoundError(f"snapshot not found: {snapshot}")
    target = _data_dir(root)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot, target)
    return {
        "ok": True,
        "snapshot": str(snapshot),
        "root": str(root),
        **snapshot_fingerprint(target),
    }


def snapshot_fingerprint(path: Path) -> dict[str, Any]:
    """Lightweight content fingerprint: file count + size + hash of sorted names/sizes.

    Enough to detect drift and to stamp the manifest; not a full content hash.
    """
    path = Path(path)
    files: list[tuple[str, int]] = []
    if path.exists():
        for p in sorted(path.rglob("*")):
            if p.is_file():
                files.append((str(p.relative_to(path)), p.stat().st_size))
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {
        "file_count": len(files),
        "total_bytes": sum(size for _, size in files),
        "fingerprint": digest,
    }
