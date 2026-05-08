"""AppConfig dataclass and load_config(): wraps settings.yaml with typed property accessors."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    settings_path: Path
    raw: dict

    @property
    def llm(self) -> dict:
        return self.raw["llm"]

    @property
    def embeddings(self) -> dict:
        return self.raw["embeddings"]

    @property
    def memory(self) -> dict:
        return self.raw["memory"]

    @property
    def sidecar(self) -> dict:
        return self.raw.get("sidecar", {})

    @property
    def semantic_router(self) -> dict:
        return self.raw.get("semantic_router", {})

    @property
    def structured_extractor(self) -> dict:
        return self.raw.get("structured_extractor", {})

    @property
    def profile(self) -> dict:
        return self.raw.get("profile", {})

    @property
    def aging(self) -> dict:
        return self.raw.get("aging", {})

    @property
    def maintenance(self) -> dict:
        return self.raw.get("maintenance", {})

    @property
    def user(self) -> dict:
        return self.raw.get("user", {})

    @property
    def current_user(self) -> str:
        selected = (
            os.environ.get("AGENT_MEMORY_V2_USER") or self.user.get("current_profile") or "catchall"
        )
        normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(selected).strip()).strip("-._")
        return normalized or "catchall"

    @property
    def embedding_dim(self) -> int:
        embeddings_dim = self.embeddings.get("dimensions")
        if embeddings_dim is not None:
            return int(embeddings_dim)
        return int(self.memory["embedding_dim"])

    @property
    def prompting(self) -> dict:
        return self.raw["prompting"]

    def resolve_path(self, value: str) -> Path:
        rendered = str(value).format(user_id=self.current_user)
        if (
            "{user_id}" not in str(value)
            and self.current_user != "catchall"
            and rendered.startswith("data/")
        ):
            rendered = f"data/users/{self.current_user}/{rendered.removeprefix('data/')}"
        return (self.root_dir / rendered).resolve()


def load_config(settings_path: str | Path | None = None) -> AppConfig:
    package_dir = Path(__file__).resolve().parents[2]
    resolved_settings = (
        Path(settings_path) if settings_path else package_dir / "config/settings.yaml"
    )
    with resolved_settings.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise RuntimeError("settings.yaml must load to a dictionary")
    return AppConfig(
        root_dir=package_dir,
        settings_path=resolved_settings.resolve(),
        raw=raw,
    )
