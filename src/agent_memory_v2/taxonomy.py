from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    pass

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "taxonomy.yaml"


@dataclass(frozen=True)
class TaxonomyKey:
    key: str
    tier1: str
    tier2: str
    mode: str          # scalar | additive | task
    cls: str           # fact | preference | task | context | ephemeral
    durable: bool
    description: str
    examples: tuple[str, ...]
    rule_patterns: tuple[str, ...]
    aliases: tuple[str, ...]
    taxonomy_version: int


@dataclass(frozen=True)
class Taxonomy:
    version: int
    keys: tuple[TaxonomyKey, ...]

    def get(self, key: str) -> TaxonomyKey | None:
        for k in self.keys:
            if k.key == key:
                return k
            if key in k.aliases:
                return k
        return None

    def durable_profile_keys(self) -> set[str]:
        return {k.key for k in self.keys if k.durable}

    def keys_by_mode(self, mode: str) -> list[TaxonomyKey]:
        return [k for k in self.keys if k.mode == mode]

    def to_fact_patterns(self) -> list[tuple[re.Pattern, str]]:
        result: list[tuple[re.Pattern, str]] = []
        for tk in self.keys:
            for raw in tk.rule_patterns:
                try:
                    result.append((re.compile(raw, re.IGNORECASE), tk.key))
                except re.error:
                    pass
        return result

    def to_prototypes(self) -> tuple:
        from agent_memory_v2.semantic_router import SemanticPrototype
        protos = []
        for tk in self.keys:
            if not tk.examples:
                continue
            protos.append(
                SemanticPrototype(
                    candidate_key=tk.key,
                    candidate_class=tk.cls,
                    description=tk.description,
                    durable_candidate=tk.durable,
                    examples=tk.examples,
                )
            )
        return tuple(protos)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    resolved = path or _DEFAULT_PATH
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    version = int(data.get("version", 1))
    keys: list[TaxonomyKey] = []
    for entry in data.get("keys", []):
        raw_key = str(entry["key"])
        parts = raw_key.split(".", 1)
        tier1 = entry.get("tier1", parts[0])
        tier2 = entry.get("tier2", parts[1] if len(parts) > 1 else "")
        keys.append(
            TaxonomyKey(
                key=raw_key,
                tier1=tier1,
                tier2=tier2,
                mode=str(entry.get("mode", "scalar")),
                cls=str(entry.get("class", "fact")),
                durable=bool(entry.get("durable", False)),
                description=str(entry.get("description", "")),
                examples=tuple(entry.get("examples") or []),
                rule_patterns=tuple(entry.get("rule_patterns") or []),
                aliases=tuple(entry.get("aliases") or []),
                taxonomy_version=version,
            )
        )
    return Taxonomy(version=version, keys=tuple(keys))


_cached: Taxonomy | None = None


def get_taxonomy(path: Path | None = None) -> Taxonomy:
    global _cached
    if path is None and _cached is not None:
        return _cached
    taxonomy = load_taxonomy(path)
    if path is None:
        _cached = taxonomy
    return taxonomy
