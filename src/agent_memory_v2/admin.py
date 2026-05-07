"""Admin CLI: prune, rebuild, backup, restore, seed, and inspect the live memory stores."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from agent_memory_v2.aging import (
    age_bucket,
    age_days,
    normalize_task_value,
    prune_dry_run_decision,
    resolved_task_values,
)
from agent_memory_v2.classifier import classify_text
from agent_memory_v2.config import AppConfig, load_config
from agent_memory_v2.embeddings import build_encoder
from agent_memory_v2.models import MemoryRecord
from agent_memory_v2.ollama import OllamaClient, OllamaProfile
from agent_memory_v2.profile import UserProfileStore
from agent_memory_v2.semantic_router import route_semantic_candidate
from agent_memory_v2.store import MemoryStore
from agent_memory_v2.structured_extractor import TextGenerator, extract_structured_memory


def _build_store(config: AppConfig) -> MemoryStore:
    return MemoryStore(
        index_path=config.resolve_path(config.memory["index_path"]),
        metadata_path=config.resolve_path(config.memory["metadata_path"]),
        embedding_dim=config.embedding_dim,
    )


def _build_sidecar_store(config: AppConfig) -> MemoryStore | None:
    if not config.sidecar.get("enabled", False):
        return None
    return MemoryStore(
        index_path=config.resolve_path(config.sidecar["index_path"]),
        metadata_path=config.resolve_path(config.sidecar["metadata_path"]),
        embedding_dim=config.embedding_dim,
    )


def _build_profile_store(config: AppConfig) -> UserProfileStore | None:
    if not config.profile.get("enabled", False):
        return None
    return UserProfileStore(config.resolve_path(config.profile["path"]))


def _classification_source_text(item: dict) -> str:
    summary = (item.get("summary") or "").strip()
    if summary:
        return summary

    text = (item.get("text") or "").strip()
    if item.get("role") == "turn":
        marker = "User:"
        agent_marker = "\nAgent:"
        if text.startswith(marker):
            body = text[len(marker) :]
            if agent_marker in body:
                body = body.split(agent_marker, 1)[0]
            return body.strip()
    return text


def _build_sidecar_record(record: MemoryRecord) -> MemoryRecord:
    memory_class = record.metadata.get("memory_class", "fact")
    extracted = record.metadata.get("extracted_value")
    text = str(extracted or record.summary or record.text).strip()
    return MemoryRecord(
        memory_id=f"{record.memory_id}:sidecar",
        role=memory_class,
        text=text,
        summary=record.summary,
        timestamp=record.timestamp,
        conversation_id=record.conversation_id,
        turn_id=record.turn_id,
        message_id=record.message_id,
        metadata={
            **record.metadata,
            "kind": "sidecar_memory",
            "source_memory_id": record.memory_id,
        },
    )


def _normalize_embedding_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _sidecar_vector_text(record: MemoryRecord) -> str:
    value = record.text or record.summary or ""
    memory_class = str(record.metadata.get("memory_class", record.role) or "").strip()
    profile_key = str(record.metadata.get("profile_key") or "").strip()
    parts: list[str] = [value]
    key_parts = [part for part in re.split(r"[._]+", profile_key) if part]
    tail_key = key_parts[-1] if key_parts else ""
    if tail_key and tail_key != "general":
        parts.append(tail_key)
    if memory_class == "preference":
        parts.append("prefer")
    elif memory_class == "task":
        parts.append("remind")
    return _normalize_embedding_text(" ".join(parts))


def _should_run_semantic_router(metadata: dict) -> bool:
    return not metadata.get("durable", False) and metadata.get("memory_class") in {"turn", "message"}


def _build_extractor_generator(config: AppConfig) -> OllamaClient:
    llm_cfg = config.llm
    return OllamaClient(
        OllamaProfile(
            host=llm_cfg["host"],
            model=llm_cfg["model"],
            temperature=0.0,
            max_tokens=int(llm_cfg.get("max_tokens", 400)),
            timeout_seconds=int(llm_cfg.get("timeout_seconds", 60)),
            raw_prompt=bool(llm_cfg.get("raw_prompt", False)),
        )
    )


def _reclassified_metadata(
    item: dict,
    config: AppConfig | None = None,
    *,
    encoder=None,
    extractor_generator: TextGenerator | None = None,
) -> dict:
    metadata = dict(item.get("metadata") or {})
    source_text = _classification_source_text(item)
    default_class = "turn" if item.get("role") == "turn" else item.get("role", "message")
    classification = classify_text(source_text, default_class=default_class)
    metadata["memory_class"] = classification.memory_class
    metadata["extracted_value"] = classification.extracted_value
    metadata["classification_confidence"] = classification.confidence
    metadata["durable"] = classification.durable
    metadata["durability_reason"] = classification.durability_reason
    metadata["profile_key"] = classification.profile_key
    metadata["classification_source"] = "rule"

    if config is None or encoder is None:
        return metadata
    router_cfg = config.semantic_router
    if not router_cfg.get("enabled", False) or not _should_run_semantic_router(metadata):
        return metadata
    route = route_semantic_candidate(
        source_text,
        encoder,
        threshold=float(router_cfg.get("threshold", 0.72)),
    )
    if route is None:
        return metadata
    if router_cfg.get("debug_metadata", True):
        metadata["semantic_candidate"] = route.to_metadata()

    extractor_cfg = config.structured_extractor
    if not extractor_cfg.get("enabled", False):
        return metadata
    if not route.above_threshold or not route.durable_candidate:
        return metadata

    generator = extractor_generator or _build_extractor_generator(config)
    extraction = extract_structured_memory(
        source_text,
        route,
        generator,
        admission_threshold=float(extractor_cfg.get("admission_threshold", 0.75)),
        allowed_profile_keys=set(extractor_cfg.get("allowed_profile_keys", [])),
        max_value_chars=int(extractor_cfg.get("max_value_chars", 160)),
    )
    metadata["structured_extraction"] = extraction.to_metadata()
    if not extraction.accepted:
        return metadata

    metadata["rule_classification"] = {
        "memory_class": classification.memory_class,
        "extracted_value": classification.extracted_value,
        "classification_confidence": classification.confidence,
        "durable": classification.durable,
        "durability_reason": classification.durability_reason,
        "profile_key": classification.profile_key,
    }
    promoted = extraction.to_classification_result()
    metadata["memory_class"] = promoted.memory_class
    metadata["extracted_value"] = promoted.extracted_value
    metadata["classification_confidence"] = promoted.confidence
    metadata["durable"] = promoted.durable
    metadata["durability_reason"] = promoted.durability_reason
    metadata["profile_key"] = promoted.profile_key
    metadata["classification_source"] = "structured_extractor"
    return metadata


def get_store_stats(config: AppConfig) -> dict:
    store = _build_store(config)
    sidecar_store = _build_sidecar_store(config)
    profile_store = _build_profile_store(config)
    records = store.records
    class_counts: dict[str, int] = {}
    durable_counts = {"durable": 0, "ephemeral": 0}
    for record in records:
        memory_class = record.metadata.get("memory_class", record.role)
        class_counts[memory_class] = class_counts.get(memory_class, 0) + 1
        if record.metadata.get("durable", False):
            durable_counts["durable"] += 1
        else:
            durable_counts["ephemeral"] += 1
    return {
        "count": len(records),
        "class_counts": class_counts,
        "durable_counts": durable_counts,
        "index_path": str(store.index_path),
        "metadata_path": str(store.metadata_path),
        "interaction_log_path": str(config.resolve_path(config.memory["interaction_log_path"])),
        "sidecar": {
            "enabled": bool(config.sidecar.get("enabled", False)),
            "count": len(sidecar_store.records) if sidecar_store is not None else 0,
            "index_path": str(sidecar_store.index_path) if sidecar_store is not None else None,
            "metadata_path": str(sidecar_store.metadata_path) if sidecar_store is not None else None,
        },
        "profile": {
            "enabled": bool(config.profile.get("enabled", False)),
            "path": str(profile_store.path) if profile_store is not None else None,
            "counts": (profile_store.load().get("counts") if profile_store is not None else {}),
        },
        "last_message_id": records[-1].message_id if records else None,
        "last_timestamp": records[-1].timestamp if records else None,
    }


def get_aging_report(config: AppConfig) -> dict:
    store = _build_store(config)
    sidecar_store = _build_sidecar_store(config)
    records = store.records
    bucket_counts: dict[str, int] = {}
    class_bucket_counts: dict[str, dict[str, int]] = {}
    now_dt = datetime.now(UTC)

    for record in records:
        days = age_days(record.timestamp, now_dt=now_dt)
        bucket = age_bucket(days)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        memory_class = record.metadata.get("memory_class", record.role)
        class_bucket_counts.setdefault(memory_class, {})
        class_bucket_counts[memory_class][bucket] = class_bucket_counts[memory_class].get(bucket, 0) + 1

    sidecar_bucket_counts: dict[str, int] = {}
    sidecar_groups: dict[str, int] = {}
    if sidecar_store is not None:
        for record in sidecar_store.records:
            bucket = age_bucket(age_days(record.timestamp, now_dt=now_dt))
            sidecar_bucket_counts[bucket] = sidecar_bucket_counts.get(bucket, 0) + 1
            group_key = f"{record.role}:{record.metadata.get('profile_key') or record.memory_id}"
            sidecar_groups[group_key] = sidecar_groups.get(group_key, 0) + 1

    return {
        "count": len(records),
        "bucket_counts": bucket_counts,
        "class_bucket_counts": class_bucket_counts,
        "sidecar_bucket_counts": sidecar_bucket_counts,
        "sidecar_superseded_groups": {key: count for key, count in sidecar_groups.items() if count > 1},
        "policy": config.aging,
    }


def _task_prune_decisions(records: list[MemoryRecord], config: AppConfig) -> dict[str, dict]:
    resolved_values = resolved_task_values(records)
    prune_cfg = config.aging.get("prune", {}) or {}
    task_ttl = float(prune_cfg.get("task_ttl_days", 90))
    resolved_task_ttl = float(prune_cfg.get("resolved_task_ttl_days", 7))
    now_dt = datetime.now(UTC)
    decisions: dict[str, dict] = {}

    for record in records:
        memory_class = record.metadata.get("memory_class", record.role)
        if memory_class != "task":
            continue
        age = age_days(record.timestamp, now_dt=now_dt)
        normalized = normalize_task_value(record.metadata.get("extracted_value") or record.summary or record.text)
        resolved_ts = resolved_values.get(normalized)
        if resolved_ts:
            resolved_age = age_days(resolved_ts, now_dt=now_dt)
            if resolved_age is not None and resolved_age >= resolved_task_ttl:
                decisions[record.memory_id] = {
                    "memory_id": record.memory_id,
                    "message_id": record.message_id,
                    "memory_class": memory_class,
                    "durable": bool(record.metadata.get("durable", False)),
                    "age_days": round(age, 3) if age is not None else None,
                    "age_bucket": age_bucket(age),
                    "decision": "prune",
                    "reason": "resolved_task",
                }
                continue
            decisions[record.memory_id] = {
                "memory_id": record.memory_id,
                "message_id": record.message_id,
                "memory_class": memory_class,
                "durable": bool(record.metadata.get("durable", False)),
                "age_days": round(age, 3) if age is not None else None,
                "age_bucket": age_bucket(age),
                "decision": "review",
                "reason": "recently_resolved_task",
            }
            continue
        if age is not None and age >= task_ttl:
            decisions[record.memory_id] = {
                "memory_id": record.memory_id,
                "message_id": record.message_id,
                "memory_class": memory_class,
                "durable": bool(record.metadata.get("durable", False)),
                "age_days": round(age, 3) if age is not None else None,
                "age_bucket": age_bucket(age),
                "decision": "prune",
                "reason": "expired_task",
            }
    return decisions


def _sidecar_prune_candidates(config: AppConfig) -> list[dict]:
    sidecar_store = _build_sidecar_store(config)
    if sidecar_store is None:
        return []
    grouped: dict[tuple[str, str], list[MemoryRecord]] = {}
    for record in sidecar_store.records:
        profile_key = record.metadata.get("profile_key")
        if not profile_key:
            continue
        grouped.setdefault((record.role, str(profile_key)), []).append(record)

    candidates: list[dict] = []
    for (role, profile_key), records in grouped.items():
        ordered = sorted(records, key=lambda item: (item.timestamp, item.memory_id), reverse=True)
        keep = ordered[0]
        for record in ordered[1:]:
            age = age_days(record.timestamp)
            candidates.append(
                {
                    "memory_id": record.memory_id,
                    "message_id": record.message_id,
                    "memory_class": role,
                    "durable": True,
                    "age_days": round(age, 3) if age is not None else None,
                    "age_bucket": age_bucket(age),
                    "decision": "prune",
                    "reason": "superseded_sidecar_entry",
                    "profile_key": profile_key,
                    "kept_memory_id": keep.memory_id,
                }
            )
    return candidates


def prune_sidecar_dry_run(config: AppConfig, *, limit: int | None = None) -> dict:
    candidates = _sidecar_prune_candidates(config)
    if limit is None:
        limit = int((config.aging.get("prune", {}) or {}).get("dry_run_limit", 100))
    return {
        "count": len(candidates),
        "summary": {"prune": len(candidates), "keep": 0, "review": 0},
        "reason_counts": {"superseded_sidecar_entry": len(candidates)} if candidates else {},
        "candidates": candidates[:limit],
        "policy": config.aging.get("sidecar", {}),
    }


def prune_dry_run(config: AppConfig, *, limit: int | None = None) -> dict:
    payload = _prune_decision_payload(config)
    if limit is None:
        limit = int((config.aging.get("prune", {}) or {}).get("dry_run_limit", 100))
    return {
        **payload,
        "candidates": payload["candidates"][:limit],
    }


def _prune_decision_payload(config: AppConfig) -> dict:
    store = _build_store(config)
    now_dt = datetime.now(UTC)
    base_decisions = [
        {
            **prune_dry_run_decision(record, config=config, now_dt=now_dt),
            "timestamp": record.timestamp,
            "summary": record.summary,
            "conversation_id": record.conversation_id,
        }
        for record in store.records
    ]
    task_overrides = _task_prune_decisions(store.records, config)
    decisions = []
    for item in base_decisions:
        override = task_overrides.get(item["memory_id"])
        decisions.append({**item, **override} if override else item)
    summary = {"keep": 0, "prune": 0, "review": 0}
    reason_counts: dict[str, int] = {}
    for item in decisions:
        summary[item["decision"]] = summary.get(item["decision"], 0) + 1
        reason = item["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    sorted_decisions = sorted(
        decisions,
        key=lambda item: (
            {"prune": 0, "review": 1, "keep": 2}.get(str(item["decision"]), 3),
            -(item["age_days"] or 0.0),
        ),
    )
    return {
        "count": len(decisions),
        "summary": summary,
        "reason_counts": reason_counts,
        "candidates": sorted_decisions,
        "policy": config.aging.get("prune", {}),
    }


def prune_sidecar_store(config: AppConfig) -> dict:
    sidecar_store = _build_sidecar_store(config)
    profile_store = _build_profile_store(config)
    if sidecar_store is None:
        return {"pruned": 0, "kept": 0, "reason_counts": {}, "sidecar_enabled": False}

    candidates = _sidecar_prune_candidates(config)
    prune_ids = {item["memory_id"] for item in candidates if item["decision"] == "prune"}
    if not prune_ids:
        if profile_store is not None:
            profile_store.rebuild_from_records(sidecar_store.records)
        return {
            "pruned": 0,
            "kept": len(sidecar_store.records),
            "reason_counts": {},
            "sidecar_enabled": True,
            "archive_path": str(
                config.resolve_path(
                    str((config.aging.get("sidecar", {}) or {}).get("archive_path", "data/archive/pruned_sidecar.jsonl"))
                )
            ),
        }

    kept_records = [record for record in sidecar_store.records if record.memory_id not in prune_ids]
    pruned_records = [record for record in sidecar_store.records if record.memory_id in prune_ids]
    archive_path = config.resolve_path(
        str((config.aging.get("sidecar", {}) or {}).get("archive_path", "data/archive/pruned_sidecar.jsonl"))
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("a", encoding="utf-8") as handle:
        for record in pruned_records:
            handle.write(json.dumps(record.__dict__) + "\n")

    encoder = build_encoder(
        provider=config.embeddings.get("provider", "hash"),
        model_name=config.embeddings.get("model", ""),
        dimensions=config.embedding_dim,
        host=config.embeddings.get("host") or config.llm.get("host"),
        timeout_seconds=int(
            config.embeddings.get("timeout_seconds", config.llm.get("timeout_seconds", 60))
        ),
    )
    sidecar_store.reset()
    for record in kept_records:
        sidecar_store.add(record, encoder.encode(_sidecar_vector_text(record)))
    if profile_store is not None:
        profile_store.rebuild_from_records(sidecar_store.records)

    return {
        "pruned": len(prune_ids),
        "kept": len(kept_records),
        "reason_counts": {"superseded_sidecar_entry": len(prune_ids)},
        "sidecar_enabled": True,
        "archive_path": str(archive_path),
    }


def prune_store(config: AppConfig) -> dict:
    store = _build_store(config)
    decisions = _prune_decision_payload(config)
    prune_ids = {
        item["memory_id"]
        for item in decisions["candidates"]
        if item["decision"] == "prune"
    }
    if not prune_ids:
        return {
            "pruned": 0,
            "kept": len(store.records),
            "reason_counts": decisions["reason_counts"],
            "index_path": str(store.index_path),
            "metadata_path": str(store.metadata_path),
        }

    kept_records = [record for record in store.records if record.memory_id not in prune_ids]
    pruned_records = [record for record in store.records if record.memory_id in prune_ids]
    encoder = build_encoder(
        provider=config.embeddings.get("provider", "hash"),
        model_name=config.embeddings.get("model", ""),
        dimensions=config.embedding_dim,
        host=config.embeddings.get("host") or config.llm.get("host"),
        timeout_seconds=int(
            config.embeddings.get("timeout_seconds", config.llm.get("timeout_seconds", 60))
        ),
    )

    archive_path = config.resolve_path(
        str((config.aging.get("prune", {}) or {}).get("archive_path", "data/archive/pruned_memories.jsonl"))
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("a", encoding="utf-8") as handle:
        for record in pruned_records:
            handle.write(json.dumps(record.__dict__) + "\n")

    store.reset()
    for record in kept_records:
        store.add(record, encoder.encode(record.summary or record.text))

    return {
        "pruned": len(prune_ids),
        "kept": len(kept_records),
        "reason_counts": decisions["reason_counts"],
        "archive_path": str(archive_path),
        "index_path": str(store.index_path),
        "metadata_path": str(store.metadata_path),
    }


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except Exception:
        return None


def _matches_filters(
    record: MemoryRecord,
    *,
    memory_class: str | None = None,
    query: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> bool:
    if memory_class and record.metadata.get("memory_class", record.role) != memory_class:
        return False

    if query:
        haystack = "\n".join(
            [
                record.text,
                record.summary,
                str(record.metadata.get("extracted_value") or ""),
            ]
        ).lower()
        if query.lower() not in haystack:
            return False

    record_ts = _parse_ts(record.timestamp)
    from_ts = _parse_ts(date_from) if date_from else None
    to_ts = _parse_ts(date_to) if date_to else None
    if from_ts is not None and (record_ts is None or record_ts < from_ts):
        return False
    return not (to_ts is not None and (record_ts is None or record_ts > to_ts))


def list_memories(
    config: AppConfig,
    *,
    limit: int = 10,
    memory_class: str | None = None,
    query: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    store = _build_store(config)
    filtered = [
        record
        for record in store.records
        if _matches_filters(
            record,
            memory_class=memory_class,
            query=query,
            date_from=date_from,
            date_to=date_to,
        )
    ]
    items = filtered[-limit:]
    payload = []
    for record in items:
        item = dict(record.__dict__)
        item["memory_class"] = record.metadata.get("memory_class", record.role)
        item["extracted_value"] = record.metadata.get("extracted_value")
        item["durable"] = bool(record.metadata.get("durable", False))
        item["durability_reason"] = record.metadata.get("durability_reason")
        payload.append(item)
    return payload


def list_sidecar_memories(config: AppConfig, *, limit: int = 10) -> list[dict]:
    sidecar_store = _build_sidecar_store(config)
    if sidecar_store is None:
        return []
    items = sidecar_store.records[-limit:]
    payload = []
    for record in items:
        item = dict(record.__dict__)
        item["memory_class"] = record.metadata.get("memory_class", record.role)
        item["extracted_value"] = record.metadata.get("extracted_value")
        item["durable"] = bool(record.metadata.get("durable", False))
        item["durability_reason"] = record.metadata.get("durability_reason")
        payload.append(item)
    return payload


def get_profile(config: AppConfig) -> dict:
    profile_store = _build_profile_store(config)
    if profile_store is None:
        return {}
    return profile_store.load()


def reset_store(config: AppConfig) -> dict:
    store = _build_store(config)
    sidecar_store = _build_sidecar_store(config)
    profile_store = _build_profile_store(config)
    store.reset()
    if sidecar_store is not None:
        sidecar_store.reset()
    if profile_store is not None:
        profile_store.reset()
    interaction_log = config.resolve_path(config.memory["interaction_log_path"])
    if interaction_log.exists():
        interaction_log.unlink()
    return {
        "reset": True,
        "index_path": str(store.index_path),
        "metadata_path": str(store.metadata_path),
        "interaction_log_path": str(interaction_log),
        "sidecar_index_path": str(sidecar_store.index_path) if sidecar_store is not None else None,
        "sidecar_metadata_path": str(sidecar_store.metadata_path) if sidecar_store is not None else None,
        "profile_path": str(profile_store.path) if profile_store is not None else None,
    }


def rebuild_store(config: AppConfig) -> dict:
    store = _build_store(config)
    sidecar_store = _build_sidecar_store(config)
    profile_store = _build_profile_store(config)
    store.reset()
    if sidecar_store is not None:
        sidecar_store.reset()
    interaction_log = config.resolve_path(config.memory["interaction_log_path"])
    encoder = build_encoder(
        provider=config.embeddings.get("provider", "hash"),
        model_name=config.embeddings.get("model", ""),
        dimensions=config.embedding_dim,
        host=config.embeddings.get("host") or config.llm.get("host"),
        timeout_seconds=int(
            config.embeddings.get("timeout_seconds", config.llm.get("timeout_seconds", 60))
        ),
    )
    extractor_generator = None
    if config.structured_extractor.get("enabled", False):
        extractor_generator = _build_extractor_generator(config)

    rebuilt = 0
    sidecar_rebuilt = 0
    if interaction_log.exists():
        with interaction_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                summary = item.get("summary") or item.get("text") or ""
                if not summary:
                    continue
                record = MemoryRecord(
                    memory_id=item["message_id"],
                    role=item["role"],
                    text=item["text"],
                    summary=summary,
                    timestamp=item["timestamp"],
                    conversation_id=item["conversation_id"],
                    turn_id=item["turn_id"],
                    message_id=item["message_id"],
                    metadata=_reclassified_metadata(
                        item,
                        config,
                        encoder=encoder,
                        extractor_generator=extractor_generator,
                    ),
                )
                vector = encoder.encode(summary)
                store.add(record, vector)
                if sidecar_store is not None:
                    allowed = set(config.sidecar.get("store_classes", ["preference", "fact"]))
                    if record.metadata.get("durable", False) and record.metadata.get("memory_class") in allowed:
                        sidecar_record = _build_sidecar_record(record)
                        sidecar_store.add(
                            sidecar_record,
                            encoder.encode(_sidecar_vector_text(sidecar_record)),
                        )
                        sidecar_rebuilt += 1
                rebuilt += 1
    if profile_store is not None and sidecar_store is not None:
        profile_store.rebuild_from_records(sidecar_store.records)
    return {
        "rebuilt": rebuilt,
        "sidecar_rebuilt": sidecar_rebuilt,
        "interaction_log_path": str(interaction_log),
        "index_path": str(store.index_path),
        "metadata_path": str(store.metadata_path),
        "sidecar_index_path": str(sidecar_store.index_path) if sidecar_store is not None else None,
        "sidecar_metadata_path": str(sidecar_store.metadata_path) if sidecar_store is not None else None,
        "profile_path": str(profile_store.path) if profile_store is not None else None,
    }


def rebuild_profile(config: AppConfig) -> dict:
    profile_store = _build_profile_store(config)
    sidecar_store = _build_sidecar_store(config)
    if profile_store is None:
        return {"rebuilt_profile": False, "reason": "profile_disabled"}
    records = sidecar_store.records if sidecar_store is not None else []
    profile = profile_store.rebuild_from_records(records)
    return {
        "rebuilt_profile": True,
        "profile_path": str(profile_store.path),
        "counts": profile.get("counts", {}),
    }


def _read_archive_records(archive_path: Path) -> list[dict]:
    if not archive_path.exists():
        return []
    records: list[dict] = []
    with archive_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                with contextlib.suppress(json.JSONDecodeError):
                    records.append(json.loads(stripped))
    return records


def inspect_archive(config: AppConfig) -> dict:
    prune_cfg = config.aging.get("prune", {}) or {}
    main_path = config.resolve_path(
        str(prune_cfg.get("archive_path", "data/archive/pruned_memories.jsonl"))
    )
    sidecar_cfg = config.aging.get("sidecar", {}) or {}
    sidecar_path = config.resolve_path(
        str(sidecar_cfg.get("archive_path", "data/archive/pruned_sidecar.jsonl"))
    )

    def _summarise(records: list[dict]) -> dict:
        if not records:
            return {"count": 0}
        class_counts: dict[str, int] = {}
        key_counts: dict[str, int] = {}
        timestamps = []
        for r in records:
            mc = (r.get("metadata") or {}).get("memory_class", r.get("role", "unknown"))
            class_counts[mc] = class_counts.get(mc, 0) + 1
            pk = (r.get("metadata") or {}).get("profile_key")
            if pk:
                key_counts[pk] = key_counts.get(pk, 0) + 1
            ts = r.get("timestamp")
            if ts:
                timestamps.append(ts)
        timestamps.sort()
        return {
            "count": len(records),
            "memory_class_breakdown": class_counts,
            "top_profile_keys": dict(sorted(key_counts.items(), key=lambda kv: -kv[1])[:10]),
            "oldest": timestamps[0] if timestamps else None,
            "newest": timestamps[-1] if timestamps else None,
        }

    return {
        "main_archive": {"path": str(main_path), **_summarise(_read_archive_records(main_path))},
        "sidecar_archive": {"path": str(sidecar_path), **_summarise(_read_archive_records(sidecar_path))},
    }


def restore_from_archive(
    config: AppConfig,
    *,
    memory_id: str | None = None,
    since: str | None = None,
    memory_class: str | None = None,
    sidecar: bool = False,
) -> dict:
    from agent_memory_v2.embeddings import build_encoder
    from agent_memory_v2.models import MemoryRecord

    prune_cfg = config.aging.get("prune", {}) or {}
    sidecar_cfg = config.aging.get("sidecar", {}) or {}
    if sidecar:
        archive_path = config.resolve_path(
            str(sidecar_cfg.get("archive_path", "data/archive/pruned_sidecar.jsonl"))
        )
        store = _build_sidecar_store(config)
    else:
        archive_path = config.resolve_path(
            str(prune_cfg.get("archive_path", "data/archive/pruned_memories.jsonl"))
        )
        store = _build_store(config)

    if store is None:
        return {"error": "store not available", "restored": 0, "skipped_duplicate": 0, "skipped_filter": 0}

    encoder = build_encoder(
        provider=config.embeddings.get("provider", "hash"),
        model_name=config.embeddings.get("model", ""),
        dimensions=config.embedding_dim,
        host=config.embeddings.get("host") or config.llm.get("host"),
        timeout_seconds=int(
            config.embeddings.get("timeout_seconds", config.llm.get("timeout_seconds", 60))
        ),
    )

    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            return {"error": f"Invalid --since date: {since}", "restored": 0, "skipped_duplicate": 0, "skipped_filter": 0}

    existing_ids = {r.memory_id for r in store.records}
    raw_records = _read_archive_records(archive_path)

    restored = 0
    skipped_dup = 0
    skipped_filter = 0

    for raw in raw_records:
        record = MemoryRecord(**raw)

        if memory_id and record.memory_id != memory_id:
            skipped_filter += 1
            continue
        if memory_class and (raw.get("metadata") or {}).get("memory_class", record.role) != memory_class:
            skipped_filter += 1
            continue
        if since_dt:
            ts = record.timestamp
            try:
                ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if ts_dt < since_dt:
                    skipped_filter += 1
                    continue
            except (ValueError, AttributeError):
                skipped_filter += 1
                continue

        if record.memory_id in existing_ids:
            skipped_dup += 1
            continue

        embed_text = record.summary or record.text
        vector = encoder.encode(embed_text)
        store.add(record, vector)
        existing_ids.add(record.memory_id)
        restored += 1

    return {
        "restored": restored,
        "skipped_duplicate": skipped_dup,
        "skipped_filter": skipped_filter,
        "archive_path": str(archive_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Admin tooling for agent_memory_v2.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser("stats")
    stats.add_argument("--json", action="store_true")

    listing = subparsers.add_parser("list")
    listing.add_argument("--limit", type=int, default=10)
    listing.add_argument("--memory-class")
    listing.add_argument("--query")
    listing.add_argument("--date-from")
    listing.add_argument("--date-to")
    listing.add_argument("--json", action="store_true")

    list_sidecar = subparsers.add_parser("list-sidecar")
    list_sidecar.add_argument("--limit", type=int, default=10)
    list_sidecar.add_argument("--json", action="store_true")

    profile = subparsers.add_parser("profile")
    profile.add_argument("--json", action="store_true")

    aging_report = subparsers.add_parser("aging-report")
    aging_report.add_argument("--json", action="store_true")

    prune_dry = subparsers.add_parser("prune-dry-run")
    prune_dry.add_argument("--limit", type=int, default=None)
    prune_dry.add_argument("--json", action="store_true")

    prune = subparsers.add_parser("prune")
    prune.add_argument("--force", action="store_true")
    prune.add_argument("--json", action="store_true")

    reset = subparsers.add_parser("reset")
    reset.add_argument("--force", action="store_true")

    rebuild = subparsers.add_parser("rebuild")
    rebuild.add_argument("--force", action="store_true")

    rebuild_profile_cmd = subparsers.add_parser("rebuild-profile")
    rebuild_profile_cmd.add_argument("--force", action="store_true")

    inspect_archive_cmd = subparsers.add_parser("inspect-archive")
    inspect_archive_cmd.add_argument("--json", action="store_true")

    restore_cmd = subparsers.add_parser("restore-from-archive")
    restore_cmd.add_argument("--memory-id")
    restore_cmd.add_argument("--since")
    restore_cmd.add_argument("--memory-class")
    restore_cmd.add_argument("--sidecar", action="store_true")
    restore_cmd.add_argument("--force", action="store_true")
    restore_cmd.add_argument("--json", action="store_true")

    check_schema_cmd = subparsers.add_parser("check-schema")
    check_schema_cmd.add_argument("--json", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config()

    if args.command == "stats":
        payload = get_store_stats(config)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "list":
        payload = list_memories(
            config,
            limit=args.limit,
            memory_class=args.memory_class,
            query=args.query,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "list-sidecar":
        payload = list_sidecar_memories(config, limit=args.limit)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "profile":
        print(json.dumps(get_profile(config), indent=2))
        return 0

    if args.command == "aging-report":
        print(json.dumps(get_aging_report(config), indent=2))
        return 0

    if args.command == "prune-dry-run":
        print(json.dumps(prune_dry_run(config, limit=args.limit), indent=2))
        return 0

    if args.command == "prune":
        if not args.force:
            print("Refusing to prune without --force")
            return 2
        print(json.dumps(prune_store(config), indent=2))
        return 0

    if args.command == "reset":
        if not args.force:
            print("Refusing to reset without --force")
            return 2
        print(json.dumps(reset_store(config), indent=2))
        return 0

    if args.command == "rebuild":
        if not args.force:
            print("Refusing to rebuild without --force")
            return 2
        print(json.dumps(rebuild_store(config), indent=2))
        return 0

    if args.command == "rebuild-profile":
        if not args.force:
            print("Refusing to rebuild profile without --force")
            return 2
        print(json.dumps(rebuild_profile(config), indent=2))
        return 0

    if args.command == "inspect-archive":
        print(json.dumps(inspect_archive(config), indent=2))
        return 0

    if args.command == "restore-from-archive":
        if not args.force:
            print("Refusing to restore without --force")
            return 2
        result = restore_from_archive(
            config,
            memory_id=args.memory_id,
            since=args.since,
            memory_class=args.memory_class,
            sidecar=args.sidecar,
        )
        print(json.dumps(result, indent=2))
        return 0 if "error" not in result else 1

    if args.command == "check-schema":
        from agent_memory_v2.models import SCHEMA_VERSION
        store = _build_store(config)
        stale = [r.memory_id for r in store.records if r.metadata.get("schema_version", 0) != SCHEMA_VERSION]
        payload = {
            "current_schema_version": SCHEMA_VERSION,
            "total_records": len(store.records),
            "stale_records": len(stale),
            "stale_ids": stale[:20],
        }
        print(json.dumps(payload, indent=2))
        return 1 if stale else 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
