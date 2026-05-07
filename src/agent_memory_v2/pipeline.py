"""Central MemoryPipeline: orchestrates ingest, recall, prompt construction, and the chat loop."""

from __future__ import annotations

__all__ = [
    "MemoryPipeline",
    "summarize_for_memory",
    "summarize_turn_for_memory",
    "should_run_semantic_router",
    "clean_memory_text",
    "resolve_timezone",
    "relative_time_label",
    "recency_bonus",
    "format_recalled_item",
    "apply_char_budget",
    "build_temporal_context",
    "run_ollama_preflight",
]

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from agent_memory_v2.aging import age_penalty, effective_age_days, parse_timestamp
from agent_memory_v2.classifier import (
    ClassificationResult,
    class_priority,
    classify_text,
    durability_bonus,
)
from agent_memory_v2.config import AppConfig, load_config
from agent_memory_v2.embeddings import EmbeddingEncoder, build_encoder
from agent_memory_v2.maintenance import maintenance_status, mark_interaction, run_maintenance
from agent_memory_v2.models import SCHEMA_VERSION, MemoryRecord, Message, RecallResult
from agent_memory_v2.ollama import OllamaClient, OllamaProfile
from agent_memory_v2.profile import UserProfileStore
from agent_memory_v2.semantic_router import SemanticRouteResult, route_semantic_candidate
from agent_memory_v2.sentiment import SentimentResult, detect_sentiment
from agent_memory_v2.store import MemoryStore
from agent_memory_v2.structured_extractor import extract_structured_memory


def summarize_for_memory(message: Message) -> str:
    return message.text.strip()


def summarize_turn_for_memory(user_message: Message, agent_message: Message) -> str:
    del agent_message
    return user_message.text.strip()


def _classification_metadata(result: ClassificationResult) -> dict:
    return {
        "memory_class": result.memory_class,
        "extracted_value": result.extracted_value,
        "classification_confidence": result.confidence,
        "durable": result.durable,
        "durability_reason": result.durability_reason,
        "profile_key": result.profile_key,
    }


def should_run_semantic_router(result: ClassificationResult) -> bool:
    return not result.durable and result.memory_class in {"turn", "message"}


def _should_store_in_sidecar(config: AppConfig, metadata: dict) -> bool:
    sidecar_cfg = config.sidecar
    if not sidecar_cfg.get("enabled", False):
        return False
    if not metadata.get("durable", False):
        return False
    allowed = set(sidecar_cfg.get("store_classes", ["preference", "fact"]))
    return metadata.get("memory_class") in allowed


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


def _normalize_embedding_text(text: str) -> str:  # internal — used only within pipeline
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


def clean_memory_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"Agent:\s*Your response:\s*", "Agent: ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?im)^\s*Your response:\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*Response:\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def resolve_timezone(config: AppConfig, profile: dict | None = None):
    if profile is not None:
        tz_pref = (
            profile.get("preferences", {})
            .get("preference.timezone", {})
            .get("value")
        )
        if tz_pref:
            try:
                return ZoneInfo(str(tz_pref))
            except (ValueError, KeyError):
                pass
    tz_name = config.raw.get("app", {}).get("timezone")
    if tz_name:
        try:
            return ZoneInfo(tz_name)
        except (ValueError, KeyError):
            pass
    return datetime.now().astimezone().tzinfo



def relative_time_label(past_dt: datetime, now_dt: datetime, tz) -> str:
    try:
        past_local = past_dt.astimezone(tz) if tz else past_dt
        now_local = now_dt.astimezone(tz) if tz else now_dt
    except (TypeError, OverflowError, OSError):
        past_local = past_dt
        now_local = now_dt

    delta = now_local - past_local
    minutes = delta.total_seconds() / 60.0
    day_delta = (now_local.date() - past_local.date()).days

    if minutes <= 10:
        return "just now"
    if past_local.date() == now_local.date():
        return "earlier today"
    if day_delta == 1:
        return "yesterday"
    if 2 <= day_delta <= 6:
        return "recently"
    if 7 <= day_delta <= 13:
        return "around a week ago"
    if 14 <= day_delta <= 20:
        return "2 weeks ago"
    if 21 <= day_delta <= 27:
        return "3 weeks ago"

    months = (now_local.year - past_local.year) * 12 + (now_local.month - past_local.month)
    if months <= 1:
        return "a month ago"
    if months < 12:
        return f"{months} months ago"
    years = months // 12
    return f"{years} years ago"


def recency_bonus(ts: str | None, now_dt: datetime | None = None) -> float:
    parsed = parse_timestamp(ts)
    if parsed is None:
        return 0.0

    now_dt = now_dt or datetime.now(UTC)
    delta_seconds = max((now_dt - parsed).total_seconds(), 0.0)
    if delta_seconds <= 6 * 60 * 60:
        return 0.02
    if delta_seconds <= 24 * 60 * 60:
        return 0.015
    if delta_seconds <= 7 * 24 * 60 * 60:
        return 0.01
    if delta_seconds <= 30 * 24 * 60 * 60:
        return 0.005
    return 0.0


def _store_kind_bonus(kind: str | None) -> float:
    if kind == "sidecar_memory":
        return 0.025
    return 0.0


def _age_penalty_for_record(record: MemoryRecord, config: AppConfig) -> float:
    return age_penalty(
        memory_class=record.metadata.get("memory_class", record.role),
        durable=bool(record.metadata.get("durable", False)),
        age_days_value=effective_age_days(record),
        config=config,
    )


def format_recalled_item(item: dict, config: AppConfig, profile: dict | None = None) -> str:
    role = item["role"]
    score = item["score"]
    text = clean_memory_text(item["text"])
    tz = resolve_timezone(config, profile)
    now_dt = datetime.now(UTC)

    suffix_parts: list[str] = [f"score={score:.3f}"]
    parsed = parse_timestamp(item.get("timestamp"))
    if parsed is not None:
        relative = relative_time_label(parsed, now_dt, tz)
        absolute = parsed.astimezone(tz).strftime("%A, %d %b %Y, %H:%M (%Z)")
        suffix_parts.append(f"relative={relative}")
        suffix_parts.append(f"at={absolute}")

    return f"- [{role}] {text} ({', '.join(suffix_parts)})"


def _partition_recalled_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    factual: list[dict] = []
    contextual: list[dict] = []
    for item in items:
        if item.get("store_kind") == "sidecar_memory":
            factual.append(item)
        else:
            contextual.append(item)
    return factual, contextual


def _normalize_profile_value(value: object) -> str:
    return str(value or "").strip().lower()


def _profile_value_map(profile: dict) -> dict[str, str]:
    values: dict[str, str] = {}
    for section_name in ("preferences", "facts", "tasks"):
        section = profile.get(section_name) or {}
        for key, entry in section.items():
            values[str(key)] = _normalize_profile_value((entry or {}).get("value"))
    return values


def _dedupe_recalled_items(items: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict] = []
    for item in items:
        dedupe_key = (
            str(item.get("store_kind", "")),
            str(item.get("source_message_id", item.get("message_id", ""))),
            clean_memory_text(str(item.get("text", ""))).lower(),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result.append(item)
    return result


def apply_char_budget(
    factual: list[dict], contextual: list[dict], budget: int
) -> tuple[list[dict], list[dict], bool]:
    """Drop trailing items once accumulated text exceeds budget.

    Always includes the first factual item regardless of size so callers
    always receive something useful even when a single memory is verbose.
    """
    if budget <= 0:
        return factual, contextual, False
    used = 0
    out_factual: list[dict] = []
    for i, item in enumerate(factual):
        chars = len(str(item.get("text", "")))
        if i > 0 and used + chars > budget:
            return out_factual, [], True
        used += chars
        out_factual.append(item)
    out_contextual: list[dict] = []
    for item in contextual:
        chars = len(str(item.get("text", "")))
        if used + chars > budget:
            return out_factual, out_contextual, True
        used += chars
        out_contextual.append(item)
    return out_factual, out_contextual, False


def _contextual_prompt_filter(items: list[dict], *, allow_empty: bool = False) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    dropped: list[dict] = []
    for item in items:
        if bool(item.get("durable", False)):
            kept.append(item)
            continue
        dropped.append(item)
    if kept:
        return kept, dropped
    if allow_empty:
        return [], dropped
    return items, []


def _should_replace_recalled_item(existing: dict, candidate: dict) -> bool:
    existing_kind = str(existing.get("store_kind", ""))
    candidate_kind = str(candidate.get("store_kind", ""))
    return candidate_kind == "sidecar_memory" and existing_kind != "sidecar_memory"


def build_temporal_context(config: AppConfig, profile: dict | None = None) -> str:
    prompting = config.prompting
    tz = resolve_timezone(config, profile)
    now_local = datetime.now(tz)
    lines: list[str] = []

    if prompting.get("inject_now", True):
        absolute_now = now_local.strftime("%A, %d %b %Y, %H:%M (%Z)")
        iso_now = now_local.isoformat()
        lines.append(f"As of now: {absolute_now}.")
        lines.append(f"Absolute timestamp: {iso_now}.")

    temporal_cfg = prompting.get("temporal_context", {}) or {}
    if temporal_cfg.get("enabled", True):
        today = now_local.strftime("%A, %d %b %Y")
        yesterday = (now_local - timedelta(days=1)).strftime("%A, %d %b %Y")
        tomorrow = (now_local + timedelta(days=1)).strftime("%A, %d %b %Y")
        lines.append(
            f"Day frame of reference: today is {today}; yesterday was {yesterday}; tomorrow is {tomorrow}."
        )

    return "\n".join(lines)


def _format_profile(profile: dict) -> str:
    lines: list[str] = []
    for section_name, label in (("preferences", "Preferences"), ("facts", "Facts"), ("tasks", "Tasks")):
        items = profile.get(section_name) or {}
        if not items:
            continue
        lines.append(f"{label}:")
        for key, entry in items.items():
            value = entry.get("value")
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _format_sentiment(sentiment: SentimentResult) -> str:
    cues = ", ".join(sentiment.cues) if sentiment.cues else "none"
    return (
        f"Detected user sentiment: {sentiment.label} "
        f"(confidence={sentiment.confidence:.2f}, cues={cues}).\n"
        f"Response tuning: {sentiment.guidance}"
    )


class MemoryPipeline:
    def __init__(
        self,
        config: AppConfig,
        *,
        encoder: EmbeddingEncoder | None = None,
        store: MemoryStore | None = None,
        sidecar_store: MemoryStore | None = None,
        profile_store: UserProfileStore | None = None,
        ollama: OllamaClient | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder or build_encoder(
            provider=config.embeddings.get("provider", "hash"),
            model_name=config.embeddings.get("model", ""),
            dimensions=config.embedding_dim,
            host=config.embeddings.get("host") or config.llm.get("host"),
            timeout_seconds=int(
                config.embeddings.get("timeout_seconds", config.llm.get("timeout_seconds", 60))
            ),
        )
        self.store = store or MemoryStore(
            index_path=config.resolve_path(config.memory["index_path"]),
            metadata_path=config.resolve_path(config.memory["metadata_path"]),
            embedding_dim=config.embedding_dim,
        )
        self.sidecar_store = None
        if config.sidecar.get("enabled", False):
            self.sidecar_store = sidecar_store or MemoryStore(
                index_path=config.resolve_path(config.sidecar["index_path"]),
                metadata_path=config.resolve_path(config.sidecar["metadata_path"]),
                embedding_dim=config.embedding_dim,
            )
        self.profile_store = None
        if config.profile.get("enabled", False):
            self.profile_store = profile_store or UserProfileStore(
                config.resolve_path(config.profile["path"])
            )
        self.interaction_log_path = config.resolve_path(config.memory["interaction_log_path"])
        llm_cfg = config.llm
        self.ollama = ollama or OllamaClient(
            OllamaProfile(
                host=llm_cfg["host"],
                model=llm_cfg["model"],
                temperature=float(llm_cfg["temperature"]),
                max_tokens=int(llm_cfg["max_tokens"]),
                timeout_seconds=int(llm_cfg["timeout_seconds"]),
                raw_prompt=bool(llm_cfg.get("raw_prompt", False)),
            )
        )

    def ingest(self, message: Message) -> MemoryRecord:
        summary = summarize_for_memory(message)
        classification = classify_text(message.text, default_class="message")
        return self._store_memory(
            role=message.role,
            text=message.text,
            summary=summary,
            timestamp=message.timestamp,
            conversation_id=message.conversation_id,
            turn_id=message.turn_id,
            message_id=message.message_id,
            metadata={
                "kind": "message_memory",
                "user_id": self.config.current_user,
                **self._memory_classification_metadata(message.text, classification),
            },
        )

    def ingest_turn(self, user_message: Message, agent_message: Message) -> MemoryRecord:
        summary = summarize_turn_for_memory(user_message, agent_message)
        text = f"User: {user_message.text.strip()}\nAgent: {agent_message.text.strip()}"
        classification = classify_text(user_message.text, default_class="turn")
        return self._store_memory(
            role="turn",
            text=text,
            summary=summary,
            embed_text=text,
            timestamp=agent_message.timestamp,
            conversation_id=user_message.conversation_id,
            turn_id=user_message.turn_id,
            message_id=user_message.message_id,
            metadata={
                "kind": "turn_memory",
                "user_message_id": user_message.message_id,
                "agent_message_id": agent_message.message_id,
                "user_id": self.config.current_user,
                **self._memory_classification_metadata(user_message.text, classification),
            },
        )

    def _semantic_candidate(
        self,
        text: str,
        classification: ClassificationResult,
    ) -> SemanticRouteResult | None:
        router_cfg = self.config.semantic_router
        if not router_cfg.get("enabled", False):
            return None
        if not should_run_semantic_router(classification):
            return None
        return route_semantic_candidate(
            text,
            self.encoder,
            threshold=float(router_cfg.get("threshold", 0.72)),
        )

    def _structured_extraction_metadata(
        self,
        text: str,
        route: SemanticRouteResult | None,
    ) -> dict:
        extractor_cfg = self.config.structured_extractor
        if not extractor_cfg.get("enabled", False):
            return {}
        if route is None or not route.above_threshold or not route.durable_candidate:
            return {}
        allowed_profile_keys = set(
            extractor_cfg.get(
                "allowed_profile_keys",
                [
                    "identity.location",
                    "identity.name",
                    "identity.occupation",
                    "identity.origin",
                    "identity.dietary",
                    "identity.health",
                    "identity.relationship",
                    "identity.birthday",
                    "identity.employer",
                    "identity.allergy",
                    "preference.general",
                    "preference.communication",
                    "preference.schedule",
                    "task.general",
                ],
            )
        )
        try:
            extraction = extract_structured_memory(
                text,
                route,
                self.ollama,
                admission_threshold=float(extractor_cfg.get("admission_threshold", 0.75)),
                allowed_profile_keys=allowed_profile_keys,
                max_value_chars=int(extractor_cfg.get("max_value_chars", 160)),
            )
        except Exception:
            return {}
        metadata = {"structured_extraction": extraction.to_metadata()}
        if extraction.accepted:
            metadata.update(_classification_metadata(extraction.to_classification_result()))
            metadata["classification_source"] = "structured_extractor"
        return metadata

    def _memory_classification_metadata(self, text: str, classification: ClassificationResult) -> dict:
        metadata = _classification_metadata(classification)
        metadata["classification_source"] = "rule"
        route = self._semantic_candidate(text, classification)
        if route is not None and self.config.semantic_router.get("debug_metadata", True):
            metadata["semantic_candidate"] = route.to_metadata()
        extraction_metadata = self._structured_extraction_metadata(text, route)
        if extraction_metadata.get("classification_source") == "structured_extractor":
            metadata["rule_classification"] = _classification_metadata(classification)
        metadata.update(extraction_metadata)
        return metadata

    @staticmethod
    def _taxonomy_version() -> int:
        try:
            from agent_memory_v2.taxonomy import get_taxonomy
            return get_taxonomy().version
        except (ImportError, AttributeError):
            return 0

    def _store_memory(
        self,
        *,
        role: str,
        text: str,
        summary: str,
        timestamp: str,
        conversation_id: str,
        turn_id: str,
        message_id: str,
        metadata: dict,
        embed_text: str | None = None,
    ) -> MemoryRecord:
        vector = self.encoder.encode(embed_text if embed_text is not None else summary)
        metadata = {**metadata, "schema_version": SCHEMA_VERSION, "taxonomy_version": self._taxonomy_version()}
        record = MemoryRecord(
            memory_id=message_id,
            role=role,
            text=text,
            summary=summary,
            timestamp=timestamp,
            conversation_id=conversation_id,
            turn_id=turn_id,
            message_id=message_id,
            metadata=metadata,
        )
        self.store.add(record, vector)
        if self.sidecar_store is not None and _should_store_in_sidecar(self.config, metadata):
            sidecar_record = _build_sidecar_record(record)
            sidecar_vector = self.encoder.encode(_sidecar_vector_text(sidecar_record))
            self.sidecar_store.add(sidecar_record, sidecar_vector)
            if self.profile_store is not None:
                self.profile_store.update_from_record(sidecar_record)
        self._append_log(
            {
                "role": role,
                "text": text,
                "summary": summary,
                "timestamp": timestamp,
                "message_id": message_id,
                "turn_id": turn_id,
                "conversation_id": conversation_id,
                "metadata": metadata,
            }
        )
        if self.config.maintenance.get("enabled", True):
            mark_interaction(self.config)
        return record

    def recall(self, message: Message) -> list[dict]:
        query_route = self._query_intent_route(message.text)
        vector = self.encoder.encode(message.text)
        results = self.store.search(
            vector,
            top_k=int(self.config.memory["top_k"]) + 1,
            similarity_threshold=float(self.config.memory["similarity_threshold"]),
            exclude_message_id=message.message_id,
        )
        if self.sidecar_store is not None:
            for query_text in self._sidecar_query_texts(message.text, query_route):
                results.extend(
                    self.sidecar_store.search(
                        self.encoder.encode(query_text),
                        top_k=int(self.config.sidecar.get("top_k", self.config.memory["top_k"])),
                        similarity_threshold=float(
                            self.config.sidecar.get(
                                "similarity_threshold", self.config.memory["similarity_threshold"]
                            )
                        ),
                        exclude_message_id=message.message_id,
                    )
                )
        ranked = sorted(results, key=lambda item: self._rank_key(item, query_route), reverse=True)
        deduped: list[dict] = []
        recalled_items: list[RecallResult] = []
        seen_positions: dict[str, int] = {}
        seen_sources: set[str] = set()
        for item in ranked:
            payload = self._recall_payload(item)
            source_id = payload["source_message_id"]
            if source_id in seen_sources:
                position = seen_positions[source_id]
                if _should_replace_recalled_item(deduped[position], payload):
                    deduped[position] = payload
                    recalled_items[position] = item
                continue
            if len(deduped) >= int(self.config.memory["top_k"]):
                break
            seen_positions[source_id] = len(deduped)
            seen_sources.add(source_id)
            deduped.append(payload)
            recalled_items.append(item)
        self._track_recall(recalled_items)
        return deduped

    def _track_recall(self, items: list[RecallResult]) -> None:
        now_iso = datetime.now(UTC).isoformat()
        for item in items:
            record = item.record
            recall_count = int(record.metadata.get("recall_count", 0)) + 1
            updates = {"last_recalled_at": now_iso, "recall_count": recall_count}
            store = (
                self.sidecar_store
                if record.metadata.get("kind") == "sidecar_memory"
                else self.store
            )
            store.update_record_metadata(record.memory_id, updates)

    def _query_intent_route(self, text: str) -> SemanticRouteResult | None:
        router_cfg = self.config.semantic_router
        if not router_cfg.get("enabled", False):
            return None
        return route_semantic_candidate(
            text,
            self.encoder,
            threshold=float(router_cfg.get("threshold", 0.72)),
        )

    def _query_intent_bonus(
        self, item, query_route: SemanticRouteResult | None
    ) -> float:
        if query_route is None or not query_route.above_threshold or not query_route.durable_candidate:
            return 0.0
        record_profile_key = item.record.metadata.get("profile_key")
        if record_profile_key and record_profile_key == query_route.candidate_key:
            return 0.04
        return 0.0

    def _sidecar_query_texts(
        self, text: str, query_route: SemanticRouteResult | None
    ) -> list[str]:
        texts: list[str] = []
        raw = str(text or "").strip()
        normalized = _normalize_embedding_text(raw)
        if raw:
            texts.append(raw)
        if normalized and normalized != raw:
            texts.append(normalized)
        if query_route is not None and query_route.durable_candidate:
            matched_example = _normalize_embedding_text(str(query_route.matched_example or ""))
            if matched_example:
                texts.append(matched_example)
        unique: list[str] = []
        seen: set[str] = set()
        for item in texts:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _rank_key(self, item, query_route: SemanticRouteResult | None = None) -> tuple[float, float]:
        rank_score = (
            float(item.score)
            + class_priority(item.record.metadata.get("memory_class", item.record.role))
            + durability_bonus(bool(item.record.metadata.get("durable", False)))
            + _store_kind_bonus(item.record.metadata.get("kind"))
            + recency_bonus(item.record.timestamp)
            + _age_penalty_for_record(item.record, self.config)
            + self._query_intent_bonus(item, query_route)
        )
        return rank_score, float(item.score)

    def _recall_payload(self, item) -> dict:
        return {
            "score": item.score,
            "rank_score": self._rank_key(item)[0],
            "durability_bonus": durability_bonus(bool(item.record.metadata.get("durable", False))),
            "recency_bonus": recency_bonus(item.record.timestamp),
            "age_penalty": _age_penalty_for_record(item.record, self.config),
            "store_kind_bonus": _store_kind_bonus(item.record.metadata.get("kind")),
            "role": item.record.role,
            "text": item.record.text,
            "timestamp": item.record.timestamp,
            "message_id": item.record.message_id,
            "source_message_id": item.record.metadata.get("source_memory_id", item.record.message_id),
            "memory_class": item.record.metadata.get("memory_class", item.record.role),
            "extracted_value": item.record.metadata.get("extracted_value"),
            "profile_key": item.record.metadata.get("profile_key"),
            "durable": bool(item.record.metadata.get("durable", False)),
            "durability_reason": item.record.metadata.get("durability_reason"),
            "store_kind": item.record.metadata.get("kind", "memory"),
        }

    def prompt_context(self, recalled: list[dict]) -> dict:
        prompting_cfg = self.config.prompting
        factual, contextual = _partition_recalled_items(recalled)
        profile = {}
        profile_values: dict[str, str] = {}
        if self.profile_store is not None and self.config.profile.get("inject", True):
            profile = self.profile_store.load()
            profile_values = _profile_value_map(profile)

        dedupe_profile_facts = bool(prompting_cfg.get("dedupe_profile_facts", True))
        if dedupe_profile_facts and profile_values:
            filtered_factual: list[dict] = []
            for item in factual:
                profile_key = str(item.get("profile_key") or "")
                if profile_key and profile_key in profile_values:
                    continue
                filtered_factual.append(item)
            factual = filtered_factual

        contextual = _dedupe_recalled_items(contextual)
        dropped_contextual: list[dict] = []
        if bool(prompting_cfg.get("drop_nondurable_contextual", True)):
            contextual, dropped_contextual = _contextual_prompt_filter(
                contextual,
                allow_empty=bool(profile_values) or bool(factual),
            )

        factual_limit = int(prompting_cfg.get("factual_max_items", 3))
        contextual_limit = int(prompting_cfg.get("contextual_max_items", 2))
        factual = factual[:factual_limit]
        contextual = contextual[:contextual_limit]

        char_budget = int(prompting_cfg.get("max_context_chars", 0))
        char_budget_applied = False
        if char_budget > 0:
            factual, contextual, char_budget_applied = apply_char_budget(
                factual, contextual, char_budget
            )

        return {
            "profile": profile,
            "factual": factual,
            "contextual": contextual,
            "dropped_contextual": dropped_contextual,
            "char_budget_applied": char_budget_applied,
        }

    def build_prompt(self, message: Message, recalled: list[dict]) -> str:
        input_heading = self.config.prompting["input_heading"]
        profile_for_tz = self.profile_store.load() if self.profile_store is not None else None
        temporal_context = build_temporal_context(self.config, profile_for_tz)
        prompt_context = self.prompt_context(recalled)
        sentiment = detect_sentiment(message.text)
        factual = prompt_context["factual"]
        contextual = prompt_context["contextual"]
        profile_block = ""
        if self.profile_store is not None and self.config.profile.get("inject", True):
            profile = prompt_context["profile"]
            rendered_profile = _format_profile(profile)
            if rendered_profile:
                profile_block = f"User profile:\n{rendered_profile}\n\n"

        sections: list[str] = []
        if factual:
            factual_lines = [format_recalled_item(item, self.config, profile_for_tz) for item in factual]
            sections.append("Relevant durable facts:\n" + "\n".join(factual_lines))
        if contextual:
            context_lines = [format_recalled_item(item, self.config, profile_for_tz) for item in contextual]
            sections.append("Relevant conversation context:\n" + "\n".join(context_lines))
        if not sections:
            fallback_heading = (
                "Additional recalled memory" if profile_block else self.config.prompting["memory_heading"]
            )
            sections.append(f"{fallback_heading}:\n- none")
        memory_block = "\n\n".join(sections)
        return (
            "You are a helpful assistant with access to relevant prior memory and temporal context.\n\n"
            f"{temporal_context}\n\n"
            f"{_format_sentiment(sentiment)}\n\n"
            f"{profile_block}"
            f"{memory_block}\n\n"
            f"{input_heading}:\n{message.text}\n\n"
            "Respond to the user directly. Use the user profile and recalled memory if they are relevant."
        )

    def merged_recall(self, message: Message) -> dict:
        recalled = self.recall(message)
        factual, contextual = _partition_recalled_items(recalled)
        return {
            "query": message.text,
            "factual_count": len(factual),
            "contextual_count": len(contextual),
            "factual": factual,
            "contextual": contextual,
            "merged": recalled,
        }

    def respond(self, message: Message) -> str:
        recalled = self.recall(message)
        prompt = self.build_prompt(message, recalled)
        try:
            response = self.ollama.generate(prompt)
        except Exception:
            return "I'm unable to respond right now — please try again."
        return response

    def maintenance_status(self) -> dict:
        return maintenance_status(self.config)

    def _append_log(self, payload: dict) -> None:
        self.interaction_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.interaction_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def run_ollama_preflight(config: AppConfig) -> dict:
    llm_cfg = config.llm
    preflight_cfg = llm_cfg.get("preflight", {})
    if not preflight_cfg.get("enabled", True):
        return {"skipped": True}

    client = OllamaClient(
        OllamaProfile(
            host=llm_cfg["host"],
            model=llm_cfg["model"],
            temperature=0.0,
            max_tokens=int(preflight_cfg.get("max_tokens", 4)),
            timeout_seconds=int(preflight_cfg.get("timeout_seconds", llm_cfg["timeout_seconds"])),
            raw_prompt=bool(llm_cfg.get("raw_prompt", False)),
        )
    )
    return client.healthcheck(run_generate=bool(preflight_cfg.get("generate", True)))


def main() -> None:
    from uuid import uuid4

    parser = argparse.ArgumentParser(description="Interactive chat CLI for agent_memory_v2.")
    parser.add_argument("--user", default=None, help="User/profile to target. Defaults to catchall.")
    args = parser.parse_args()
    if args.user:
        os.environ["AGENT_MEMORY_V2_USER"] = args.user
    config = load_config()
    try:
        preflight = run_ollama_preflight(config)
    except Exception as exc:
        print(f"Ollama preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if not preflight.get("skipped"):
        reachable = preflight.get("reachable", False)
        model_present = preflight.get("model_present", False)
        version = preflight.get("version", "")
        print(
            f"Ollama preflight OK: reachable={reachable} model_present={model_present} version={version}"
        )
        if not reachable or not model_present or preflight.get("generate_ok") is False:
            print(json.dumps(preflight, indent=2), file=sys.stderr)
            raise SystemExit(2)

    if config.maintenance.get("startup_check", True):
        status = maintenance_status(config)
        if status.get("maintenance_due"):
            reasons = ",".join(status.get("due_reasons", []))
            print(f"Startup maintenance due for user={config.current_user}: reasons={reasons}")
            if config.maintenance.get("startup_run_if_due", True):
                result = run_maintenance(config)
                print(
                    "Startup maintenance completed: "
                    f"prune={result['actions'].get('prune', {}).get('pruned', 0)} "
                    f"sidecar={result['actions'].get('prune_sidecar', {}).get('pruned', 0)}"
                )

    pipeline = MemoryPipeline(config)
    print(f"Agent Memory V2 CLI. User profile: {config.current_user}. Type 'exit' to quit.")
    while True:
        text = input("You: ").strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        turn_id = str(uuid4())
        user_message = Message(role="user", text=text, turn_id=turn_id)
        reply = pipeline.respond(user_message)
        agent_message = Message(
            role="agent",
            text=reply,
            conversation_id=user_message.conversation_id,
            turn_id=turn_id,
        )
        pipeline.ingest_turn(user_message, agent_message)
        status = pipeline.maintenance_status()
        if status.get("maintenance_due"):
            reasons = ",".join(status.get("due_reasons", []))
            print(f"[maintenance due] reasons={reasons} run `make maintain`")
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
