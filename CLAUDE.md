# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For resumable agent work, start with `AGENTS.md`. Complex work should create or update an execution plan in `docs/exec-plans/active/`.

## Commands

### Install

```bash
pip install -e '.[dev]'          # base install (tests, linting)
pip install -e '.[dev,embeddings]'  # adds sentence-transformers for live embedding
```

Scripts default to `AGENT_MEMORY_V2_PYTHON=/tmp/agent_memory_v2_venv/bin/python`. Override with:

```bash
export AGENT_MEMORY_V2_PYTHON=$(which python)
```

### Testing

```bash
make test                        # run full test suite (unit + integration)
bash scripts/test.sh -k "test_name"   # run a single test by name
bash scripts/test.sh tests/unit/test_pipeline.py  # run one file
```

CI runs `make test` then `make eval-all`. The eval suite forces hash embeddings (no Ollama needed) — safe in CI.

### Linting

Ruff is the only linter. Run directly:

```bash
python -m ruff check src/ tests/
python -m ruff check --fix src/ tests/
```

### Evaluation

```bash
make eval-all          # deterministic eval — no Ollama required (uses hash encoder)
make eval-classification
make eval-semantic
make eval-sentiment
make eval-recall
make eval-prompt
make eval-history      # show score history across runs
make eval-compare      # compare latest vs prior baseline
```

Live evals require a running Ollama stack:

```bash
make live-eval-all
make live-eval-memory
make live-eval-sentiment
```

Agent tool-loop evals:

```bash
make agent-eval-run ARGS="--scenario preference_recall --provider fake --save-all"
make agent-eval-all ARGS="--provider fake --record-history --save-all"
make agent-eval-history
make agent-eval-compare
OPENAI_API_KEY=... make agent-eval-run ARGS="--scenario preference_recall --provider openai --save-all"
ANTHROPIC_API_KEY=... make agent-eval-run ARGS="--scenario preference_recall --provider anthropic --save-all"
```

### Pipeline debugging (stage-by-stage)

```bash
make classify ARGS="--text 'I prefer oat milk.'"
make classify ARGS="--text 'I am based in Edinburgh.' --semantic"
make classify ARGS="--text 'I am based in Edinburgh.' --extract"
make recall ARGS="--text 'What do I prefer?'"
make prompt ARGS="--text 'What do I prefer?'"   # shows recall, context, dropped items, sentiment
make generate ARGS="--prompt 'Reply with exactly OK.'"
make ingest ARGS="--text 'I prefer oat milk.' --reply 'Noted.'"
```

### Admin / maintenance

```bash
make stats
make list ARGS="--memory-class preference"
make list ARGS="--query dentist"
make list-sidecar ARGS="--limit 10"
make profile
make aging-report
make prune-dry-run
make prune
make maintenance-status
make maintain
make rebuild-profile
make rebuild          # full rebuild from interaction logs (reclassifies + re-extracts)
make reset            # wipe live store (destructive)
make check-schema     # print schema-version summary for live stores
make doctor           # Ollama + embedding connectivity check
```

### Backup / restore / seed

```bash
make backup ARGS="--output backups/state.zip"
make restore ARGS="--input backups/state.zip --force"
make seed ARGS="--seed-file seeds/generic_seed.jsonl --user demo"
make sanitize-publish   # remove runtime state before publishing to GitHub
```

### Multi-user

```bash
make chat ARGS="--user mark"
bash scripts/admin.sh --user mark stats
export AGENT_MEMORY_V2_USER=mark  # sets user for helper commands
```

### Scenarios (qualitative review)

```bash
make scenario-list
make scenario-run ARGS="--scenario <name>"
make scenario-show ARGS="--scenario <name>"
make scenario-compare ARGS="--scenario <name>"
```

### Harness validation

```bash
bash scripts/validate-harness.sh
```

---

## Architecture

### Memory pipeline overview

`MemoryPipeline` in `pipeline.py` is the central class. A complete turn flows:

1. **Classify** — `classifier.py` applies regex patterns to detect preference, fact, task, or ephemeral turn. Patterns are loaded from `config/taxonomy.yaml` at import time (with hardcoded fallback).
2. **Semantic route** (optional) — if the classifier returns a non-durable `turn`, `semantic_router.py` embeds the text and cosine-matches it against prototype examples from the taxonomy. Above-threshold hits become durable candidates.
3. **Structured extraction** (optional) — above-threshold durable candidates go through `structured_extractor.py`, which asks Ollama to extract a constrained JSON value. Accepted extractions are stored in sidecar.
4. **Store** — `store.py` (`MemoryStore`) persists a FAISS index plus a JSON sidecar of `MemoryRecord` objects. Every record's metadata is stamped with `schema_version` and `taxonomy_version`.
5. **Sidecar** — durable records (fact/preference/task) are written to a second `MemoryStore` instance (the sidecar). The sidecar is embedded using the extracted value prefixed by its profile key for better semantic surface area.
6. **Profile update** — `profile.py` `update_from_record()` does an O(1) incremental update of `data/profile/user_profile.json` from the new sidecar record. Full `rebuild_from_records()` is only used in maintenance/admin.

**Recall** (`MemoryPipeline.recall()`):
- Embeds the query and searches both the main store and sidecar with cosine similarity.
- Scores are boosted by: class priority (fact > preference > task > turn), durability, recency, and an optional query-intent bonus when the semantic router matches the query to a known profile key.
- Returns deduplicated results split into `factual` (sidecar) and `contextual` (main) lists.

**Prompt construction** (`MemoryPipeline.build_prompt()`):
- Loads the user profile for per-user timezone resolution.
- Applies count limits (`factual_max_items`, `contextual_max_items`) then a character budget (`max_context_chars: 3200`) — always keeping at least one factual item.
- Injects profile, temporal context, sentiment signal, factual items, and conversational context as separate prompt sections.
- Returns the assembled prompt string.

### Taxonomy (`config/taxonomy.yaml` + `taxonomy.py`)

The taxonomy is the single source of truth for all extractable memory keys. It drives:
- Regex fact patterns for `classifier.py`
- Semantic prototype examples for `semantic_router.py`
- Allowed profile keys for `structured_extractor.py`
- Compaction mode (`scalar` = latest overwrites, `additive` = accumulates `all_values` list) for `profile.py`

`get_taxonomy()` returns a module-level singleton; tests that need a custom taxonomy must set `taxonomy._cached = None` before loading.

Each key has `tier1` (domain) and `tier2` (concept) that must match its `key` prefix (e.g. `identity.location` → `tier1: identity`, `tier2: location`).

### Dual-store layout

| Store | Contents | Index | Metadata |
|---|---|---|---|
| Main store | All ingested turns (full `User: … Agent: …` text) | `data/memory/memory.index` | `data/memory/memory_metadata.json` |
| Sidecar | Durable facts/preferences/tasks only (extracted value, prefixed by profile key) | `data/sidecar/facts.index` | `data/sidecar/facts_metadata.json` |
| Profile | Derived JSON view of sidecar, injected into every prompt | — | `data/profile/user_profile.json` |

Pruned records from either store are archived to `data/archive/` as JSONL before deletion. Use `make inspect-archive` / `make restore-from-archive` to inspect or recover them.

### Ollama resilience

`ollama.py` wraps every outbound call in `_with_retry()` (3 attempts, exponential backoff on `ConnectionError`, `Timeout`, 5xx). `pipeline.respond()` returns a canned fallback string on Ollama failure; `_structured_extraction_metadata()` returns `{}` so ingestion continues without structured extraction.

### Schema versioning

`SCHEMA_VERSION = 1` in `models.py` is stamped into every stored record's metadata. On load, `store._check_schema_versions()` emits `warnings.warn` for any record with a mismatched version — run `make rebuild` to re-stamp.

### Embeddings

Two providers are supported, switched via `config/settings.yaml` (`embeddings.provider`):
- `ollama` — `nomic-embed-text` via Ollama, 768-dim (live default)
- `hash` — deterministic `HashEmbeddingEncoder` built into `embeddings.py` — used by CI and all unit tests, no Ollama required

Switch with `make use-ollama-embeddings` / `make use-hash-embeddings`. After switching, run `make rebuild` to re-embed existing records under the new provider.

### Aging and maintenance

`aging.py` applies decay penalties to recall scores based on time since creation (or `last_recalled_at` if the record has been recalled recently). Records recalled ≥ `min_recall_count_to_keep` times are protected from pruning.

`maintenance.py` runs a deferred maintenance cycle (configurable interval, default 30 min) that: prunes stale ephemeral turns, resolves/expires tasks, compacts the sidecar (keeping latest per profile key), and rebuilds the profile. It runs automatically at chat startup if due.

### Sentiment

`sentiment.py` classifies the current user utterance into `neutral / positive / negative / distressed / urgent`. A `_is_negated()` helper suppresses cues when a negation word appears within 3 tokens before the matched keyword. The result is injected into the prompt as response-tuning guidance.

---

## Configuration

All runtime config lives in `config/settings.yaml`. Key sections:

- `embeddings.provider` — `ollama` or `hash`
- `structured_extractor.enabled` / `admission_threshold` — controls hybrid extraction
- `semantic_router.enabled` / `threshold` — controls semantic routing
- `prompting.max_context_chars` — character budget for recalled items in prompt (3200 default)
- `aging.prune.min_recall_count_to_keep` — protects frequently recalled memories
- `app.timezone` — fallback timezone; per-user timezone overrides via `preference.timezone` profile key

`AppConfig` in `config.py` wraps the YAML dict. All paths are resolved relative to `root_dir` via `config.resolve_path()`.

---

## Key invariants

- **Taxonomy tier1/tier2 must match the key prefix.** `contextual.world_fact` requires `tier1: contextual`, not `tier1: context`.
- **Sidecar embedding text** is `"{profile_key}: {extracted_value}"` to maximise semantic surface area for short extracted values.
- **`ingest_turn` embeds the full `User: … Agent: …` text** for richer recall, but classifies on user text only.
- **`update_from_record` is the hot path** for profile writes (O(1)); `rebuild_from_records` is for maintenance/admin only.
- **Evals must not use Ollama.** `eval_cli.py` hard-overrides `embeddings.provider = "hash"` — do not remove this.
- **Agent evals use isolated hash-backed storage.** This keeps tool-loop/model behavior separate from live Ollama embeddings and runtime state.
