# Internal README

This document preserves detailed internal/operator notes for `agent_memory_v2`.
For the public overview see `README.md`. For Claude Code architecture guidance see `CLAUDE.md`.

## Install

```bash
pip install -e '.[dev]'            # base install (tests + linting)
pip install -e '.[dev,embeddings]' # adds sentence-transformers for local embedding models
```

The project-local venv at `.venv/` is auto-detected by all scripts. No manual
environment variable is needed after `pip install -e '.[dev]'` inside `.venv`.

To override the Python binary used by all scripts:

```bash
export AGENT_MEMORY_V2_PYTHON=/path/to/your/python
```

Scripts resolve Python in this order:
1. `$AGENT_MEMORY_V2_PYTHON` (explicit override)
2. `.venv/bin/python` in the project root (auto-detected)
3. `/tmp/agent_memory_v2_venv/bin/python` (legacy fallback)

## Environment Variables

```bash
AGENT_MEMORY_V2_PYTHON=/path/to/python   # override Python binary
AGENT_MEMORY_V2_USER=mark                # override active user profile
OLLAMA_HOST=http://127.0.0.1:11434       # Ollama host (default)
OLLAMA_MODEL=llama3:8b                   # Ollama model (default)
```

## Command Reference

All commands are available via `make <target>` or directly via `bash scripts/<script>.sh`.

### Setup and Health

```bash
make doctor                    # Ollama + embedding connectivity check
make smoke                     # Ollama reachability and model presence
make smoke-generate            # Ollama generation check with longer timeout
make embedding-smoke           # embedding model health check
make install-embedding-model   # pull the configured embedding model
make use-ollama-embeddings ARGS="--host http://localhost:11434 --model nomic-embed-text --dimensions 768"
make use-hash-embeddings       # switch to deterministic hash encoder (no Ollama needed)
make preflight                 # full preflight check before chat
```

### Chat

```bash
make chat                      # start interactive chat (catchall user)
make chat ARGS="--user mark"   # start chat for named user
```

### Stage-by-Stage Debug

```bash
make classify ARGS="--text 'I prefer oat milk.'"
make classify ARGS="--text 'I am based in Edinburgh.' --semantic"
make classify ARGS="--text 'I am based in Edinburgh.' --extract"
make recall ARGS="--text 'What do I prefer?'"
make prompt ARGS="--text 'What do I prefer?'"    # shows recall, context, dropped items, sentiment
make generate ARGS="--prompt 'Reply with exactly OK.'"
make ingest ARGS="--text 'I prefer oat milk.' --reply 'Noted.'"
make ingest ARGS="--text 'I am based in Edinburgh.' --reply 'Noted.' --conversation-id debug"
```

### Store Inspection

```bash
make stats
make list
make list ARGS="--memory-class preference"
make list ARGS="--query oat milk"
make list ARGS="--date-from 2026-05-01T00:00:00+00:00 --date-to 2026-05-01T23:59:59+00:00"
make list-sidecar ARGS="--limit 10"
make profile
make check-schema              # print schema-version summary for live stores
```

### Aging, Pruning, and Maintenance

```bash
make aging-report
make prune-dry-run
make prune                     # archive-then-delete stale ephemeral turns
make maintenance-status
make maintain                  # run full maintenance cycle (prune + compact + rebuild)
make rebuild-profile           # rebuild derived profile from sidecar records
make rebuild                   # full rebuild from interaction logs (reclassifies + re-extracts)
make reset                     # wipe live store (destructive)
```

### Archive

```bash
bash scripts/admin.sh inspect-archive
bash scripts/admin.sh restore-from-archive --since 2026-01-01
bash scripts/admin.sh restore-from-archive --memory-class fact
bash scripts/admin.sh restore-from-archive --sidecar
```

### Backup, Restore, Seed

```bash
make backup ARGS="--output backups/state-$(date +%Y%m%d).zip"
make restore ARGS="--input backups/state-20260506.zip --force"
make seed ARGS="--seed-file seeds/generic_seed.jsonl --user demo --conversation-id seed"
make sanitize-publish          # strip runtime state before publishing to GitHub
```

### Evaluation

```bash
make eval-all                  # all deterministic evals (no Ollama)
make eval-classification
make eval-semantic
make eval-sentiment
make eval-profile
make eval-recall
make eval-prompt
make eval-all ARGS="--record-history"
make eval-history
make eval-compare

make live-eval-all             # all live evals (requires running Ollama)
make live-eval-memory
make live-eval-sentiment
make live-eval-all ARGS="--record-history --save-all"
make live-eval-history
make live-eval-compare
```

### Scenarios

```bash
make scenario-list
make scenario-run ARGS="--scenario negative_sentiment_preference"
make scenario-show ARGS="--run-id <run-id>"
make scenario-compare ARGS="--run-a <run-a> --run-b <run-b>"
```

### Multi-User

```bash
make chat ARGS="--user mark"
bash scripts/admin.sh --user mark stats
bash scripts/admin.sh --user mark profile
export AGENT_MEMORY_V2_USER=mark   # scopes all subsequent commands to that user
```

## Taxonomy

`config/taxonomy.yaml` is the single source of truth for all extractable memory keys.
It is the canonical definition of what the system can remember and drives:

- Regex fact patterns used by `classifier.py`
- Semantic prototype examples used by `semantic_router.py`
- Allowed profile keys for `structured_extractor.py`
- Compaction mode (`scalar` / `additive` / `task`) for `profile.py`

When adding a new memory category, add it to `taxonomy.yaml` first.
The classifier, router, and extractor all load from it at startup (module-level singleton with hardcoded fallback).

After editing `taxonomy.yaml`, clear the taxonomy cache in any running test by setting `taxonomy._cached = None`.

## Schema Versioning

`SCHEMA_VERSION = 1` in `models.py` is stamped into every new `MemoryRecord`'s metadata.
On store load, `_check_schema_versions()` emits `warnings.warn` for records with a stale version.
Run `make rebuild` to re-stamp all records after a schema change.

## Embedding Providers

| Provider | Command | Notes |
|---|---|---|
| `ollama` | `make use-ollama-embeddings` | 768-dim, requires running Ollama |
| `hash` | `make use-hash-embeddings` | deterministic, used by all unit tests and CI |

After switching providers, run `make rebuild` to re-embed existing records.
The eval suite hard-overrides to `hash` — never remove that override.

## Maintenance Cycle

Running `make maintain` executes all configured maintenance tasks under a lock:

1. Prune stale ephemeral turn memories from the main store (archive before delete)
2. Resolve completed tasks after their grace period
3. Expire stale unresolved tasks by policy
4. Compact the sidecar (keep latest record per `profile_key`, archive superseded entries)
5. Rebuild the derived profile from the compacted sidecar

Chat startup checks whether maintenance is due and runs it automatically if `startup_run_if_due: true`.

## Recommended Routines

**Before a chat session:**
```bash
make doctor
make backup ARGS="--output backups/pre-session.zip"
make chat
```

**After changes to taxonomy or classification:**
```bash
make test
make eval-all
make rebuild  # re-classify and re-extract historical interaction logs
make profile
```

**Debugging a recall or prompt issue:**
```bash
make recall ARGS="--text 'your query'"
make prompt ARGS="--text 'your query'"
make list ARGS="--query keyword"
make list-sidecar
make aging-report
make prune-dry-run
```
