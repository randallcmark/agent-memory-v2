# Internal README

This document preserves the detailed internal/operator README that was used
during active build-out of `agent_memory_v2`.

For the public GitHub-facing overview, see `README.md`.

# Agent Memory V2

`agent_memory_v2` is a clean rebuild of the original agent memory project.

The objective is unchanged:

1. process interactions,
2. persist useful memory,
3. retrieve relevant memory,
4. inject retrieved memory into subsequent prompts,
5. run against a local Ollama-hosted `llama3:8b` model.

## Scope of Milestone 1

Milestone 1 establishes a narrow, testable baseline:

- single local LLM provider: Ollama
- single embedding path: sentence-transformers
- single vector store: FAISS
- deterministic memory record model
- prompt injection based on recalled memory
- simple CLI loop for manual verification

Current default embedding provider is the local Ollama model
`nomic-embed-text`, which is used for the live baseline alongside
`llama3:8b` for generation.

This version deliberately excludes advanced features for now:

- sidecar anchor stores
- multi-provider abstraction
- LLM summarization-based preprocessing
- warm-start optimization
- rich memory classification beyond base metadata

## Install

Base install for development and tests:

```bash
pip install -e '.[dev]'
```

Embedding model support:

```bash
pip install -e '.[dev,embeddings]'
```

The embedding dependency is separated so the test harness and project skeleton
can be bootstrapped before model-layer compatibility is finalized.

The built-in hash encoder is still available as a fallback. The optional
`sentence-transformers` extra is reserved for later enablement on a compatible
Python/Torch platform.

## Ollama Verification

## Command-Line Drivers

You do not need to remember the full Python module commands anymore.

From the project root:

```bash
make test
make smoke
make smoke-generate
make embedding-smoke
make install-embedding-model
make use-ollama-embeddings
make use-hash-embeddings
make e2e
make preflight
make chat
make ingest
make classify
make recall
make prompt
make generate
make backup
make restore
make seed
make sanitize-publish
make stats
make list
make list-sidecar
make profile
make aging-report
make prune-dry-run
make prune
make maintenance-status
make maintain
make rebuild-profile
make reset
make rebuild
make doctor
```

Equivalent shell wrappers are available under `scripts/`:

```bash
bash scripts/test.sh
bash scripts/ollama-smoke.sh
bash scripts/ollama-generate-smoke.sh
bash scripts/embedding-smoke.sh
bash scripts/install-embedding-model.sh
bash scripts/use-ollama-embeddings.sh
bash scripts/use-hash-embeddings.sh
bash scripts/e2e-smoke.sh
bash scripts/preflight.sh
bash scripts/chat.sh
bash scripts/ingest.sh
bash scripts/classify.sh
bash scripts/recall.sh
bash scripts/prompt.sh
bash scripts/generate.sh
bash scripts/seed.sh
bash scripts/sanitise.sh
bash scripts/state.sh export
bash scripts/state.sh import
bash scripts/admin.sh stats
bash scripts/admin.sh list
bash scripts/admin.sh list-sidecar
bash scripts/admin.sh profile
bash scripts/admin.sh aging-report
bash scripts/admin.sh prune-dry-run
bash scripts/admin.sh prune --force
bash scripts/maintenance.sh status
bash scripts/maintenance.sh run
bash scripts/admin.sh rebuild-profile --force
bash scripts/admin.sh reset --force
bash scripts/admin.sh rebuild --force
bash scripts/doctor.sh
```

`scripts/admin.sh` is now the consolidated operator entrypoint for:

1. admin inspection and rebuild commands
2. maintenance status and maintenance runs
3. backup and restore
4. config helper commands such as embedding-provider switches

Environment variables:

```bash
AGENT_MEMORY_V2_PYTHON=/tmp/agent_memory_v2_venv/bin/python
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3:8b
```

The scripts default to the working local venv at `/tmp/agent_memory_v2_venv/bin/python`.

For make targets that need arguments, pass them via `ARGS`, for example:

```bash
make use-ollama-embeddings ARGS="--host http://localhost:11434 --model nomic-embed-text --timeout-seconds 60 --dimensions 768"
make chat ARGS="--user mark"
make classify ARGS="--text 'I prefer oat milk.'"
make ingest ARGS="--text 'I prefer oat milk.' --reply 'Noted.'"
make seed ARGS="--seed-file seeds/generic_seed.jsonl --user demo --conversation-id seed"
make recall ARGS="--text 'What do I prefer?'"
make prompt ARGS="--text 'What do I prefer?'"
make generate ARGS="--prompt 'Reply with exactly OK.'"
make backup ARGS="--output backups/state-2026-03-29.zip"
make restore ARGS="--input backups/state-2026-03-29.zip --force"
make sanitize-publish
make list ARGS="--memory-class preference"
make list-sidecar ARGS="--limit 10"
make profile
make aging-report
make prune-dry-run
make prune
make maintenance-status
make maintain
make rebuild-profile
make list ARGS="--query dentist"
make list ARGS="--date-from 2026-03-30T00:00:00+00:00 --date-to 2026-03-30T23:59:59+00:00"
```

`make prompt` now returns both the final `prompt` and a `prompt_context`
object so you can distinguish:

1. what recall returned
2. what prompt construction kept
3. what prompt construction deliberately dropped

`make prompt` also exposes an explicit sentiment analysis block for the current
utterance so prompt tuning is inspectable rather than implicit.

`make aging-report` returns the current age buckets for the main store and
sidecar store. `make prune-dry-run` returns recommended keep/review/prune
decisions without modifying any stored state. `make prune` applies the first
conservative pruning rule: stale ephemeral turn memories in the main store,
archiving them to the configured prune archive path before removal.

`make maintenance-status` shows whether deferred maintenance is due.
`make maintain` acquires the maintenance lock, runs the configured maintenance
tasks, and updates the persisted maintenance state.

User/profile selection:

1. default behavior uses the `catchall` profile
2. `make chat ARGS="--user mark"` starts the chat loop against the `mark` profile
3. `bash scripts/admin.sh --user mark stats` runs admin tasks against the `mark` profile
4. `AGENT_MEMORY_V2_USER=mark ...` can be used with other helper commands when needed

The current maintenance cycle can:

1. prune stale ephemeral turn memories
2. resolve completed tasks after the configured grace period
3. expire stale unresolved tasks by policy
4. compact the sidecar by keeping only the latest durable entry per `profile_key`
5. rebuild the derived profile after compaction

Publishing and seeding:

1. `make seed` loads generic, non-sensitive seed data into the selected user profile
2. `make sanitize-publish` removes publish-unsafe runtime state in preparation for GitHub publication
3. the safe publication flow is documented in [PUBLISHING.md](/Volumes/Media/Repository/agent_memory_v2/PUBLISHING.md)

## Progress Tracking

Track roadmap and current status in:

- [ROADMAP.md](/Volumes/Media/Repository/agent_memory_v2/ROADMAP.md)
- [STATUS.md](/Volumes/Media/Repository/agent_memory_v2/STATUS.md)
- [TOOLS.md](/Volumes/Media/Repository/agent_memory_v2/TOOLS.md)

Basic service and model check:

```bash
make smoke
```

Generation check with a longer timeout:

```bash
make smoke-generate
```

Pipeline-level smoke test:

```bash
make e2e
```

Interactive chat:

```bash
make chat
```

## Layout

```text
agent_memory_v2/
├── config/
│   └── settings.yaml
├── src/
│   └── agent_memory_v2/
│       ├── __init__.py
│       ├── config.py
│       ├── embeddings.py
│       ├── models.py
│       ├── ollama.py
│       ├── pipeline.py
│       └── store.py
├── tests/
│   ├── integration/
│   └── unit/
└── pyproject.toml
```

## Workflows

Two workflows are now documented explicitly:

1. clean operational flow for normal use
2. rich debug flow for qualitative analysis and troubleshooting

See [TOOLS.md](/Volumes/Media/Repository/agent_memory_v2/TOOLS.md) for:

- command-by-command workflows
- state seeding and snapshot/restore
- module traceability for each tool
- guidance on interpreting recall and prompt outputs

The next work items are tracked in `ROADMAP.md` and `STATUS.md`.
