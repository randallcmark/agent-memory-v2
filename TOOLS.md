# Tools and Workflows

This project supports two distinct usage modes:

1. clean operational flow for normal use
2. rich debug flow for qualitative analysis and troubleshooting

The consolidated operator entrypoint is:

```bash
bash scripts/admin.sh <command> [args...]
```

You can scope admin and maintenance commands to a named user with:

```bash
bash scripts/admin.sh --user mark stats
```

## Operational Flow

Use these commands when you want the system to behave like a running product.

Health and setup:

```bash
make smoke
make smoke-generate
make embedding-smoke
make doctor
```

Normal operation:

```bash
make chat
make stats
make list
make list-sidecar
make profile
make aging-report
make prune-dry-run
make prune
make maintenance-status
make maintain
```

Named user operation:

```bash
make chat ARGS="--user mark"
```

State management:

```bash
make backup ARGS="--output backups/state-latest.zip"
make restore ARGS="--input backups/state-latest.zip --force"
make rebuild
make rebuild-profile
make reset
```

Recommended operational routine:

1. `make doctor`
2. `make backup ARGS="--output backups/pre-session.zip"`
3. `make chat`
4. `make stats`
5. `make backup ARGS="--output backups/post-session.zip"`

## Debug Flow

Use these commands when you want to inspect one stage at a time.

Seed analysis data without chat:

```bash
make ingest ARGS="--text 'I prefer oat milk.' --reply 'Noted.'"
```

Classification only:

```bash
make classify ARGS="--text 'I prefer oat milk.'"
```

Recall only:

```bash
make recall ARGS="--text 'What do I prefer?'"
```

This now returns:

1. `factual`: durable sidecar results
2. `contextual`: turn-memory results
3. `merged`: final combined ordering used by the pipeline

Prompt assembly only:

```bash
make prompt ARGS="--text 'What do I prefer?'"
```

This now builds separate prompt sections for:

1. `User profile`
1. `Relevant durable facts`
2. `Relevant conversation context`

It also returns `prompt_context`, which shows:

1. `profile`: the exact derived profile considered for injection
2. `factual`: the factual items that survived prompt filtering
3. `contextual`: the contextual items that survived prompt filtering
4. `dropped_contextual`: contextual items that were recalled but deliberately not injected
5. `sentiment`: the explicit input sentiment signal and response-tuning guidance

Raw generation only:

```bash
make generate ARGS="--prompt 'Reply with exactly OK.'"
```

Store inspection:

```bash
make list ARGS="--memory-class preference --limit 5"
make list-sidecar ARGS="--limit 10"
make profile
make list ARGS="--query oat"
make list ARGS="--date-from 2026-03-29T00:00:00+00:00 --date-to 2026-03-29T23:59:59+00:00"
make stats
```

Seed and publish-safe prep:

```bash
make seed ARGS="--seed-file seeds/generic_seed.jsonl --user demo --conversation-id seed"
make sanitize-publish
```

Recommended debug routine:

1. `make backup ARGS="--output backups/pre-debug.zip"`
2. `make ingest ARGS="--text 'My name is Mark.' --reply 'Noted.'"`
3. `make classify ARGS="--text 'My name is Mark.'"`
4. `make recall ARGS="--text 'What is my name?'"`
5. `make prompt ARGS="--text 'What is my name?'"`
6. `make generate ARGS="--prompt-file /path/to/prompt.txt"`
7. `make list ARGS="--memory-class fact --limit 10"`
8. `make list-sidecar ARGS="--limit 10"`
9. `make profile`
10. `make aging-report`
11. `make prune-dry-run`
12. `make prune`
13. `make maintenance-status`
14. `make maintain`
15. `make restore ARGS="--input backups/pre-debug.zip --force"`

## Traceability

When a stage looks wrong, use this map to jump directly to the owning module.

`make smoke`, `make smoke-generate`:
- ownership: [ollama.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/ollama.py)
- entrypoint: [ollama_smoke.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/ollama_smoke.py)

`make embedding-smoke`:
- ownership: [ollama.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/ollama.py)
- embedding path: [embeddings.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/embeddings.py)
- entrypoint: [embedding_smoke.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/embedding_smoke.py)

`make classify`:
- ownership: [classifier.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/classifier.py)
- entrypoint: [classify_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/classify_cli.py)

`make ingest`:
- ownership: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)
- message model: [models.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/models.py)
- entrypoint: [ingest_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/ingest_cli.py)

`make recall`:
- ownership: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)
- store path: [store.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/store.py)
- entrypoint: [recall_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/recall_cli.py)

`make prompt`:
- ownership: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)
- sentiment signal: [sentiment.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/sentiment.py)
- entrypoint: [prompt_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/prompt_cli.py)

`make generate`:
- ownership: [ollama.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/ollama.py)
- entrypoint: [generate_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/generate_cli.py)

`make stats`, `make list`, `make rebuild`, `make reset`:
- ownership: [admin.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/admin.py)
- store path: [store.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/store.py)

`make aging-report`, `make prune-dry-run`, `make prune`:
- ownership: [aging.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/aging.py)
- admin surface: [admin.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/admin.py)
- recall scoring: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)

`make maintenance-status`, `make maintain`:
- ownership: [maintenance.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/maintenance.py)
- chat hook: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)
- state persistence: [state_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/state_cli.py)

`make list-sidecar`:
- ownership: [admin.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/admin.py)
- durable store path: [store.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/store.py)

`make profile`, `make rebuild-profile`:
- ownership: [profile.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/profile.py)
- admin surface: [admin.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/admin.py)
- prompt injection: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)

`make backup`, `make restore`:
- ownership: [state_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/state_cli.py)

`make seed`, `make sanitize-publish`:
- ownership: [seed_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/seed_cli.py)
- ownership: [sanitise_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/sanitise_cli.py)
- publish flow: [PUBLISHING.md](/Volumes/Media/Repository/agent_memory_v2/PUBLISHING.md)

`make chat`:
- ownership: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)

`make e2e`, `make doctor`:
- orchestration: [e2e_smoke.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/e2e_smoke.py)
- orchestration: [doctor.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/doctor.py)

## Interpreting Output

For recall and prompt analysis, focus on these fields:

- `score`: raw semantic similarity
- `rank_score`: score after class, durability, and recency adjustments
- `memory_class`: `preference`, `fact`, `task`, `turn`, or `message`
- `durable`: whether the memory is treated as lasting
- `durability_reason`: why it was or was not treated as durable
- `recency_bonus`: ranking bonus for freshness
- `age_penalty`: ranking penalty for older memories under the current aging policy
- `durability_bonus`: ranking bonus or penalty for durability
- `store_kind`: whether a result came from `sidecar_memory` or `turn_memory`
- `store_kind_bonus`: ranking bonus for sidecar results

For profile analysis, focus on:

- `preferences`
- `facts`
- `tasks`
- field keys such as `preference.general` or `identity.name`

For prompt-selection analysis, focus on:

- `prompt_context.profile`
- `prompt_context.factual`
- `prompt_context.contextual`
- `prompt_context.dropped_contextual`

This separates:

1. what recall found
2. what prompt construction actually kept
3. what prompt construction intentionally excluded

If the wrong memory is ranked first:

1. run `make recall ...`
2. inspect `score`, `rank_score`, `memory_class`, `durable`, `age_penalty`
3. inspect the stored record with `make list ...`
4. inspect store age distribution with `make aging-report`
5. inspect prune recommendations with `make prune-dry-run`
6. apply the conservative prune policy with `make prune` if the dry run looks correct
7. inspect the final prompt with `make prompt ...`
8. inspect the owning module from the traceability map above

Current prune behavior:

- `make prune` only removes stale ephemeral `turn` or `message` records from the main store
- removed records are appended to the configured archive path before deletion
- durable profile-backed memories are retained

Current deferred maintenance behavior:

- chat startup performs a maintenance check before entering the interaction loop
- completed turns mark maintenance state and increment `new_records_since_run`
- the chat loop only reports when maintenance is due; it does not run maintenance inline
- `make maintain` performs the configured background-safe tasks under a lock
- the maintenance cycle now includes main-store prune, sidecar compaction, and profile rebuild
- task memories can now be pruned either because they were explicitly completed or because they expired by policy

## Current Limitation

Debug output is intentionally rich, but some stored historical turn text still contains residual assistant-side boilerplate. That cleanup work should improve:

1. recalled memory readability
2. prompt readability
3. model-behavior analysis from `make prompt` and `make generate`
