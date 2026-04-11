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

Classification plus semantic candidate routing:

```bash
make classify ARGS="--text 'I am based in Edinburgh in the UK.' --semantic"
make classify ARGS="--text 'The Meadows has cherry blossom trees.' --semantic"
make classify ARGS="--text 'What day is it today?' --semantic"
```

The semantic path reports the best `semantic_candidate` for generic non-durable
rule-classifier misses.

Structured extraction:

```bash
make classify ARGS="--text 'I am based in Edinburgh in the UK.' --extract"
make ingest ARGS="--text 'I am based in Edinburgh in the UK.' --reply 'Noted.' --conversation-id extraction-debug"
make list ARGS="--query Edinburgh --limit 5"
make list-sidecar ARGS="--limit 10"
make profile
```

The `--extract` path runs the Phase 2 constrained JSON extractor after semantic
routing finds an above-threshold durable candidate. A record is promoted into
the sidecar/profile only when the extractor returns valid JSON, a supported
`profile_key`, a compact `extracted_value`, and confidence above the configured
admission threshold. Rejected extraction attempts remain traceable in
`structured_extraction` metadata.

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

Deterministic regression evaluation:

```bash
make eval-classification
make eval-semantic
make eval-sentiment
make eval-profile
make eval-recall
make eval-prompt
make eval-all
make eval-history
make eval-compare
```

These eval commands run against isolated temporary storage and force the
deterministic hash embedding path so they are stable regression checks rather
than live-environment Ollama quality probes.

To persist compact deterministic summaries and compare runs:

```bash
make eval-all ARGS="--record-history"
make eval-history
make eval-compare
```

Live Ollama evaluation:

```bash
make live-eval-memory
make live-eval-sentiment
make live-eval-all
make live-eval-history
make live-eval-compare
```

These commands run against the real local runtime stack:

1. real Ollama generation model
2. real configured embedding provider
3. isolated temporary memory/profile/sidecar state
4. checked-in live eval dataset

Artifacts for failed cases, or for all cases when `--save-all` is used, are
written under `artifacts/live_eval/...` and include:

1. recalled items
2. selected prompt context
3. final prompt
4. raw model response

To persist compact live-run summaries and compare them over time:

```bash
make live-eval-all ARGS="--record-history --save-all"
make live-eval-history
make live-eval-compare
```

Scenario-driven qualitative review:

```bash
make scenario-list
make scenario-run ARGS="--scenario negative_sentiment_preference"
make scenario-run ARGS="--scenario semantic_location_candidate"
make scenario-run ARGS="--scenario semantic_world_context_not_profile"
make scenario-run ARGS="--scenario semantic_location_correction_latest_wins"
make scenario-show ARGS="--run-id 20260329_211535_negative_sentiment_preference"
make scenario-compare ARGS="--run-a <run-a> --run-b <run-b>"
```

These commands are intended for subjective review rather than score-based
regression. Each scenario run stores a bundle under `artifacts/scenarios/...`
containing:

1. setup-turn ingestion details
2. merged recall output
3. prompt context
4. final prompt
5. live model response
6. scenario review notes

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
4. `make classify ARGS="--text 'I am based in Edinburgh in the UK.' --semantic"`
5. `make recall ARGS="--text 'What is my name?'"`
6. `make prompt ARGS="--text 'What is my name?'"`
7. `make generate ARGS="--prompt-file /path/to/prompt.txt"`
8. `make list ARGS="--memory-class fact --limit 10"`
9. `make list-sidecar ARGS="--limit 10"`
10. `make profile`
11. `make aging-report`
12. `make prune-dry-run`
13. `make prune`
14. `make maintenance-status`
15. `make maintain`
16. `make restore ARGS="--input backups/pre-debug.zip --force"`

Recommended regression routine:

1. `make eval-classification`
2. `make eval-semantic`
3. `make eval-sentiment`
4. `make eval-profile`
5. `make eval-recall`
6. `make eval-prompt`
7. `make eval-all`
8. `make eval-all ARGS="--record-history"`
9. `make eval-compare`

Recommended live-quality routine:

1. `make smoke`
2. `make embedding-smoke`
3. `make live-eval-memory`
4. `make live-eval-sentiment`
5. `make live-eval-all`
6. `make live-eval-all ARGS="--record-history --save-all"`
7. `make live-eval-compare`

Recommended semantic-extraction review routine:

1. `make classify ARGS="--text 'I am based in Edinburgh in the UK.' --semantic"`
2. `make classify ARGS="--text 'I am based in Edinburgh in the UK.' --extract"`
3. `make classify ARGS="--text 'The Meadows has cherry blossom trees.' --extract"`
4. `make ingest ARGS="--text 'I am based in Edinburgh in the UK.' --reply 'Noted.' --conversation-id semantic-debug"`
5. `make list ARGS="--query Edinburgh --limit 5"`
6. `make list-sidecar ARGS="--limit 10"`
7. `make profile`
8. `make scenario-run ARGS="--scenario semantic_location_candidate"`
9. `make scenario-show ARGS="--run-id <semantic_location_run_id>"`
10. `make scenario-run ARGS="--scenario semantic_location_correction_latest_wins"`
11. `make rebuild`
12. `make list-sidecar ARGS="--limit 10"`
13. `make profile`

For Phase 2, the expected qualitative result for durable user facts is an
accepted `structured_extraction`, a promoted `memory_class`, and a corresponding
sidecar/profile write. Non-durable semantic candidates such as local-world facts
should still remain out of the sidecar/profile.

Rebuild now replays the same hybrid extraction policy from the interaction log,
so `make rebuild` should preserve accepted structured extractions rather than
falling back to the older rule-only classifier behavior.

Recommended qualitative routine:

1. `make scenario-list`
2. `make scenario-run ARGS="--scenario preference_recall"`
3. `make scenario-run ARGS="--scenario conflicting_fact_latest_wins"`
4. `make scenario-show ARGS="--run-id <saved-run-id>"`
5. `make scenario-compare ARGS="--run-a <run-a> --run-b <run-b>"`

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

`make eval-classification`, `make eval-sentiment`, `make eval-profile`, `make eval-recall`, `make eval-prompt`, `make eval-all`, `make eval-history`, `make eval-compare`:
- ownership: [eval_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/eval_cli.py)
- dataset: [baseline.json](/Volumes/Media/Repository/agent_memory_v2/evals/baseline.json)
- pipeline logic under test: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)

`make live-eval-memory`, `make live-eval-sentiment`, `make live-eval-all`, `make live-eval-history`, `make live-eval-compare`:
- ownership: [live_eval_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/live_eval_cli.py)
- dataset: [live_ollama.json](/Volumes/Media/Repository/agent_memory_v2/evals/live_ollama.json)
- generation path: [ollama.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/ollama.py)
- prompt path: [pipeline.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/pipeline.py)

`make scenario-list`, `make scenario-run`, `make scenario-show`, `make scenario-compare`:
- ownership: [scenario_cli.py](/Volumes/Media/Repository/agent_memory_v2/src/agent_memory_v2/scenario_cli.py)
- dataset: [scenarios.json](/Volumes/Media/Repository/agent_memory_v2/evals/scenarios.json)
- artifact output: `artifacts/scenarios/...`

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
