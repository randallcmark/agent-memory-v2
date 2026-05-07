# Tools and Workflows

This project supports two distinct usage modes:

1. **Operational flow** — normal use with chat, maintenance, and backup
2. **Debug flow** — stage-by-stage analysis and regression evaluation

The consolidated operator entrypoint is:

```bash
bash scripts/admin.sh <command> [args...]
```

Scope commands to a named user with `--user`:

```bash
bash scripts/admin.sh --user mark stats
```

---

## Operational Flow

### Health and setup

```bash
make doctor
make smoke
make smoke-generate
make embedding-smoke
make preflight
```

### Normal operation

```bash
make chat
make chat ARGS="--user mark"
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

### State management

```bash
make backup ARGS="--output backups/state-latest.zip"
make restore ARGS="--input backups/state-latest.zip --force"
make rebuild         # full replay of interaction logs (reclassifies + re-extracts)
make rebuild-profile # rebuild profile from current sidecar records only
make reset           # wipe live store (destructive)
make check-schema    # print schema-version summary for live stores
```

### Archive inspection and recovery

```bash
bash scripts/admin.sh inspect-archive
bash scripts/admin.sh restore-from-archive --since 2026-01-01
bash scripts/admin.sh restore-from-archive --memory-class fact --sidecar
```

### Recommended operational routine

1. `make doctor`
2. `make backup ARGS="--output backups/pre-session.zip"`
3. `make chat`
4. `make stats`
5. `make backup ARGS="--output backups/post-session.zip"`

---

## Debug Flow

### Seed data without chat

```bash
make ingest ARGS="--text 'I prefer oat milk.' --reply 'Noted.'"
make ingest ARGS="--text 'I am based in Edinburgh.' --reply 'Noted.' --conversation-id debug"
```

### Classification

```bash
make classify ARGS="--text 'I prefer oat milk.'"
```

Classification plus semantic candidate routing (for rule misses):

```bash
make classify ARGS="--text 'I am based in Edinburgh in the UK.' --semantic"
make classify ARGS="--text 'The Meadows has cherry blossom trees.' --semantic"
make classify ARGS="--text 'What day is it today?' --semantic"
```

The semantic path reports the best `semantic_candidate` for generic non-durable
rule-classifier misses.

### Structured extraction

```bash
make classify ARGS="--text 'I am based in Edinburgh in the UK.' --extract"
```

`--extract` runs the Phase 2 constrained JSON extractor after semantic routing finds
an above-threshold durable candidate. Records are promoted into the sidecar/profile
only when the extractor returns valid JSON, a supported `profile_key`, a compact
`extracted_value`, and confidence above the admission threshold. Rejected attempts
remain traceable in `structured_extraction` metadata.

### Recall

```bash
make recall ARGS="--text 'What do I prefer?'"
```

Returns:
1. `factual`: durable sidecar results
2. `contextual`: turn-memory results
3. `merged`: final combined ordering used by the pipeline

### Prompt assembly

```bash
make prompt ARGS="--text 'What do I prefer?'"
```

Returns the assembled prompt plus a `prompt_context` object showing:
1. `profile`: the derived profile considered for injection
2. `factual`: factual items that survived prompt filtering and the character budget
3. `contextual`: contextual items that survived filtering
4. `dropped_contextual`: contextual items recalled but intentionally not injected
5. `char_budget_applied`: whether the character budget truncated any items
6. `sentiment`: the explicit input sentiment signal and response-tuning guidance

### Raw generation

```bash
make generate ARGS="--prompt 'Reply with exactly OK.'"
```

---

## Evaluation

### Deterministic (no Ollama required)

These run against isolated temporary storage with hash embeddings:

```bash
make eval-classification
make eval-semantic
make eval-sentiment
make eval-profile
make eval-recall
make eval-prompt
make eval-all
make eval-all ARGS="--record-history"
make eval-history
make eval-compare
```

### Live Ollama evaluation

These run against the local Ollama stack with isolated temporary storage:

```bash
make live-eval-memory
make live-eval-sentiment
make live-eval-all
make live-eval-all ARGS="--record-history --save-all"
make live-eval-history
make live-eval-compare
```

Artifacts for failed (or all, with `--save-all`) cases are written under
`artifacts/live_eval/...` and include: recalled items, prompt context,
final prompt, and raw model response.

### Scenario-driven qualitative review

```bash
make scenario-list
make scenario-run ARGS="--scenario negative_sentiment_preference"
make scenario-run ARGS="--scenario semantic_location_candidate"
make scenario-run ARGS="--scenario semantic_location_correction_latest_wins"
make scenario-show ARGS="--run-id <run-id>"
make scenario-compare ARGS="--run-a <run-a> --run-b <run-b>"
```

Each scenario run stores a bundle under `artifacts/scenarios/...` containing:
setup-turn ingestion details, merged recall output, prompt context, final prompt,
live model response, and scenario review notes.

### Agent tool-loop evaluation

These run scenarios through a provider-neutral agent harness with memory tools.
The fake provider is deterministic and safe for local validation:

```bash
make agent-eval-run ARGS="--scenario preference_recall --provider fake --save-all"
make agent-eval-all ARGS="--provider fake --record-history --save-all"
make agent-eval-history
make agent-eval-compare
```

OpenAI-backed runs require `OPENAI_API_KEY`:

```bash
OPENAI_API_KEY=... make agent-eval-run ARGS="--scenario preference_recall --provider openai --save-all"
OPENAI_API_KEY=... make agent-eval-all ARGS="--provider openai --record-history --save-all"
```

Anthropic-backed Claude runs require `ANTHROPIC_API_KEY`. Prefer the provider name
`anthropic`; `claude` is retained as a compatibility alias:

```bash
ANTHROPIC_API_KEY=... make agent-eval-run ARGS="--scenario preference_recall --provider anthropic --save-all"
ANTHROPIC_API_KEY=... make agent-eval-all ARGS="--provider anthropic --record-history --save-all"
```

Artifacts are written under `artifacts/agent_eval/...` and include provider
metadata, model name, tool calls, tool results, final answer, invalid-call
count, memory state, latency, and git metadata. Compact history is written under
`artifacts/eval_history/agent_eval`.

---

## Recommended Routines

**Regression after code changes:**
1. `make test`
2. `make eval-all`
3. `make eval-all ARGS="--record-history"`
4. `make eval-compare`

**Taxonomy or classification changes:**
1. `make test`
2. `make eval-all`
3. `make rebuild`
4. `make profile`
5. `make scenario-run ARGS="--scenario semantic_location_candidate"`

**Semantic extraction debug:**
1. `make classify ARGS="--text '...' --semantic"`
2. `make classify ARGS="--text '...' --extract"`
3. `make ingest ARGS="--text '...' --reply 'Noted.' --conversation-id debug"`
4. `make list ARGS="--query <keyword>"`
5. `make list-sidecar`
6. `make profile`

**Recall problem debug:**
1. `make recall ARGS="--text '...'"`
2. Inspect `score`, `rank_score`, `memory_class`, `durable`, `age_penalty`
3. `make list ARGS="--query <keyword>"`
4. `make aging-report`
5. `make prune-dry-run`
6. `make prompt ARGS="--text '...'"`

---

## Traceability

| Command | Owning module |
|---|---|
| `make smoke`, `make smoke-generate` | `ollama.py`, `ollama_smoke.py` |
| `make embedding-smoke` | `ollama.py`, `embeddings.py`, `embedding_smoke.py` |
| `make classify` | `classifier.py`, `semantic_router.py`, `structured_extractor.py`, `classify_cli.py` |
| `make ingest` | `pipeline.py`, `models.py`, `ingest_cli.py` |
| `make recall` | `pipeline.py`, `store.py`, `recall_cli.py` |
| `make prompt` | `pipeline.py`, `sentiment.py`, `prompt_cli.py` |
| `make generate` | `ollama.py`, `generate_cli.py` |
| `make eval-*` | `eval_cli.py`, `evals/baseline.json`, `pipeline.py` |
| `make live-eval-*` | `live_eval_cli.py`, `evals/live_ollama.json`, `ollama.py` |
| `make scenario-*` | `scenario_cli.py`, `evals/scenarios.json` |
| `make agent-eval-*` | `agent_eval_cli.py`, `agent_eval.py`, `openai_provider.py`, `evals/scenarios.json` |
| `make stats`, `make list`, `make rebuild`, `make reset` | `admin.py`, `store.py` |
| `make aging-report`, `make prune-dry-run`, `make prune` | `aging.py`, `admin.py` |
| `make maintenance-status`, `make maintain` | `maintenance.py`, `pipeline.py` |
| `make list-sidecar` | `admin.py`, `store.py` |
| `make profile`, `make rebuild-profile` | `profile.py`, `admin.py`, `pipeline.py` |
| `make check-schema` | `admin.py`, `store.py`, `models.py` |
| `inspect-archive`, `restore-from-archive` | `admin.py` |
| `make backup`, `make restore` | `state_cli.py` |
| `make seed`, `make sanitize-publish` | `seed_cli.py`, `sanitise_cli.py` |
| `make chat` | `pipeline.py` |
| `make e2e`, `make doctor` | `e2e_smoke.py`, `doctor.py` |

---

## Interpreting Output

**Recall scoring fields:**
- `score`: raw cosine similarity
- `rank_score`: score after class, durability, recency, and query-intent adjustments
- `memory_class`: `preference`, `fact`, `task`, `turn`, or `message`
- `durable`: whether the memory is treated as lasting
- `store_kind`: `sidecar_memory` or `turn_memory`
- `age_penalty`: recall penalty for older memories
- `recency_bonus`: recall bonus for freshness
- `recall_count`: how many times this record has been recalled

**Profile fields:**
- `preferences`, `facts`, `tasks`: the three profile sections
- Additive keys (e.g. `identity.allergy`) include `all_values` alongside the latest `value`

**Prompt context fields:**
- `prompt_context.factual`: durable items selected for the prompt
- `prompt_context.contextual`: conversational items selected
- `prompt_context.dropped_contextual`: recalled but excluded items
- `prompt_context.char_budget_applied`: true if character budget truncated results
