# Execution Plan: agent-memory-v2 as a Claude Controller (Experiment Harness)

Status: **draft / awaiting build start**
Owner: Mark
Created: 2026-05-31
Branch (proposed): `experiment/memory-controller`

## 1. Goal of THIS plan

Get to a state where **agent-memory-v2 acts as a controller wrapped around Claude**:
it intercepts each turn, recalls + injects curated memory, calls Claude for
generation, ingests the result, and **journals everything** — across snapshot-able
session lifecycles (cold start, seeded, specific, compiled-and-resumed).

This is the **controller-side build**. The eventual comparison is
*agent-memory-v2-as-controller* **vs.** *an MCP-tool-driven memory framework*,
both driving Claude. The MCP arm is **deferred** (see §2). This plan delivers the
curated-memory controller, a no-memory control, and the data/transcript scaffolding
that the MCP arm will later reuse.

**Out of scope (deliberately):** no automated scoring, no LLM judge, no computed
conclusions. The deliverable is **clean transcripts + structured per-turn data**;
Mark owns the retrospective analysis.

## 2. Decisions locked (planning session, 2026-05-31)

| Decision | Choice | Consequence |
|---|---|---|
| What plays "Claude" | **Anthropic API model**, framework orchestrates | Reuse `ClaudeProvider`; framework stays the orchestrator; no MCP needed for this build |
| Comparison target | agent-memory-v2 controller **vs. MCP-driven memory** (later) | This plan builds the controller side + control baseline; MCP arm deferred |
| Control arm | **Yes** — no-memory baseline | Quantifies what memory buys and what it costs |
| Embeddings | **nomic-embed-text via Ollama, embeddings ONLY** | `llama3:8b` (generation) retired; Ollama still required, but only to serve recall embeddings |
| Persona | **Deferred**; keep minimal | Injection must feel like a *helper, not a personality* — see invariant in §5 |
| Evaluation | **None in-harness** — transcripts + data only | No judge model; Mark analyses retrospectively |
| Repeatability | **Fixed temperature, N iterations** | One temperature value; repeat each cell N times for spread; no temperature sweep |
| Fork vs adapt | **Adapt on a branch** | New code under `src/agent_memory_v2/experiment/`; high reuse, no fork drift |

### Why MCP is deferred (the orchestrator-inversion point)
MCP would make the `claude` CLI the orchestrator and demote agent-memory-v2 to a
subordinate tool server — the opposite of how the framework is designed (it wraps
the turn and runs inline). The future MCP arm is precisely the *contrast condition*
(agent decides its own memory over MCP) and will reuse this plan's scenario format,
journaling schema, and snapshot lifecycles. Building the controller cleanly first
makes that arm a drop-in.

## 3. The arms

| Arm | Memory control | Mechanism |
|---|---|---|
| **A — Curated controller** | Deterministic pipeline recalls + injects + ingests around every Claude turn | `recall` → `build_prompt(msg, recalled)` → `ClaudeProvider`(no tools) → `ingest_turn` |
| **C — Control (no memory)** | None | `ClaudeProvider` with base system prompt only; no store, no injection |
| **B — MCP agent memory** | *(deferred)* model manages its own memory over MCP | future; reuses this harness's scenarios + journal + lifecycles |

This plan implements **A and C**. (Arm letters kept stable for the eventual writeup.)

## 4. Reuse map (verified against source)

- `claude_provider.py` — `ClaudeProvider.next_response(instructions, input_items, tools)` over the Messages API. With empty `tools` it returns plain `output_text` — that is the generation path for A and C. `raw` carries the full API response (usage check is a Phase-0 verify item).
- `state_cli.py` — `export_state`/`import_state` zip main+sidecar+profile+interaction-log+maintenance+settings with a manifest. **Snapshot primitive** for lifecycles.
- `eval_history.py` — `git_metadata` (provenance) and history helpers if useful for run indexing.
- `pipeline.py` (verified signatures):
  - `recall(message: Message) -> list[dict]`
  - `build_prompt(message: Message, recalled: list[dict]) -> str` *(returns the assembled prompt string directly)*
  - `ingest_turn(user_message: Message, agent_message: Message) -> MemoryRecord`
  - `respond(message: Message) -> str` *(recall + build_prompt + **Ollama llama3** generate — Arm A replicates its recall/build_prompt but calls `ClaudeProvider` instead; this is the seam we exploit)*
  - `_store_memory(...)` *(needed only when hand-authoring "specific" snapshots)*
- `agent_eval.py` — not used by A/C, but its tool-loop is the template for the deferred MCP/agentic arm. Leave untouched.

## 5. Injection invariant — "helper, not personality" (from decision #1)

Arm A must not let curated memory re-characterise Claude. Constraints on the
injected block:
- Presented as **supplementary context the assistant may draw on**, not as identity
  or behavioural directives ("You are…", tone instructions, etc. are forbidden in
  the injected section).
- A short labelled section (e.g. `Known about the user (use only if relevant):`)
  placed so it reads as reference material, not a persona overlay.
- **Verify at build time** what `build_prompt` currently emits. If its existing
  framing is heavier than "helper," add a thin experiment-level prompt assembler
  that takes `recalled` + the user turn and produces the helper-framed prompt,
  rather than mutating pipeline behaviour. Record the exact framing in the manifest
  so the personality-neutrality claim is auditable.

## 6. New components (under `src/agent_memory_v2/experiment/`)

1. **`config.py`** — `experiment_config(base, root_dir)`: clone settings, isolate all
   store/profile/log paths under a per-run `root_dir`, **keep `embeddings.provider="ollama"`
   / `nomic-embed-text` / 768-dim**. (Do NOT reuse `agent_eval.isolated_agent_config` —
   it force-sets `hash`.) Generation is not configured here; it goes through `ClaudeProvider`.
2. **`controller.py`** — the wrapper that *is* the deliverable. Per user turn:
   `recall` → assemble helper-framed prompt (§5) → `ClaudeProvider.next_response(tools=[])`
   → capture response + usage → `ingest_turn` → journal. Arm C variant skips
   recall/inject/ingest entirely.
3. **`snapshots.py`** — wrap `state_cli.export_state`/`import_state` (or a fast
   `copytree` of the run root's `data/` tree): `snapshot_save(root, name)`,
   `snapshot_load(name, root)`, `snapshot_fingerprint(root)` (record counts for the
   journal). Library under `experiments/snapshots/<name>/`.
4. **`journal.py`** — JSONL per-turn writer (schema §8) + per-run `manifest.json`.
5. **`scenarios.py`** — load the new scenario format (§9).
6. **`cli.py`** — `exp-run` (one arm × scenario × lifecycle × N), `exp-matrix`
   (the grid), `exp-build-snapshots` (build seeded/specific libraries).
   **No `exp-report`** — analysis is external.

New Make targets: `make exp-run ARGS=...`, `make exp-matrix`, `make exp-build-snapshots`.

## 7. Session model & lifecycles (the snapshot/reset requirement)

**Session = a contiguous Claude context** (the `input_items` carried within it).
Memory's value shows up *across* session boundaries; discriminating turns drop the
in-context history and force reliance on the store.

| Lifecycle | Setup | What it exercises |
|---|---|---|
| **cold_start** | Fresh empty run root | Acquisition from scratch within a session |
| **seeded** | `snapshot_load("seed_<persona>")` first | Recall over a realistic background corpus |
| **specific** | `snapshot_load("specific_<scenario>")` (hand-authored minimal facts, no distractors) | Precise recall, cleanest condition |
| **compiled_resumed** | Phase 1 session builds memory → `snapshot_save` → **new session, empty in-context history** → `snapshot_load` → Phase 2 | Cross-session persistence + use of resumed memory |

`compiled_resumed` is the headline lifecycle and maps to Mark's "compiled and
resumed memories." Arm C here is the memoryless baseline.

Snapshot libraries built once by `exp-build-snapshots`:
- **seeded**: replay a persona's turns through the Arm A controller, then snapshot.
- **specific**: hand-author exact N target records via `_store_memory`, then snapshot.

## 8. Journaling schema (`journal.jsonl`, one record per turn)

```json
{
  "exp_id": "...", "arm": "A|C", "scenario": "...", "lifecycle": "...",
  "iteration": 0, "session_id": "...", "turn_index": 0, "ts": "ISO8601",
  "user_text": "...",
  "recalled_items": [ ... ],                // Arm A: raw recall() output (with scores)
  "injected_context": "...|null",           // Arm A: the helper-framed block actually injected
  "final_prompt": "...",                    // exact system+messages sent to Claude
  "response": "...",
  "usage": {"input_tokens": 0, "output_tokens": 0},  // from ClaudeProvider raw, if present
  "duration_ms": 0,
  "memory_state_after": {"main_count": 0, "sidecar_count": 0, "profile": {...}},
  "scenario_meta": { ... }                  // ground-truth labels carried verbatim for Mark's later analysis
}
```

No score field — labels travel with the data; scoring is external.

Per-run `manifest.json`: Claude model id, temperature, `nomic-embed-text` model,
taxonomy_version, settings hash, snapshot id + fingerprint, git sha (`git_metadata`),
iteration seed, **and the exact injection framing string** (for the §5 audit).

Output layout:
`artifacts/experiments/<exp_id>/<arm>/<scenario>/<lifecycle>/iter_<n>/`
→ `journal.jsonl`, `manifest.json`, `final_memory_snapshot/`.

## 9. Scenario catalog (`evals/experiment_scenarios/`, new schema)

Do **not** overload `evals/scenarios.json`. New format:
```json
{
  "name": "...", "description": "...",
  "seed": "seed_persona|null",
  "phases": [{"session": "s1", "turns": ["...", "..."], "snapshot_after": true}],
  "probes": [{"phase": "s2", "turn": 0,
              "expected_contains": [...], "expected_not_contains": [...]}],
  "ground_truth_memories": [{"key": "...", "value": "..."}]
}
```
`expected_contains` / `ground_truth_memories` are **carried into the journal as
labels for Mark's later analysis**, not auto-scored here.

Initial catalog (designed to separate curated-memory from no-memory, and to stress
the eventual MCP contrast):
- **single_fact_recall** — one fact, asked back later.
- **preference_application** — preference stated, must be *applied* implicitly later.
- **distractor_haystack** — target fact buried among irrelevant turns (recall precision; stresses A's char budget).
- **contradiction_update** — fact later updated ("actually I moved to Berlin"); stale value should not resurface.
- **cross_session_resume** — `compiled_resumed`; learned in session 1, asked in session 2.
- **no_memory_needed** — answerable without memory + a distracting stored fact (cost/leakage of always-on injection).
- **volume_growth** — long multi-turn; context/token growth curve.

## 10. Run matrix

`arms {A, C} × scenarios (7) × applicable lifecycles × iterations (N)`.
Fixed temperature (one value), N iterations per cell for spread. Pin model id,
temperature, embedding model, snapshot id, git sha in every manifest.

## 11. Confounds & controls

- **Always-on injection token cost** — Arm A pays an injection tax every turn; Arm C
  is cheapest. Captured in `usage`; it's a real property to surface, not normalise away.
- **Model sampling nondeterminism** — N iterations + fixed temperature + pinned model id.
- **Embedding nondeterminism** — nomic via Ollama is near-deterministic but Ollama
  must be running for any run; manifest records the embedding model.
- **State leakage** — unique run root + unique `session_id` per iteration; never touch
  the live `data/` dir; snapshots are copied in, never referenced in place.
- **Personality drift** — §5 invariant + manifest records exact injection framing.

## 12. Phased build

- **Phase 0 — Verify — DONE (2026-05-31):**
  - Ollama **up**, `nomic-embed-text` present → embeddings path good.
  - `ANTHROPIC_API_KEY` **not set in shell** → build is fine; live runs need it exported.
  - **`build_prompt` is heavier than "helper".** It hardcodes `"You are a helpful
    assistant with access to relevant prior memory…"`, injects sentiment
    `Response tuning:` directives via `_format_sentiment`, and folds the user input
    into one string. This violates the §5 invariant → **Arm A uses a thin
    experiment assembler** (`controller.py`) that reuses retrieval
    (`recall` + `prompt_context`) but renders a neutral reference-framed memory
    block and keeps system/user separated. We do NOT call `build_prompt`.
  - Reuse seam confirmed: `prompt_context(recalled)->{profile,factual,contextual,...}`
    already applies dedupe + char budget; the assembler formats its output.
  - `ClaudeProvider.next_response(instructions=system, input_items=[{role,content}], tools=[])`
    returns `output_text` and `raw` (raw carries Messages-API `usage`; surfaced into journal).
  - Conventions: argparse module + `pyproject [project.scripts]` + `scripts/*.sh` + Make target.
    `Message` is frozen (`role,text,timestamp,message_id,conversation_id,turn_id`).
  - `settings.yaml` already defaults embeddings to ollama/nomic/768 — `experiment_config`
    isolates paths, doesn't flip provider.
- **Phase 1 — Scaffold + Arm C:** `experiment/config.py`, `journal.py`, `manifest`,
  `controller.py` (C path), `cli.py exp-run`. Smallest end-to-end: Claude turn in,
  journaled transcript out.
- **Phase 2 — Arm A:** `recall` → helper-framed prompt → Claude → `ingest_turn`,
  with `recalled_items` + `injected_context` journaled.
- **Phase 3 — Lifecycles:** `snapshots.py` + `exp-build-snapshots` (seeded/specific)
  + two-phase `compiled_resumed`.
- **Phase 4 — Scenario catalog + matrix:** author the 7 scenarios; `exp-matrix`.
- **Phase 5 — Dry run & handoff:** run a small grid, eyeball transcripts/journals,
  document the data format so Mark can start retrospective analysis.

## 13. Remaining open question

- **Personas (deferred):** kept minimal per decision #1; revisit only if the seeded
  lifecycle needs a richer background corpus. No blocker for Phases 0–3.
