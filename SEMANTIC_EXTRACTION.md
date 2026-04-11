# Hybrid Durable-Memory Extraction

This note defines the next extraction-quality upgrade for `agent_memory_v2`.

The goal is to improve durable-memory capture without relying only on regex
patterns and without allowing a pure semantic model to write noisy records into
the sidecar.

## Problem

The current classifier is precise for obvious cases but misses important
variations such as:

- `I'm based in Edinburgh`
- `I usually work from home on Fridays`
- `Actually, I live in London now`

Pure semantic similarity is not enough to solve this safely because semantic
similarity does not directly answer:

1. whether a statement is durable
2. whether it belongs in the user profile
3. what exact value should be extracted
4. whether it supersedes an older value

## Proposed Architecture

Use a three-stage admission pipeline for durable memory:

1. rule-first extraction
2. semantic candidate detection
3. structured fallback extraction

The final write decision remains policy-driven.

## Stage 1: Rule-First Extraction

Keep the current rulebase for high-confidence cases:

- explicit preferences
- explicit identity facts
- explicit tasks
- obvious ephemeral turns

If a rule match has high confidence, the current behavior stands:

1. assign `memory_class`
2. assign `durable`
3. assign `profile_key`
4. assign `extracted_value`
5. decide sidecar admission

## Stage 2: Semantic Candidate Detection

When Stage 1 falls back to generic `turn`, run a semantic routing step against a
small set of prototype classes.

Initial prototype classes:

- `identity.location`
- `identity.name`
- `identity.occupation`
- `identity.origin`
- `preference.general`
- `task.general`
- `ephemeral.smalltalk`
- `ephemeral.question`
- `contextual.world_fact`

This stage does not write memory directly.

It only answers:

1. is this utterance close enough to a durable-memory family to justify further extraction?
2. which family should the extractor try first?

Expected output:

```json
{
  "candidate_class": "identity.location",
  "similarity": 0.82,
  "above_threshold": true
}
```

## Stage 3: Structured Fallback Extraction

If the semantic router says the utterance is a durable-memory candidate, run a
structured extractor.

The first implementation should use the local Ollama model in a constrained
JSON-extraction prompt.

Expected output shape:

```json
{
  "memory_class": "fact",
  "durable": true,
  "profile_key": "identity.location",
  "extracted_value": "Edinburgh",
  "confidence": 0.78,
  "supersedes_profile_key": "identity.location"
}
```

This stage is only allowed to promote a generic `turn` into durable memory if:

1. JSON is valid
2. confidence is above threshold
3. `profile_key` is one of the allowed durable keys
4. extracted value is non-empty and reasonably compact

## Sidecar Admission Policy

Sidecar writes remain strict.

A record should be admitted only if all of these are true:

1. `durable == true`
2. `memory_class` is in the sidecar allowlist
3. `confidence >= admission_threshold`
4. `profile_key` is present for profile-backed memory

Initial thresholds:

- semantic candidate threshold: `0.72`
- structured extractor admission threshold: `0.75`
- below threshold: keep in main store only

## Supersession Rules

For profile-backed memory:

- the newest accepted value wins
- older values with the same `profile_key` remain candidates for archive or sidecar compaction

Typical examples:

- `I live in Bristol`
- `Actually, I live in London now`

Desired behavior:

1. first writes `identity.location = Bristol`
2. second writes `identity.location = London`
3. profile shows only `London`
4. sidecar compaction can archive or suppress the older Bristol value

## What This Should Catch

Good candidates for the hybrid path:

- `I'm based in Edinburgh`
- `I'm located in London`
- `I usually work from home on Fridays`
- `Actually, I live in London now`
- `Weekends work best for me`

Examples that should still stay out of the sidecar by default:

- `The Meadows has cherry blossom trees`
- `Today is the first day of spring here`
- `Do you know where to find cherry blossom in Edinburgh?`

These may still be useful main-store memory, but they are not clean profile
facts.

## Implementation Phases

### Phase 1

- [x] add semantic prototype definitions
- [x] add semantic candidate router
- [x] add debug output for candidate matches
- [x] no writes yet

### Phase 2

- [x] add structured Ollama fallback extractor
- [x] run it only when rules fall through to generic `turn`
- [x] keep sidecar admission gated by confidence

### Phase 3

- [x] add rebuild support for hybrid extraction
- [x] add eval cases for missed durable facts like `I'm based in Edinburgh`
- [x] add scenario cases for naturally phrased durable facts
- [x] add scenario cases for corrections and updates

## Evaluation Plan

This feature should be measured in three layers:

1. deterministic evals
   - expected class/profile key/value for known examples
2. live Ollama evals
   - does the model answer correctly after hybrid extraction?
3. scenario review
   - does the extracted durable memory feel correct and non-noisy?

Primary quality goals:

- fewer false negatives for durable user facts
- no significant increase in false-positive sidecar admissions
- cleaner profile construction for naturally phrased user statements

## Detailed Implementation Handoff

This section is the concrete implementation plan for the next development pass.

### Current Integration Points

The current classification path is:

1. `pipeline.ingest(...)`
2. `pipeline.ingest_turn(...)`
3. `classifier.classify_text(...)`
4. `_classification_metadata(...)`
5. `_should_store_in_sidecar(...)`
6. `MemoryStore.add(...)`

Important files:

- `src/agent_memory_v2/classifier.py`
- `src/agent_memory_v2/pipeline.py`
- `src/agent_memory_v2/embeddings.py`
- `src/agent_memory_v2/config.py`
- `config/settings.yaml`
- `src/agent_memory_v2/classify_cli.py`
- `src/agent_memory_v2/eval_cli.py`
- `evals/baseline.json`
- `evals/scenarios.json`

The semantic router must initially be observable only. It should not promote
records into durable memory until the structured extractor and confidence gates
exist.

### Phase 1 Objective

Add semantic candidate routing for rule misses, without changing sidecar writes.

Phase 1 should answer:

1. if the rule classifier returns generic `turn`, does the utterance look semantically close to a durable-memory family?
2. which durable-memory family is the best candidate?
3. is the similarity above the configured candidate threshold?

Phase 1 must not:

1. change `memory_class`
2. set `durable=true`
3. write new records to the sidecar
4. alter profile contents

### Proposed New Module

Add:

- `src/agent_memory_v2/semantic_router.py`

Suggested contents:

```python
@dataclass(frozen=True)
class SemanticPrototype:
    candidate_key: str
    candidate_class: str
    examples: tuple[str, ...]
    durable_candidate: bool


@dataclass(frozen=True)
class SemanticRouteResult:
    candidate_key: str | None
    candidate_class: str | None
    score: float
    threshold: float
    above_threshold: bool
    durable_candidate: bool
    matched_example: str | None
```

Core function:

```python
def route_semantic_candidate(
    text: str,
    encoder: EmbeddingEncoder,
    *,
    threshold: float,
) -> SemanticRouteResult:
    ...
```

Implementation detail:

- encode the user utterance
- encode prototype examples
- use cosine similarity via dot product because project encoders normalize vectors
- compare against all examples
- choose the single best prototype/example pair
- return a result even when below threshold

### Initial Prototype Set

Start small and explicit.

`identity.location` examples:

- `I live in Edinburgh.`
- `I'm based in Edinburgh.`
- `I am located in London.`
- `I moved to Bristol.`
- `I'm in Manchester now.`

`identity.name` examples:

- `My name is Mark.`
- `I'm called Alex.`
- `You can call me Sam.`

`identity.occupation` examples:

- `I work as a product designer.`
- `I'm an architect.`
- `My job is software engineer.`

`identity.origin` examples:

- `I'm from Scotland.`
- `I grew up in Wales.`

`preference.general` examples:

- `I prefer oat milk.`
- `I like concise answers.`
- `My favourite editor is Neovim.`
- `Mornings work best for me.`

`task.general` examples:

- `Remind me to renew my passport.`
- `I need to file my tax return.`
- `Don't let me forget to call the dentist.`

`contextual.world_fact` examples:

- `The Meadows has cherry blossom trees.`
- `Spring starts today here.`
- `Princes Street Gardens has flowers.`

`ephemeral.question` examples:

- `What day is it today?`
- `How do I get to the park?`
- `Where can I find cherry blossom?`

The first durable candidate keys should be:

- `identity.location`
- `identity.name`
- `identity.occupation`
- `identity.origin`
- `preference.general`
- `task.general`

The first non-durable candidate keys should be:

- `contextual.world_fact`
- `ephemeral.question`

### Config Changes

Add to `config/settings.yaml`:

```yaml
semantic_router:
  enabled: true
  threshold: 0.72
  debug_metadata: true
```

Add `AppConfig.semantic_router` property in `config.py`.

### Pipeline Integration

Initial integration should be metadata-only.

In `pipeline.ingest(...)` and `pipeline.ingest_turn(...)`:

1. call `classify_text(...)` as today
2. if result is a non-durable generic `turn` or `message`
3. if `semantic_router.enabled`
4. call `route_semantic_candidate(...)`
5. add route result to metadata under:

```json
{
  "semantic_candidate": {
    "candidate_key": "identity.location",
    "candidate_class": "fact",
    "score": 0.81,
    "threshold": 0.72,
    "above_threshold": true,
    "durable_candidate": true,
    "matched_example": "I'm based in Edinburgh."
  }
}
```

Do not use this metadata for sidecar admission in Phase 1.

### CLI / Debug Surface

Extend `make classify` output so it can show semantic routing on demand.

Preferred approach:

1. keep default `classify_text(...)` behavior unchanged for low-level rule checks
2. add a new CLI flag to `classify_cli.py`:

```bash
make classify ARGS="--text 'I am based in Edinburgh' --semantic"
```

Output should include:

- rule classification
- semantic candidate result
- whether the semantic route would be considered durable candidate

This is important because the router is not yet changing storage behavior.

### Tests

Add unit tests for `semantic_router.py` using the deterministic hash encoder
only where possible.

Minimum tests:

1. returns a route result for empty/non-empty text safely
2. identifies `I'm based in Edinburgh` as closest to `identity.location`
3. identifies `The Meadows has cherry blossom trees` as `contextual.world_fact` or non-durable candidate
4. identifies `What day is it today?` as `ephemeral.question` or non-durable candidate
5. below-threshold result does not mark `above_threshold=true`

Add pipeline tests:

1. generic rule miss stores `semantic_candidate` metadata when router enabled
2. router metadata does not make the memory durable in Phase 1
3. sidecar count remains unchanged for semantic-only candidate in Phase 1

Add CLI tests:

1. `classify --semantic` includes semantic route data

### Eval / Scenario Updates

Add deterministic eval cases that are expected to remain generic in Phase 1 but
include semantic candidate metadata when tested via the new router-specific path:

- `I'm based in Edinburgh in the UK.`
- `The Meadows has cherry blossom trees.`
- `What day is it today?`

Add scenario cases:

- `semantic_location_candidate`
- `semantic_world_context_not_profile`

These should be subjective review scenarios, not strict sidecar assertions yet.

### Acceptance Criteria For Phase 1

Phase 1 is complete when:

1. [x] `make test` passes
2. [x] `make classify ARGS="--text 'I am based in Edinburgh in the UK.' --semantic"` shows `identity.location` above threshold
3. [x] ingesting `I am based in Edinburgh` stores semantic candidate metadata
4. [x] that same ingest does not write to sidecar yet
5. [x] `make scenario-run ARGS="--scenario semantic_location_candidate"` produces an artifact showing the semantic candidate

### Phase 2 Handoff Preview

Phase 2 adds:

1. [x] `structured_extractor.py`
2. [x] constrained Ollama JSON prompt
3. [x] confidence-gated promotion from semantic candidate to durable memory
4. [x] sidecar/profile admission only when extraction passes policy

Phase 2 should start with one target:

- [x] promote `I'm based in Edinburgh` into:

```json
{
  "memory_class": "fact",
  "durable": true,
  "profile_key": "identity.location",
  "extracted_value": "Edinburgh"
}
```

### Phase 2 Acceptance Criteria

Phase 2 is complete when:

1. [x] `make test` passes
2. [x] `make classify ARGS="--text 'I am based in Edinburgh in the UK.' --extract"` returns an accepted structured extraction
3. [x] accepted extraction promotes the stored record to durable `fact`
4. [x] accepted extraction writes to sidecar/profile through the existing admission path
5. [x] rejected extraction remains a generic non-durable memory with rejection metadata
6. [x] `make scenario-run ARGS="--scenario semantic_location_candidate"` produces an artifact with sidecar/profile-backed recall

### Known Risks

Semantic routing can create false confidence. Keep Phase 1 metadata-only so
that we can evaluate false positives before allowing writes.

Potential false positives:

- location questions misread as user location
- topical interests misread as durable preferences
- world facts misread as profile facts

Mitigation:

- include non-durable prototypes
- keep a relatively high threshold
- require structured extraction before sidecar writes
- evaluate with scenarios before enabling promotion
