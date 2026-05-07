# Agent Memory V2 Roadmap

## Completed

### Core Memory Loop
- [x] Local Ollama generation with `llama3:8b`
- [x] Interactive CLI chat
- [x] Persistent FAISS-backed memory store
- [x] Turn-aware ingestion (full `User: … Agent: …` text embedded for richer recall)
- [x] Recall and prompt injection with temporal grounding

### Classification and Extraction
- [x] Rule-based memory classification: preference, fact, task, ephemeral turn
- [x] Hybrid extraction: semantic router for rule misses → constrained Ollama JSON extractor for durable candidates
- [x] Rebuild replays hybrid extraction from interaction logs
- [x] Versioned taxonomy (`config/taxonomy.yaml`) as single source of truth for all memory keys
  - Drives regex fact patterns, semantic prototype examples, allowed profile keys, compaction mode

### Dual-Store and Profile Layer
- [x] Sidecar store for durable facts/preferences/tasks
- [x] Multi-source recall merge (factual vs contextual, with class/durability/recency ranking)
- [x] Query-intent recall bonus (semantic router on query boosts matching profile-key records)
- [x] Derived user profile injected as a structured prompt section
- [x] Additive profile compaction for set-valued keys (allergies, relationships, etc.)
- [x] Incremental profile update as O(1) hot path; full rebuild reserved for maintenance

### Prompt Construction
- [x] Profile, temporal context, sentiment, factual, and contextual prompt sections
- [x] Character-budget gate (`max_context_chars`) after count-based slicing
- [x] Per-user timezone from `preference.timezone` profile key
- [x] Sentiment detection with negation suppression and expanded keyword coverage

### Resilience and Versioning
- [x] Ollama retry with exponential backoff (`_with_retry`, 3 attempts)
- [x] Graceful degradation in `respond()` (fallback string) and structured extraction (`{}`)
- [x] Schema versioning stamped on every record; load-time warning for stale records

### Aging and Maintenance
- [x] Age-aware recall scoring with `last_recalled_at` as effective age baseline
- [x] Recall-count protection for frequently recalled memories
- [x] Ephemeral turn pruning with archive-before-delete
- [x] `inspect-archive` and `restore-from-archive` admin commands
- [x] Task resolution and expiry by policy
- [x] Sidecar compaction (latest per profile key)
- [x] Deferred maintenance runner with persisted state and locking
- [x] Startup maintenance check; auto-runs maintenance if due

### Operations and Tooling
- [x] Multi-user profile segregation
- [x] Consolidated `scripts/admin.sh` entrypoint
- [x] Backup, restore, seed, and sanitise-for-publish tooling
- [x] Schema check command (`make check-schema`)
- [x] Stage-by-stage debug tools: classify, recall, prompt, generate, ingest

### Evaluation and Quality
- [x] Deterministic eval harness: classification, semantic routing, sentiment, profile, recall, prompt
- [x] Eval history and score-trend tracking
- [x] Live Ollama eval layer with failure artifact capture
- [x] Scenario-driven qualitative review with saved artifact bundles
- [x] Provider-neutral agent tool-loop eval harness with fake, OpenAI, and Anthropic providers
- [x] Agent operating harness with resumable execution plans
- [x] GitHub Actions CI: `make test` + `make eval-all` on push/PR to main

---

## Remaining / Planned

### Quality and Retrieval
- [ ] Reviewer notes or annotation workflow on top of saved scenario artifacts
- [ ] Richer qualitative evaluation UI beyond scenario compare

### Architecture
- [ ] Warm start (preload profile/recent sidecar into context at session open)
- [ ] Rich extractor pipeline (multi-step, multi-model extraction strategies)
- [ ] Advanced migration tooling (automated schema version upgrades, not just warnings)

### Parity with V1 Ambitions
- [ ] Prompt profile system (per-user system prompt tuning)
- [ ] Runtime multi-provider abstraction beyond Ollama
