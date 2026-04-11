# Agent Memory V2 Roadmap

## Current Baseline

- [x] Local Ollama generation works with `llama3:8b`
- [x] Interactive CLI works
- [x] Persistent FAISS-backed memory store works
- [x] Turn-aware memory ingestion works
- [x] Recall and prompt injection work
- [x] Global temporal context is injected
- [x] Per-memory temporal context is injected
- [x] First-pass rule-based memory classification is implemented
- [x] Admin tooling exists for stats, list, reset, rebuild, doctor

## Must-Have For Useful Beta

### Milestone 1: Real Embeddings
- [x] Implement a real local embedding provider path
- [x] Add embedding-model health tooling
- [x] Verify the chosen embedding path against the local machine setup
- [x] Replace the hash encoder as the default production path

### Milestone 2: Better Rebuild and Reclassification
- [x] Rebuild old interaction logs using current classification rules
- [x] Ensure rebuild preserves or refreshes classification metadata
- [x] Add regression tests for rebuild behavior

### Milestone 3: Better Classification Coverage
- [x] Add first-pass classes: `preference`, `fact`, `task`, `turn`
- [x] Expand pattern coverage for more user facts and preferences
- [x] Distinguish durable facts from ephemeral dialogue more reliably

### Milestone 4: Better Recall Ranking
- [x] Prefer classified memories over generic turns when scores are close
- [x] Add more nuanced ranking weights and recency handling
- [x] Add ranking regression fixtures

### Milestone 5: Better Admin and Query Tooling
- [x] Add stats/list/reset/rebuild/doctor commands
- [x] Add filtering by memory class
- [x] Add query-by-keyword and date-range inspection

## Next Quality Upgrades

- [x] Sidecar factual store
- [x] Multi-source recall merge
- [x] User profile layer
- [x] Better prompt strategy separation by memory class
- [x] Memory aging and pruning policies
- [x] Evaluation harness for recall-quality comparisons
- [ ] Hybrid durable-memory extraction: rules first, semantic routing second, structured fallback extraction third

Evaluation harness progress:
- [x] Checked-in generic eval dataset
- [x] Stage runners for classification, sentiment, profile, recall, and prompt behavior
- [x] Aggregate `eval-all` runner
- [x] Machine-readable JSON output
- [x] Baseline history and score trend tracking over time

Live Ollama evaluation progress:
- [x] Checked-in live Ollama eval dataset
- [x] Live memory-use eval runner
- [x] Live sentiment-behavior eval runner
- [x] Aggregate `live-eval-all` runner
- [x] Failure artifact capture for recalled items, final prompt, and model response

Qualitative workflow progress:
- [x] Checked-in scenario dataset for subjective review
- [x] Scenario runner with saved artifact bundles
- [x] Scenario show/compare tooling
- [ ] Reviewer notes or annotation workflow on top of saved scenario artifacts

Hybrid extraction progress:
- [x] Prototype durable-memory semantic classes
- [x] Semantic candidate router for rule misses
- [x] Metadata-only semantic debug output for `make classify --semantic`
- [x] Unit and pipeline coverage proving semantic candidates do not write to sidecar in Phase 1
- [x] Constrained Ollama fallback extractor for durable-memory candidates
- [x] Sidecar admission gating by extracted confidence
- [x] Rebuild support for hybrid extraction
- [x] Scenario coverage for naturally phrased durable facts such as `I'm based in Edinburgh`
- [x] Deterministic eval coverage for semantic candidate metadata

Current aging progress:
- [x] Age-aware recall scoring
- [x] Aging report tooling
- [x] Prune dry-run tooling
- [x] Destructive prune for stale ephemeral turn memories
- [x] Archive-before-delete for pruned ephemeral turn memories
- [x] Deferred maintenance runner with persisted state and locking
- [ ] Archive actions
- [x] Task resolution or expiry actions
- [x] Sidecar aging strategy

## Additional Requirements And Recommended Phase

### Immediate Follow-Up: Operational Hardening

- [x] Startup maintenance check and age-related tasks
  - reason: this is a direct extension of the maintenance work already in place
  - scope:
    - run `maintenance-status` during startup paths
    - optionally trigger or recommend `maintain` when overdue
    - ensure non-chat entrypoints also surface overdue maintenance

- [x] Consolidated admin shell entrypoint
  - reason: this is a tooling consolidation task, not a model/architecture task
  - scope:
    - provide a single operator-facing admin script instead of relying primarily on `make <command>`
    - route stats, list, reset, rebuild, aging, maintenance, and profile commands through that entrypoint

### Next Quality Upgrade: Multi-User Segregation

- [x] Multi-user profile selection and storage segregation
  - reason: this changes storage layout, profile lookup, prompt injection, admin tooling, and startup behavior
  - scope:
    - explicit user/profile selection on start
    - catchall/default profile when no user is specified
    - per-user memory, sidecar, profile, and maintenance state segregation
    - admin/debug commands that target a chosen user profile

### After Multi-User: Sentiment-Aware Response Tuning

- [x] Input sentiment detection and prompt/response tuning
  - reason: this is best added after multi-user support so sentiment signals can be inspected per user/session
  - scope:
    - detect user sentiment on input
    - pass sentiment signal into prompt construction
    - add regression/evaluation coverage so the behavior is measurable rather than subjective
  - note:
    - this should not rely only on vague system-prompt wording; it needs an explicit signal path and tests

### Operational Publishing And Repo Hygiene

- [x] Generic seed dataset for demos, smoke tests, and qualitative evaluation
  - reason: this supports both the evaluation harness and GitHub publication
  - scope:
    - clean, non-sensitive seed interactions
    - reusable fixtures for recall/prompt/profile/maintenance testing

- [x] Sanitiser script for repository publication
  - reason: this should exist before pushing operational state or seeded data to GitHub
  - scope:
    - strip or rewrite sensitive local state
    - prepare publish-safe fixtures and example data
    - verify no user-specific/private runtime files are staged

- [x] GitHub publication and maintenance workflow
  - reason: this depends on the sanitiser and generic seed data
  - scope:
    - publish v2 to the target GitHub repository
    - preserve a clean maintenance workflow from the repo
    - document the safe push/update path

## Later Parity-With-V1 Ambitions

- [ ] Warm start
- [ ] Rich extractor pipeline
- [ ] Prompt profile system
- [ ] Advanced operational and migration tooling
- [x] Baseline history and score trend tracking over time
