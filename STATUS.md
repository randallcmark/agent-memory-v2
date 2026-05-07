# Agent Memory V2 Status

## Last Updated
2026-05-06

## Summary

All milestones through the second structural review are now complete. The system
has a working local memory loop, a deterministic test and eval harness, and
GitHub Actions CI running on every push and PR to main.

### Implemented

- Ollama connectivity, interactive chat, and persistent FAISS-backed memory
- Turn-aware ingestion storing `User: … Agent: …` full turn text for richer embedding
- First-pass rule-based classification: preference, fact, task, ephemeral turn
- Hybrid extraction: semantic router for rule misses → constrained Ollama JSON extractor for durable candidates
- Dual-store layout: main store (all turns) + sidecar (durable facts/preferences/tasks)
- Multi-source recall with class priority, durability, recency, and query-intent bonuses
- Derived user profile injected as a structured prompt section
- Temporal context and per-user timezone via `preference.timezone` profile key
- Sentiment detection with negation suppression and expanded keyword coverage
- Character-budget-gated prompt construction (`max_context_chars: 3200`)
- Aging-aware recall scoring with `last_recalled_at` as effective age baseline
- Pruning with archive-before-delete; `inspect-archive` and `restore-from-archive` admin commands
- Deferred maintenance runner: prune, sidecar compaction, task resolution/expiry, profile rebuild
- Incremental profile update (`update_from_record`) as O(1) hot path
- Schema versioning stamped on every record; load-time warning for stale records
- Versioned taxonomy (`config/taxonomy.yaml`) as the single source of truth for all memory keys
  - Drives: regex fact patterns, semantic prototype examples, allowed profile keys, compaction mode
  - Additive keys accumulate `all_values` lists; scalar keys overwrite
- Ollama resilience: `_with_retry` with exponential backoff; graceful degradation in `respond()` and structured extraction
- Multi-user profile segregation via `--user` / `AGENT_MEMORY_V2_USER`
- Deterministic regression eval harness (forces hash embeddings, no Ollama required)
- Live Ollama eval layer with failure artifact capture
- Scenario-driven qualitative review with saved artifact bundles
- Provider-neutral agent tool-loop eval harness with fake, OpenAI, and Anthropic providers
- Agent operating harness with resumable execution plans under `docs/exec-plans/`
- GitHub Actions CI: `make test` + `make eval-all` on every push/PR to main
- `CLAUDE.md` with full architecture guidance for Claude Code sessions

### Verified Commands

```bash
make test
make eval-all
make eval-classification
make eval-semantic
make eval-sentiment
make eval-profile
make eval-recall
make eval-prompt
make eval-history
make eval-compare
make chat
make classify ARGS="--text 'I am based in Edinburgh in the UK.' --semantic"
make classify ARGS="--text 'I am based in Edinburgh in the UK.' --extract"
make recall ARGS="--text 'Where do I live?'"
make prompt ARGS="--text 'Where do I live?'"
make ingest ARGS="--text 'I prefer oat milk.' --reply 'Noted.'"
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
make rebuild
make reset
make backup
make restore
make doctor
make check-schema
make scenario-list
make scenario-run
make scenario-show
make scenario-compare
make live-eval-memory
make live-eval-sentiment
make live-eval-all
make agent-eval-run ARGS="--scenario preference_recall --provider fake --save-all"
bash scripts/validate-harness.sh
```

### Known Limitations

- Classification is still rule-based; facts phrased outside current patterns fall to generic `turn` storage. Hybrid extraction partially compensates for durable facts.
- Structured extraction depends on `llama3:8b` returning valid constrained JSON; malformed or low-confidence outputs leave the original memory non-durable (rejection is traceable in metadata).
- Stored historical turn text can contain residual assistant boilerplate from early sessions before prompt cleanup was in place.
- Multi-user support exists at the storage/profile/admin-routing layer; some helper entrypoints still require `--user` or `AGENT_MEMORY_V2_USER` to target a non-default profile.
- Qualitative review is scenario-driven and manual; there is no annotation or reviewer-notes workflow on top of saved artifacts.
- OpenAI agent evals require `OPENAI_API_KEY` and are not part of CI-safe validation.
- Anthropic agent evals require `ANTHROPIC_API_KEY` and are not part of CI-safe validation.
