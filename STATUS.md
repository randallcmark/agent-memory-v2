# Agent Memory V2 Status

## Last Updated
2026-03-29

## Summary

V2 has reached a working operational baseline:

- Ollama connectivity is verified
- interactive chat works
- end-to-end memory recall works
- temporal prompt context is restored
- first-pass memory classification is active
- admin tooling is available
- Ollama-backed embeddings are installed and verified via `nomic-embed-text`
- embedding/store dimensions are now synchronized at `768`
- rebuild now reclassifies historical logs using current rules
- classification coverage now includes a wider set of preference, fact, and task patterns
- recall ranking now includes class priority plus recency weighting
- admin inspection now supports filtering by class, keyword, and date range
- durable memories are now distinguished from ephemeral dialogue in classification, stats, and recall scoring
- independent analysis tooling now exists for ingest, classify, recall, prompt build, raw generate, and state backup/restore
- operational and debug workflows are now documented separately in `TOOLS.md`, with direct module traceability for each stage
- a dedicated sidecar store now persists durable preference/fact memories separately from generic turn memory
- recall and prompt construction now use structured multi-source merge, separating durable facts from conversational context
- a derived user profile is now rebuilt from sidecar memories and injected as a separate prompt section
- prompt construction now suppresses durable-fact duplicates already represented in the user profile
- prompt construction now drops low-value ephemeral context when profile or durable-fact grounding is sufficient
- `make prompt` now exposes the selected prompt context and dropped contextual items for direct debugging
- a first aging-policy pass now exists: recall scoring includes age decay, and admin tooling exposes `make aging-report` plus `make prune-dry-run`
- conservative main-store pruning now exists for stale ephemeral turn memories via `make prune`
- pruned records are now archived before removal using the configured prune archive path
- deferred maintenance now has persisted state, a lock file, a status command, and a separate runner command
- completed turns now mark maintenance state and surface a non-blocking `maintenance due` notice in the chat loop
- maintenance now resolves completed tasks by pruning matched task memories after their resolution grace period
- maintenance now expires stale unresolved tasks by policy
- maintenance now compacts the sidecar by pruning superseded entries for the same durable `profile_key`
- chat startup now performs a maintenance check and can run deferred maintenance before entering the interaction loop
- the consolidated `scripts/admin.sh` entrypoint now routes admin, maintenance, backup/restore, and config helper tasks
- the system now supports catchall-by-default storage plus explicit named-user segregation via `--user` or `AGENT_MEMORY_V2_USER`
- prompt construction now includes an explicit sentiment signal and response-tuning guidance derived from the current user utterance
- generic publish-safe seed data and a dedicated seed loader now exist for demos and GitHub-safe reseeding
- a sanitiser command and a documented publication workflow now exist for preparing the repo for GitHub

Milestones 1 through 5 are now complete. The first prompt-strategy quality
upgrade is also complete. The default embedding provider is now
the local Ollama embedding model rather than the deterministic hash baseline,
rebuild is safe to use after provider changes, recall ordering is more robust
for fresh classified memories, and the store is easier to inspect during
debugging.

## Verified Commands

```bash
make test
make smoke
make smoke-generate
make embedding-smoke
make preflight
make e2e
make chat
make ingest
make classify
make recall
make prompt
make generate
make backup
make restore
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

## Current Risks

- Existing historical records that do not express a clear durable fact still remain generic `turn` memories after rebuild, which is now expected behavior rather than a schema gap
- Stored assistant text can still contain some residual prompt boilerplate
- Classification is still intentionally rule-based and will miss facts/preferences phrased outside the current patterns
- Memory aging is still partial: task resolution/expiry and sidecar compaction now exist, but broader archive workflows beyond prune archives are not implemented yet
- Multi-user support currently exists at the storage/profile/admin-routing layer; broader user selection coverage across every helper entrypoint still depends on passing `--user` or `AGENT_MEMORY_V2_USER`
- The GitHub workflow is prepared and documented, but no remote push has been performed from this session
