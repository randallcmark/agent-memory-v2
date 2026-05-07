# Agent Memory V2

`agent_memory_v2` is a local-first persistent memory layer for an LLM agent.

The project objective is to make a locally running model more useful across
sessions by giving it a structured memory system that can:

1. process interactions,
2. extract durable user information,
3. persist memory across sessions,
4. retrieve relevant prior memory,
5. inject that memory back into future prompts.

The current runtime target is a fully local stack built around:

- `llama3:8b` via Ollama for generation
- `nomic-embed-text` via Ollama for embeddings
- FAISS for vector storage

## Why This Exists

Most local chat setups are stateless. Once a session ends, the agent forgets
preferences, facts, tasks, and prior context.

This project is an attempt to fix that with a practical memory architecture for
local agents:

- conversation memory for recent contextual recall
- durable fact/preference storage for long-lived user data
- profile derivation from recalled durable memory
- prompt injection with temporal grounding and sentiment-aware tuning
- maintenance flows for aging, pruning, and rebuilds

## Current Capabilities

- local Ollama-backed chat with persistent memory across sessions
- multi-user profile segregation
- rule-based memory classification with hybrid semantic/structured extraction fallback
- durable sidecar storage for facts, preferences, and tasks
- structured multi-source recall with class, durability, and recency ranking
- per-user timezone and communication preference support
- derived user profile injection into every prompt
- temporal context and sentiment-aware response guidance
- Ollama resilience: retry with exponential backoff, graceful degradation on failure
- character-budget-gated prompt construction
- maintenance, rebuild, backup, restore, and archive inspection tooling
- schema versioning with load-time migration warnings
- deterministic regression evaluation (no Ollama required)
- live Ollama evaluation for real memory-use and sentiment-behavior checks
- scenario-driven qualitative review with saved prompt/recall/response artifacts
- provider-neutral agent tool-loop evaluation with fake, OpenAI, and Anthropic providers
- CI via GitHub Actions on every push and pull request to main

## Quick Start

```bash
pip install -e '.[dev]'
make doctor
make chat
```

Agent and maintainer routing starts in `AGENTS.md`.

For named-user operation:

```bash
make chat ARGS="--user mark"
```

Useful inspection commands:

```bash
make stats
make list
make list-sidecar
make profile
```

## Debug and Analysis

Stage-by-stage tooling for inspecting the pipeline independently:

```bash
make classify ARGS="--text 'I am based in Edinburgh.' --extract"
make recall ARGS="--text 'Where do I live?'"
make prompt ARGS="--text 'Where do I live?'"
make ingest ARGS="--text 'I prefer oat milk.' --reply 'Noted.'"
make scenario-run
make scenario-show
```

## Repo Structure

- `src/agent_memory_v2/`: application source
- `config/`: runtime configuration (`settings.yaml`, `taxonomy.yaml`)
- `scripts/`: shell entrypoints
- `tests/`: unit and integration tests
- `evals/`: deterministic and live eval datasets
- `seeds/`: generic non-sensitive seed data
- `docs/`: agent operating harness, architecture notes, execution plans, and quality records

## Documentation

- `AGENTS.md`: short routing map for agents and maintainers
- `CLAUDE.md`: architecture and command reference for Claude Code sessions
- `TOOLS.md`: operational and debug workflows with stage-by-stage traceability
- `STATUS.md`: current implementation state and verified commands
- `ROADMAP.md`: planned work and completed milestones
- `INTERNAL_README.md`: detailed internal/operator notes
- `PUBLISHING.md`: repo sanitisation and publication flow
