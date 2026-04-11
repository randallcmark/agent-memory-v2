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

The current baseline supports:

- local Ollama-backed chat
- persistent memory across sessions
- multi-user profile segregation
- rule-based memory classification
- durable sidecar storage for facts and preferences
- structured multi-source recall
- derived user profile injection
- temporal context injection
- sentiment-aware response guidance
- maintenance, rebuild, backup, restore, and inspection tooling
- deterministic regression evaluation for core memory stages
- live Ollama evaluation for real memory-use and sentiment-behavior checks
- scenario-driven qualitative review with saved prompt/recall/response artifacts

## Repo Structure

- `src/agent_memory_v2/`: application code
- `config/`: runtime configuration
- `scripts/`: shell entrypoints
- `tests/`: unit and integration coverage
- `seeds/`: generic non-sensitive seed data

## Quick Start

From the project root:

```bash
make doctor
make chat
```

Useful commands:

```bash
make chat
make stats
make list
make list-sidecar
make profile
make doctor
```

For named-user operation:

```bash
make chat ARGS="--user mark"
```

## Debug and Analysis

The project also includes stage-by-stage tooling so you can inspect the pipeline
independently:

```bash
make classify
make recall
make prompt
make generate
make ingest
make scenario-run
make scenario-show
make scenario-compare
```

These tools are intended for qualitative analysis, debugging, and regression
work rather than normal end-user operation.

## Documentation

- `TOOLS.md`: operational and debug workflows
- `STATUS.md`: current implementation state
- `ROADMAP.md`: planned work and completed milestones
- `SEMANTIC_EXTRACTION.md`: concrete design for the next hybrid durable-memory extraction upgrade
- `PUBLISHING.md`: repo sanitisation and publication flow
- `skills/`: project-contained Codex skills for operations and quality workflows
- `INTERNAL_README.md`: detailed internal/operator notes preserved from active development

## Status

This project is under active development. The core local memory loop is working,
and the current focus is improving quality, evaluation, and retrieval behavior.
