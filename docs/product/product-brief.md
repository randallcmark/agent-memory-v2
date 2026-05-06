# Product Brief

`agent-memory-v2` is a local-first persistent memory layer for LLM agents.

The project objective is to make a local or agent-backed assistant useful across sessions by giving it a structured memory system that can:

1. process interactions,
2. extract durable user information,
3. persist memory across sessions,
4. retrieve relevant prior memory,
5. inject that memory back into future prompts,
6. evaluate memory behavior with reproducible artifacts.

## Current Direction

The current baseline is a fixed memory system with a developer-defined taxonomy, dual stores, a derived user profile, and deterministic/live/scenario evals.

The next research direction is an agent-eval harness that lets stronger instruction-following models operate memory through tools. This supports comparison with AIPCS-style memory, where the agent has more control over what and when to store.

## Non-Goals

- This repo is not a consumer UI product.
- This repo does not currently implement full AIPCS schema autonomy.
- OpenAI-backed evals are optional and credential-gated; deterministic tests must remain runnable without paid APIs.
