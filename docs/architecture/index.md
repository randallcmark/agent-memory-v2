# Architecture

`agent-memory-v2` is a local-first persistent memory layer for LLM agents. It stores interaction memory, extracts durable facts/preferences/tasks, recalls relevant context, derives a user profile, and injects memory back into prompts.

## Runtime Pipeline

`MemoryPipeline` is the central orchestration class:

1. Classify user text with taxonomy-driven rules.
2. Route generic non-durable turns through semantic candidates when enabled.
3. Use structured extraction for durable semantic candidates.
4. Store all turns in the main FAISS-backed memory store.
5. Store durable facts/preferences/tasks in the sidecar store.
6. Update the derived profile incrementally.
7. Recall from sidecar and main stores with class, durability, recency, store-kind, and query-intent scoring.
8. Build prompts with temporal context, sentiment, profile, factual memory, and contextual memory.

## Stores

| Store | Purpose |
|---|---|
| Main memory | All ingested turns, embedded for contextual recall. |
| Sidecar memory | Durable facts, preferences, and tasks. |
| Profile | Derived JSON view of durable sidecar memory. |
| Archive | JSONL archive of pruned records before deletion. |

## Evaluation Layers

| Layer | Purpose | Commands |
|---|---|---|
| Deterministic eval | Model-independent memory mechanics using hash embeddings. | `make eval-all` |
| Live Ollama eval | End-to-end response behavior using local Ollama models. | `make live-eval-all` |
| Scenario review | Qualitative prompt, recall, and response artifacts. | `make scenario-run` |
| Agent eval | Provider-neutral tool-loop behavior for agent-class models. | `make agent-eval-run` |

## Provider Model

Ollama is the existing generation and embedding provider. The agent-eval harness adds provider-neutral agent behavior with OpenAI and Anthropic as API-backed providers. Provider code must stay behind interfaces so future Gemini or local-agent providers can be added without changing memory mechanics.

## Key Docs

- `CLAUDE.md`: detailed architecture and command reference.
- `TOOLS.md`: operator and debug workflows.
- `STATUS.md`: implementation state and known limitations.
- `docs/architecture/boundaries.md`: ownership and dependency rules.
