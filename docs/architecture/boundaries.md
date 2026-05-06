# Boundaries

These boundaries keep memory mechanics, model calls, and evaluation artifacts separable.

## Ownership

| Subsystem | Owns | Must Not Own |
|---|---|---|
| `MemoryPipeline` | Classification, storage, recall, prompt construction, profile updates. | Provider-specific API protocols or eval artifact layout. |
| Stores | FAISS indices, metadata JSON, archive/recovery primitives. | Model calls or scenario scoring. |
| Taxonomy | Durable memory keys, regex patterns, semantic examples, compaction modes. | Runtime provider selection. |
| Providers | External/local model calls and provider-specific request/response parsing. | Memory storage rules or profile compaction policy. |
| Eval CLIs | Isolated configs, scenario/eval orchestration, artifact writing, history summaries. | Core memory scoring rules beyond test harness assertions. |
| Config | Runtime settings and path resolution. | Secrets; provider keys must come from environment variables. |

## Dependency Rules

- Provider modules may depend on `requests` and standard library types.
- Core memory modules must not depend on OpenAI-specific classes.
- Agent-eval code may depend on `MemoryPipeline`, scenario loading, and eval-history helpers.
- Deterministic evals must remain free of Ollama and OpenAI dependencies.
- Unit tests must use fake providers for external model behavior.

## Artifact Rules

- Generated eval artifacts stay under `artifacts/`.
- Agent-eval artifacts use `artifacts/agent_eval/`.
- Compact agent-eval history uses `artifacts/eval_history/agent_eval/`.
- Do not commit generated artifacts unless a task explicitly asks for a curated fixture.
