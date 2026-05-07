# Execution Plan: OpenAI Agent Eval Harness

Status: In Progress

Owner: Agent

Created: 2026-05-06

Last Updated: 2026-05-07

## Goal

Add a provider-neutral mini agent evaluation harness with OpenAI as the first API-backed agent-class provider, while preserving existing deterministic, live Ollama, and scenario eval behavior.

## Non-Goals

- Do not implement Gemini or cloud GPU providers.
- Do not replace the existing `MemoryPipeline`.
- Do not change deterministic eval semantics.
- Do not implement AIPCS schema evolution beyond reserving `memory_evolve` as unsupported.

## Context

- `AGENTS.md` routes future agents through the harness.
- `CLAUDE.md` documents the current memory pipeline and eval architecture.
- `TOOLS.md` documents current operational and debug commands.
- `docs/architecture/index.md` defines Layer 1 memory mechanics and Layer 2 agent-eval behavior.
- AIPCS evaluation planning requires an agent-class reference beyond `llama3:8b`.

## Acceptance Criteria

- Provider-neutral agent eval runner exists with fake, OpenAI, and Anthropic providers.
- Agent tools include `memory_write`, `memory_query`, `memory_inspect`, `answer`, and unsupported `memory_evolve`.
- CLI supports `run`, `run-all`, `history`, and `compare`.
- Artifacts are written under `artifacts/agent_eval/...`.
- Compact history is written under `artifacts/eval_history/agent_eval`.
- Tests cover fake provider flow, invalid tool calls, tool-call budget, unsupported evolve, artifact payload, and CLI history/run/compare behavior.
- Existing `make test` and `make eval-all` still pass.

## Plan

1. Add provider-neutral agent-eval data types and runner.
2. Add an OpenAI Responses API provider using `requests` and environment configuration.
3. Add an Anthropic Messages API provider using `requests` and environment configuration.
4. Add fake provider support for deterministic tests and local validation.
5. Add CLI, shell wrapper, Make targets, and pyproject entrypoint.
6. Add tests for runner, providers, and CLI behavior.
7. Update docs with new commands and validation expectations.
8. Run harness validation, tests, deterministic evals, and fake-provider agent eval.

## Progress Log

- 2026-05-06: Created plan while adopting the repo harness.
- 2026-05-06: Adopted template harness files, added provider-neutral agent eval runner, OpenAI provider, fake provider, CLI, Make targets, docs, and tests.
- 2026-05-07: Added Anthropic provider support after OpenAI API-key access was not available for Plus-only accounts; restored OpenAI default model to `gpt-5.1`.

## Decisions

- Use `requests` instead of the OpenAI SDK to avoid adding a dependency in the first pass.
- Keep OpenAI calls credential-gated through `OPENAI_API_KEY`.
- Keep Anthropic calls credential-gated through `ANTHROPIC_API_KEY`.
- Use `anthropic` as the preferred provider name and keep `claude` as a CLI compatibility alias.
- Keep generated artifacts local under `artifacts/`.
- Use fake providers for CI-safe tests and validation.
- Use isolated hash-backed storage for agent eval runs so provider/tool-loop behavior is separated from live Ollama embeddings.

## Validation

Commands to run before completion:

```sh
bash scripts/validate-harness.sh
make test
make eval-all
make agent-eval-run ARGS="--scenario preference_recall --provider fake --save-all"
```

Manual OpenAI validation when credentials are available:

```sh
OPENAI_API_KEY=... make agent-eval-run ARGS="--scenario preference_recall --provider openai --save-all"
OPENAI_API_KEY=... make agent-eval-all ARGS="--provider openai --record-history --save-all"
```

Manual Anthropic validation when credentials are available:

```sh
ANTHROPIC_API_KEY=... make agent-eval-run ARGS="--scenario preference_recall --provider anthropic --save-all"
ANTHROPIC_API_KEY=... make agent-eval-all ARGS="--provider anthropic --record-history --save-all"
```

## Risks

- OpenAI API shape may evolve; isolate provider parsing and keep raw responses in traces.
- Provider outputs may include sensitive user context; keep artifacts local and sanitize before publishing.
- Tool-loop behavior may vary by model; record model, latency, invalid calls, and raw provider output in artifacts.
