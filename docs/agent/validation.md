# Validation

This file is the authoritative validation contract for `agent-memory-v2`. Run commands from the repository root.

## Harness Validation

Run after changing harness docs or validation rules:

```sh
bash scripts/validate-harness.sh
```

## Project Validation

Core regression suite:

```sh
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
make test
make eval-all
```

These Ruff commands intentionally use the pinned project version, `ruff==0.15.12`.

Live model validation when Ollama is available:

```sh
make doctor
make live-eval-all ARGS="--record-history --save-all"
```

Agent-eval validation after agent harness changes:

```sh
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

## Validation Standard

- Run the narrowest relevant tests during implementation.
- Run `make test` and `make eval-all` before completion when feasible.
- Use fake providers for CI-safe agent harness checks.
- Do not require OpenAI, Anthropic, or Ollama credentials for deterministic CI.
- If validation cannot run, report why and record persistent gaps in `docs/quality/technical-debt.md`.
