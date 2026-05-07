# Contributing to agent-memory-v2

## Getting started

```bash
git clone https://github.com/<your-fork>/agent-memory-v2
cd agent-memory-v2
pip install -e '.[dev]'          # base install
pip install -e '.[dev,embeddings]'  # adds sentence-transformers
```

## Running tests

```bash
make test                        # full suite (unit + integration)
bash scripts/test.sh -k "name"   # single test by name
```

The test suite uses a deterministic hash encoder — no Ollama required. CI runs `make test` then `make eval-all`.

## Linting

[Ruff](https://docs.astral.sh/ruff/) is the only linter:

```bash
python -m ruff check src/ tests/
python -m ruff check --fix src/ tests/
```

Pre-commit will run this automatically if you install the hooks:

```bash
pip install pre-commit
pre-commit install
```

## Running evals

```bash
make eval-all          # deterministic, no Ollama required
make agent-eval-all ARGS="--provider fake --record-history --save-all"
```

See `CLAUDE.md` for the full eval command reference.

## Making a change

1. Fork the repo and create a branch from `main`.
2. Write or update tests — the test suite must stay green.
3. Run `python -m ruff check --fix src/ tests/` before committing.
4. Open a pull request using the template. Include a short description of *why* the change is needed, not just what changed.

## Code style

- Python 3.12+, `from __future__ import annotations` in every module.
- All public symbols must appear in the module's `__all__`.
- Prefer specific exception types over bare `except Exception`.
- No comments that describe *what* code does — only *why* when the reason is non-obvious.
- New modules need a one-line module-level docstring.

## Taxonomy changes

If you add or modify entries in `config/taxonomy.yaml`, run:

```bash
make rebuild    # re-classifies and re-embeds all stored records
make eval-all   # verify eval pass rates haven't regressed
```

## Questions

Open a GitHub issue tagged `question`. For security issues, see [SECURITY.md](SECURITY.md) if present, otherwise email the maintainer directly.
