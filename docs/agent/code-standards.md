# Code Standards

These rules apply to all changes in this repository. They are the agent-facing complement to `CONTRIBUTING.md`. Follow them without deviation unless an execution plan documents an explicit exception.

## Enforcement

Ruff 0.15.12 is the repository lint and format contract. `python -m ruff check src/ tests/` and `python -m ruff format --check src/ tests/` must pass before any work is complete. The rule set is `["E", "W", "F", "I", "UP", "B", "SIM"]` with `E501` and `B008` ignored. Run `ruff check --fix` and `ruff format` to auto-fix what ruff can.

Pre-commit hooks enforce Ruff lint + format on every commit if installed. Pull requests that do not conform to Ruff 0.15.12 fail CI.

## Module structure

- Every source module must have a one-line module docstring as the first non-import statement.
- Every module that defines a public API must export `__all__`. Omit `__all__` only from CLI entrypoints and internal-only utility modules.
- Use `from __future__ import annotations` at the top of every source file.

## Naming

- Public functions and classes: `snake_case` / `PascalCase` as usual.
- Private implementation helpers: `_snake_case`. Do not import `_`-prefixed names from outside their defining module. If a test needs it, the function is public — remove the prefix.
- Named constants replacing magic numbers: `_UPPER_SNAKE_CASE` (module-private) or `UPPER_SNAKE_CASE` (exported). Never leave bare numeric literals for thresholds, scores, or counts.

## Exception handling

- Never use bare `except:` or `except Exception: pass` silently.
- For expected no-ops (file already gone, optional parse skip): use `contextlib.suppress(SpecificError)`.
- For degraded-mode fallbacks (taxonomy load fails, config key missing): use `warnings.warn(message, stacklevel=2)` before returning the fallback value.
- Catch the narrowest applicable type: `ValueError`, `KeyError`, `OSError`, etc. — not `Exception`.
- The custom hierarchy in `exceptions.py` (`AgentMemoryError` → `MemoryStoreError`, `EmbeddingError`, `ConfigError`, `TaxonomyError`) is available for raise sites and callers that need to distinguish error origins.

## Comments

- Default to no comments. Only add one when the **why** is non-obvious: a hidden constraint, a workaround for a specific external bug, an invariant that would surprise a reader.
- Never comment what the code does — well-named identifiers already do that.
- Never add multi-line docstrings to functions. One short line maximum.

## Generics

Use PEP 695 type parameter syntax (`def fn[T](...) -> T:`) — not `TypeVar`. Python 3.12 is the minimum.

## Imports

- `zoneinfo` is always available on Python 3.12+. Import directly; no try/except compatibility shim.
- Circular imports that exist today (e.g. `taxonomy` imported inside functions in `classifier`, `profile`, `structured_extractor`) are a known constraint. Do not restructure them without an execution plan. Keep the `warnings.warn()` on the exception fallback path.

## Tests

- Shared fixtures and stubs belong in `tests/conftest.py`. Do not duplicate helpers across test files.
- Tests must not import `_`-prefixed names. Promote to public or test through the public interface.
- All tests must pass with the hash embedding encoder. Never add a test that requires a live Ollama connection.
- Run `make test` and `make eval-all` before marking any task complete.

## Taxonomy changes

Adding or changing a taxonomy key requires all four of these to stay consistent: `config/taxonomy.yaml`, regex patterns in `classifier.py` (or taxonomy `rule_patterns`), semantic examples in `taxonomy.yaml`, and `tests/unit/test_taxonomy.py`. Run `make eval-classification` after changes.
