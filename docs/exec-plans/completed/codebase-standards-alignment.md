# Execution Plan: Codebase Standards Alignment

**Status:** Completed — 2026-05-07

## Goal

Audit the full codebase for drift and inconsistencies accumulated across maintenance spurts. Produce a consistent, legible, GitHub-friendly codebase with codified standards an agent or contributor can follow without prior context.

## Non-goals

- No behavior changes to memory mechanics, recall scoring, or pipeline logic.
- No new features.
- No dependency additions.

## Context

- Architecture: `docs/architecture/index.md`
- Quality tracking: `docs/quality/technical-debt.md`
- Agent rules: `docs/agent/ai-feature-rules.md`, `docs/agent/security-rules.md`

## Acceptance Criteria

- [x] `make test` passes (0 failures)
- [x] `make eval-all` passes (0 failures)
- [x] `python -m ruff check src/ tests/` reports 0 violations
- [x] `python -m ruff format --check src/ tests/` reports 0 violations
- [x] All source modules have a one-line module docstring
- [x] Core modules expose `__all__`
- [x] No bare `except Exception: pass` without `warnings.warn()`
- [x] Magic numbers replaced with named constants
- [x] No `_`-prefixed functions imported by tests
- [x] GitHub contribution infrastructure in place

All criteria met at completion.

## Validation Commands

```sh
make test
make eval-all
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
```

## Steps

### 1. Custom exception hierarchy
- Created `src/agent_memory_v2/exceptions.py`
- `AgentMemoryError` base → `MemoryStoreError`, `EmbeddingError`, `ConfigError`, `TaxonomyError`

### 2. Bare except → specific exception types
- `aging.py`: `except Exception` → `except ValueError` in `parse_timestamp()`
- `maintenance.py`: `except Exception` → `except ValueError` in `_parse_ts()`
- `pipeline.py`: fixed `_resolve_timezone`, `_relative_time_label`, `_taxonomy_version`
- `claude_provider.py`, `openai_provider.py`: `except Exception` → `except (ValueError, TypeError)` on JSON parse
- `profile.py`, `structured_extractor.py`: fallback paths now emit `warnings.warn()`
- `classifier.py`, `semantic_router.py`: fallback paths now emit `warnings.warn()`
- Used `contextlib.suppress()` for true no-op cases (`taxonomy.py`, `admin.py`, `maintenance.py`)

### 3. Module docstrings
- Added one-line docstrings to all 38 source files.

### 4. Named constants
- `classifier.py`: `_CONFIDENCE_PREFERENCE`, `_CONFIDENCE_FACT`, `_CONFIDENCE_TASK`, `_CONFIDENCE_TURN`, `_DURABILITY_BONUS`, `_DURABILITY_PENALTY`
- `semantic_router.py`: `DEFAULT_THRESHOLD = 0.72`

### 5. Shared test fixtures (`tests/conftest.py`)
- Created with `make_record`, `make_store`, `make_config`, `StubEncoder`, `StubOllama`, `StubExtractionOllama`, `QueueExtractionOllama`
- Removed duplicated helpers from `test_store.py` and `test_pipeline.py`

### 6. Public API delineation (`pipeline.py`)
- 8 `_`-prefixed functions tested externally promoted to public
- `_parse_timestamp` removed (thin wrapper; tests import `parse_timestamp` from `aging` directly)
- `classify_cli.py` updated to new function name

### 7. `__all__` on core modules
- Added to: `aging`, `classifier`, `embeddings`, `exceptions`, `models`, `ollama`, `pipeline`, `profile`, `semantic_router`, `sentiment`, `store`, `structured_extractor`, `taxonomy`

### 8. GitHub contribution infrastructure
- `CONTRIBUTING.md` — setup, testing, linting, code style, taxonomy changes
- `.pre-commit-config.yaml` — ruff lint + format hooks
- `.github/pull_request_template.md`
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`

### 9. Ruff expanded rule set
- `pyproject.toml` updated: `select = ["E", "W", "F", "I", "UP", "B", "SIM"]`
- Added `isort` first-party config: `known-first-party = ["agent_memory_v2"]`

## Decisions and Tradeoffs

- **Named constants vs config**: Confidence values kept as module constants (not `settings.yaml`) — they are internal classifier weights, not operator-tunable parameters.
- **Public vs private functions in pipeline.py**: Functions directly imported by tests were semantically public already; renaming eliminated the lie rather than changing the API contract.
- **`exceptions.py` hierarchy**: Defined but not yet wired into all raise sites — that is a separate task to avoid behavior changes in this pass.

## Progress Log

- 2026-05-07: All 8 tasks completed. `make test` 243 passed, `make eval-all` all cases passed, ruff 0 violations.
- 2026-05-08: Follow-up review cleanup aligned `ruff-format` with repository state, added Ruff lint/format checks to CI, promoted remaining tested helper imports to public names, and added missing `exceptions.__all__`.
- 2026-05-08: Clarified that Ruff 0.15.12 is the repository lint/format contract, not a backward-compatibility minimum.
