# Agent Memory V2 Agent Instructions

This file is a map for agents working in this repository. Keep it short enough to stay in context.

## Start Here

1. Read this file.
2. Read `docs/agent/index.md`.
3. Read `CLAUDE.md` for architecture and command details.
4. Read `TOOLS.md` for operational, debug, and evaluation workflows.
5. Check `STATUS.md` for current implementation state and known limitations.
6. For complex work, create or update an execution plan in `docs/exec-plans/active/`.

## Operating Rules

- Do not invent memory behavior; repo-local docs, config, tests, and eval artifacts are the source of truth.
- Do not make broad refactors without an execution plan.
- Prefer small, reviewable changes that preserve existing deterministic and live eval behavior.
- Update docs when behavior, commands, artifact formats, or validation expectations change.
- Run the project validation commands before proposing completion.
- If an agent gets stuck, improve the harness: add the missing doc, command, check, or route.
- Preserve user work. Do not revert unrelated changes.

## Routing

- Project overview: `docs/product/product-brief.md`
- User/operator journeys: `docs/product/user-journeys.md`
- Architecture: `docs/architecture/index.md`
- Boundaries: `docs/architecture/boundaries.md`
- Task protocol: `docs/agent/task-protocol.md`
- Code standards: `docs/agent/code-standards.md`
- Validation: `docs/agent/validation.md`
- Technical debt: `docs/quality/technical-debt.md`
- Active plans: `docs/exec-plans/active/`
