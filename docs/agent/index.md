# Agent Operating Index

This is the operating reference for agents working in `agent-memory-v2`.

## Principles

- Treat memory behavior as research evidence: preserve traces, artifacts, and exact commands.
- Keep deterministic memory mechanics separate from live model behavior.
- Prefer existing pipeline, config, CLI, and eval patterns before adding abstractions.
- Make complex work resumable through `docs/exec-plans/active/`.
- Improve the harness when repo context is missing or ambiguous.

## Task Routes

| Task type | Start with |
|---|---|
| Understand the project | `docs/product/product-brief.md` |
| Understand user/operator workflows | `docs/product/user-journeys.md` |
| Architecture work | `docs/architecture/index.md` |
| Boundary or dependency work | `docs/architecture/boundaries.md` |
| Complex implementation | `docs/agent/task-protocol.md` |
| Validation | `docs/agent/validation.md` |
| Documentation maintenance | `docs/agent/doc-maintenance.md` |
| Quality/debt review | `docs/quality/technical-debt.md` |

## Default Workflow

1. Read the smallest relevant docs.
2. Create or update an execution plan for complex work.
3. Make the smallest coherent change.
4. Add tests near changed behavior.
5. Update docs if behavior or commands changed.
6. Run the documented validation commands.
7. Summarize changes, validation, and remaining risk.
