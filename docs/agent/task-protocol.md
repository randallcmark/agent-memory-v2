# Task Protocol

Use this protocol to keep agent work reviewable and recoverable.

## Intake

Before editing, identify:

- The requested outcome.
- The files or subsystems likely involved.
- The source of product or architecture truth.
- The validation commands required before completion.
- Whether an execution plan is required.

If behavior is underspecified, ask for the missing behavior or record an explicit assumption in the execution plan before implementing.

## Planning

Use an execution plan for complex work. The plan is a working artifact, not a ceremony. It must capture:

- Goal and non-goals.
- Relevant context links.
- Step-by-step approach.
- Acceptance criteria.
- Validation commands.
- Decisions and tradeoffs discovered during work.
- Progress log.

Keep active plans in `docs/exec-plans/active/`. Move completed plans to `docs/exec-plans/completed/` after the work lands.

## Implementation

- Prefer existing patterns over new abstractions.
- Keep changes scoped to the task.
- Add tests near the changed behavior.
- Avoid broad formatting churn unless formatting is the task.
- Do not introduce dependencies without checking architecture and validation docs.
- Update product, architecture, tool, or validation docs in the same change when behavior changes.

## Review

Before proposing completion, inspect the diff as a reviewer:

- Does the implementation match the requested behavior?
- Did it invent behavior not captured in docs or acceptance criteria?
- Are boundaries and dependency directions respected?
- Are failure modes handled?
- Are validation commands documented and run?
- Did the task reveal a missing harness rule or check?

## Completion

Report:

- What changed.
- Which validation commands ran and their result.
- Any commands not run and why.
- Remaining risks or follow-up work.
