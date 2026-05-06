# Documentation Maintenance

Keep docs accurate enough that another agent can resume work without reconstructing the repo from scratch.

## Update Docs When

- Commands, Make targets, or scripts change.
- Artifact locations or JSON shapes change.
- Architecture boundaries change.
- A new evaluation mode or provider is added.
- Known limitations are resolved or newly discovered.
- An execution plan starts, changes scope, or completes.

## Rules

- Prefer durable docs over chat-only context.
- Keep `AGENTS.md` short and route to deeper docs.
- Keep `CLAUDE.md`, `TOOLS.md`, and `STATUS.md` aligned with actual commands.
- Do not leave template placeholders in adopted harness files.
- Keep raw generated artifacts in `artifacts/`; summarize durable findings in docs when needed.
