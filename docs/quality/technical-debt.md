# Technical Debt

Track known gaps here so agents can improve the system incrementally.

## Active Debt

| ID | Area | Problem | Impact | Suggested Fix | Status |
|---|---|---|---|---|---|
| TD-001 | Scenario review | Qualitative scenario artifacts do not have a reviewer-notes or annotation workflow. | Harder to compare subjective response quality across runs. | Add structured reviewer annotations or a lightweight review CLI. | Open |
| TD-002 | Structured extraction | Durable semantic extraction depends on local model JSON reliability. | Malformed or low-confidence outputs can leave useful facts non-durable. | Compare extractor behavior across providers and preserve rejection traces. | Open |
| TD-003 | Multi-provider support | Core runtime is Ollama-first. | Harder to compare local models with API-backed agent-class models. | Add provider-neutral agent eval harness before expanding runtime providers. | In Progress |
| TD-004 | Harness adoption | Agent operating harness was added after the initial implementation. | Older docs may duplicate or drift from new routes. | Keep `AGENTS.md`, `CLAUDE.md`, `TOOLS.md`, and `STATUS.md` aligned as commands evolve. | Open |

## Cleanup Rules

- Prefer small targeted cleanup PRs.
- Link debt items to execution plans when work is complex.
- Remove or close debt entries when the fix lands.
- Promote recurring debt into validation checks.
