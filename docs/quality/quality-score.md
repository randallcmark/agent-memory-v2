# Quality Score

A recurring snapshot of project health. Review after major milestones or before publishing research results.

## Scorecard

| Area | Grade | Evidence | Next Action |
|---|---|---|---|
| Memory pipeline | A | Deterministic evals cover classification, semantic routing, sentiment, profile, recall, and prompt stages. | Keep evals green as providers are added. |
| Evaluation reliability | B | Deterministic, live, and scenario artifacts exist. | Add agent-eval artifacts and reviewer annotations. |
| Provider abstraction | C | Ollama abstraction exists; agent-class provider abstraction is new. | Complete OpenAI-backed agent eval harness. |
| Documentation harness | B | Repo now has `AGENTS.md` and resumable execution plans. | Keep docs synchronized with implementation changes. |
| Publication safety | B | Sanitise workflow exists. | Confirm OpenAI artifacts are excluded from publishable state. |

## Notes

Grades are evidence-based and should be updated when validation or architecture changes.
