# Security Rules

This project stores user memory and model traces. Treat artifacts as potentially sensitive.

## Data Handling

- Do not commit runtime memory stores, generated artifacts, backups, or API keys.
- Keep OpenAI and other provider credentials in environment variables.
- Before publishing, run the sanitisation workflow documented in `PUBLISHING.md`.
- Assume raw provider outputs and scenario traces may contain sensitive user context.

## External Calls

- OpenAI calls require `OPENAI_API_KEY`.
- Do not make external provider calls in tests unless explicitly marked manual.
- Validate tool-call payloads before executing memory operations.
- Record provider errors in artifacts without exposing secret values.

## Memory Integrity

- Prefer append/trace artifacts over silent mutation.
- Use isolated temp storage for evals and scenarios.
- Do not run destructive admin commands such as `reset`, `prune`, or restore unless the user explicitly requested them.
