# AI Feature Rules

AI behavior in this repo must be observable, testable, and separable from memory mechanics.

## Rules

- Distinguish deterministic memory behavior from live model behavior in docs and results.
- Use isolated temporary storage for eval and scenario runs.
- Save prompts, recalled context, tool calls, raw model responses, and final answers for qualitative review.
- Keep provider-specific code behind a provider interface.
- Do not require paid API credentials for unit tests or deterministic evals.
- Treat model output as untrusted input; validate tool calls and structured JSON before acting.
- Record invalid tool calls and refusals as artifacts instead of hiding them.

## Provider Guidance

- Ollama remains the local default for current live evals.
- OpenAI is the first agent-class API provider for the mini agent harness.
- Claude, Gemini, and cloud GPU providers are deferred until the OpenAI harness produces comparable traces.
