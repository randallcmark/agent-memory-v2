# User Journeys

## Chat With Memory

The operator starts `make chat`, interacts with the assistant, and expects durable preferences, facts, and tasks to be available in later turns or sessions.

## Debug Memory Stages

The evaluator inspects individual stages with:

- `make classify`
- `make recall`
- `make prompt`
- `make ingest`
- `make list`
- `make list-sidecar`
- `make profile`

## Run Deterministic Evals

The maintainer runs `make eval-all` to verify model-independent memory mechanics with hash embeddings and isolated temporary storage.

## Run Live Evals

The evaluator runs `make live-eval-all ARGS="--record-history --save-all"` to inspect response behavior using Ollama generation and embeddings.

## Review Scenarios

The evaluator runs `make scenario-run ARGS="--scenario <name>"` and reviews saved artifacts containing ingestion details, recall, prompt context, final prompt, and response.

## Run Agent Evals

The evaluator runs `make agent-eval-run ARGS="--scenario <name> --provider fake"` for local validation or `--provider openai` for API-backed agent-class behavior. The run produces a trace of model/tool-loop decisions and memory operations.
