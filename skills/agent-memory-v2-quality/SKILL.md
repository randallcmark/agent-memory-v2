---
name: agent-memory-v2-quality
description: Evaluate and qualitatively review the local agent_memory_v2 project. Use when asked to run deterministic evals, live Ollama evals, compare eval history, run saved qualitative scenarios, inspect prompts or recalled memory, or review response quality in /Volumes/Media/Repository/agent_memory_v2.
---

# Agent Memory V2 Quality

Use this skill when the task is about measuring or reviewing the quality of `agent_memory_v2`.

Project root:

```bash
cd /Volumes/Media/Repository/agent_memory_v2
```

Deterministic regression checks:

```bash
make eval-classification
make eval-sentiment
make eval-profile
make eval-recall
make eval-prompt
make eval-all
make eval-all ARGS="--record-history"
make eval-history
make eval-compare
```

Live Ollama quality checks:

```bash
make smoke
make embedding-smoke
make live-eval-memory
make live-eval-sentiment
make live-eval-all
make live-eval-all ARGS="--record-history --save-all"
make live-eval-history
make live-eval-compare
```

Scenario-driven qualitative review:

```bash
make scenario-list
make scenario-run ARGS="--scenario preference_recall"
make scenario-show ARGS="--run-id <saved-run-id>"
make scenario-compare ARGS="--run-a <run-a> --run-b <run-b>"
```

Low-level inspection tools:

```bash
make classify ARGS="--text 'I prefer oat milk.'"
make recall ARGS="--text 'What do I prefer?'"
make prompt ARGS="--text 'What do I prefer?'"
make generate ARGS="--prompt 'Reply with exactly OK.'"
```

Use this review order:

1. deterministic evals for regression
2. live evals for real model behavior
3. scenario runs for subjective review
4. low-level classify/recall/prompt inspection if a case looks wrong

When reviewing output, focus on:

1. whether the right memory was selected
2. whether prompt context is clean
3. whether the model answered directly
4. whether sentiment guidance improved or degraded tone

If the task is about running or maintaining the system rather than reviewing quality, use `agent-memory-v2-ops` instead.
