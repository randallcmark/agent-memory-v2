#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${AGENT_MEMORY_V2_PYTHON:-/tmp/agent_memory_v2_venv/bin/python}"

exec "$PYTHON_BIN" -m agent_memory_v2.live_eval_cli "$@"
