#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

run_python_module agent_memory_v2.e2e_smoke \
  --host "${OLLAMA_HOST}" \
  --model "${OLLAMA_MODEL}" \
  --timeout-seconds 60 \
  "$@"
