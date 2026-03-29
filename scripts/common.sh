#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AGENT_MEMORY_V2_PYTHON:-/tmp/agent_memory_v2_venv/bin/python}"
PYTEST_CACHE_DIR="${AGENT_MEMORY_V2_PYTEST_CACHE:-/tmp/agent_memory_v2_pytest_cache}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3:8b}"

require_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    echo "Set AGENT_MEMORY_V2_PYTHON or create the local venv first." >&2
    exit 2
  fi
}

run_python_module() {
  require_python
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -m "$@")
}
