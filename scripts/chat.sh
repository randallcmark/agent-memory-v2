#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

if [[ "${1:-}" == "--user" && -n "${2:-}" ]]; then
  export AGENT_MEMORY_V2_USER="${2}"
  shift 2
fi

run_python_module agent_memory_v2.pipeline "$@"
