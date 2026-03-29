#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

run_python_module agent_memory_v2.sanitise_cli "$@"
