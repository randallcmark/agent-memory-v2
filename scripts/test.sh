#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_python
cd "${PROJECT_ROOT}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${PYTHON_BIN}" -m pytest -o "cache_dir=${PYTEST_CACHE_DIR}" "$@"
