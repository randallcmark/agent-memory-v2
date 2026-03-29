#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

if [[ "${1:-}" == "--user" && -n "${2:-}" ]]; then
  export AGENT_MEMORY_V2_USER="${2}"
  shift 2
fi

command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
  echo "Usage: bash scripts/admin.sh [--user USER] <command> [args...]" >&2
  exit 2
fi
shift

case "${command_name}" in
  stats|list|list-sidecar|profile|aging-report|prune-dry-run|prune|reset|rebuild|rebuild-profile)
    run_python_module agent_memory_v2.admin "${command_name}" "$@"
    ;;
  backup|export)
    run_python_module agent_memory_v2.state_cli export "$@"
    ;;
  restore|import)
    run_python_module agent_memory_v2.state_cli import "$@"
    ;;
  maintenance-status|status)
    run_python_module agent_memory_v2.maintenance status "$@"
    ;;
  maintain|run-maintenance)
    run_python_module agent_memory_v2.maintenance run "$@"
    ;;
  use-hash-embeddings|use-ollama-embeddings)
    run_python_module agent_memory_v2.config_tool "${command_name}" "$@"
    ;;
  *)
    echo "Unknown admin command: ${command_name}" >&2
    exit 2
    ;;
esac
