#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() {
  printf 'harness validation failed: %s\n' "$1" >&2
  exit 1
}

required_files=(
  "AGENTS.md"
  "README.md"
  "CLAUDE.md"
  "TOOLS.md"
  "STATUS.md"
  "docs/agent/index.md"
  "docs/agent/task-protocol.md"
  "docs/agent/validation.md"
  "docs/agent/doc-maintenance.md"
  "docs/agent/ai-feature-rules.md"
  "docs/agent/security-rules.md"
  "docs/architecture/index.md"
  "docs/architecture/boundaries.md"
  "docs/product/product-brief.md"
  "docs/product/user-journeys.md"
  "docs/exec-plans/template.md"
  "docs/quality/technical-debt.md"
  "docs/quality/quality-score.md"
)

for path in "${required_files[@]}"; do
  [[ -s "$path" ]] || fail "missing or empty required file: $path"
done

[[ -d docs/exec-plans/active ]] || fail "missing docs/exec-plans/active"
[[ -d docs/exec-plans/completed ]] || fail "missing docs/exec-plans/completed"

if grep -R "docs/agents/" AGENTS.md docs README.md CLAUDE.md TOOLS.md >/dev/null 2>&1; then
  fail "found stale docs/agents route; use docs/agent"
fi

if grep -R "TODO(project)" AGENTS.md docs README.md CLAUDE.md TOOLS.md >/dev/null 2>&1; then
  fail "found TODO(project) marker"
fi

if find docs -type f -name '*.md' -empty | grep -q .; then
  fail "empty markdown files remain under docs"
fi

printf 'harness validation passed\n'
