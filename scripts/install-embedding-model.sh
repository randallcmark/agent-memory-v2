#!/usr/bin/env bash
set -euo pipefail

OLLAMA_MODEL="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"
ollama pull "${OLLAMA_MODEL}"
