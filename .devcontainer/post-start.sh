#!/usr/bin/env bash
set -euo pipefail

warn() {
  printf 'WARNING post-start: %s\n' "$*" >&2
}

run_python() {
  if command -v uv >/dev/null 2>&1; then
    UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-cache}" uv run python "$@"
    return $?
  fi
  return 127
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HELPER="$SCRIPT_DIR/hf-ai-sync.py"

if ! command -v uv >/dev/null 2>&1; then
  warn "uv is unavailable; skipping Hugging Face AI state sync."
  exit 0
fi

if [[ ! -f "$HELPER" ]]; then
  warn "missing sync helper at $HELPER; skipping Hugging Face AI state sync."
  exit 0
fi

run_python "$HELPER" pull --repo-root "$REPO_ROOT" || warn "Hugging Face AI state sync failed; continuing."
