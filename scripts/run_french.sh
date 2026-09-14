#!/usr/bin/env bash
set -euo pipefail
FACTRISK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$FACTRISK_ROOT/scripts/common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
# Preserve the historical positional execution API; flag-based stages are dry-run
# unless --execute is explicitly supplied.
if [[ "${1:-}" =~ ^(direct|evidence)$ && "${2:-}" =~ ^(qwen|seamless)$ ]]; then
  if [[ $# -gt 4 ]]; then
    echo 'Legacy form: run_french.sh direct|evidence qwen|seamless [shard] [shards]; use flag form for --group.' >&2
    exit 2
  fi
  role="$1" backend="$2" shard="${3:-0}"
  extra=()
  if [[ $# -ge 4 ]]; then extra+=(--shards "$4"); fi
  exec "$FACTRISK_PY" -u -m factrisk.french.french_pipeline "$role" --backend "$backend" \
    --shard "$shard" "${extra[@]}" --execute
fi
exec "$FACTRISK_PY" -u -m factrisk.french.french_pipeline "$@"
