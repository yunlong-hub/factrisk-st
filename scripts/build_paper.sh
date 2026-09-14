#!/usr/bin/env bash
set -euo pipefail
FACTRISK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTRISK_PY="${FACTRISK_PY:-/workspace/yunlong/anaconda3/envs/factst/bin/python}"
cd "$FACTRISK_ROOT"
export PYTHONPATH="$FACTRISK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mode=technical
if [[ "${1:-}" == --mode && $# == 2 && "${2:-}" =~ ^(technical|submission)$ ]]; then
  mode="$2"
elif [[ $# != 0 ]]; then
  echo 'Usage: scripts/build_paper.sh [--mode technical|submission]' >&2
  exit 2
fi
"$FACTRISK_PY" -c 'from factrisk.eval.readiness import require_experiments; require_experiments()'
"$FACTRISK_PY" -m factrisk.paper.publication
"$FACTRISK_PY" -m factrisk.paper.method_overview
cd papers/factriskst
mkdir -p build
../../tools/tectonic/tectonic --keep-logs --keep-intermediates --outdir build main.tex > build/compile.log 2>&1
cp build/main.pdf main.pdf
cd "$FACTRISK_ROOT"
"$FACTRISK_PY" -m factrisk.eval.readiness --mode "$mode"
