#!/usr/bin/env bash
set -euo pipefail
FACTRISK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTRISK_PY="${FACTRISK_PY:-/workspace/yunlong/anaconda3/envs/factst/bin/python}"
cd "$FACTRISK_ROOT"
export PYTHONPATH="$FACTRISK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$FACTRISK_PY" -m factrisk.pipeline.workflow "$@"
