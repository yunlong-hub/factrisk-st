#!/usr/bin/env bash
set -euo pipefail
FACTRISK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTRISK_PY="${FACTRISK_PY:-/workspace/yunlong/anaconda3/envs/factst/bin/python}"
cd "$FACTRISK_ROOT"
export PYTHONPATH="$FACTRISK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
exec "$FACTRISK_PY" -u -m factrisk.pipeline.finish_experiments "$@"
