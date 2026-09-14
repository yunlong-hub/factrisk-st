#!/usr/bin/env bash
set -euo pipefail

FACTRISK_ROOT="${FACTRISK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FACTRISK_PY="${FACTRISK_PY:-/workspace/yunlong/anaconda3/envs/factst/bin/python}"
CONFIG="${CONFIG:-$FACTRISK_ROOT/configs/paper.yaml}"

cd "$FACTRISK_ROOT"
export PYTHONPATH="$FACTRISK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$FACTRISK_PY" ]]; then
  echo "Python interpreter is not executable: $FACTRISK_PY" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Configuration does not exist: $CONFIG" >&2
  exit 2
fi
