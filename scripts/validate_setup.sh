#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

cd "${PROJECT_ROOT}"

echo "[1/4] Checking shell syntax"
bash -n scripts/*.sh

echo "[2/4] Compiling Python sources"
"${FACTRISK_PY}" -m compileall -q src/factrisk tests

echo "[3/4] Running unit tests"
"${FACTRISK_PY}" -m pytest -q

echo "[4/4] Auditing the frozen formal-data protocol (read-only)"
"${FACTRISK_PY}" -m factrisk.cli.prepare \
  --config configs/paper.yaml \
  --dry-run

echo "Setup validation passed."
