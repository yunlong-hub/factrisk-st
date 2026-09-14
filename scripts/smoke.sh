#!/usr/bin/env bash
set -euo pipefail

FACTRISK_ROOT="${FACTRISK_ROOT:-/workspace/yunlong/ST/factrisk-st}"
CONFIG="$FACTRISK_ROOT/configs/smoke.yaml"
source "$FACTRISK_ROOT/scripts/common.sh"

"$FACTRISK_PY" -m factrisk.cli.prepare --config "$CONFIG"
"$FACTRISK_PY" -m factrisk.cli.import_factst --config "$CONFIG"
"$FACTRISK_PY" -m factrisk.cli.features --config "$CONFIG"
"$FACTRISK_PY" -m factrisk.cli.evaluate --config "$CONFIG"
"$FACTRISK_PY" -m factrisk.cli.paper_assets --config "$CONFIG"
"$FACTRISK_PY" -m pytest

echo "FactRisk-ST engineering smoke completed: $FACTRISK_ROOT/exp/factrisk/smoke"

