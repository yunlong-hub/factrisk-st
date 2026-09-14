from __future__ import annotations

import argparse
import json

from factrisk.datasets.data import prepare_factst_data
from factrisk.core.io import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare speaker-disjoint FactRisk-ST audio data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_factst_data(read_yaml(args.config), dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

