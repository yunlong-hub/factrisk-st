from __future__ import annotations

import argparse
import json

from factrisk.datasets.importers import import_factst_predictions
from factrisk.core.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Import completed FACT-ST outputs for smoke.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = import_factst_predictions(read_yaml(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

