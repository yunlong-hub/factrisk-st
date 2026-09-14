from __future__ import annotations

import argparse
import json

from factrisk.method.features import build_features
from factrisk.core.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen FactRisk-ST labels/features.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = build_features(read_yaml(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

