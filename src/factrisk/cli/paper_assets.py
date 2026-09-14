from __future__ import annotations

import argparse
import json

from factrisk.core.io import read_yaml
from factrisk.paper.assets import build_paper_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FactRisk-ST paper result assets.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = build_paper_assets(read_yaml(args.config), args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

