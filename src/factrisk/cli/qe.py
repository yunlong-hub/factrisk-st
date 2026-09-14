from __future__ import annotations

import argparse
import json

from factrisk.core.io import read_yaml
from factrisk.method.qe import run_comet_qe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the COMET-QE text-only baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence-file")
    args = parser.parse_args()
    report = run_comet_qe(read_yaml(args.config), evidence_file=args.evidence_file)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
