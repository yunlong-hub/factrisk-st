from __future__ import annotations

import argparse
import json

from factrisk.method.extension_eval import evaluate_extension
from factrisk.core.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an extension ST model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["seamless_m4t", "qwen3_omni"], required=True)
    args = parser.parse_args()
    report = evaluate_extension(read_yaml(args.config), model_name=args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
