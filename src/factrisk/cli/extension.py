from __future__ import annotations

import argparse
import json

from factrisk.backends.extensions import run_extension_inference
from factrisk.core.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an independent extension ST model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["seamless_m4t", "qwen3_omni"], required=True)
    args = parser.parse_args()
    report = run_extension_inference(read_yaml(args.config), model_name=args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
