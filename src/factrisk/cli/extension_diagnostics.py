from __future__ import annotations

import argparse
import json

from factrisk.method.extension_diagnostics import diagnose_extension
from factrisk.core.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extension translation outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["seamless_m4t", "qwen3_omni"])
    args = parser.parse_args()
    print(
        json.dumps(
            diagnose_extension(read_yaml(args.config), model_name=args.model),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
