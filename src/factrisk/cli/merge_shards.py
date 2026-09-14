from __future__ import annotations

import argparse
import json

from factrisk.core.io import read_yaml
from factrisk.core.sharding import merge_inference_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge inference shards.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", choices=["direct", "evidence"], required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    report = merge_inference_shards(
        read_yaml(args.config), kind=args.kind, num_shards=args.num_shards
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
