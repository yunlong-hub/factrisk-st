from __future__ import annotations

import argparse
import json

from factrisk.core.io import read_yaml
from factrisk.backends.whisper import run_evidence_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Whisper→NLLB evidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    report = run_evidence_inference(
        read_yaml(args.config),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
