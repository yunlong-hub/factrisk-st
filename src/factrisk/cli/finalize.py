from __future__ import annotations

import argparse
import json

from factrisk.core.io import read_yaml
from factrisk.pipeline.release import finalize_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze the paper release.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_release(read_yaml(args.config)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
