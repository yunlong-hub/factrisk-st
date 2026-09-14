from __future__ import annotations

import argparse
import json
import warnings

from sklearn.exceptions import ConvergenceWarning

from factrisk.core.io import read_yaml
from factrisk.method.risk import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/calibrate/evaluate FactRisk risk models.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = train_and_evaluate(read_yaml(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

