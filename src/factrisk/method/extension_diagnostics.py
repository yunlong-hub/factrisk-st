from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from factrisk.method.features import critical_fact_labels, sentence_chrf
from factrisk.core.io import read_jsonl, write_json


def diagnose_extension(config: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    project = config["project"]
    output_dir = Path(project["output_dir"])
    manifest = {
        str(row["id"]): row
        for row in read_jsonl(Path(project["data_dir"]) / "manifest.jsonl")
    }
    prediction_path = output_dir / "extensions" / model_name / "predictions.jsonl"
    predictions = read_jsonl(prediction_path)
    if not predictions:
        raise ValueError(f"No extension predictions: {prediction_path}")
    if len({str(row["id"]) for row in predictions}) != len(predictions):
        raise ValueError(f"Duplicate extension prediction IDs: {model_name}")

    labeled = []
    for prediction in predictions:
        sample_id = str(prediction["id"])
        if sample_id not in manifest:
            raise ValueError(f"Unknown extension prediction ID: {sample_id}")
        item = manifest[sample_id]
        translation = str(prediction["translation"]).strip()
        correct, unsupported, severe = critical_fact_labels(
            translation,
            item["expected_slot"],
            item["contrast_slot"],
            str(item["fact_type"]),
        )
        labeled.append(
            {
                "condition": str(item["condition"]),
                "fact_type": str(item["fact_type"]),
                "split": str(item["split"]),
                "severe": float(severe),
                "unsupported": float(unsupported),
                "chrf": sentence_chrf(str(item["reference"]), translation),
                "slot_correct": bool(correct),
            }
        )

    report = {
        "protocol": project["protocol"],
        "model": model_name,
        "num_rows": len(labeled),
        "splits": sorted({row["split"] for row in labeled}),
        "overall": _aggregate(labeled),
        "by_split": _group(labeled, "split"),
        "by_condition": _group(labeled, "condition"),
        "by_fact_type": _group(labeled, "fact_type"),
    }
    write_json(output_dir / "extensions" / model_name / "diagnostics.json", report)
    return report


def _group(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: _aggregate(values) for name, values in sorted(groups.items())}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "severe_error_rate": float(np.mean([row["severe"] for row in rows])),
        "unsupported_rate": float(np.mean([row["unsupported"] for row in rows])),
        "chrf": float(np.mean([row["chrf"] for row in rows])),
    }
