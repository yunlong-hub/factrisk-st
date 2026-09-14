from __future__ import annotations

from pathlib import Path
from typing import Any

from factrisk.core.io import read_jsonl, write_json, write_jsonl


def import_factst_predictions(config: dict[str, Any]) -> dict[str, Any]:
    project = config["project"]
    if project["protocol"] != "engineering_smoke":
        raise ValueError("Completed FACT-ST prediction import is restricted to engineering_smoke")
    sources = config["sources"]
    data_dir = Path(project["data_dir"])
    output_dir = Path(project["output_dir"])
    manifest = read_jsonl(data_dir / "manifest.jsonl")
    clean = {str(row["id"]): row for row in read_jsonl(sources["clean_predictions"])}
    intervention = {
        str(row["id"]): row for row in read_jsonl(sources["intervention_predictions"])
    }
    outputs = []
    missing = []
    for row in manifest:
        pair_id = str(row["pair_id"])
        side = str(row["side"])
        condition = str(row["condition"])
        translation = _translation(clean, intervention, pair_id, side, condition)
        if not translation:
            missing.append(str(row["id"]))
            continue
        probe_texts = []
        for probe_condition in ("local_mask", "irrelevant_mask"):
            text = _translation(clean, intervention, pair_id, side, probe_condition)
            if text and text != translation:
                probe_texts.append(
                    {"name": f"smoke_{probe_condition}", "translation": text}
                )
        outputs.append(
            {
                "id": str(row["id"]),
                "translation": translation,
                "probe_translations": probe_texts,
                "sample_translations": [translation] + [
                    item["translation"] for item in probe_texts
                ],
                "confidence": {},
                "backend": "completed_factst_engineering_smoke",
                "formal_eligible": False,
            }
        )
    if missing:
        raise ValueError(f"Missing completed FACT-ST predictions for {len(missing)} rows")
    prediction_path = output_dir / "predictions" / "factst_precomputed.jsonl"
    write_jsonl(prediction_path, outputs)
    report = {
        "protocol": project["protocol"],
        "formal_eligible": False,
        "num_manifest_rows": len(manifest),
        "num_predictions": len(outputs),
        "output": str(prediction_path),
    }
    write_json(output_dir / "predictions" / "import_report.json", report)
    return report


def _translation(
    clean: dict[str, dict[str, Any]],
    intervention: dict[str, dict[str, Any]],
    pair_id: str,
    side: str,
    condition: str,
) -> str:
    if condition == "clean":
        return _text(clean.get(pair_id, {}).get(side))
    suffix = {
        "local_mask": "key_mask",
        "irrelevant_mask": "irrelevant_mask",
        "sham_mask": "sham_mask",
    }.get(condition)
    if not suffix:
        return ""
    values = intervention.get(pair_id, {}).get("interventions", {})
    return _text(values.get(f"{side}_{suffix}"))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("translation", value.get("text", ""))).strip()
    return ""

