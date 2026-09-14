from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from factrisk.core.io import read_jsonl, sha256_file, write_json


def finalize_release(config: dict[str, Any]) -> dict[str, Any]:
    project = config["project"]
    if project["protocol"] != "formal":
        raise ValueError("Only the formal protocol can be finalized")
    output_dir = Path(project["output_dir"])
    paper_dir = Path(config["paper"]["output_dir"])
    snapshot_path = Path(config["paper"]["result_snapshot"])
    pdf_path = paper_dir / "main.pdf"
    metrics_path = output_dir / "results" / "metrics.json"

    with snapshot_path.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not snapshot.get("formal_eligible"):
        raise ValueError("Result snapshot is not formal-eligible")

    stale_sources = []
    for source in snapshot["sources"]:
        path = Path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            stale_sources.append(str(path))
    if stale_sources:
        raise ValueError(f"Snapshot source hashes are stale: {stale_sources[:5]}")

    expected_counts = {
        output_dir / "predictions" / "qwen2_audio.jsonl": 6398,
        output_dir / "evidence" / "whisper_nllb.jsonl": 6398,
        Path(config["models"]["optional_qe"]["score_file"]): 6398,
        output_dir / "features.jsonl": 6398,
        output_dir / "results" / "risk_predictions.jsonl": 6398,
        output_dir / "results" / "selective_outputs.jsonl": 6398,
        output_dir / "extensions" / "seamless_m4t" / "predictions.jsonl": 6398,
        output_dir / "extensions" / "qwen3_omni" / "predictions.jsonl": 1274,
    }
    observed_counts = {}
    for path, expected in expected_counts.items():
        count = len(read_jsonl(path))
        observed_counts[str(path)] = count
        if count != expected:
            raise ValueError(f"Release count mismatch for {path}: {count}/{expected}")

    audit_path = paper_dir / "audit" / "stratified_audit_sample.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if not audit_rows or any(
        row["human_slot_correct"] or row["human_unsupported"] or row["human_severity"]
        for row in audit_rows
    ):
        raise ValueError("Audit sheet must be nonempty with explicitly blank human labels")

    if not pdf_path.is_file() or pdf_path.stat().st_size < 10_000:
        raise ValueError("Compiled paper PDF is missing or implausibly small")
    page_count = None
    pdf_text = ""
    try:
        import fitz

        document = fitz.open(pdf_path)
        page_count = document.page_count
        pdf_text = "\n".join(page.get_text() for page in document)
    except ImportError:
        pass
    for forbidden in ("pending", "Results are inserted automatically", "??"):
        if forbidden.lower() in pdf_text.lower():
            raise ValueError(f"Unresolved paper placeholder: {forbidden}")

    artifacts = [
        pdf_path,
        snapshot_path,
        metrics_path,
        output_dir / "features.validation.json",
        audit_path,
        *expected_counts.keys(),
        output_dir / "extensions" / "seamless_m4t" / "results" / "metrics.json",
        output_dir / "extensions" / "qwen3_omni" / "diagnostics.json",
    ]
    report = {
        "release_ready": True,
        "protocol": "formal",
        "paper_pdf": str(pdf_path),
        "paper_pages": page_count,
        "paper_words_extracted": len(pdf_text.split()) if pdf_text else None,
        "snapshot_sources_verified": len(snapshot["sources"]),
        "row_counts": observed_counts,
        "audit_rows": len(audit_rows),
        "claim_gate_eligible": bool(metrics["claim_gates"].get("eligible")),
        "claim_gate_passed": bool(metrics["claim_gates"].get("passed")),
        "claim_gate_checks": metrics["claim_gates"].get("checks", {}),
        "artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in dict.fromkeys(artifacts)
            if path.is_file()
        ],
    }
    write_json(paper_dir / "release_manifest.json", report)
    return report
