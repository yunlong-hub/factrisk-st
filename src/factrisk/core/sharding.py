from __future__ import annotations

from pathlib import Path
from typing import Any

from factrisk.core.io import read_jsonl, write_json, write_jsonl


KINDS = {
    "direct": ("predictions", "qwen2_audio", "num_predictions"),
    "evidence": ("evidence", "whisper_nllb", "num_evidence_rows"),
}


def merge_inference_shards(
    config: dict[str, Any], *, kind: str, num_shards: int
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"Unknown shard kind: {kind}")
    if num_shards < 2:
        raise ValueError("Shard merge requires num_shards >= 2")
    project = config["project"]
    output_dir = Path(project["output_dir"])
    data_dir = Path(project["data_dir"])
    subdir, stem, count_key = KINDS[kind]
    manifest = read_jsonl(data_dir / "manifest.jsonl")
    expected_ids = [str(row["id"]) for row in manifest]
    expected_set = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    shard_reports = []
    for shard_index in range(num_shards):
        suffix = f"shard-{shard_index:05d}-of-{num_shards:05d}"
        shard_path = output_dir / subdir / f"{stem}.{suffix}.jsonl"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        rows = read_jsonl(shard_path)
        expected_count = sum(
            1 for index in range(len(manifest)) if index % num_shards == shard_index
        )
        if len(rows) != expected_count:
            raise ValueError(
                f"Incomplete shard {shard_index}: {len(rows)}/{expected_count}"
            )
        for row in rows:
            row_id = str(row.get("id", ""))
            if row_id not in expected_set:
                raise ValueError(f"Unexpected ID in {shard_path}: {row_id}")
            if row_id in by_id:
                raise ValueError(f"Duplicate ID across shards: {row_id}")
            by_id[row_id] = row
        shard_reports.append(
            {"shard_index": shard_index, "path": str(shard_path), "rows": len(rows)}
        )
    missing = [row_id for row_id in expected_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Missing {len(missing)} IDs after merge; first={missing[0]}")
    canonical = output_dir / subdir / f"{stem}.jsonl"
    write_jsonl(canonical, (by_id[row_id] for row_id in expected_ids))
    report = {
        "protocol": project["protocol"],
        "kind": kind,
        "num_shards": num_shards,
        "num_manifest_rows": len(manifest),
        count_key: len(by_id),
        "output": str(canonical),
        "shards": shard_reports,
    }
    report_name = "inference_report.json" if kind == "direct" else "evidence_report.json"
    write_json(output_dir / subdir / report_name, report)
    return report
