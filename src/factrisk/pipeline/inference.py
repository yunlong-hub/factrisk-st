from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from factrisk.core.io import append_jsonl, completed_ids, read_jsonl, write_json
from factrisk.backends.qwen import Qwen2AudioScorer


def run_direct_inference(
    config: dict[str, Any],
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    project = config["project"]
    data_dir = Path(project["data_dir"])
    output_dir = Path(project["output_dir"])
    model_config = config["models"]["direct_st"]
    backend = str(model_config["backend"])
    if backend != "qwen2_audio":
        raise ValueError(f"Unsupported direct ST backend: {backend}")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index}/{num_shards}")
    all_rows = read_jsonl(data_dir / "manifest.jsonl")
    rows = [row for index, row in enumerate(all_rows) if index % num_shards == shard_index]
    output = (
        output_dir / "predictions" / "qwen2_audio.jsonl"
        if num_shards == 1
        else output_dir
        / "predictions"
        / f"qwen2_audio.shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    )
    done = completed_ids(output)
    pending = [row for row in rows if str(row["id"]) not in done]
    scorer = Qwen2AudioScorer(model_config)
    batch_size = int(model_config.get("batch_size", 1))
    started = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        outputs = scorer.infer_rows(batch)
        append_jsonl(output, outputs)
        completed = len(done) + offset + len(batch)
        elapsed = max(time.monotonic() - started, 1e-6)
        speed = (offset + len(batch)) / elapsed
        print(
            f"[direct-st] {completed}/{len(rows)} rows; "
            f"{speed:.3f} rows/s; resume-safe output={output}",
            flush=True,
        )
    report = {
        "protocol": project["protocol"],
        "backend": backend,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "num_manifest_rows": len(all_rows),
        "num_shard_rows": len(rows),
        "num_predictions": len(completed_ids(output)),
        "output": str(output),
    }
    report_path = (
        output_dir / "predictions" / "inference_report.json"
        if num_shards == 1
        else output_dir
        / "predictions"
        / f"inference_report.shard-{shard_index:05d}-of-{num_shards:05d}.json"
    )
    write_json(report_path, report)
    return report
