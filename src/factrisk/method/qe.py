from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from factrisk.core.contracts import qe_fingerprint, validated_predictions
from factrisk.core.io import append_jsonl, completed_ids, read_jsonl, read_yaml, sha256_file, write_json


def run_comet_qe(
    config: dict[str, Any], *, evidence_file: str | Path | None = None
) -> dict[str, Any]:
    """Score direct translations with a text-only COMET-QE baseline.

    The source is the independently decoded Whisper transcript, not the gold
    transcript.  This keeps the baseline deployable while preventing access to
    label-only manifest fields.
    """

    project = config["project"]
    output_dir = Path(project["output_dir"])
    qe_config = config["models"]["optional_qe"]
    model_path = Path(qe_config["model_path"])
    prediction_path = output_dir / "predictions" / "qwen2_audio.jsonl"
    evidence_path = (
        Path(evidence_file)
        if evidence_file is not None
        else output_dir / "evidence" / "whisper_nllb.jsonl"
    )
    score_path = Path(qe_config["score_file"])

    predictions = {str(row["id"]): row for row in read_jsonl(prediction_path)}
    evidence = {str(row["id"]): row for row in read_jsonl(evidence_path)}
    if set(predictions) != set(evidence):
        only_predictions = len(set(predictions) - set(evidence))
        only_evidence = len(set(evidence) - set(predictions))
        raise ValueError(
            "COMET-QE requires one evidence transcript per prediction: "
            f"prediction_only={only_predictions}, evidence_only={only_evidence}"
        )

    done = completed_ids(score_path)
    pending_ids = [sample_id for sample_id in predictions if sample_id not in done]
    if pending_ids:
        from comet import load_from_checkpoint

        model = load_from_checkpoint(str(model_path), local_files_only=True)
        batch_size = int(qe_config.get("batch_size", 32))
        chunk_size = int(qe_config.get("chunk_size", 1024))
        started = time.monotonic()
        for offset in range(0, len(pending_ids), chunk_size):
            ids = pending_ids[offset : offset + chunk_size]
            samples = [
                {
                    "src": str(evidence[sample_id]["asr_transcript"]),
                    "mt": str(predictions[sample_id]["translation"]),
                }
                for sample_id in ids
            ]
            result = model.predict(
                samples,
                batch_size=batch_size,
                gpus=1,
                accelerator="gpu",
                progress_bar=True,
                num_workers=int(qe_config.get("num_workers", 4)),
            )
            scores = [float(value) for value in result.scores]
            if len(scores) != len(ids):
                raise RuntimeError(f"COMET-QE returned {len(scores)} scores for {len(ids)} rows")
            append_jsonl(
                score_path,
                [
                    {
                        "id": sample_id,
                        "score": score,
                        "source": "whisper_asr_transcript",
                        "backend": "wmt20-comet-qe-da",
                    }
                    for sample_id, score in zip(ids, scores, strict=True)
                ],
            )
            elapsed = max(time.monotonic() - started, 1e-6)
            completed = len(done) + offset + len(ids)
            print(
                f"[comet-qe] {completed}/{len(predictions)} rows; "
                f"{(offset + len(ids)) / elapsed:.3f} rows/s; output={score_path}",
                flush=True,
            )

    report = {
        "protocol": project["protocol"],
        "backend": "wmt20-comet-qe-da",
        "source": "whisper_asr_transcript",
        "num_predictions": len(predictions),
        "num_scores": len(completed_ids(score_path)),
        "model_path": str(model_path),
        "evidence_path": str(evidence_path),
        "output": str(score_path),
    }
    write_json(score_path.with_suffix(".report.json"), report)
    return report


def run(config_path: str | Path) -> None:
    """Run frozen QE scoring with exact-input cache validation."""
    cfg = read_yaml(config_path)
    out = Path(cfg["project"]["output_dir"])
    rows = read_jsonl(Path(cfg["project"]["data_dir"]) / "manifest.jsonl")
    predictions = validated_predictions(
        out / "predictions/qwen2_audio.jsonl", rows, "direct", cfg["models"]["direct_st"]
    )
    evidence_path = cfg.get("sources", {}).get(
        "revision_evidence_file", out / "evidence/whisper_nllb.jsonl"
    )
    evidence = validated_predictions(
        evidence_path, rows, "evidence", cfg["models"]["evidence"]
    )
    path = out / "qe/comet_qe.jsonl"
    old = read_jsonl(path) if path.exists() else []
    done = {row["id"]: row for row in old}
    if len(done) != len(old) or set(done) - set(predictions):
        raise ValueError("Invalid QE cache IDs")

    model_path = cfg["models"]["optional_qe"]["model_path"]
    model_hash = sha256_file(model_path)
    for key, value in done.items():
        expected = qe_fingerprint(predictions[key], evidence[key])
        if value.get("input_fingerprint") != expected or value.get("model_sha256") != model_hash:
            raise ValueError(f"Stale QE cache: {key}")

    pending = [row["id"] for row in rows if row["id"] not in done]
    if pending:
        import torch
        from comet import load_from_checkpoint

        started = time.monotonic()
        model = load_from_checkpoint(model_path, local_files_only=True)
        for start in range(0, len(pending), 256):
            keys = pending[start : start + 256]
            samples = [
                {"src": evidence[key]["asr_transcript"], "mt": predictions[key]["translation"]}
                for key in keys
            ]
            batch_started = time.monotonic()
            result = model.predict(
                samples, batch_size=16, gpus=1, accelerator="gpu",
                num_workers=0, progress_bar=False,
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - batch_started
            append_jsonl(path, [
                {
                    "id": key,
                    "score": float(score),
                    "input_fingerprint": qe_fingerprint(predictions[key], evidence[key]),
                    "model_sha256": model_hash,
                    "source": "whisper_asr_transcript",
                    "seconds_per_item": elapsed / len(keys),
                }
                for key, score in zip(keys, result.scores, strict=True)
            ])
            print(f"QE {len(done) + start + len(keys)}/{len(rows)}", flush=True)
        write_json(out / "qe/timing.json", {
            "total_seconds": time.monotonic() - started,
            "n_new": len(pending),
            "includes_load": True,
        })
    write_json(out / "qe/status.json", {
        "status": "complete", "rows": len(rows), "new_human_labels": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
