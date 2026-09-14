from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from factrisk.core.io import append_jsonl, completed_ids, read_jsonl, write_json


def run_extension_inference(config: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    project = config["project"]
    model_config = config["models"]["extensions"].get(model_name)
    if not model_config:
        raise ValueError(f"Unknown extension model: {model_name}")
    all_rows = read_jsonl(Path(project["data_dir"]) / "manifest.jsonl")
    requested_splits = set(str(value) for value in model_config.get("splits", []))
    rows = (
        [row for row in all_rows if str(row["split"]) in requested_splits]
        if requested_splits
        else all_rows
    )
    output_dir = Path(project["output_dir"]) / "extensions" / model_name
    output = output_dir / "predictions.jsonl"
    done = completed_ids(output)
    pending = [row for row in rows if str(row["id"]) not in done]
    translator = _translator(config, model_name, model_config)
    batch_size = int(model_config.get("batch_size", 1))
    started = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        requests: list[dict[str, Any]] = []
        owners: list[tuple[int, str, str]] = []
        for row_index, row in enumerate(batch):
            requests.append(_request(row, str(row["audio"])))
            owners.append((row_index, "base", "base"))
            for probe in row.get("probes", []):
                requests.append(_request(row, str(probe["audio"])))
                owners.append((row_index, "probe", str(probe["name"])))
        translations = translator.translate_batch(requests)
        if len(translations) != len(requests):
            raise RuntimeError(
                f"{model_name} returned {len(translations)}/{len(requests)} translations"
            )
        outputs = [
            {
                "id": str(row["id"]),
                "translation": "",
                "probe_translations": [],
                "sample_translations": [],
                "confidence": {},
                "backend": model_name,
                "formal_eligible": True,
            }
            for row in batch
        ]
        for translation, (owner, role, name) in zip(translations, owners):
            if role == "base":
                outputs[owner]["translation"] = str(translation).strip()
            else:
                outputs[owner]["probe_translations"].append(
                    {"name": name, "translation": str(translation).strip()}
                )
        if any(not row["translation"] for row in outputs):
            raise ValueError(f"{model_name} generated an empty base translation")
        append_jsonl(output, outputs)
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"[extension:{model_name}] {len(done) + offset + len(batch)}/{len(rows)} "
            f"rows; {(offset + len(batch)) / elapsed:.3f} rows/s; output={output}",
            flush=True,
        )
    expected_ids = {str(row["id"]) for row in rows}
    observed_ids = completed_ids(output)
    missing_ids = expected_ids - observed_ids
    extra_ids = observed_ids - expected_ids
    if missing_ids or extra_ids:
        raise RuntimeError(
            f"Incomplete {model_name} extension output: missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}"
        )
    report = {
        "protocol": project["protocol"],
        "model": model_name,
        "num_manifest_rows": len(all_rows),
        "selected_splits": sorted(requested_splits) if requested_splits else ["all"],
        "num_selected_rows": len(rows),
        "num_predictions": len(observed_ids),
        "output": str(output),
    }
    write_json(output_dir / "inference_report.json", report)
    return report


def _request(row: dict[str, Any], audio_path: str) -> dict[str, Any]:
    return {
        "audio_path": audio_path,
        "src_lang": str(row["source_language"]),
        "tgt_lang": str(row["target_language"]),
    }


def _translator(
    config: dict[str, Any], model_name: str, model_config: dict[str, Any]
) -> Any:
    factst_repo = str(config["sources"]["factst_repo"])
    if factst_repo not in sys.path:
        sys.path.insert(0, factst_repo)
    if model_name == "seamless_m4t":
        from factst.models.seamless_m4t import SeamlessM4TTranslator

        return SeamlessM4TTranslator(
            str(model_config["model_path"]),
            torch_dtype=str(model_config.get("dtype", "bfloat16")),
            max_new_tokens=int(model_config.get("max_new_tokens", 192)),
        )
    if model_name == "qwen3_omni":
        from factst.models.qwen3_omni import Qwen3OmniTranslator

        return Qwen3OmniTranslator(
            str(model_config["model_path"]),
            torch_dtype=str(model_config.get("dtype", "bfloat16")),
            attn_implementation=str(model_config.get("attn_implementation", "sdpa")),
            max_new_tokens=int(model_config.get("max_new_tokens", 192)),
            disable_talker=True,
        )
    raise ValueError(f"Unsupported extension model: {model_name}")
