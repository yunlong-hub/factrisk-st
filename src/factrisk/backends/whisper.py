from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from factrisk.core.io import append_jsonl, completed_ids, read_jsonl, write_json


WHISPER_LANGUAGES = {
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "ca": "catalan",
    "it": "italian",
    "ru": "russian",
    "zh": "chinese",
}

NLLB_LANGUAGES = {
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ca": "cat_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
}


class WhisperNLLBEvidence:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = "cuda:0"
        self.asr_processor = None
        self.asr_model = None
        self.mt_tokenizer = None
        self.mt_model = None

    def load(self) -> None:
        if self.asr_model is not None:
            return
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoProcessor,
            AutoTokenizer,
            WhisperForConditionalGeneration,
        )

        dtype = getattr(torch, str(self.config.get("dtype", "bfloat16")))
        self.asr_processor = AutoProcessor.from_pretrained(self.config["whisper_path"])
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            self.config["whisper_path"], dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.mt_tokenizer = AutoTokenizer.from_pretrained(self.config["nllb_path"])
        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config["nllb_path"], dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.asr_model.eval()
        self.mt_model.eval()

    def infer(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        self.load()
        assert self.asr_processor is not None
        assert self.asr_model is not None
        assert self.mt_tokenizer is not None
        assert self.mt_model is not None
        import librosa
        import torch

        source_languages = {str(row["source_language"]).lower() for row in rows}
        target_languages = {str(row["target_language"]).lower() for row in rows}
        if len(source_languages) != 1 or len(target_languages) != 1:
            raise ValueError("Evidence batches must contain one language direction")
        source_language = source_languages.pop()
        target_language = target_languages.pop()
        sample_rate = int(self.asr_processor.feature_extractor.sampling_rate)
        audios = [
            librosa.load(str(row["audio"]), sr=sample_rate, mono=True)[0] for row in rows
        ]
        inputs = self.asr_processor(
            audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        model_dtype = next(self.asr_model.parameters()).dtype
        inputs = {
            key: (
                value.to(self.device, dtype=model_dtype)
                if key == "input_features"
                else value.to(self.device)
            )
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = self.asr_model.generate(
                **inputs,
                language=WHISPER_LANGUAGES[source_language],
                task="transcribe",
                max_new_tokens=int(self.config.get("max_new_tokens", 96)),
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        transcripts = [
            text.strip()
            for text in self.asr_processor.batch_decode(
                generated.sequences, skip_special_tokens=True
            )
        ]
        asr_nll = _transition_nll(self.asr_model, generated)
        asr_entropy = _batch_entropy(generated.scores, len(rows))

        self.mt_tokenizer.src_lang = NLLB_LANGUAGES[source_language]
        mt_inputs = self.mt_tokenizer(
            transcripts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        target_token = self.mt_tokenizer.convert_tokens_to_ids(
            NLLB_LANGUAGES[target_language]
        )
        with torch.inference_mode():
            translated = self.mt_model.generate(
                **mt_inputs,
                forced_bos_token_id=target_token,
                max_new_tokens=int(self.config.get("max_new_tokens", 96)),
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        cascade = [
            text.strip()
            for text in self.mt_tokenizer.batch_decode(
                translated.sequences, skip_special_tokens=True
            )
        ]
        mt_nll = _transition_nll(self.mt_model, translated)
        return [
            {
                "id": str(row["id"]),
                "asr_transcript": transcripts[index],
                "cascade_translation": cascade[index],
                "asr_sequence_nll": asr_nll[index] if index < len(asr_nll) else None,
                "asr_token_entropy": (
                    asr_entropy[index] if index < len(asr_entropy) else None
                ),
                "cascade_sequence_nll": (
                    mt_nll[index] if index < len(mt_nll) else None
                ),
                "backend": "whisper_nllb",
                "formal_eligible": True,
            }
            for index, row in enumerate(rows)
        ]


def run_evidence_inference(
    config: dict[str, Any],
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    project = config["project"]
    data_dir = Path(project["data_dir"])
    output_dir = Path(project["output_dir"])
    model_config = config["models"]["evidence"]
    if str(model_config["backend"]) != "whisper_nllb":
        raise ValueError("Only whisper_nllb evidence is currently implemented")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index}/{num_shards}")
    all_rows = read_jsonl(data_dir / "manifest.jsonl")
    rows = [row for index, row in enumerate(all_rows) if index % num_shards == shard_index]
    output = (
        output_dir / "evidence" / "whisper_nllb.jsonl"
        if num_shards == 1
        else output_dir
        / "evidence"
        / f"whisper_nllb.shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    )
    done = completed_ids(output)
    pending = [row for row in rows if str(row["id"]) not in done]
    scorer = WhisperNLLBEvidence(model_config)
    batch_size = int(model_config.get("batch_size", 8))
    started = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        append_jsonl(output, scorer.infer(batch))
        elapsed = max(time.monotonic() - started, 1e-6)
        speed = (offset + len(batch)) / elapsed
        print(
            f"[evidence] {len(done) + offset + len(batch)}/{len(rows)} rows; "
            f"{speed:.3f} rows/s; resume-safe output={output}",
            flush=True,
        )
    report = {
        "protocol": project["protocol"],
        "backend": model_config["backend"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "num_manifest_rows": len(all_rows),
        "num_shard_rows": len(rows),
        "num_evidence_rows": len(completed_ids(output)),
        "output": str(output),
    }
    report_path = (
        output_dir / "evidence" / "evidence_report.json"
        if num_shards == 1
        else output_dir
        / "evidence"
        / f"evidence_report.shard-{shard_index:05d}-of-{num_shards:05d}.json"
    )
    write_json(report_path, report)
    return report


def _transition_nll(model: Any, generated: Any) -> list[float]:
    import numpy as np

    try:
        scores = model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
    except Exception:
        return []
    values = scores.detach().float().cpu().numpy()
    return [
        float(-valid.mean()) if len(valid := row[np.isfinite(row)]) else float("nan")
        for row in values
    ]


def _batch_entropy(scores: Any, batch_size: int) -> list[float]:
    import torch

    totals = [0.0] * batch_size
    counts = [0] * batch_size
    for logits in scores:
        probabilities = torch.softmax(logits.float(), dim=-1)
        # `generate` uses -inf to enforce vocabulary constraints.  The
        # mathematically valid 0 log 0 contribution is zero, whereas the
        # direct `exp(log_p) * log_p` expression produces 0 * -inf = NaN.
        entropy = torch.special.entr(probabilities).sum(dim=-1).detach().cpu().tolist()
        for index, value in enumerate(entropy[:batch_size]):
            totals[index] += float(value)
            counts[index] += 1
    return [totals[index] / max(counts[index], 1) for index in range(batch_size)]
