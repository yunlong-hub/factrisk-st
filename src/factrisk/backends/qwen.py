from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np


LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ca": "Catalan",
    "it": "Italian",
    "ru": "Russian",
    "zh": "Chinese",
}


class Qwen2AudioScorer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        dtype = getattr(torch, str(self.config.get("dtype", "bfloat16")))
        path = str(self.config["model_path"])
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def infer_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        requests: list[dict[str, Any]] = []
        request_owner: list[tuple[int, str, str]] = []
        for row_index, row in enumerate(rows):
            requests.append(
                {
                    "audio": row["audio"],
                    "source_language": row["source_language"],
                    "target_language": row["target_language"],
                }
            )
            request_owner.append((row_index, "base", "base"))
            for probe in row.get("probes", []):
                requests.append(
                    {
                        "audio": probe["audio"],
                        "source_language": row["source_language"],
                        "target_language": row["target_language"],
                    }
                )
                request_owner.append((row_index, "probe", str(probe["name"])))
        deterministic = self._generate(
            requests,
            num_beams=int(self.config.get("num_beams", 2)),
            num_return_sequences=max(1, int(self.config.get("num_beams", 2))),
            do_sample=False,
        )
        base_requests = [
            {
                "audio": row["audio"],
                "source_language": row["source_language"],
                "target_language": row["target_language"],
            }
            for row in rows
        ]
        sample_count = int(self.config.get("num_samples", 0))
        samples = (
            self._generate(
                base_requests,
                num_beams=1,
                num_return_sequences=sample_count,
                do_sample=True,
                temperature=float(self.config.get("sample_temperature", 0.8)),
            )
            if sample_count > 1
            else []
        )
        outputs: list[dict[str, Any]] = [
            {
                "id": str(row["id"]),
                "translation": "",
                "probe_translations": [],
                "sample_translations": [],
                "confidence": {},
                "backend": "qwen2_audio",
                "formal_eligible": True,
            }
            for row in rows
        ]
        for result, (row_index, role, name) in zip(deterministic, request_owner):
            if role == "base":
                outputs[row_index]["translation"] = result["translation"]
                outputs[row_index]["confidence"] = result["confidence"]
                outputs[row_index]["beam_translations"] = result["beam_translations"]
            else:
                outputs[row_index]["probe_translations"].append(
                    {"name": name, "translation": result["translation"]}
                )
        for row_index, result in enumerate(samples):
            outputs[row_index]["sample_translations"] = result["beam_translations"]
        return outputs

    def _generate(
        self,
        requests: list[dict[str, Any]],
        *,
        num_beams: int,
        num_return_sequences: int,
        do_sample: bool,
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        self.load()
        assert self.processor is not None
        assert self.model is not None
        import librosa
        import torch

        sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        prompts, audios = [], []
        for request in requests:
            source = LANGUAGE_NAMES.get(
                str(request["source_language"]).lower(), str(request["source_language"])
            )
            target = LANGUAGE_NAMES.get(
                str(request["target_language"]).lower(), str(request["target_language"])
            )
            prompt = f"Translate the speech from {source} to {target}. Output only the translation."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": str(Path(request["audio"]))},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            prompts.append(
                self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
            )
            audio, _ = librosa.load(str(request["audio"]), sr=sampling_rate, mono=True)
            audios.append(audio)
        processor_parameters = inspect.signature(self.processor.__call__).parameters
        audio_key = "audio" if "audio" in processor_parameters else "audios"
        if audio_key not in processor_parameters:
            raise TypeError("Qwen audio processor exposes neither audio= nor audios=")
        inputs = self.processor(
            text=prompts,
            **{audio_key: audios},
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 192)),
            "do_sample": do_sample,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "return_dict_in_generate": True,
            "output_scores": bool(self.config.get("collect_scores", True)),
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = int(inputs["input_ids"].shape[1])
        new_ids = generated.sequences[:, prompt_length:]
        decoded = [
            text.strip()
            for text in self.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ]
        per_sequence_nll = _transition_nll(self.model, generated)
        group_size = num_return_sequences
        results = []
        for request_index in range(len(requests)):
            start, end = request_index * group_size, (request_index + 1) * group_size
            translations = decoded[start:end]
            nll_values = per_sequence_nll[start:end]
            confidence: dict[str, float] = {}
            if nll_values:
                confidence["sequence_nll"] = float(nll_values[0])
            if not do_sample and generation_kwargs['output_scores']:
                confidence["token_entropy"] = _request_entropy(
                    generated.scores, request_index, num_beams
                )
                confidence["beam_margin"] = _beam_margin(generated, start, end, nll_values)
            results.append(
                {
                    "translation": translations[0] if translations else "",
                    "beam_translations": translations,
                    "confidence": confidence,
                }
            )
        return results


def _transition_nll(model: Any, generated: Any) -> list[float]:
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
    result = []
    for row in values:
        valid = row[np.isfinite(row)]
        result.append(float(-valid.mean()) if len(valid) else float("nan"))
    return result


def _request_entropy(scores: Any, request_index: int, num_beams: int) -> float:
    import torch

    entropies = []
    start, end = request_index * num_beams, (request_index + 1) * num_beams
    for logits in scores:
        subset = logits[start:end].float()
        log_probs = torch.log_softmax(subset, dim=-1)
        probs = torch.exp(log_probs)
        # Generation processors can forbid tokens with -inf. 0 * -inf is NaN.
        entropy = torch.special.entr(probs).sum(dim=-1)
        entropies.extend(entropy.detach().cpu().tolist())
    return float(sum(entropies) / len(entropies)) if entropies else float("nan")


def _beam_margin(generated: Any, start: int, end: int, nll_values: list[float]) -> float:
    sequence_scores = getattr(generated, "sequences_scores", None)
    if sequence_scores is not None and end - start >= 2:
        values = sequence_scores[start:end].detach().float().cpu().tolist()
        return float(values[0] - values[1])
    if len(nll_values) >= 2 and all(math.isfinite(value) for value in nll_values[:2]):
        return float(nll_values[1] - nll_values[0])
    return float("nan")
