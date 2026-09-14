from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from factrisk.backends.audio import signal_features
from factrisk.datasets.data import validate_speaker_splits
from factrisk.core.io import read_jsonl, write_json, write_jsonl
from factrisk.core.text import (
    fact_signature,
    normalized_distance,
    pairwise_mean_distance,
    slot_correct,
    slot_polarity,
    text_is_negative,
    token_coverage,
    unsupported_contrast,
)


FEATURE_NAMES = [
    "sequence_nll",
    "token_entropy",
    "beam_margin_risk",
    "output_length",
    "asr_sequence_nll",
    "asr_token_entropy",
    "cascade_sequence_nll",
    "audio_duration",
    "audio_rms_db",
    "audio_zero_fraction",
    "audio_spectral_entropy",
    "audio_clipping_fraction",
    "perturb_mean_distance",
    "perturb_max_distance",
    "sample_mean_distance",
    "sample_fact_disagreement",
    "evidence_text_distance",
    "evidence_fact_mismatch",
    "evidence_coverage_gap",
    "qe_risk",
]


def build_features(config: dict[str, Any]) -> dict[str, Any]:
    project = config["project"]
    data_dir = Path(project["data_dir"])
    output_dir = Path(project["output_dir"])
    manifest = read_jsonl(data_dir / "manifest.jsonl")
    validate_speaker_splits(manifest)
    prediction_override = config.get("sources", {}).get("direct_predictions")
    prediction_path = (
        Path(prediction_override)
        if prediction_override
        else output_dir / "predictions" / "factst_precomputed.jsonl"
        if project["protocol"] == "engineering_smoke"
        else output_dir / "predictions" / "qwen2_audio.jsonl"
    )
    predictions = {str(row["id"]): row for row in read_jsonl(prediction_path)}
    missing_predictions = [str(row["id"]) for row in manifest if str(row["id"]) not in predictions]
    if missing_predictions:
        raise ValueError(f"Missing direct ST outputs for {len(missing_predictions)} manifest rows")
    evidence_override = config.get("sources", {}).get("evidence_predictions")
    evidence_path = (
        Path(evidence_override)
        if evidence_override
        else output_dir / "evidence" / "whisper_nllb.jsonl"
    )
    evidence = (
        {str(row["id"]): row for row in read_jsonl(evidence_path)}
        if evidence_path.is_file()
        else {}
    )
    if str(project["protocol"]).startswith("formal") and len(evidence) != len(manifest):
        raise ValueError(
            f"Formal protocol requires complete evidence: {len(evidence)}/{len(manifest)}"
        )
    qe_scores = _load_qe_scores(config)
    rows = [
        _feature_row(
            item,
            predictions[str(item["id"])],
            evidence.get(str(item["id"])),
            qe_scores.get(str(item["id"])),
        )
        for item in manifest
    ]
    validation = _validate_feature_rows(rows, config)
    feature_path = output_dir / "features.jsonl"
    write_jsonl(feature_path, rows)
    write_json(output_dir / "features.validation.json", validation)
    return {
        "protocol": project["protocol"],
        "num_rows": len(rows),
        "num_features": len(FEATURE_NAMES),
        "output": str(feature_path),
        "validation": validation,
    }


def _feature_row(
    manifest: dict[str, Any],
    prediction: dict[str, Any],
    evidence: dict[str, Any] | None,
    qe_score: float | None,
) -> dict[str, Any]:
    if not isinstance(prediction.get("translation"), str):
        raise ValueError(f"Missing or non-text translation: {manifest['id']}")
    # An observed empty decoding is a valid system output, not a missing row.
    # Keep it in coverage; number labeling records the omitted reference value.
    translation = prediction["translation"].strip()
    confidence = prediction.get("confidence", {})
    probes = [
        str(item.get("translation", "")).strip()
        for item in prediction.get("probe_translations", [])
        if str(item.get("translation", "")).strip()
    ]
    samples = [
        str(value).strip()
        for value in prediction.get("sample_translations", [])
        if str(value).strip()
    ]
    perturb_distances = [normalized_distance(translation, value) for value in probes]
    signatures = [
        fact_signature(value, str(manifest["fact_type"]))
        for value in ([translation] + samples)
    ]
    disagreement = (
        (len(set(signatures)) - 1) / max(len(signatures) - 1, 1)
        if len(signatures) > 1
        else float("nan")
    )
    evidence_translation = (
        str(evidence.get("cascade_translation", "")).strip() if evidence else ""
    )
    if evidence_translation:
        precision, recall = token_coverage(evidence_translation, translation)
        evidence_distance = normalized_distance(evidence_translation, translation)
        evidence_mismatch = float(
            fact_signature(evidence_translation, str(manifest["fact_type"]))
            != fact_signature(translation, str(manifest["fact_type"]))
        )
        coverage_gap = 1.0 - min(precision, recall)
    else:
        evidence_distance = evidence_mismatch = coverage_gap = float("nan")
    expected = manifest["expected_slot"]
    contrast = manifest["contrast_slot"]
    fact_type = str(manifest["fact_type"])
    correct, unsupported, severe = critical_fact_labels(
        translation, expected, contrast, fact_type
    )
    audio_values = _cached_signal_features(str(manifest["audio"]))
    values = {
        "sequence_nll": _number(confidence.get("sequence_nll")),
        "token_entropy": _number(confidence.get("token_entropy")),
        "beam_margin_risk": -_number(confidence.get("beam_margin")),
        "output_length": float(len(translation.split())),
        "asr_sequence_nll": _number(evidence.get("asr_sequence_nll") if evidence else None),
        "asr_token_entropy": _number(evidence.get("asr_token_entropy") if evidence else None),
        "cascade_sequence_nll": _number(
            evidence.get("cascade_sequence_nll") if evidence else None
        ),
        **audio_values,
        "perturb_mean_distance": (
            float(np.mean(perturb_distances)) if perturb_distances else float("nan")
        ),
        "perturb_max_distance": (
            float(np.max(perturb_distances)) if perturb_distances else float("nan")
        ),
        "sample_mean_distance": pairwise_mean_distance(samples),
        "sample_fact_disagreement": disagreement,
        "evidence_text_distance": evidence_distance,
        "evidence_fact_mismatch": evidence_mismatch,
        "evidence_coverage_gap": coverage_gap,
        "qe_risk": 1.0 - qe_score if qe_score is not None else float("nan"),
    }
    return {
        "id": str(manifest["id"]),
        "pair_id": str(manifest["pair_id"]),
        "side": str(manifest["side"]),
        "split": str(manifest["split"]),
        "speaker_id": str(manifest["speaker_id"]),
        "condition": str(manifest["condition"]),
        "evidence_insufficient": bool(manifest["evidence_insufficient"]),
        "fact_type": fact_type,
        "tts_engine": str(manifest["tts_engine"]),
        "translation": translation,
        "evidence_translation": evidence_translation,
        "reference": str(manifest["reference"]),
        "slot_correct": bool(correct),
        "unsupported": bool(unsupported),
        "severe_fact_error": int(severe),
        "label_reason": (
            "wrong_slot+unsupported"
            if not correct and unsupported
            else "wrong_or_missing_slot"
            if not correct
            else "unsupported_contrast"
            if unsupported
            else "correct"
        ),
        "label_rule_version": "agcal_fact_v1_number_negation",
        "features": values,
    }


def _slot_correct_with_contrast(
    text: str,
    expected: Any,
    contrast: Any,
    fact_type: str,
) -> bool:
    if fact_type.lower() != "negation":
        return slot_correct(text, expected, fact_type)
    expected_polarity = slot_polarity(expected)
    contrast_polarity = slot_polarity(contrast)
    if expected_polarity is None and contrast_polarity is not None:
        expected_polarity = not contrast_polarity
    if expected_polarity is not None:
        return text_is_negative(text) == expected_polarity
    return slot_correct(text, expected, fact_type)


def critical_fact_labels(
    translation: str, expected: Any, contrast: Any, fact_type: str
) -> tuple[bool, bool, bool]:
    """Apply the frozen Fact-ST number/negation label rule to one output."""

    correct = _slot_correct_with_contrast(translation, expected, contrast, fact_type)
    unsupported = unsupported_contrast(translation, expected, contrast, fact_type)
    return correct, unsupported, (not correct) or unsupported


@lru_cache(maxsize=65536)
def _cached_signal_features(path: str) -> dict[str, float]:
    return signal_features(path)


def sentence_chrf(reference: str, hypothesis: str, *, beta: float = 2.0) -> float:
    scores = []
    for order in range(1, 7):
        ref = Counter(_char_ngrams(reference, order))
        hyp = Counter(_char_ngrams(hypothesis, order))
        overlap = sum((ref & hyp).values())
        precision = overlap / max(sum(hyp.values()), 1)
        recall = overlap / max(sum(ref.values()), 1)
        denominator = beta * beta * precision + recall
        scores.append(
            (1.0 + beta * beta) * precision * recall / denominator
            if denominator > 0
            else 0.0
        )
    return 100.0 * sum(scores) / len(scores)


def _char_ngrams(text: str, order: int) -> list[str]:
    normalized = " ".join(str(text).lower().split())
    return [
        normalized[index : index + order]
        for index in range(max(0, len(normalized) - order + 1))
    ]


def _load_qe_scores(config: dict[str, Any]) -> dict[str, float]:
    qe = config.get("models", {}).get("optional_qe", {})
    if not qe.get("enabled", False):
        return {}
    path = qe.get("score_file")
    if not path or not Path(path).is_file():
        raise FileNotFoundError("optional_qe.enabled=true but score_file is missing")
    scores = {}
    for row in read_jsonl(path):
        scores[str(row["id"])] = float(row["score"])
    return scores


def _validate_feature_rows(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    label_counts = {
        split: Counter(
            int(row["severe_fact_error"]) for row in rows if row["split"] == split
        )
        for split in ("train", "calibration", "test")
    }
    for split in ("train", "calibration", "test"):
        if split_counts[split] == 0:
            raise ValueError(f"Empty split: {split}")
        if len(label_counts[split]) < 2:
            raise ValueError(f"Split {split} has only one risk label: {label_counts[split]}")
    required = config.get("claim_gates", {}).get("require_complete_conditions", [])
    present_conditions = set(row["condition"] for row in rows)
    missing_conditions = sorted(set(required) - present_conditions)
    if str(config["project"]["protocol"]).startswith("formal") and missing_conditions:
        raise ValueError(f"Formal feature data is missing conditions: {missing_conditions}")
    availability = {
        feature: sum(math.isfinite(_number(row["features"].get(feature))) for row in rows)
        for feature in FEATURE_NAMES
    }
    return {
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": {
            split: {str(key): value for key, value in sorted(counts.items())}
            for split, counts in label_counts.items()
        },
        "feature_nonmissing_counts": availability,
        "conditions": sorted(present_conditions),
        "missing_required_conditions": missing_conditions,
    }


def _number(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
