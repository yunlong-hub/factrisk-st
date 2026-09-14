from __future__ import annotations

import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from factrisk.method.features import sentence_chrf
from factrisk.core.io import read_jsonl, stable_int, write_json, write_jsonl
from factrisk.core.text import NEGATIVE_MARKERS, NUMBER_WORDS


TEXT_FEATURES = [
    "sequence_nll",
    "token_entropy",
    "beam_margin_risk",
    "output_length",
]
ACOUSTIC_FEATURES = [
    "asr_sequence_nll",
    "asr_token_entropy",
    "audio_duration",
    "audio_rms_db",
    "audio_zero_fraction",
    "audio_spectral_entropy",
    "audio_clipping_fraction",
]
STABILITY_FEATURES = [
    "perturb_mean_distance",
    "perturb_max_distance",
    "sample_mean_distance",
    "sample_fact_disagreement",
]
EVIDENCE_FEATURES = [
    "cascade_sequence_nll",
    "evidence_text_distance",
    "evidence_fact_mismatch",
    "evidence_coverage_gap",
]

METHOD_FEATURES = {
    "sequence_probability": ["sequence_nll"],
    "token_entropy": ["token_entropy"],
    "beam_margin": ["beam_margin_risk"],
    "asr_confidence": ["asr_sequence_nll", "asr_token_entropy"],
    "qe": ["qe_risk"],
    "self_consistency": ["sample_mean_distance", "sample_fact_disagreement"],
    "perturbation_only": ["perturb_mean_distance", "perturb_max_distance"],
    "entailment_only": [
        "evidence_text_distance",
        "evidence_fact_mismatch",
        "evidence_coverage_gap",
    ],
    "ablation_text": TEXT_FEATURES,
    "ablation_text_acoustic": TEXT_FEATURES + ACOUSTIC_FEATURES,
    "ablation_text_acoustic_stability": (
        TEXT_FEATURES + ACOUSTIC_FEATURES + STABILITY_FEATURES
    ),
    "logreg_model_agnostic": (
        ACOUSTIC_FEATURES
        + ["perturb_mean_distance", "perturb_max_distance"]
        + EVIDENCE_FEATURES
    ),
}

FULL_FEATURES = TEXT_FEATURES + ACOUSTIC_FEATURES + STABILITY_FEATURES + EVIDENCE_FEATURES

# Frozen after speaker-grouped cross-validation on the training split.  Keeping
# these values here and in the release snapshot makes the confirmatory test run
# independent of test-set model selection.
INTERACTION_MODEL_DEFAULTS = {
    "n_estimators": 600,
    "min_samples_leaf": 20,
    "max_features": 0.7,
}


def train_and_evaluate(config: dict[str, Any]) -> dict[str, Any]:
    project = config["project"]
    risk_config = config["risk"]
    output_dir = Path(project["output_dir"])
    rows = read_jsonl(output_dir / "features.jsonl")
    splits = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "calibration", "test")
    }
    _validate_training_splits(splits)
    requested = [str(name) for name in risk_config["models"]]
    development_selection = _development_model_selection(
        splits["train"], risk_config, int(project["seed"]), project["protocol"]
    )
    if (
        development_selection
        and risk_config.get("model_selection", {}).get("enforce_primary", False)
        and development_selection["selected"] != str(risk_config["primary_model"])
    ):
        raise RuntimeError(
            "Configured primary model was not selected by train-only grouped CV: "
            f"{development_selection['selected']}"
        )
    methods: dict[str, Any] = {}
    availability: dict[str, Any] = {}
    risk_predictions: dict[str, dict[str, float]] = defaultdict(dict)

    for method in requested:
        feature_names = _method_features(method, rows, project["protocol"])
        reason = _availability_reason(feature_names, splits)
        if reason:
            availability[method] = {
                "available": False,
                "reason": reason,
                "features": feature_names,
            }
            continue
        availability[method] = {
            "available": True,
            "reason": None,
            "features": feature_names,
        }
        estimator = _build_estimator(method, int(project["seed"]), risk_config)
        x_train, y_train = _matrix(splits["train"], feature_names)
        estimator.fit(x_train, y_train)
        raw_calibration = estimator.predict_proba(
            _matrix(splits["calibration"], feature_names)[0]
        )[:, 1]
        calibrator = _fit_platt(raw_calibration, _labels(splits["calibration"]))
        split_scores = {}
        for split_name, split_rows in splits.items():
            raw = estimator.predict_proba(_matrix(split_rows, feature_names)[0])[:, 1]
            calibrated = _apply_platt(calibrator, raw)
            split_scores[split_name] = calibrated
            for row, score in zip(split_rows, calibrated):
                risk_predictions[str(row["id"])][method] = float(score)
        methods[method] = _evaluate_method(
            method,
            splits,
            split_scores,
            risk_config,
            feature_names,
        )

    if not methods:
        raise RuntimeError("No requested risk method has usable features")
    primary = str(risk_config["primary_model"])
    if primary not in methods:
        raise RuntimeError(f"Primary risk model is unavailable: {availability.get(primary)}")
    comparator = _select_comparator(methods, primary)
    bootstrap = _cluster_bootstrap_difference(
        splits["test"],
        np.asarray([risk_predictions[str(row["id"])][primary] for row in splits["test"]]),
        np.asarray([risk_predictions[str(row["id"])][comparator] for row in splits["test"]]),
        samples=int(risk_config.get("bootstrap_samples", 1000)),
        seed=int(project["seed"]),
    )
    bootstrap_by_method = {
        method: _cluster_bootstrap_difference(
            splits["test"],
            np.asarray(
                [risk_predictions[str(row["id"])][primary] for row in splits["test"]]
            ),
            np.asarray(
                [risk_predictions[str(row["id"])][method] for row in splits["test"]]
            ),
            samples=int(risk_config.get("bootstrap_samples", 1000)),
            seed=int(project["seed"]),
        )
        for method in methods
        if method != primary
    }
    risk_rows = []
    for row in rows:
        risk_rows.append(
            {
                "id": row["id"],
                "split": row["split"],
                "speaker_id": row["speaker_id"],
                "condition": row["condition"],
                "severe_fact_error": row["severe_fact_error"],
                "unsupported": row["unsupported"],
                "risks": risk_predictions.get(str(row["id"]), {}),
            }
        )
    results_dir = output_dir / "results"
    write_jsonl(results_dir / "risk_predictions.jsonl", risk_rows)
    _write_selective_outputs(
        rows,
        risk_predictions,
        primary,
        methods[primary]["thresholds"],
        results_dir / "selective_outputs.jsonl",
    )
    write_json(results_dir / "availability.json", availability)
    result = {
        "protocol": project["protocol"],
        "seed": int(project["seed"]),
        "primary_method": primary,
        "development_model_selection": development_selection,
        "preselected_comparator": comparator,
        "methods": methods,
        "primary_vs_comparator_cluster_bootstrap": bootstrap,
        "primary_vs_method_cluster_bootstrap": bootstrap_by_method,
        "claim_gates": _claim_gates(config, methods, primary, comparator, bootstrap),
    }
    write_json(results_dir / "metrics.json", result)
    write_json(results_dir / "development_model_selection.json", development_selection)
    write_json(results_dir / "selective.json", {
        method: values["selective"] for method, values in methods.items()
    })
    write_json(results_dir / "condition_analysis.json", {
        method: values["by_condition"] for method, values in methods.items()
    })
    return result


def ece_score(labels: np.ndarray, risks: np.ndarray, bins: int = 15) -> float:
    if len(labels) == 0:
        return float("nan")
    order = np.argsort(risks)
    chunks = np.array_split(order, min(bins, len(order)))
    return float(
        sum(
            len(chunk) / len(labels)
            * abs(float(labels[chunk].mean()) - float(risks[chunk].mean()))
            for chunk in chunks
            if len(chunk)
        )
    )


def aurc_score(labels: np.ndarray, risks: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    order = np.argsort(risks, kind="stable")
    cumulative_risk = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    return float(np.mean(cumulative_risk))


def _evaluate_method(
    method: str,
    splits: dict[str, list[dict[str, Any]]],
    scores: dict[str, np.ndarray],
    risk_config: dict[str, Any],
    feature_names: list[str],
) -> dict[str, Any]:
    calibration_labels = _labels(splits["calibration"])
    test_rows = splits["test"]
    test_labels = _labels(test_rows)
    test_risks = scores["test"]
    target_coverage = float(risk_config["target_coverage"])
    tau2 = _coverage_threshold(scores["calibration"], target_coverage)
    accepted_cal = scores["calibration"] < tau2
    tau1 = (
        float(np.quantile(scores["calibration"][accepted_cal], 1.0 - float(risk_config["mark_fraction"])))
        if np.any(accepted_cal)
        else tau2
    )
    fixed_keep = _fixed_coverage_mask(test_risks, target_coverage)
    operational_keep = test_risks < tau2
    metrics = {
        "features": feature_names,
        "calibration_aurc": aurc_score(calibration_labels, scores["calibration"]),
        "ece": ece_score(test_labels, test_risks, int(risk_config["ece_bins"])),
        "brier": float(np.mean(np.square(test_risks - test_labels))),
        "aurc": aurc_score(test_labels, test_risks),
        "auroc": _safe_auroc(test_labels, test_risks),
        "auprc": float(average_precision_score(test_labels, test_risks)),
        "thresholds": {"mark": tau1, "abstain": tau2},
        "selective": {
            "fixed_target_coverage": _selective_metrics(
                test_rows, fixed_keep, target_coverage
            ),
            "calibration_threshold": _selective_metrics(
                test_rows, operational_keep, target_coverage
            ),
        },
        "by_condition": {},
        "by_fact_type": {},
    }
    for field in ("condition", "fact_type"):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(test_rows):
            groups[str(row[field])].append(index)
        destination = "by_condition" if field == "condition" else "by_fact_type"
        for group, indices in sorted(groups.items()):
            index_array = np.asarray(indices, dtype=int)
            group_labels = test_labels[index_array]
            group_risks = test_risks[index_array]
            keep = group_risks < tau2
            metrics[destination][group] = {
                "n": len(indices),
                "base_severe_error_rate": float(group_labels.mean()),
                "ece": ece_score(group_labels, group_risks, int(risk_config["ece_bins"])),
                "abstain_rate": float(1.0 - keep.mean()),
                "selective_severe_error_rate": (
                    float(group_labels[keep].mean()) if np.any(keep) else None
                ),
            }
    return metrics


def _selective_metrics(
    rows: list[dict[str, Any]],
    keep: np.ndarray,
    target_coverage: float,
) -> dict[str, Any]:
    labels = _labels(rows)
    unsupported = np.asarray([float(row["unsupported"]) for row in rows])
    quality = np.asarray(
        [sentence_chrf(str(row["reference"]), str(row["translation"])) for row in rows]
    )
    correct_clean = np.asarray(
        [
            row["condition"] == "clean" and not bool(row["severe_fact_error"])
            for row in rows
        ],
        dtype=bool,
    )
    rejected = ~keep
    return {
        "target_coverage": target_coverage,
        "coverage": float(keep.mean()),
        "num_kept": int(keep.sum()),
        "severe_error_rate": float(labels[keep].mean()) if np.any(keep) else None,
        "unsupported_rate": (
            float(unsupported[keep].mean()) if np.any(keep) else None
        ),
        "retained_chrf": float(quality[keep].mean()) if np.any(keep) else None,
        "all_output_chrf": float(quality.mean()),
        "all_severe_error_rate": float(labels.mean()),
        "all_unsupported_rate": float(unsupported.mean()),
        "clean_over_abstention_rate": (
            float(np.sum(rejected & correct_clean) / np.sum(correct_clean))
            if np.any(correct_clean)
            else None
        ),
    }


def _build_estimator(
    method: str, seed: int, risk_config: dict[str, Any] | None = None
) -> Pipeline:
    if method == "mlp_audio_grounded":
        classifier: Any = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            random_state=seed,
            early_stopping=False,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    elif method == "hgb_audio_grounded":
        classifier = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", classifier),
        ]
    elif method == "interaction_audio_grounded":
        configured = (risk_config or {}).get("interaction_model", {})
        params = {**INTERACTION_MODEL_DEFAULTS, **configured}
        classifier = ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            class_weight="balanced",
            n_jobs=int(params.get("n_jobs", 8)),
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", classifier),
        ]
    else:
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    return Pipeline(steps)


def _fit_platt(raw_risk: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration split must contain both risk labels")
    logits = _logit(raw_risk).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=2000)
    calibrator.fit(logits, labels)
    return calibrator


def _apply_platt(calibrator: LogisticRegression, raw_risk: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(_logit(raw_risk).reshape(-1, 1))[:, 1]


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _method_features(method: str, rows: list[dict[str, Any]], protocol: str) -> list[str]:
    if method in {
        "logreg_audio_grounded",
        "mlp_audio_grounded",
        "hgb_audio_grounded",
        "interaction_audio_grounded",
    }:
        if protocol == "formal":
            return FULL_FEATURES
        return [
            feature
            for feature in FULL_FEATURES
            if any(math.isfinite(_feature_value(row, feature)) for row in rows)
        ]
    if method not in METHOD_FEATURES:
        raise ValueError(f"Unknown risk method: {method}")
    return METHOD_FEATURES[method]


def _development_model_selection(
    train_rows: list[dict[str, Any]],
    risk_config: dict[str, Any],
    seed: int,
    protocol: str,
) -> dict[str, Any]:
    """Select the risk-head family without using calibration or test rows.

    Every validation fold contains speakers absent from its fitting fold.  The
    calibration split remains reserved for Platt scaling and action thresholds.
    """

    selection = risk_config.get("model_selection", {})
    candidates = [str(value) for value in selection.get("candidates", [])]
    if not candidates:
        return {}
    groups = np.asarray([str(row["speaker_id"]) for row in train_rows])
    folds = min(int(selection.get("folds", 5)), len(np.unique(groups)))
    if folds < 2:
        raise ValueError("Grouped development selection requires at least two speakers")
    labels = _labels(train_rows)
    splitter = GroupKFold(n_splits=folds)
    results: dict[str, Any] = {}
    for method in candidates:
        feature_names = _method_features(method, train_rows, protocol)
        matrix, _ = _matrix(train_rows, feature_names)
        estimator = _build_estimator(method, seed, risk_config)
        predictions = cross_val_predict(
            estimator,
            matrix,
            labels,
            groups=groups,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        fold_scores = []
        fold_speakers = []
        for _, validation in splitter.split(matrix, labels, groups):
            fold_scores.append(aurc_score(labels[validation], predictions[validation]))
            fold_speakers.append(len(np.unique(groups[validation])))
        results[method] = {
            "pooled_aurc": aurc_score(labels, predictions),
            "mean_fold_aurc": float(np.mean(fold_scores)),
            "std_fold_aurc": float(np.std(fold_scores)),
            "fold_aurc": fold_scores,
            "fold_speakers": fold_speakers,
            "features": feature_names,
        }
    selected = min(results, key=lambda method: results[method]["pooled_aurc"])
    return {
        "scope": "train_only_speaker_grouped_cross_validation",
        "selection_metric": "pooled_aurc",
        "num_rows": len(train_rows),
        "num_speakers": len(np.unique(groups)),
        "folds": folds,
        "selected": selected,
        "candidates": results,
    }


def _availability_reason(
    feature_names: list[str],
    splits: dict[str, list[dict[str, Any]]],
) -> str | None:
    if not feature_names:
        return "no usable features"
    for feature in feature_names:
        for split in ("train", "calibration", "test"):
            available = sum(
                math.isfinite(_feature_value(row, feature)) for row in splits[split]
            )
            if available == 0:
                return f"{feature} is entirely missing in {split}"
    return None


def _matrix(
    rows: list[dict[str, Any]], feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[_feature_value(row, feature) for feature in feature_names] for row in rows],
        dtype=np.float64,
    )
    return matrix, _labels(rows)


def _feature_value(row: dict[str, Any], feature: str) -> float:
    value = row.get("features", {}).get(feature)
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["severe_fact_error"]) for row in rows], dtype=np.int64)


def _coverage_threshold(scores: np.ndarray, coverage: float) -> float:
    if not 0.0 < coverage < 1.0:
        raise ValueError("Coverage must lie strictly between zero and one")
    return float(np.quantile(scores, coverage, method="higher"))


def _fixed_coverage_mask(scores: np.ndarray, coverage: float) -> np.ndarray:
    keep_count = max(1, min(len(scores), int(round(coverage * len(scores)))))
    order = np.argsort(scores, kind="stable")
    keep = np.zeros(len(scores), dtype=bool)
    keep[order[:keep_count]] = True
    return keep


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def _select_comparator(methods: dict[str, Any], primary: str) -> str:
    full_heads = {
        "logreg_audio_grounded",
        "mlp_audio_grounded",
        "hgb_audio_grounded",
        "interaction_audio_grounded",
        "logreg_model_agnostic",
    }
    candidates = [
        method
        for method in methods
        if method != primary
        and not method.startswith("ablation_")
        and method not in full_heads
    ]
    if not candidates:
        candidates = [method for method in methods if method != primary]
    if not candidates:
        return primary
    # Comparator selection uses calibration AURC only.
    return min(candidates, key=lambda method: methods[method]["calibration_aurc"])


def _cluster_bootstrap_difference(
    rows: list[dict[str, Any]],
    primary_scores: np.ndarray,
    comparator_scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[str(row["speaker_id"])].append(index)
    names = sorted(clusters)
    labels = _labels(rows)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.asarray(
            [index for name in sampled for index in clusters[str(name)]], dtype=int
        )
        differences.append(
            aurc_score(labels[indices], primary_scores[indices])
            - aurc_score(labels[indices], comparator_scores[indices])
        )
    point = aurc_score(labels, primary_scores) - aurc_score(labels, comparator_scores)
    return {
        "metric": "aurc_primary_minus_comparator",
        "num_speakers": len(names),
        "num_bootstrap_samples": samples,
        "point": point,
        "ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
    }


def _claim_gates(
    config: dict[str, Any],
    methods: dict[str, Any],
    primary: str,
    comparator: str,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    if config["project"]["protocol"] != "formal":
        return {"eligible": False, "reason": "engineering smoke results cannot support claims"}
    gate_config = config["claim_gates"]
    primary_fixed = methods[primary]["selective"]["fixed_target_coverage"]
    checks = {
        "aurc_better_than_preselected_comparator": (
            methods[primary]["aurc"] < methods[comparator]["aurc"]
        ),
        "cluster_bootstrap_aurc_ci_below_zero": bootstrap["ci95"][1] < 0.0,
        "clean_over_abstention_within_limit": (
            primary_fixed["clean_over_abstention_rate"]
            <= float(gate_config["max_clean_over_abstention"])
        ),
        "retained_quality_within_tolerance": (
            primary_fixed["retained_chrf"]
            >= primary_fixed["all_output_chrf"]
            - float(gate_config["quality_drop_tolerance_chrf"])
        ),
    }
    for baseline in gate_config.get("require_aurc_improvement_over", []):
        if baseline in methods:
            checks[f"aurc_better_than_{baseline}"] = (
                methods[primary]["aurc"] < methods[baseline]["aurc"]
            )
    return {"eligible": True, "passed": all(checks.values()), "checks": checks}


def _validate_training_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    speakers = {
        split: {str(row["speaker_id"]) for row in rows}
        for split, rows in splits.items()
    }
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        overlap = speakers[left] & speakers[right]
        if overlap:
            raise ValueError(f"Speaker leakage between {left} and {right}: {len(overlap)}")
    for split, rows in splits.items():
        if not rows or len(set(_labels(rows).tolist())) < 2:
            raise ValueError(f"Risk training split {split} is empty or single-class")


def _write_selective_outputs(
    rows: list[dict[str, Any]],
    risks: dict[str, dict[str, float]],
    method: str,
    thresholds: dict[str, float],
    path: Path,
) -> None:
    outputs = []
    for row in rows:
        row_id = str(row["id"])
        risk = float(risks[row_id][method])
        if risk >= float(thresholds["abstain"]):
            action = "abstain"
            rendered = "[ABSTAIN] The critical speech evidence is unclear; please repeat."
            uncertain_spans: list[str] = []
        elif risk >= float(thresholds["mark"]):
            action = "mark_uncertain"
            rendered, uncertain_spans = _mark_uncertain(
                str(row["translation"]),
                str(row.get("evidence_translation", "")),
                str(row["fact_type"]),
            )
        else:
            action = "translate"
            rendered = str(row["translation"])
            uncertain_spans = []
        outputs.append(
            {
                "id": row_id,
                "split": row["split"],
                "risk": risk,
                "action": action,
                "output": rendered,
                "uncertain_spans": uncertain_spans,
                "risk_method": method,
            }
        )
    write_jsonl(path, outputs)


def _mark_uncertain(
    translation: str,
    evidence_translation: str,
    fact_type: str,
) -> tuple[str, list[str]]:
    if fact_type == "number":
        number_words = "|".join(
            sorted((re.escape(value) for value in NUMBER_WORDS), key=len, reverse=True)
        )
        pattern = re.compile(
            rf"(?<!\w)(?:[+-]?\d+(?:[.,]\d+)?|{number_words})(?!\w)",
            flags=re.IGNORECASE,
        )
    elif fact_type == "negation":
        markers = "|".join(
            sorted((re.escape(value) for value in NEGATIVE_MARKERS), key=len, reverse=True)
        )
        pattern = re.compile(rf"(?<!\w)(?:{markers})(?!\w)", flags=re.IGNORECASE)
    else:
        return f"<uncertain>{translation}</uncertain>", [translation]
    spans = [match.group(0) for match in pattern.finditer(translation)]
    if not spans:
        return f"{translation} <uncertain>[critical fact]</uncertain>", []
    rendered = pattern.sub(lambda match: f"<uncertain>{match.group(0)}</uncertain>", translation)
    return rendered, spans
