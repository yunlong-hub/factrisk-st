import numpy as np

from factrisk.method.risk import (
    _build_estimator,
    _development_model_selection,
    _mark_uncertain,
    aurc_score,
    ece_score,
)


def test_perfect_ranking_has_better_aurc() -> None:
    labels = np.asarray([0, 0, 1, 1])
    good = np.asarray([0.1, 0.2, 0.8, 0.9])
    bad = np.asarray([0.9, 0.8, 0.2, 0.1])
    assert aurc_score(labels, good) < aurc_score(labels, bad)


def test_perfect_risk_has_zero_ece() -> None:
    labels = np.asarray([0, 0, 1, 1])
    risks = labels.astype(float)
    assert ece_score(labels, risks, bins=2) == 0.0


def test_uncertain_span_marker_never_uses_gold() -> None:
    rendered, spans = _mark_uncertain(
        "There were twelve passengers.",
        "There were thirteen passengers.",
        "number",
    )
    assert "<uncertain>twelve</uncertain>" in rendered
    assert spans == ["twelve"]


def test_mid_risk_action_marks_even_when_evidence_agrees() -> None:
    rendered, spans = _mark_uncertain(
        "There were twelve passengers.",
        "There were twelve passengers.",
        "number",
    )
    assert "<uncertain>twelve</uncertain>" in rendered
    assert spans == ["twelve"]


def test_interaction_estimator_uses_frozen_regularized_forest() -> None:
    estimator = _build_estimator("interaction_audio_grounded", 7)
    forest = estimator.named_steps["classifier"]
    assert forest.n_estimators == 600
    assert forest.min_samples_leaf == 20
    assert forest.max_features == 0.7


def test_development_selection_is_speaker_grouped() -> None:
    rows = []
    for speaker in range(4):
        for index in range(8):
            label = int(index >= 4)
            rows.append(
                {
                    "speaker_id": f"s{speaker}",
                    "severe_fact_error": label,
                    "features": {"sequence_nll": float(label) + index / 100.0},
                }
            )
    result = _development_model_selection(
        rows,
        {
            "model_selection": {
                "candidates": ["sequence_probability"],
                "folds": 4,
            }
        },
        seed=7,
        protocol="engineering_smoke",
    )
    assert result["num_speakers"] == 4
    assert result["candidates"]["sequence_probability"]["fold_speakers"] == [1, 1, 1, 1]
