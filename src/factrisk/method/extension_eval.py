from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from factrisk.method.features import build_features
from factrisk.core.io import write_yaml
from factrisk.paper.assets import build_paper_assets
from factrisk.method.risk import train_and_evaluate


def evaluate_extension(config: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    resolved = deepcopy(config)
    formal_output = Path(config["project"]["output_dir"])
    extension_dir = formal_output / "extensions" / model_name
    prediction_path = extension_dir / "predictions.jsonl"
    evidence_path = formal_output / "evidence" / "whisper_nllb.jsonl"
    resolved["project"]["protocol"] = f"formal_extension_{model_name}"
    resolved["project"]["output_dir"] = str(extension_dir)
    resolved["sources"]["direct_predictions"] = str(prediction_path)
    resolved["sources"]["evidence_predictions"] = str(evidence_path)
    # The formal QE scores belong to Qwen2-Audio outputs and must not be
    # attached to a different extension model by shared sample ID.
    resolved["models"]["optional_qe"]["enabled"] = False
    resolved["risk"]["primary_model"] = "logreg_model_agnostic"
    resolved["risk"]["models"] = [
        "asr_confidence",
        "perturbation_only",
        "entailment_only",
        "logreg_model_agnostic",
    ]
    # The extension intentionally tests the frozen architecture-independent
    # logistic head.  Its predictions do not expose the generator confidence
    # and stochastic-sample features required by the primary interaction head.
    resolved["risk"]["model_selection"] = {
        "scope": "not_applicable_frozen_extension_head",
        "candidates": [],
        "enforce_primary": False,
    }
    resolved["paper"]["output_dir"] = str(extension_dir / "paper_assets")
    resolved["paper"]["result_snapshot"] = str(
        extension_dir / "paper_assets" / "results_snapshot.json"
    )
    config_path = extension_dir / "resolved_config.yaml"
    write_yaml(config_path, resolved)
    feature_report = build_features(resolved)
    metrics = train_and_evaluate(resolved)
    assets = build_paper_assets(resolved, config_path)
    return {
        "model": model_name,
        "features": feature_report,
        "primary_metrics": metrics["methods"]["logreg_model_agnostic"],
        "assets": assets,
    }
