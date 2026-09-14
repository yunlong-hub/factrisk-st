from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from factrisk.core.io import read_jsonl, sha256_file, stable_int, write_json


DISPLAY_NAMES = {
    "sequence_probability": "Sequence probability",
    "token_entropy": "Token entropy",
    "beam_margin": "Beam margin",
    "asr_confidence": "ASR confidence",
    "qe": "COMET-QE (WMT20)",
    "self_consistency": "Self-consistency",
    "perturbation_only": "Audio perturbation only",
    "entailment_only": "Evidence consistency only",
    "ablation_text": "Text only",
    "ablation_text_acoustic": "Text + acoustic",
    "ablation_text_acoustic_stability": "Text + acoustic + stability",
    "logreg_model_agnostic": "Architecture-independent FactRisk",
    "logreg_audio_grounded": "FactRisk (logistic)",
    "hgb_audio_grounded": "FactRisk (boosted trees)",
    "interaction_audio_grounded": "FactRisk-Interact (ours)",
    "mlp_audio_grounded": "FactRisk (MLP)",
}


def build_paper_assets(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    project = config["project"]
    output_dir = Path(project["output_dir"])
    paper_dir = Path(config["paper"]["output_dir"])
    tables_dir = paper_dir / "tables" / "generated"
    figures_dir = paper_dir / "figures" / "generated"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "results" / "metrics.json"
    selective_path = output_dir / "results" / "selective.json"
    condition_path = output_dir / "results" / "condition_analysis.json"
    availability_path = output_dir / "results" / "availability.json"
    prediction_path = output_dir / "results" / "risk_predictions.jsonl"
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    with selective_path.open(encoding="utf-8") as handle:
        selective = json.load(handle)
    with condition_path.open(encoding="utf-8") as handle:
        conditions = json.load(handle)
    with availability_path.open(encoding="utf-8") as handle:
        availability = json.load(handle)
    predictions = read_jsonl(prediction_path)

    _risk_table(metrics["methods"], tables_dir)
    _development_selection_table(metrics, tables_dir)
    _bootstrap_table(metrics, tables_dir)
    _ablation_table(metrics["methods"], tables_dir)
    _selective_table(selective, tables_dir)
    _condition_table(
        conditions[metrics["primary_method"]],
        metrics["primary_method"],
        tables_dir,
    )
    _fact_type_table(
        metrics["methods"][metrics["primary_method"]]["by_fact_type"],
        tables_dir,
    )
    if project["protocol"] == "formal":
        _extension_table(metrics, output_dir, tables_dir)
    _risk_coverage_svg(metrics, predictions, figures_dir / "risk_coverage.svg")
    _reliability_svg(
        metrics["primary_method"], predictions, figures_dir / "reliability.svg"
    )
    _method_overview_svg(figures_dir / "method_overview.svg")
    _convert_svgs_to_pdf(figures_dir)
    _write_macros(metrics, tables_dir / "result_macros.tex")
    audit_paths = _write_audit_sheet(config, metrics, paper_dir / "audit")

    source_paths = [
        Path(config_path),
        Path(project["data_dir"]) / "manifest.jsonl",
        output_dir / "predictions" / "qwen2_audio.jsonl",
        output_dir / "evidence" / "whisper_nllb.jsonl",
        output_dir / "features.jsonl",
        metrics_path,
        selective_path,
        condition_path,
        availability_path,
        prediction_path,
        output_dir / "results" / "development_model_selection.json",
    ]
    qe_config = config.get("models", {}).get("optional_qe", {})
    if qe_config.get("enabled"):
        if qe_config.get("score_file"):
            source_paths.append(Path(qe_config["score_file"]))
        if qe_config.get("model_path"):
            checkpoint = Path(qe_config["model_path"])
            source_paths.extend([checkpoint, checkpoint.parents[1] / "hparams.yaml"])
    if project["protocol"] == "formal":
        repository_root = Path(__file__).resolve().parents[3]
        source_paths.extend(
            [repository_root / "pyproject.toml", repository_root / "configs" / "paper.yaml"]
            + sorted((repository_root / "src" / "factrisk").rglob("*.py"))
            + sorted((repository_root / "scripts").glob("*.sh"))
            + [repository_root / "papers" / "factriskst" / "main.tex"]
            + sorted((repository_root / "papers" / "factriskst" / "sections").glob("*.tex"))
            + [repository_root / "papers" / "factriskst" / "references.bib"]
        )
        for model_name in ("seamless_m4t", "qwen3_omni"):
            extension_dir = output_dir / "extensions" / model_name
            source_paths.extend(
                [
                    extension_dir / "predictions.jsonl",
                    extension_dir / "diagnostics.json",
                    extension_dir / "results" / "metrics.json",
                ]
            )
    source_paths = list(dict.fromkeys(path.resolve() for path in source_paths))
    source_paths.extend(
        Path(config["sources"][key])
        for key in ("pair_manifest", "interventions")
        if config.get("sources", {}).get(key)
    )
    snapshot = {
        "protocol": project["protocol"],
        "formal_eligible": project["protocol"] == "formal",
        "seed": int(project["seed"]),
        "primary_method": metrics["primary_method"],
        "comparator": metrics["preselected_comparator"],
        "claim_gates": metrics["claim_gates"],
        "constants": {
            "target_coverage": float(config["risk"]["target_coverage"]),
            "ece_bins": int(config["risk"]["ece_bins"]),
            "bootstrap_samples": int(config["risk"]["bootstrap_samples"]),
            "fact_types": list(config["data"]["fact_types"]),
        },
        "availability": availability,
        "results": metrics,
        "sources": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in source_paths
            if path.is_file()
        ],
        "generated": sorted(
            str(path)
            for path in list(tables_dir.glob("*"))
            + list(figures_dir.glob("*"))
            + audit_paths
        ),
    }
    snapshot_path = Path(config["paper"]["result_snapshot"])
    write_json(snapshot_path, snapshot)
    return {
        "protocol": project["protocol"],
        "paper_dir": str(paper_dir),
        "snapshot": str(snapshot_path),
        "num_source_files": len(snapshot["sources"]),
        "generated": snapshot["generated"],
    }


def _risk_table(methods: dict[str, Any], destination: Path) -> None:
    headers = ["Method", "ECE ↓", "Brier ↓", "AURC ↓", "AUROC ↑", "AUPRC ↑"]
    order = [
        "sequence_probability",
        "token_entropy",
        "beam_margin",
        "asr_confidence",
        "qe",
        "self_consistency",
        "perturbation_only",
        "entailment_only",
        "logreg_audio_grounded",
        "hgb_audio_grounded",
        "interaction_audio_grounded",
        "mlp_audio_grounded",
    ]
    rows = [
        [
            DISPLAY_NAMES.get(method, method),
            _format(values["ece"]),
            _format(values["brier"]),
            _format(values["aurc"]),
            _format(values["auroc"]),
            _format(values["auprc"]),
        ]
        for method in order
        if (values := methods.get(method)) is not None
    ]
    _write_table_bundle(destination / "risk_prediction", headers, rows)


def _development_selection_table(metrics: dict[str, Any], destination: Path) -> None:
    selection = metrics.get("development_model_selection", {})
    candidates = selection.get("candidates", {})
    if not candidates:
        return
    order = [
        "logreg_audio_grounded",
        "hgb_audio_grounded",
        "interaction_audio_grounded",
    ]
    headers = ["Risk head", "Train-CV AURC ↓", "Fold mean ↓", "Fold std."]
    rows = [
        [
            DISPLAY_NAMES.get(method, method),
            _format(values["pooled_aurc"]),
            _format(values["mean_fold_aurc"]),
            _format(values["std_fold_aurc"]),
        ]
        for method in order
        if (values := candidates.get(method)) is not None
    ]
    _write_table_bundle(destination / "development_selection", headers, rows)


def _ablation_table(methods: dict[str, Any], destination: Path) -> None:
    order = [
        "ablation_text",
        "ablation_text_acoustic",
        "ablation_text_acoustic_stability",
        "logreg_audio_grounded",
        "interaction_audio_grounded",
    ]
    selected = [(method, methods[method]) for method in order if method in methods]
    if not selected:
        return
    headers = ["Features", "ECE ↓", "Brier ↓", "AURC ↓", "AUROC ↑"]
    rows = [
        [
            DISPLAY_NAMES.get(method, method),
            _format(values["ece"]),
            _format(values["brier"]),
            _format(values["aurc"]),
            _format(values["auroc"]),
        ]
        for method, values in selected
    ]
    _write_table_bundle(destination / "feature_ablation", headers, rows)


def _bootstrap_table(metrics: dict[str, Any], destination: Path) -> None:
    comparisons = metrics.get("primary_vs_method_cluster_bootstrap", {})
    order = [
        "sequence_probability",
        "token_entropy",
        "qe",
        "self_consistency",
        "perturbation_only",
        "entailment_only",
        "ablation_text",
    ]
    headers = ["Comparator", "Comparator AURC", "FactRisk $-$ comparator", "Speaker 95% CI"]
    rows = []
    for method in order:
        if method not in comparisons:
            continue
        values = comparisons[method]
        low, high = values["ci95"]
        rows.append(
            [
                DISPLAY_NAMES.get(method, method),
                _format(metrics["methods"][method]["aurc"]),
                _format(values["point"]),
                f"[{low:.4f}, {high:.4f}]",
            ]
        )
    _write_table_bundle(destination / "bootstrap_comparisons", headers, rows)


def _selective_table(selective: dict[str, Any], destination: Path) -> None:
    headers = [
        "Method",
        "Coverage",
        "chrF ↑",
        "Severe Error ↓",
        "Unsupported ↓",
        "Over-abstention ↓",
    ]
    first = next(iter(selective.values()))["fixed_target_coverage"]
    rows = [
        [
            "No abstention",
            "100.0%",
            _format(first["all_output_chrf"], 2),
            _percent(first["all_severe_error_rate"]),
            _percent(first["all_unsupported_rate"]),
            "0.0%",
        ]
    ]
    order = [
        "sequence_probability",
        "token_entropy",
        "beam_margin",
        "asr_confidence",
        "qe",
        "self_consistency",
        "perturbation_only",
        "entailment_only",
        "logreg_audio_grounded",
        "hgb_audio_grounded",
        "interaction_audio_grounded",
        "mlp_audio_grounded",
    ]
    for method in order:
        if method not in selective:
            continue
        values = selective[method]
        fixed = values["fixed_target_coverage"]
        rows.append(
            [
                DISPLAY_NAMES.get(method, method),
                _percent(fixed["coverage"]),
                _format(fixed["retained_chrf"], 2),
                _percent(fixed["severe_error_rate"]),
                _percent(fixed["unsupported_rate"]),
                _percent(fixed["clean_over_abstention_rate"]),
            ]
        )
    _write_table_bundle(destination / "selective_translation", headers, rows)


def _fact_type_table(facts: dict[str, Any], destination: Path) -> None:
    headers = ["Fact type", "N", "Base error", "Abstain", "Selective error", "Reduction"]
    rows = []
    for fact_type, values in sorted(facts.items()):
        base = float(values["base_severe_error_rate"])
        selective = values["selective_severe_error_rate"]
        reduction = base - float(selective) if selective is not None else None
        rows.append(
            [
                fact_type,
                str(values["n"]),
                _percent(base),
                _percent(values["abstain_rate"]),
                _percent(selective),
                _percent(reduction),
            ]
        )
    _write_table_bundle(destination / "fact_type_analysis", headers, rows)


def _condition_table(
    conditions: dict[str, Any], method: str, destination: Path
) -> None:
    headers = [
        "Condition",
        "N",
        "Base error",
        "ECE",
        "Abstain",
        "Selective error",
    ]
    rows = [
        [
            condition.replace("_", " "),
            str(values["n"]),
            _percent(values["base_severe_error_rate"]),
            _format(values["ece"]),
            _percent(values["abstain_rate"]),
            _percent(values["selective_severe_error_rate"]),
        ]
        for condition, values in conditions.items()
    ]
    _write_table_bundle(destination / f"condition_{method}", headers, rows)


def _extension_table(
    metrics: dict[str, Any], output_dir: Path, destination: Path
) -> None:
    headers = [
        "System",
        "Scope",
        "Base chrF",
        "Base error",
        "Risk estimator",
        "Coverage",
        "Selective error",
    ]
    primary = metrics["methods"][metrics["primary_method"]]["selective"][
        "fixed_target_coverage"
    ]
    rows = [
        [
            "Qwen2-Audio",
            "confirmatory test",
            _format(primary["all_output_chrf"], 2),
            _percent(primary["all_severe_error_rate"]),
            "full FactRisk",
            _percent(primary["coverage"]),
            _percent(primary["severe_error_rate"]),
        ]
    ]
    for model_name, display_name in (
        ("seamless_m4t", "SeamlessM4T-v2"),
        ("qwen3_omni", "Qwen3-Omni"),
    ):
        diagnostic_path = output_dir / "extensions" / model_name / "diagnostics.json"
        if not diagnostic_path.is_file():
            continue
        with diagnostic_path.open(encoding="utf-8") as handle:
            diagnostic = json.load(handle)
        test_values = diagnostic["by_split"]["test"]
        extension_metrics_path = output_dir / "extensions" / model_name / "results" / "metrics.json"
        if extension_metrics_path.is_file():
            with extension_metrics_path.open(encoding="utf-8") as handle:
                extension_metrics = json.load(handle)
            extension_primary = extension_metrics["methods"][
                extension_metrics["primary_method"]
            ]["selective"]["fixed_target_coverage"]
            risk_name = "model-agnostic FactRisk"
            coverage = _percent(extension_primary["coverage"])
            selective_error = _percent(extension_primary["severe_error_rate"])
        else:
            risk_name = "diagnostic only"
            coverage = "--"
            selective_error = "--"
        rows.append(
            [
                display_name,
                "test" if model_name == "seamless_m4t" else "test-only",
                _format(test_values["chrf"], 2),
                _percent(test_values["severe_error_rate"]),
                risk_name,
                coverage,
                selective_error,
            ]
        )
    _write_table_bundle(destination / "extension_analysis", headers, rows)


def _write_table_bundle(base: Path, headers: list[str], rows: list[list[str]]) -> None:
    with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    markdown = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    markdown.extend("| " + " | ".join(row) + " |" for row in rows)
    base.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex = [
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        "\\toprule",
        " & ".join(_latex(value) for value in headers) + " \\\\",
        "\\midrule",
    ]
    latex.extend(" & ".join(_latex(value) for value in row) + " \\\\" for row in rows)
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    base.with_suffix(".tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def _risk_coverage_svg(
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    path: Path,
) -> None:
    test = [row for row in predictions if row["split"] == "test"]
    width, height, margin = 720, 440, 55
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    colors = itertools.cycle(
        ["#3366cc", "#dc3912", "#109618", "#990099", "#ff9900", "#0099c6", "#7a4eab"]
    )
    polylines = []
    legends = []
    plot_methods = [
        "sequence_probability",
        "token_entropy",
        "asr_confidence",
        "qe",
        "self_consistency",
        "perturbation_only",
        metrics["primary_method"],
    ]
    plot_methods = list(dict.fromkeys(method for method in plot_methods if method in metrics["methods"]))
    for color, method in zip(colors, plot_methods):
        available = [row for row in test if method in row["risks"]]
        if not available:
            continue
        ordered = sorted(available, key=lambda row: row["risks"][method])
        labels = np.asarray([float(row["severe_fact_error"]) for row in ordered])
        risks = np.cumsum(labels) / np.arange(1, len(labels) + 1)
        points = []
        for index, risk in enumerate(risks, start=1):
            coverage = index / len(risks)
            x = margin + coverage * plot_width
            y = height - margin - min(1.0, risk) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>'
        )
        legends.append((color, DISPLAY_NAMES.get(method, method)))
    svg = _svg_header(width, height, "Risk–coverage")
    svg.extend(_axes(width, height, margin, "Coverage", "Severe fact error"))
    svg.extend(polylines)
    for index, (color, label) in enumerate(legends):
        y = 24 + index * 18
        svg.append(f'<line x1="445" y1="{y}" x2="465" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="472" y="{y + 4}" font-size="11">{_xml(label)}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _reliability_svg(method: str, predictions: list[dict[str, Any]], path: Path) -> None:
    test = [row for row in predictions if row["split"] == "test" and method in row["risks"]]
    scores = np.asarray([row["risks"][method] for row in test], dtype=float)
    labels = np.asarray([row["severe_fact_error"] for row in test], dtype=float)
    width, height, margin = 520, 440, 55
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    chunks = np.array_split(np.argsort(scores), min(10, len(scores)))
    points = []
    for chunk in chunks:
        if not len(chunk):
            continue
        confidence, observed = float(scores[chunk].mean()), float(labels[chunk].mean())
        x = margin + confidence * plot_width
        y = height - margin - observed * plot_height
        points.append((x, y))
    svg = _svg_header(width, height, f"Reliability: {DISPLAY_NAMES.get(method, method)}")
    svg.extend(_axes(width, height, margin, "Predicted severe-error risk", "Observed frequency"))
    svg.append(
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" '
        'stroke="#888" stroke-dasharray="5 4"/>'
    )
    svg.append(
        '<polyline fill="none" stroke="#dc3912" stroke-width="2.5" points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        + '"/>'
    )
    svg.extend(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#dc3912"/>' for x, y in points
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _method_overview_svg(path: Path) -> None:
    width, height = 960, 390
    boxes = [
        (25, 155, 120, 62, "Source speech x", "#e8f1fb"),
        (190, 45, 150, 62, "Frozen direct ST", "#eef7e8"),
        (380, 45, 125, 62, "Translation y", "#eef7e8"),
        (190, 150, 150, 62, "Mild audio probes", "#fff3d9"),
        (190, 255, 150, 62, "Whisper → NLLB", "#f5eafb"),
        (380, 255, 125, 62, "Evidence text", "#f5eafb"),
        (560, 85, 185, 220, "Risk estimator", "#fde9e7"),
        (790, 125, 145, 140, "Selective action", "#e6f4f1"),
    ]
    svg = _svg_header(width, height, "FactRisk-ST: reference-free number-preservation risk estimation")
    svg.append(
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#445"/></marker></defs>'
    )
    arrows = [
        (145, 173, 190, 76),
        (340, 76, 380, 76),
        (145, 186, 190, 181),
        (145, 199, 190, 286),
        (340, 286, 380, 286),
        (505, 76, 560, 130),
        (340, 181, 560, 195),
        (505, 286, 560, 260),
        (745, 195, 790, 195),
    ]
    for x1, y1, x2, y2 in arrows:
        svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#445" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    for x, y, box_width, box_height, label, color in boxes:
        svg.append(
            f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" '
            f'rx="9" fill="{color}" stroke="#445" stroke-width="1.5"/>'
        )
        svg.append(
            f'<text x="{x + box_width / 2}" y="{y + 36}" text-anchor="middle" '
            f'font-size="14" font-weight="600">{_xml(label)}</text>'
        )
    for index, label in enumerate(
        ["generation confidence", "acoustic uncertainty", "translation stability", "evidence consistency"]
    ):
        svg.append(
            f'<text x="652" y="{145 + 32 * index}" text-anchor="middle" '
            f'font-size="12">{_xml(label)}</text>'
        )
    for index, (label, color) in enumerate(
        [("translate", "#26734d"), ("mark uncertain", "#a26300"), ("abstain / repeat", "#a12b2b")]
    ):
        svg.append(
            f'<text x="862" y="{167 + 31 * index}" text-anchor="middle" '
            f'font-size="12" fill="{color}" font-weight="600">{_xml(label)}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _axes(
    width: int, height: int, margin: int, x_label: str, y_label: str
) -> list[str]:
    values = [
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="13">{_xml(x_label)}</text>',
        (
            f'<text x="15" y="{height/2}" text-anchor="middle" font-size="13" '
            f'transform="rotate(-90 15 {height/2})">{_xml(y_label)}</text>'
        ),
    ]
    for tick in range(6):
        value = tick / 5
        x = margin + value * (width - 2 * margin)
        y = height - margin - value * (height - 2 * margin)
        values.append(f'<text x="{x:.1f}" y="{height-margin+18}" text-anchor="middle" font-size="10">{value:.1f}</text>')
        values.append(f'<text x="{margin-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">{value:.1f}</text>')
    return values


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-size="15" font-weight="600">{_xml(title)}</text>',
    ]


def _write_macros(metrics: dict[str, Any], path: Path) -> None:
    primary = metrics["methods"][metrics["primary_method"]]
    fixed = primary["selective"]["fixed_target_coverage"]
    operational = primary["selective"]["calibration_threshold"]
    comparator_name = metrics["preselected_comparator"]
    comparator = metrics["methods"][comparator_name]
    comparator_fixed = comparator["selective"]["fixed_target_coverage"]
    bootstrap = metrics["primary_vs_comparator_cluster_bootstrap"]
    lines = [
        f"\\newcommand{{\\PrimaryAURC}}{{{primary['aurc']:.4f}}}",
        f"\\newcommand{{\\PrimaryECE}}{{{primary['ece']:.4f}}}",
        f"\\newcommand{{\\PrimaryCoverage}}{{{100*fixed['coverage']:.1f}\\%}}",
        f"\\newcommand{{\\PrimarySelectiveRisk}}{{{100*fixed['severe_error_rate']:.1f}\\%}}",
        f"\\newcommand{{\\PrimarySelectiveChrF}}{{{fixed['retained_chrf']:.2f}}}",
        f"\\newcommand{{\\BaseSevereRisk}}{{{100*fixed['all_severe_error_rate']:.1f}\\%}}",
        f"\\newcommand{{\\PrimaryRiskReduction}}{{{100*(fixed['all_severe_error_rate']-fixed['severe_error_rate']):.1f}}}",
        f"\\newcommand{{\\PrimaryUnsupported}}{{{100*fixed['unsupported_rate']:.1f}\\%}}",
        f"\\newcommand{{\\PrimaryOverAbstention}}{{{100*fixed['clean_over_abstention_rate']:.1f}\\%}}",
        f"\\newcommand{{\\PrimaryOperationalCoverage}}{{{100*operational['coverage']:.1f}\\%}}",
        f"\\newcommand{{\\PrimaryOperationalRisk}}{{{100*operational['severe_error_rate']:.1f}\\%}}",
        f"\\newcommand{{\\PrimaryAUROC}}{{{primary['auroc']:.4f}}}",
        f"\\newcommand{{\\PrimaryBrier}}{{{primary['brier']:.4f}}}",
        f"\\newcommand{{\\ComparatorName}}{{{DISPLAY_NAMES.get(comparator_name, comparator_name)}}}",
        f"\\newcommand{{\\ComparatorAURC}}{{{comparator['aurc']:.4f}}}",
        f"\\newcommand{{\\ComparatorSelectiveRisk}}{{{100*comparator_fixed['severe_error_rate']:.1f}\\%}}",
        f"\\newcommand{{\\BootstrapDelta}}{{{bootstrap['point']:.4f}}}",
        f"\\newcommand{{\\BootstrapLow}}{{{bootstrap['ci95'][0]:.4f}}}",
        f"\\newcommand{{\\BootstrapHigh}}{{{bootstrap['ci95'][1]:.4f}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _convert_svgs_to_pdf(figures_dir: Path) -> None:
    try:
        import cairosvg
    except ImportError:
        return
    for source in figures_dir.glob("*.svg"):
        cairosvg.svg2pdf(url=str(source), write_to=str(source.with_suffix(".pdf")))


def _write_audit_sheet(
    config: dict[str, Any], metrics: dict[str, Any], destination: Path
) -> list[Path]:
    """Create a deterministic, blank bilingual-adjudication sheet.

    This is deliberately an audit *sample*, not a model-generated substitute
    for human labels.  The human columns remain empty in the released file.
    """

    output_dir = Path(config["project"]["output_dir"])
    data_dir = Path(config["project"]["data_dir"])
    manifest = {str(row["id"]): row for row in read_jsonl(data_dir / "manifest.jsonl")}
    features = {str(row["id"]): row for row in read_jsonl(output_dir / "features.jsonl")}
    actions = {
        str(row["id"]): row
        for row in read_jsonl(output_dir / "results" / "selective_outputs.jsonl")
    }
    primary = str(metrics["primary_method"])
    risks = {
        str(row["id"]): float(row["risks"][primary])
        for row in read_jsonl(output_dir / "results" / "risk_predictions.jsonl")
        if row["split"] == "test"
    }
    groups: dict[str, list[str]] = {
        "accepted_severe": [],
        "abstained_severe": [],
        "abstained_correct": [],
        "clean_correct_control": [],
    }
    for sample_id in risks:
        feature = features[sample_id]
        action = str(actions[sample_id]["action"])
        severe = bool(feature["severe_fact_error"])
        if severe and action != "abstain":
            groups["accepted_severe"].append(sample_id)
        elif severe and action == "abstain":
            groups["abstained_severe"].append(sample_id)
        elif not severe and action == "abstain":
            groups["abstained_correct"].append(sample_id)
        elif not severe and feature["condition"] == "clean":
            groups["clean_correct_control"].append(sample_id)

    seed = int(config["project"]["seed"])
    selected: list[tuple[str, str]] = []
    for group, candidates in groups.items():
        ranked = sorted(candidates, key=lambda value: stable_int(f"audit:{group}:{value}", seed))
        per_type: dict[str, list[str]] = {}
        for sample_id in ranked:
            per_type.setdefault(str(features[sample_id]["fact_type"]), []).append(sample_id)
        chosen: list[str] = []
        for fact_type in sorted(per_type):
            chosen.extend(per_type[fact_type][:10])
        if len(chosen) < 20:
            chosen_set = set(chosen)
            chosen.extend(value for value in ranked if value not in chosen_set)
        selected.extend((group, value) for value in chosen[:20])

    destination.mkdir(parents=True, exist_ok=True)
    sheet = destination / "stratified_audit_sample.csv"
    fieldnames = [
        "audit_group",
        "id",
        "fact_type",
        "condition",
        "source_text_de",
        "translation_en",
        "evidence_translation_en",
        "reference_en",
        "expected_slot",
        "contrast_slot",
        "automatic_severe",
        "automatic_unsupported",
        "risk",
        "action",
        "human_slot_correct",
        "human_unsupported",
        "human_severity",
        "adjudicator_notes",
    ]
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, sample_id in selected:
            item = manifest[sample_id]
            feature = features[sample_id]
            writer.writerow(
                {
                    "audit_group": group,
                    "id": sample_id,
                    "fact_type": feature["fact_type"],
                    "condition": feature["condition"],
                    "source_text_de": item["source_text"],
                    "translation_en": feature["translation"],
                    "evidence_translation_en": feature["evidence_translation"],
                    "reference_en": feature["reference"],
                    "expected_slot": item["expected_slot"],
                    "contrast_slot": item["contrast_slot"],
                    "automatic_severe": feature["severe_fact_error"],
                    "automatic_unsupported": feature["unsupported"],
                    "risk": f"{risks[sample_id]:.8f}",
                    "action": actions[sample_id]["action"],
                    "human_slot_correct": "",
                    "human_unsupported": "",
                    "human_severity": "",
                    "adjudicator_notes": "",
                }
            )
    readme = destination / "README.md"
    readme.write_text(
        "# Stratified bilingual audit sample\n\n"
        "This deterministic sample contains up to 20 rows from each of four "
        "decision/error strata. The `human_*` fields are intentionally blank; "
        "the paper does not claim a completed human evaluation. Two bilingual "
        "annotators should label independently, then adjudicate disagreements.\n",
        encoding="utf-8",
    )
    return [sheet, readme]


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number:.{digits}f}" if np.isfinite(number) else "--"


def _percent(value: Any) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):.1f}%"


def _latex(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("↓", "$\\downarrow$")
        .replace("↑", "$\\uparrow$")
    )


def _xml(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
