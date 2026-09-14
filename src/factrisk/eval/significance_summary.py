"""Export existing paired intervals without refitting or selecting comparisons.

Cohort-level rows come from the frozen per-row risks; the condition breakdown for
the stress cohorts reuses the same draws, clusters, and statistic.
"""
from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from factrisk.core.io import sha256_file, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "exp/factrisk"
COHORTS = {
    "qwen_de_controlled": "corrected/number",
    "qwen_de_natural": "real_clean/frozen_transfer",
    "qwen_de_stress": "real_stress/frozen_transfer",
    "seamless_de_controlled": "seamless_controlled/number",
    "seamless_de_natural": "seamless_real/frozen_transfer",
    "qwen_fr_clean": "fr_en/qwen/numeric_clean",
    "seamless_fr_clean": "fr_en/seamless/numeric_clean",
    "qwen_fr_stress": "fr_en/stress/qwen/numeric_stress",
    "seamless_fr_stress": "fr_en/stress/seamless/numeric_stress",
}
BOOTSTRAP_REPLICATES = 2000
CONDITION_COHORTS = ("qwen_de_stress", "qwen_fr_stress", "seamless_fr_stress")
CONDITION_COMPARATOR = "et_without_acoustic"


def fast_aurc(labels, scores):
    """Same expected tie-breaking statistic as evaluation.aurc, vectorized."""
    order = np.argsort(scores, kind='stable')
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(labels)])
    means = np.add.reduceat(labels[order], starts) / counts
    expected = np.repeat(means, counts)
    return float(np.mean(np.cumsum(expected) / np.arange(1, len(labels)+1)))


def compute():
    """Share 2,000 speaker draws across all methods; never alter frozen metrics."""
    from factrisk.eval.evaluation import aurc
    cache_path = OUT/'metrics/significance_computed.json'
    previous = json.loads(cache_path.read_text()).get('cohorts', {}) if cache_path.exists() else {}
    computed, pending = {}, {}
    for cohort, folder in COHORTS.items():
        directory = OUT / folder
        paths = [directory/'metrics.json', directory/'predictions.jsonl']
        if not all(p.exists() for p in paths):
            pending[cohort] = 'Metrics or frozen per-row risks not yet available'
            continue
        if cohort in previous and all((ROOT/p).exists() and sha256_file(ROOT/p) == digest
                                      for p, digest in previous[cohort]['inputs'].items()):
            computed[cohort] = previous[cohort]
            print(f'{cohort}: reused input-hash-verified bootstrap results', flush=True)
            continue
        metrics = json.loads(paths[0].read_text())
        predictions = read_jsonl(paths[1])
        if len({r['id'] for r in predictions}) != len(predictions):
            raise ValueError(f'Duplicate predictions: {cohort}')
        groups = defaultdict(list)
        clusters = None
        if cohort.endswith('_stress'):
            path = OUT/'real_stress/features.jsonl' if cohort == 'qwen_de_stress' else directory.parent/'numeric_features.jsonl'
            paths.append(path)
            metadata = {r['id']: r for r in read_jsonl(path)}
            clusters = {r['id']: r['cluster_id'] for r in metadata.values()}
        for i, row in enumerate(predictions):
            groups[clusters[row['id']] if clusters else row['speaker_id']].append(i)
        labels = np.array([np.nan if r['label'] is None else r['label'] for r in predictions])
        known = np.isfinite(labels)
        risks, unavailable = {}, {}
        for method, metric in metrics['methods'].items():
            if 'aurc_known' not in metric:
                unavailable[method] = metric.get('reason', 'No stored AURC')
                continue
            if any(method not in r['risks'] for r in predictions):
                unavailable[method] = 'Missing per-row frozen risk'
                continue
            score = np.array([r['risks'][method]['raw'] for r in predictions])
            if not np.isfinite(score).all():
                unavailable[method] = 'Nonfinite frozen score'
                continue
            measured = fast_aurc(labels[known], score[known])
            if not np.isclose(measured, metric['aurc_known'], rtol=0, atol=1e-12):
                raise ValueError(f'Stored AURC mismatch: {cohort}/{method}')
            if not np.isclose(measured, aurc(labels[known], score[known]), rtol=0, atol=1e-12):
                raise ValueError('Tie-statistic mismatch')
            risks[method] = score
        if 'et_full' not in risks:
            raise ValueError(f'Primary risk missing: {cohort}')
        methods = [m for m in risks if m != 'et_full']
        draws = []
        rng = np.random.default_rng(20260906)
        keys = sorted(groups)
        for replicate in range(2000):
            indices = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
            indices = indices[known[indices]]
            if not len(indices):
                continue
            y = labels[indices]
            full = fast_aurc(y, risks['et_full'][indices])
            draws.append([full-fast_aurc(y, risks[m][indices]) for m in methods])
        array = np.asarray(draws)
        comparisons = {}
        for i, method in enumerate(methods):
            ci = np.quantile(array[:,i], [.025, .975]).tolist()
            existing = metrics.get('comparisons', {}).get(method, {}).get('paired_aurc_difference_ci')
            if existing and not np.allclose(ci, existing, rtol=0, atol=1e-12):
                raise ValueError(f'Existing interval mismatch: {cohort}/{method}: {ci} != {existing}')
            comparisons[method] = dict(paired_aurc_difference_ci=ci, speakers=len(keys),
                bootstrap_replicates=len(draws), bootstrap_seed=20260906,
                cluster_unit='speaker_donor_connected_component' if clusters else 'speaker',
                role='primary' if method == 'et_asr_evidence' else 'descriptive',
                existing_interval_reproduced=bool(existing))
        computed[cohort] = dict(comparisons=comparisons, unavailable=unavailable,
                               inputs={str(p.relative_to(ROOT)): sha256_file(p) for p in paths})
        print(f'{cohort}: {len(comparisons)} paired intervals; {len(keys)} clusters', flush=True)
    write_json(OUT/'metrics/significance_computed.json', dict(cohorts=computed, pending=pending,
        statistic='Known-label expected-tie AURC: Full minus comparator',
        bootstrap_replicates=2000, seed=20260906, multiple_comparison_adjustment='none',
        interpretation='ASR+evidence is primary; other nominal 95% percentile intervals are descriptive. No p-values or family-wise claims.',
        code_sha256=sha256_file(__file__)))


def record(cohort, method, primary, comparator, comparison, source):
    interval = comparison.get("paired_aurc_difference_ci")
    low, high = interval if interval else (None, None)
    if interval and not low <= high:
        raise ValueError(f"Invalid interval in {source}")
    conclusion = ("not_computed" if interval is None else
                  "full_lower_nominal_95" if high < 0 else
                  "full_higher_nominal_95" if low > 0 else
                  "cannot_exclude_no_difference")
    return dict(cohort=cohort, comparator=method,
                full_aurc=primary["aurc_known"], comparator_aurc=comparator["aurc_known"],
                delta_aurc=primary["aurc_known"]-comparator["aurc_known"],
                ci_low=low, ci_high=high, conclusion=conclusion,
                clusters=comparison.get("speakers"),
                cluster_unit=comparison.get("cluster_unit"),
                bootstrap_replicates=comparison.get("bootstrap_replicates"),
                bootstrap_seed=20260906 if interval else None,
                multiple_comparison_adjustment="none",
                source=str(source.relative_to(ROOT)), source_sha256=sha256_file(source))


def condition_interval(predictions, metadata, index, comparator):
    """Paired Full-minus-comparator AURC interval over one corruption condition."""
    labels = np.array([np.nan if predictions[i]['label'] is None else predictions[i]['label']
                       for i in index])
    known = np.isfinite(labels)
    full = np.array([predictions[i]['risks']['et_full']['raw'] for i in index])
    other = np.array([predictions[i]['risks'][comparator]['raw'] for i in index])
    groups = defaultdict(list)
    for position, i in enumerate(index):
        groups[metadata[predictions[i]['id']]['cluster_id']].append(position)
    rng = np.random.default_rng(20260906)
    keys = sorted(groups)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        sample = sample[known[sample]]
        if not len(sample):
            continue
        draws.append(fast_aurc(labels[sample], full[sample]) - fast_aurc(labels[sample], other[sample]))
    array = np.asarray(draws)
    low, high = np.quantile(array, [.025, .975]).tolist()
    return dict(n=len(index), known=int(known.sum()), clusters=len(keys),
                delta_aurc=fast_aurc(labels[known], full[known]) - fast_aurc(labels[known], other[known]),
                ci_low=low, ci_high=high, bootstrap_replicates=len(draws),
                conclusion=('full_lower_nominal_95' if high < 0 else
                            'full_higher_nominal_95' if low > 0 else
                            'cannot_exclude_no_difference'))


def write_condition_summary():
    """Per-corruption intervals for the stress cohorts, using the frozen cohort draws."""
    rows = []
    for cohort in CONDITION_COHORTS:
        folder = COHORTS[cohort]
        directory = OUT / folder
        if not (directory/'predictions.jsonl').exists():
            continue
        predictions = read_jsonl(directory/'predictions.jsonl')
        feature_path = (OUT/'real_stress/features.jsonl' if cohort == 'qwen_de_stress'
                        else directory.parent/'numeric_features.jsonl')
        metadata = {r['id']: r for r in read_jsonl(feature_path)}
        conditions = defaultdict(list)
        for i, row in enumerate(predictions):
            conditions[metadata[row['id']]['condition']].append(i)
        for condition, index in sorted(conditions.items()):
            row = condition_interval(predictions, metadata, np.asarray(index), CONDITION_COMPARATOR)
            rows.append(dict(cohort=cohort, condition=condition, comparator=CONDITION_COMPARATOR,
                             cluster_unit='speaker_donor_connected_component',
                             bootstrap_seed=20260906,
                             source=str((directory/'predictions.jsonl').relative_to(ROOT)),
                             source_sha256=sha256_file(directory/'predictions.jsonl'), **row))
    if not rows:
        print('No stress condition rows available.')
        return rows
    path = OUT/'metrics/condition_significance.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported {len(rows)} condition-level comparisons to {path.relative_to(ROOT)}.')
    return rows


def main():
    rows = []
    loaded = {}
    computed_path = OUT/'metrics/significance_computed.json'
    computed = json.loads(computed_path.read_text())['cohorts'] if computed_path.exists() else {}
    for cohort, folder in COHORTS.items():
        path = OUT / folder / "metrics.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        loaded[cohort] = value
        for method, metric in value["methods"].items():
            if method == "et_full" or "aurc_known" not in metric:
                continue
            comparison = value.get("comparisons", {}).get(method, {})
            source = path
            if not comparison and method in computed.get(cohort, {}).get('comparisons', {}):
                for input_path, digest in computed[cohort]['inputs'].items():
                    if sha256_file(ROOT/input_path) != digest:
                        raise ValueError(f'Stale computed interval: {input_path}')
                comparison = computed[cohort]['comparisons'][method]
                source = computed_path
            rows.append(record(cohort, method, value["methods"]["et_full"], metric,
                               comparison, source))
    for kind, folder in (("hypothesis_nll", "likelihood"),
                         ("attention", "corrected/number/attention")):
        path = OUT / folder / "metrics.json"
        auxiliary = json.loads(path.read_text())
        for split, cohort, analysis in (
            ("controlled", "qwen_de_controlled", "corrected/number/additional_analysis.json"),
            ("natural_clean", "qwen_de_natural", "real_clean/frozen_transfer/additional_analysis.json"),
        ):
            metric = auxiliary["metrics"][split]
            source = path
            comparison = metric.get("full_minus_likelihood", {})
            if kind == "attention":
                source = OUT / analysis
                comparison = json.loads(source.read_text()).get("attention_comparison") or {}
            rows.append(record(cohort, kind, loaded[cohort]["methods"]["et_full"], metric,
                               comparison, source))
    for cohort, folder in COHORTS.items():
        if '_fr_' not in cohort or cohort not in loaded:
            continue
        for kind, key in (('attention', 'full_minus_attention'), ('likelihood', 'full_minus_likelihood')):
            path = (OUT/folder).parent/kind/'metrics.json'
            if not path.exists():
                continue
            auxiliary = json.loads(path.read_text())['metrics']
            if key not in auxiliary:
                continue
            rows.append(record(cohort, 'hypothesis_nll' if kind == 'likelihood' else kind,
                               loaded[cohort]['methods']['et_full'], auxiliary, auxiliary[key], path))
    csv_path = OUT / "metrics/significance_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# German–English and French–English paired comparisons", "",
             "All differences are Full minus comparator on known-label AURC (lower is better).",
             "Intervals are paired cluster-bootstrap percentile 95% intervals, conditional on frozen fitted heads; added intervals are recomputed from frozen per-row risks.",
             "ASR+evidence is the fixed primary comparator; all other comparisons are descriptive.",
             "They are **not adjusted for multiple comparisons**; no family-wise superiority claim or p-value is inferred.",
             "An interval containing zero cannot exclude no difference and does not establish equivalence or non-inferiority.",
             "`not_computed` means no paired interval was saved, not evidence of no difference.",
             "The CSV includes every available registered comparator, including rows without intervals; source paths and hashes permit verification.",
             "Pending cohorts: " + ', '.join(c for c in COHORTS if c not in loaded),
             "", "| Cohort | Comparator | ΔAURC | Nominal 95% CI | Clusters | Interpretation |",
             "|---|---|---:|---|---:|---|"]
    for row in rows:
        ci = "not computed" if row["ci_low"] is None else f'[{row["ci_low"]:.8g}, {row["ci_high"]:.8g}]'
        lines.append(f'| {row["cohort"]} | {row["comparator"]} | {row["delta_aurc"]:.6f} | {ci} | {row["clusters"] or "—"} | {row["conclusion"]} |')
    (OUT / "metrics/significance_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Exported {len(rows)} comparisons; no model, threshold, or original metrics changed.")
    write_condition_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute', action='store_true', help='Compute missing paired intervals from frozen predictions')
    if parser.parse_args().compute:
        compute()
    main()
