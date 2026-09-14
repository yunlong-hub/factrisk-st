"""Descriptive frozen-head sensitivity to observed empty probe/sample text.

No model fitting, label changes or writes to primary experiment artifacts.
"""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import joblib
import numpy as np

from factrisk.eval.evaluation import apply_calibration, describe, keep_top, matrix
from factrisk.core.io import read_jsonl, sha256_file, write_json, write_jsonl
from factrisk.eval.significance_summary import COHORTS, OUT, ROOT
from factrisk.core.text import normalized_distance, pairwise_mean_distance

DEST = OUT / 'robustness/empty'
DISTANCES = ('perturb_mean_distance', 'perturb_max_distance', 'sample_mean_distance')


def observed_texts(prediction):
    """Absence/non-string is missing data; an actual string, even blank, is observed."""
    if not isinstance(prediction.get('translation'), str):
        raise ValueError('Missing/non-text base translation')
    probes, samples = [], []
    for item in prediction.get('probe_translations', []):
        if not isinstance(item, dict) or not isinstance(item.get('translation'), str):
            raise ValueError('Missing/non-text probe translation')
        probes.append(item['translation'].strip())
    for item in prediction.get('sample_translations', []):
        if not isinstance(item, str):
            raise ValueError('Missing/non-text sample translation')
        samples.append(item.strip())
    return probes, samples


def distance_features(prediction, include_empty):
    probes, samples = observed_texts(prediction)
    if not include_empty:
        probes = [p for p in probes if p]
        samples = [s for s in samples if s]
    distances = [normalized_distance(prediction['translation'].strip(), p) for p in probes]
    pairs = [normalized_distance(a, b) for a, b in itertools.combinations(samples, 2)]
    return dict(perturb_mean_distance=float(np.mean(distances)) if distances else np.nan,
                perturb_max_distance=max(distances) if distances else np.nan,
                sample_mean_distance=(float(np.mean(pairs)) if pairs else np.nan)
                if include_empty else pairwise_mean_distance(samples))


def cohort_inputs(name, folder):
    directory = OUT / folder
    if name.endswith('_controlled'):
        feature_paths = [directory.parent / 'features.jsonl']
        unresolved = directory.parent / 'labels_unresolved.jsonl'
        if unresolved.exists():
            feature_paths.append(unresolved)
    else:
        feature_paths = [directory.parent / ('numeric_features.jsonl' if '_fr_' in name else 'features.jsonl')]
    prediction_path = directory.parent / 'predictions/qwen2_audio.jsonl'
    head_dir = OUT / ('seamless_controlled' if name.startswith('seamless') else 'corrected') / 'number/checkpoints'
    return directory, feature_paths, prediction_path, head_dir


def run():
    # Write and retain protocol BEFORE examining differences or derived metrics.
    protocol = dict(version=1, role='descriptive_implementation_sensitivity',
        convention='Include actual empty strings in text edit distances; absent/non-text outputs fail closed.',
        preserved='All frozen labels, heads, calibration, thresholds, metadata-free signatures and primary results.',
        cohorts=list(COHORTS), methods='All available frozen heads (18 per cohort)',
        outcomes=['AURC delta', '90% retained-set symmetric difference', 'frozen-threshold retained-set difference', 'K/E/U/coverage'],
        training='none', selection='none', bootstrap='none; deterministic paired implementation comparison')
    protocol_path = DEST / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing protocol differs')
    write_json(protocol_path, protocol)
    report, changes, inputs = {}, [], {}
    for name, folder in COHORTS.items():
        directory, feature_paths, pred_path, head_dir = cohort_inputs(name, folder)
        paths = feature_paths + [pred_path, directory/'predictions.jsonl', directory/'metrics.json']
        inputs.update({str(p.relative_to(ROOT)): sha256_file(p) for p in paths})
        all_features = [r for path in feature_paths for r in read_jsonl(path)]
        indexed = {r['id']: r for r in all_features}
        if len(indexed) != len(all_features):
            raise ValueError(f'Duplicate features: {name}')
        saved = read_jsonl(directory/'predictions.jsonl')
        predictions = {r['id']: r for r in read_jsonl(pred_path)}
        rows = [indexed[r['id']] for r in saved]
        altered = copy.deepcopy(rows)
        affected, blanks = [], dict(base=0, probe=0, sample=0)
        for row, new, frozen in zip(rows, altered, saved):
            if row['severe_fact_error'] != frozen['label']:
                raise ValueError('Frozen label mismatch')
            pred = predictions[row['id']]
            if pred['translation'].strip() != row['translation'].strip():
                raise ValueError('Cached hypothesis mismatch')
            probes, samples = observed_texts(pred)
            blanks['base'] += int(not pred['translation'].strip())
            blanks['probe'] += probes.count('')
            blanks['sample'] += samples.count('')
            old_values = distance_features(pred, False)
            new_values = distance_features(pred, True)
            for key, value in old_values.items():
                if not np.isclose(row['features'][key], value, atol=1e-12, rtol=0, equal_nan=True):
                    raise ValueError(f'Original distance mismatch: {name}/{row["id"]}/{key}')
            changed = {k: dict(original=old_values[k], include_empty=new_values[k]) for k in DISTANCES
                       if not np.isclose(old_values[k], new_values[k], atol=1e-12, rtol=0, equal_nan=True)}
            new['features'].update(new_values)
            if changed:
                affected.append(row['id'])
                changes.append(dict(cohort=name, id=row['id'], label=row['severe_fact_error'],
                                    blank_probes=probes.count(''), blank_samples=samples.count(''), features=changed))
        methods = {}
        for path in sorted(head_dir.glob('*.joblib')):
            method = path.stem
            if method not in saved[0]['risks']:
                continue
            inputs[str(path.relative_to(ROOT))] = sha256_file(path)
            checkpoint = joblib.load(path)
            raw = checkpoint['model'].predict_proba(matrix(rows, checkpoint['features']))[:, 1]
            probability = apply_calibration(raw, checkpoint['calibration'])
            expected = np.array([[r['risks'][method]['raw'], r['risks'][method]['calibrated']] for r in saved])
            if not np.allclose(np.c_[raw, probability], expected, atol=1e-12, rtol=0):
                raise ValueError(f'Frozen risk mismatch: {name}/{method}')
            modified = checkpoint['model'].predict_proba(matrix(altered, checkpoint['features']))[:, 1]
            calibrated = apply_calibration(modified, checkpoint['calibration'])
            threshold = checkpoint['threshold']
            before, after = describe(rows, raw, probability, threshold), describe(altered, modified, calibrated, threshold)
            ids = [r['id'] for r in rows]
            top_before, top_after = keep_top(raw, .9, ids), keep_top(modified, .9, ids)
            op_before, op_after = probability < threshold, calibrated < threshold
            methods[method] = dict(original=before, include_empty=after,
                delta_aurc=after['aurc_known']-before['aurc_known'],
                risk_changed_rows=int(np.sum(abs(modified-raw)>1e-12)),
                top90_set_symmetric_difference=int(np.sum(top_before != top_after)),
                operational_set_symmetric_difference=int(np.sum(op_before != op_after)))
        report[name] = dict(n=len(rows), blank_outputs=blanks, affected_rows=affected, methods=methods)
        print(f'{name}: {len(rows)} rows, {blanks}, {len(affected)} changed, {len(methods)} heads verified', flush=True)
    for path, digest in inputs.items():
        if sha256_file(ROOT/path) != digest:
            raise ValueError(f'Input changed during analysis: {path}')
    write_jsonl(DEST/'affected_rows.jsonl', changes)
    write_json(DEST/'results.json', dict(protocol=protocol, cohorts=report, inputs=inputs,
               code_sha256=sha256_file(__file__), primary_inputs_unchanged=True))
    lines = ['# Empty-output sensitivity', '',
        'Frozen-head descriptive comparison; no refitting, relabeling, threshold changes or replacement of primary results.', '',
        '| Cohort | N | Blank base/probe/sample | Changed feature rows | Full ΔAURC | Full top90 set Δ |',
        '|---|---:|---|---:|---:|---:|']
    for name, item in report.items():
        full = item['methods']['et_full']
        blank = '/'.join(str(item['blank_outputs'][k]) for k in ('base','probe','sample'))
        lines.append(f'| {name} | {item["n"]} | {blank} | {len(item["affected_rows"])} | {full["delta_aurc"]:.8f} | {full["top90_set_symmetric_difference"]} |')
    lines += ['', 'All 18 heads per cohort reproduce the stored raw and calibrated risk within 1e-12 before the intervention. '
              'The JSON records full K/E/U, coverage, all heads and both selection rules. '
              'Signature-based number/negation features already include observed blanks and remain unchanged. '
              'These are implementation-sensitivity results, not evidence of semantic label accuracy.', '',
              'Reproduce: `PYTHONPATH=src python -m factrisk.eval.empty_sensitivity`']
    (DEST/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    run()
