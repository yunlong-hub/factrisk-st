"""Fixed-protocol training-seed robustness on cached number-only features.

Run ``python -m factrisk.eval.seed_stability``. No ST inference or test-driven selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn

from factrisk.eval.evaluation import apply_calibration, calibrate, describe, matrix
from factrisk.core.io import read_jsonl, sha256_file, write_json, write_jsonl
from factrisk.core.models import FEATURES, build_model
from factrisk.eval.significance_summary import COHORTS

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'exp/factrisk'
SEEDS = (20260906, 20260907, 20260908, 20260909, 20260910)
METHODS = ('et_full', 'et_asr_evidence', 'et_without_stability')
TRAIN = {'qwen': 'corrected', 'seamless': 'seamless_controlled'}
EXTERNAL = {
    'qwen_de_natural': 'real_clean/features.jsonl',
    'qwen_de_stress': 'real_stress/features.jsonl',
    'seamless_de_natural': 'seamless_real/features.jsonl',
    'qwen_fr_clean': 'fr_en/qwen/numeric_features.jsonl',
    'seamless_fr_clean': 'fr_en/seamless/numeric_features.jsonl',
    'qwen_fr_stress': 'fr_en/stress/qwen/numeric_features.jsonl',
    'seamless_fr_stress': 'fr_en/stress/seamless/numeric_features.jsonl',
}


def split_rows(rows):
    """Match original known-label train/cal order; retain unresolved test rows."""
    rows = [r for r in rows if r['fact_type'] == 'number']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate number IDs')
    splits = {s: [r for r in rows if r['split'] == s]
              for s in ('train', 'calibration', 'test')}
    if sum(map(len, splits.values())) != len(rows):
        raise ValueError('Unexpected split')
    for s, subset in splits.items():
        others = {r['speaker_id'] for k, rr in splits.items() if k != s for r in rr}
        if others & {r['speaker_id'] for r in subset}:
            raise ValueError('Speaker leakage')
    for s in ('train', 'calibration'):
        splits[s] = [r for r in splits[s] if r['severe_fact_error'] is not None]
        if {r['severe_fact_error'] for r in splits[s]} != {0, 1}:
            raise ValueError('Both classes required in train/calibration')
    return splits


def distribution(values):
    a = np.asarray(values, float)
    if len(a) != len(SEEDS) or not np.isfinite(a).all():
        raise ValueError('Expected all five finite seed measurements')
    return dict(mean=float(a.mean()), sample_sd=float(a.std(ddof=1)),
                minimum=float(a.min()), maximum=float(a.max()), seeds=list(SEEDS))


def load_inputs():
    training, cohorts, paths = {}, {}, []
    for backend, folder in TRAIN.items():
        source = OUT / folder / 'features.jsonl'
        rows = read_jsonl(source); paths.append(source)
        unresolved = source.parent/'labels_unresolved.jsonl'
        if unresolved.exists():
            rows += read_jsonl(unresolved); paths.append(unresolved)
        training[backend] = split_rows(rows)
        cohorts[f'{backend}_de_controlled'] = training[backend]['test']
        paths += [source.parent/'number/checkpoints'/f'{m}.joblib' for m in METHODS]
    for name, path in EXTERNAL.items():
        source = OUT/path; paths.append(source)
        rows = read_jsonl(source)
        if len({r['id'] for r in rows}) != len(rows) or any(
                r['split'] != 'test' or r['fact_type'] != 'number' for r in rows):
            raise ValueError(f'Invalid external cohort {name}')
        cohorts[name] = rows
    for folder in COHORTS.values():
        paths += [OUT/folder/'metrics.json', OUT/folder/'predictions.jsonl']
    return training, cohorts, paths


def freeze(destination, training, paths):
    source_files = [Path(__file__), ROOT/'src/factrisk/core/models.py', ROOT/'src/factrisk/eval/evaluation.py']
    content = dict(seeds=list(SEEDS), methods=list(METHODS), cohorts=list(COHORTS),
        role='Post-primary descriptive training-seed sensitivity; no best-seed selection',
        fitting='German controlled known-label train and calibration only; no external adaptation',
        statistics='Five-seed mean/sample SD/range and matched-seed deltas, not confidence intervals',
        features={m: FEATURES[m] for m in METHODS},
        hyperparameters=dict(n_estimators=600, min_samples_leaf=20, max_features=.7,
                             class_weight='balanced', calibration='nonnegative-slope Platt',
                             threshold='calibration probability 0.9 quantile; strict less-than retain'),
        inputs={str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
        code={str(p.relative_to(ROOT)): sha256_file(p) for p in source_files},
        fitted_ids={b: {s:[r['id'] for r in rows] for s, rows in split.items()}
                    for b, split in training.items()},
        environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__))
    path = destination/'protocol.json'
    if path.exists():
        saved = json.loads(path.read_text())
        # The experimental design and shared method remain frozen. Local runner
        # fixes are separately recorded, not silently rewritten into the freeze.
        old_design = json.loads(json.dumps(saved['design']))
        new_design = json.loads(json.dumps(content))
        # Code identity is the recorded file name and hash, so a pure package
        # layout move neither rewrites the freeze nor trips the reuse guard.
        old_design['code'] = {Path(key).name: value for key, value in old_design['code'].items()}
        new_design['code'] = {Path(key).name: value for key, value in new_design['code'].items()}
        runner = Path(__file__).name
        old_design['code'].pop(runner, None)
        new_design['code'].pop(runner, None)
        if old_design != new_design:
            raise ValueError('Frozen seed protocol/input/code differs; refusing reuse')
    else:
        write_json(path, dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), design=content))
    write_json(destination/'execution_code.json', dict(code=content['code'],
        note='Runner reference checks permit 1e-6 calibration rounding but require raw-risk agreement and identical deployed selections. Scientific design is unchanged.'))
    return sha256_file(path)


def check_reference(cohort, method, rows, raw, probability, threshold):
    folder = OUT/COHORTS[cohort]
    previous = {r['id']:r for r in read_jsonl(folder/'predictions.jsonl')}
    if set(previous) != {r['id'] for r in rows}:
        raise ValueError(f'Original test IDs differ: {cohort}')
    a = np.array([previous[r['id']]['risks'][method]['raw'] for r in rows])
    b = np.array([previous[r['id']]['risks'][method]['calibrated'] for r in rows])
    if any(previous[r['id']]['label'] != r['severe_fact_error'] for r in rows):
        raise ValueError('Original labels differ')
    raw_error = float(np.max(abs(a-raw)))
    probability_error = float(np.max(abs(b-probability)))
    if raw_error > 1e-12 or probability_error > 1e-6:
        raise ValueError(f'Reference-seed reproduction failed {cohort}/{method}: {raw_error}, {probability_error}')
    original_head = joblib.load(OUT/TRAIN[cohort.split('_')[0]]/'number/checkpoints'/f'{method}.joblib')
    if abs(threshold-original_head['threshold']) > 1e-6:
        raise ValueError('Reference threshold differs')
    if not np.array_equal(b < original_head['threshold'], probability < threshold):
        raise ValueError('Reference operational selection differs')
    return dict(max_absolute_raw_difference=raw_error,
                max_absolute_calibrated_difference=probability_error, ids_labels_match=True,
                operational_selection_identical=True,
                threshold_absolute_difference=abs(threshold-original_head['threshold']))


def flatten(cohort, method, seed, metric):
    fixed, operational = metric['fixed']['0.9'], metric['operational']
    return dict(cohort=cohort, method=method, seed=seed, aurc=metric['aurc_known'],
        kept90=fixed['kept'], errors90=fixed['known_errors'], unresolved90=fixed['unresolved'],
        lower90=fixed['risk_lower'], upper90=fixed['risk_upper'],
        frozen_coverage=operational['coverage'], frozen_kept=operational['kept'],
        frozen_errors=operational['known_errors'], frozen_unresolved=operational['unresolved'],
        frozen_lower=operational['risk_lower'], frozen_upper=operational['risk_upper'])


def run(destination):
    from torch.utils.tensorboard import SummaryWriter
    destination = Path(destination)
    training, cohorts, paths = load_inputs()
    protocol_hash = freeze(destination, training, paths)
    output, reference = [], {}
    write_json(destination/'status.json', dict(stage='running', protocol_sha256=protocol_hash))
    try:
        for backend, split in training.items():
            train, cal = split['train'], split['calibration']
            y_train = np.array([r['severe_fact_error'] for r in train])
            y_cal = np.array([r['severe_fact_error'] for r in cal])
            for seed in SEEDS:
                folder = destination/backend/str(seed)
                writer = SummaryWriter(str(folder/'tensorboard'))
                for method in METHODS:
                    checkpoint_path = folder/'checkpoints'/f'{method}.joblib'
                    names = FEATURES[method]
                    if checkpoint_path.exists():
                        head = joblib.load(checkpoint_path)
                        if head['protocol_sha256'] != protocol_hash:
                            raise ValueError('Stale resumed checkpoint')
                    else:
                        model = build_model(method, seed=seed)
                        xtrain, xcal = matrix(train, names), matrix(cal, names)
                        if not all(np.isfinite(x).any(axis=0).all() for x in (xtrain, xcal)):
                            raise ValueError('Entire feature channel absent')
                        started = time.monotonic()
                        model.fit(xtrain, y_train)
                        calraw = model.predict_proba(xcal)[:,1]
                        ab = calibrate(calraw, y_cal)
                        pcal = apply_calibration(calraw, ab)
                        head = dict(model=model, features=names, calibration=ab,
                            threshold=float(np.quantile(pcal, .9)), seed=seed,
                            protocol_sha256=protocol_hash, fitting_seconds=time.monotonic()-started)
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(head, checkpoint_path)
                    writer.add_scalar(f'{method}/fitting_seconds', head['fitting_seconds'], 0)
                    writer.add_scalar(f'{method}/calibration_threshold', head['threshold'], 0)
                    for name, rows in cohorts.items():
                        if not name.startswith(backend+'_'): continue
                        raw = head['model'].predict_proba(matrix(rows, names))[:,1]
                        probability = apply_calibration(raw, head['calibration'])
                        metric = describe(rows, raw, probability, head['threshold'])
                        dest = folder/name/method
                        write_json(dest/'metrics.json', metric)
                        write_jsonl(dest/'predictions.jsonl', [dict(id=r['id'], label=r['severe_fact_error'],
                            raw=float(a), calibrated=float(b)) for r,a,b in zip(rows,raw,probability)])
                        output.append(flatten(name, method, seed, metric))
                        writer.add_scalar(f'{method}/{name}/aurc', metric['aurc_known'], 0)
                        if seed == SEEDS[0]:
                            reference[name+'/'+method] = check_reference(name, method, rows, raw, probability, head['threshold'])
                    print(f'{backend}/{seed}/{method} complete', flush=True)
                writer.close()
        summary = {}
        for cohort in COHORTS:
            subset = [r for r in output if r['cohort']==cohort]
            measures = [k for k in output[0] if k not in ('cohort','method','seed')]
            summary[cohort] = dict(methods={m: {k: distribution([r[k] for r in subset if r['method']==m])
                for k in measures} for m in METHODS}, matched_deltas={})
            for comparator in METHODS[1:]:
                differences = []
                for seed in SEEDS:
                    full = next(r for r in subset if r['seed']==seed and r['method']=='et_full')
                    other = next(r for r in subset if r['seed']==seed and r['method']==comparator)
                    differences.append(dict(seed=seed, **{k:full[k]-other[k] for k in measures}))
                summary[cohort]['matched_deltas'][comparator] = dict(per_seed=differences,
                    summary={k:distribution([r[k] for r in differences]) for k in measures})
        with (destination/'per_seed.csv').open('w', newline='') as handle:
            csvwriter = csv.DictWriter(handle, fieldnames=list(output[0])); csvwriter.writeheader(); csvwriter.writerows(output)
        write_json(destination/'summary.json', dict(protocol_sha256=protocol_hash, cohorts=summary,
            reference_seed_checks=reference, measurement_rows=len(output), trained_heads=30))
        lines = ['# Training-seed stability', '',
            'Five fixed seeds; same German train/calibration, feature definitions and hyperparameters. All external cohorts are evaluation only.',
            'Run: `PYTHONPATH=src python -m factrisk.eval.seed_stability`.',
            'Values are mean ± sample SD across training seeds, not confidence intervals. Original primary results remain unchanged; no best seed is selected.',
            '', '| Cohort | Full AURC | Full − ASR+evidence range | Full − no-stability range |', '|---|---:|---:|---:|']
        for name, value in summary.items():
            a=value['methods']['et_full']['aurc']
            d=[value['matched_deltas'][m]['summary']['aurc'] for m in METHODS[1:]]
            lines.append(f'| {name} | {a["mean"]:.6f} ± {a["sample_sd"]:.6f} | [{d[0]["minimum"]:.6f}, {d[0]["maximum"]:.6f}] | [{d[1]["minimum"]:.6f}, {d[1]["maximum"]:.6f}] |')
        lines += ['', 'Per-seed K/E/U counts, error bounds and frozen-threshold coverage: `per_seed.csv`; all matched deltas: `summary.json`.',
                  'Frozen input hashes, exact fitted/split IDs, package versions, and hyperparameters: `protocol.json`. Individual heads, predictions and TensorBoard events are retained per backend/seed.']
        (destination/'README.md').write_text('\n'.join(lines)+'\n')
        write_json(destination/'status.json', dict(stage='complete', protocol_sha256=protocol_hash,
            summary_sha256=sha256_file(destination/'summary.json'), measurement_rows=len(output)))
    except Exception as exc:
        write_json(destination/'status.json', dict(stage='failed', error=repr(exc), protocol_sha256=protocol_hash))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=OUT/'robustness/seeds')
    run(parser.parse_args().destination)
