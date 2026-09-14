"""Validate robustness evidence from assets, identities and measured values."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from factrisk.eval.evaluation import describe
from factrisk.core.io import read_jsonl, sha256_file
from factrisk.eval.significance_summary import COHORTS

METHODS = ('et_full', 'et_asr_evidence', 'et_without_stability')
SEEDS = (20260906, 20260907, 20260908, 20260909, 20260910)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def same(actual, expected, message, tolerance=1e-12):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), message)
        for key in expected:
            same(actual[key], expected[key], message+'/'+key, tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(bool(np.isclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True)), message)
    else:
        require(actual == expected, message)


def verify_hashes(root, inputs):
    require(bool(inputs), 'Empty input provenance')
    for name, digest in inputs.items():
        path = Path(name) if Path(name).is_absolute() else root/name
        require(path.is_file() and sha256_file(path) == digest, f'Changed robustness input: {name}')


def load(path):
    return json.loads(path.read_text())


def require_seeds(root, base):
    folder = base/'seeds'
    protocol = load(folder/'protocol.json')
    design = protocol['design']
    digest = sha256_file(folder/'protocol.json')
    require(design['seeds'] == list(SEEDS) and design['methods'] == list(METHODS)
            and set(design['cohorts']) == set(COHORTS), 'Seed design incomplete')
    verify_hashes(root, design['inputs'])
    verify_hashes(root, load(folder/'execution_code.json')['code'])
    summary, state = load(folder/'summary.json'), load(folder/'status.json')
    require(summary['protocol_sha256'] == digest == state['protocol_sha256'], 'Seed protocol mismatch')
    require(state['summary_sha256'] == sha256_file(folder/'summary.json'), 'Seed summary changed')
    require(summary['trained_heads'] == 30 and summary['measurement_rows'] == 135,
            'Seed reported counts incomplete')
    records = list(csv.DictReader((folder/'per_seed.csv').open()))
    expected = {(c, m, s) for c in COHORTS for m in METHODS for s in SEEDS}
    require(len(records) == 135 and {(r['cohort'],r['method'],int(r['seed'])) for r in records} == expected,
            'Seed measurements incomplete or duplicated')
    require(len(list(folder.glob('*/20*/checkpoints/*.joblib'))) == 30, 'Seed head files incomplete')
    checks = summary['reference_seed_checks']
    require(set(checks) == {c+'/'+m for c in COHORTS for m in METHODS}, 'Reference seed checks incomplete')
    for name, check in checks.items():
        require(check['ids_labels_match'] and check['operational_selection_identical']
                and check['max_absolute_raw_difference'] <= 1e-12
                and check['max_absolute_calibrated_difference'] <= 1e-6
                and check['threshold_absolute_difference'] <= 1e-6, 'Reference seed failed: '+name)
    for c, m, seed in sorted(expected):
        backend = c.split('_')[0]
        head = joblib.load(folder/backend/str(seed)/'checkpoints'/f'{m}.joblib')
        require(head['seed'] == seed and head['protocol_sha256'] == digest, 'Seed head provenance mismatch')
        directory = folder/backend/str(seed)/c/m
        rows = read_jsonl(directory/'predictions.jsonl')
        original = read_jsonl(root/'exp/factrisk'/COHORTS[c]/'predictions.jsonl')
        index = {r['id']: r for r in original}
        require(len(rows) == len(index) and {r['id'] for r in rows} == set(index), 'Seed prediction IDs differ')
        require(all(r['label'] == index[r['id']]['label'] for r in rows), 'Seed labels differ')
        raw, probability = np.array([r['raw'] for r in rows]), np.array([r['calibrated'] for r in rows])
        require(np.isfinite(raw).all() and np.isfinite(probability).all(), 'Nonfinite seed risk')
        # Clean over-abstention needs the real condition metadata, not a default.
        from factrisk.eval.empty_sensitivity import cohort_inputs
        _, paths, _, _ = cohort_inputs(c, COHORTS[c])
        metadata = {r['id']:r for p in paths for r in read_jsonl(p)}
        metric = describe([metadata[r['id']] for r in rows], raw, probability, head['threshold'])
        same(load(directory/'metrics.json'), metric, 'Seed metric replay')
        record = next(r for r in records if r['cohort']==c and r['method']==m and int(r['seed'])==seed)
        same(float(record['aurc']), metric['aurc_known'], 'Seed CSV AURC mismatch')
        if seed == SEEDS[0]:
            a = np.array([index[r['id']]['risks'][m]['raw'] for r in rows])
            b = np.array([index[r['id']]['risks'][m]['calibrated'] for r in rows])
            require(np.max(abs(a-raw)) <= 1e-12 and np.max(abs(b-probability)) <= 1e-6,
                    'Actual reference seed risks differ')


def require_empty(root, base):
    result = load(base/'empty/results.json')
    same(result['protocol'], load(base/'empty/protocol.json'), 'Empty protocol differs')
    verify_hashes(root, result['inputs'])
    require(result['code_sha256'] == sha256_file(root/'src/factrisk/eval/empty_sensitivity.py'), 'Empty analysis code differs')
    require(set(result['cohorts']) == set(COHORTS), 'Empty cohort coverage incomplete')
    affected = read_jsonl(base/'empty/affected_rows.jsonl')
    expected = set()
    for cohort, row in result['cohorts'].items():
        original = load(root/'exp/factrisk'/COHORTS[cohort]/'metrics.json')['methods']
        require(len(row['methods']) == 18 and set(row['methods']) == set(original), 'Empty head coverage incomplete')
        expected.update((cohort, i) for i in row['affected_rows'])
        for method, value in row['methods'].items():
            same(value['original'], {k:original[method][k] for k in value['original']}, 'Empty original metric differs')
            same(value['delta_aurc'], value['include_empty']['aurc_known']-value['original']['aurc_known'], 'Empty delta differs')
            require(value['include_empty']['n'] == row['n'], 'Empty cohort denominator changed')
    require(len(affected) == len(expected) and {(r['cohort'],r['id']) for r in affected} == expected,
            'Empty affected row evidence incomplete')


def require_environment(root, base):
    result = load(base/'environment/replay.json')
    verify_hashes(root, result['inputs'])
    require(result['code_sha256'] == sha256_file(root/'src/factrisk/pipeline/environment_replay.py'), 'Environment replay code differs')
    require(result['lock_sha256'] == sha256_file(root/'configs/environments/evaluation.lock.txt'), 'Evaluation lock differs')
    prefix = Path(result['prefix']).resolve()
    require(str(prefix) != result['base_prefix'] and (prefix/'pyvenv.cfg').is_file(), 'No isolated runtime evidence')
    require('include-system-site-packages = false' in (prefix/'pyvenv.cfg').read_text().lower(), 'Runtime inherits system packages')
    require(Path(result['executable']).parent == Path(result['prefix'])/'bin', 'Unexpected runtime executable')
    require(all(Path(p).resolve().is_relative_to(prefix) for p in result['sys_path'] if 'site-packages' in p), 'Inherited runtime package path')
    require(result['pip_check'] == 'No broken requirements found.', 'Isolated pip check failed')
    require(set(result['cohorts']) == set(COHORTS), 'Runtime cohort replay incomplete')
    for cohort, methods in result['cohorts'].items():
        require(set(methods) == set(METHODS), 'Runtime 27-head replay incomplete')
        expected = load(root/'exp/factrisk'/COHORTS[cohort]/'metrics.json')['methods']
        for method, value in methods.items():
            require(value['max_prediction_error'] <= 1e-12, 'Runtime prediction discrepancy')
            same(value['metrics'], {k:expected[method][k] for k in value['metrics']}, 'Runtime metric discrepancy')
    require(set(result['quality']) == {'qwen','seamless'}, 'Runtime chrF replay incomplete')
    for backend, value in result['quality'].items():
        expected = load(root/'exp/factrisk/fr_en'/backend/'general_quality.json')
        require(value['n'] == expected['n'] == 1000, 'Runtime chrF cohort size differs')
        same(value['base_chrf'], expected['base_chrf'], 'Runtime base chrF differs')
        require(set(value['top90_chrf']) == set(METHODS), 'Runtime selective chrF incomplete')
        for method, score in value['top90_chrf'].items():
            same(score, expected['methods'][method]['top90_chrf'], 'Runtime chrF differs')


def require_comet(root, base):
    report = load(base/'comet/report.json')
    required = {'checkpoint','config','environment_lock','environment_manifest','evidence','predictions','scores','replay_code'}
    require(required <= set(report['assets']), 'COMET asset provenance incomplete')
    verify_hashes(root, {v['path']:v['sha256'] for v in report['assets'].values()})
    require(report['replay_input_sha256'] == sha256_file(base/'comet/inputs.json'), 'COMET replay inputs changed')
    require(report['status'] == 'passed' and report['assets_unchanged'] and not report['cache_modified'], 'COMET replay failed')
    comparisons = report['comparisons']
    require(len(comparisons) == report['count'] == 16 and len({r['id'] for r in comparisons}) == 16, 'COMET batch incomplete')
    require(bool(report['gpu']) and bool(report['versions']), 'COMET load/runtime evidence missing')
    cached = {r['id']:r for r in read_jsonl(report['assets']['scores']['path'])}
    errors = []
    for row in comparisons:
        same(row['cached'], cached[row['id']]['score'], 'COMET cached score differs')
        error = abs(row['replayed']-row['cached'])
        require(np.isfinite(error) and error <= 1e-5, 'COMET numerical replay failed')
        same(error, row['absolute_error'], 'COMET error field differs')
        errors.append(error)
    same(max(errors), report['maximum_absolute_error'], 'COMET maximum error differs')


def require_robustness(root):
    root = Path(root)
    base = root/'exp/factrisk/robustness'
    require_seeds(root, base)
    require_empty(root, base)
    require_environment(root, base)
    require_comet(root, base)
