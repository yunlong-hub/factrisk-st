"""Replay frozen risk heads and corpus chrF in an isolated installed environment."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import site
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from sacrebleu.metrics import CHRF

from factrisk.eval.empty_sensitivity import cohort_inputs
from factrisk.eval.evaluation import apply_calibration, describe, keep_top, matrix
from factrisk.core.io import read_jsonl, sha256_file, write_json
from factrisk.eval.significance_summary import COHORTS, OUT, ROOT

METHODS = ('et_full', 'et_asr_evidence', 'et_without_stability')


def assert_equal(actual, expected, path='root'):
    """Compare all nested reported fields, with explicit float tolerance."""
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f'{path}: key mismatch')
        for key in expected:
            assert_equal(actual[key], expected[key], f'{path}/{key}')
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError(f'{path}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'{path}: {actual} != {expected}')


def run():
    config = Path(sys.prefix) / 'pyvenv.cfg'
    if sys.prefix == sys.base_prefix or not config.exists():
        raise RuntimeError('Replay must run in a virtual environment')
    if 'include-system-site-packages = false' not in config.read_text().lower():
        raise RuntimeError('Inherited system packages are not allowed')
    inherited = [p for p in sys.path if 'site-packages' in p
                 and not Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())]
    if inherited:
        raise RuntimeError(f'Inherited site-packages: {inherited}')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(check.stdout + check.stderr)
    inputs, cohorts, quality = {}, {}, {}

    def record(path):
        inputs[str(path.relative_to(ROOT))] = sha256_file(path)

    for name, folder in COHORTS.items():
        directory, paths, _, heads = cohort_inputs(name, folder)
        for path in paths + [directory/'predictions.jsonl', directory/'metrics.json']:
            record(path)
        indexed = {r['id']: r for p in paths for r in read_jsonl(p)}
        saved = read_jsonl(directory/'predictions.jsonl')
        rows = [indexed[r['id']] for r in saved]
        expected_metrics = json.loads((directory/'metrics.json').read_text())['methods']
        results = {}
        for method in METHODS:
            path = heads/f'{method}.joblib'
            record(path)
            cp = joblib.load(path)
            raw = cp['model'].predict_proba(matrix(rows, cp['features']))[:, 1]
            probability = apply_calibration(raw, cp['calibration'])
            expected = np.asarray([[r['risks'][method]['raw'], r['risks'][method]['calibrated']]
                                   for r in saved])
            delta = np.abs(np.c_[raw, probability] - expected)
            if not np.all(delta <= 1e-12):
                raise ValueError(f'{name}/{method}: frozen prediction mismatch')
            measured = describe(rows, raw, probability, cp['threshold'])
            # Stored records may contain timing or other provenance beyond describe().
            assert_equal(measured, {k: expected_metrics[method][k] for k in measured}, f'{name}/{method}')
            results[method] = dict(max_prediction_error=float(delta.max()), metrics=measured)
        cohorts[name] = results
        print(f'{name}: three frozen heads and all metric fields reproduced', flush=True)

    metric = CHRF()
    for backend in ('qwen', 'seamless'):
        directory = OUT/'fr_en'/backend
        source, expected_path = directory/'features.jsonl', directory/'general_quality.json'
        record(source); record(expected_path)
        expected = json.loads(expected_path.read_text())
        indexed = {r['id']: r for r in read_jsonl(source)}
        rows = [indexed[k] for k in expected['selected_ids']]
        def chrf(keep):
            chosen = [r for r, selected in zip(rows, keep) if selected]
            return metric.corpus_score([r['translation'] for r in chosen],
                                       [[r['reference'] for r in chosen]]).score
        base = chrf(np.ones(len(rows), bool))
        assert_equal(base, expected['base_chrf'])
        scores = {}
        heads = OUT/('corrected' if backend == 'qwen' else 'seamless_controlled')/'number/checkpoints'
        for method in METHODS:
            cp = joblib.load(heads/f'{method}.joblib')
            raw = cp['model'].predict_proba(matrix(rows, cp['features']))[:, 1]
            keep = keep_top(raw, .9, [r['id'] for r in rows])
            value = chrf(keep)
            assert_equal(value, expected['methods'][method]['top90_chrf'])
            assert_equal(int(keep.sum()), expected['methods'][method]['retained'])
            scores[method] = value
        quality[backend] = dict(n=len(rows), base_chrf=base, top90_chrf=scores)
    for path, digest in inputs.items():
        if sha256_file(ROOT/path) != digest:
            raise ValueError(f'Input changed: {path}')
    write_json(OUT/'robustness/environment/replay.json', dict(
        status='complete', scope='Cached CPU risk-head and chrF replay; no ST or COMET inference rerun',
        executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
        site_packages=site.getsitepackages(), sys_path=sys.path, module_path=__file__,
        python=sys.version, packages={d.metadata['Name']: d.version for d in metadata.distributions()},
        pip_check=check.stdout.strip(), tolerance=1e-12, cohorts=cohorts, quality=quality,
        inputs=inputs, code_sha256=sha256_file(__file__),
        lock_sha256=sha256_file(ROOT/'configs/environments/evaluation.lock.txt')))


if __name__ == '__main__':
    run()
