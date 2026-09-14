"""Ranking, calibration, unresolved-label bounds, matched heads and clusters."""
from __future__ import annotations
import csv
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from factrisk.core.io import read_jsonl, sha256_file, stable_int, write_json, write_jsonl
from factrisk.method.risk import ece_score
from factrisk.core.models import FEATURES, build_model


def matrix(rows, names):
    return np.array([[r['features'].get(n, float('nan')) for n in names] for r in rows], float)


def aurc(y, score):
    """Expected AURC under random tie breaking, avoiding manifest-order bias."""
    order = np.argsort(score, kind='stable')
    s, labels = score[order], y[order]
    _, starts, counts = np.unique(s, return_index=True, return_counts=True)
    expected = labels.astype(float).copy()
    for start, count in zip(starts, counts):
        expected[start:start+count] = labels[start:start+count].mean()
    return float(np.mean(np.cumsum(expected)/np.arange(1, len(y)+1)))


def calibrate(raw, y):
    x = np.log(np.clip(raw, 1e-6, 1-1e-6)/np.clip(1-raw, 1e-6, 1))
    def loss(ab):
        z = ab[0]*x + ab[1]
        return np.mean(np.logaddexp(0, z)-y*z) + 1e-8 * ab[0]**2
    fit = minimize(loss, [1., 0.], bounds=[(0., None), (None, None)], method='L-BFGS-B')
    if not fit.success:
        raise RuntimeError(f'Calibration failed: {fit.message}')
    return fit.x


def apply_calibration(raw, ab):
    x = np.log(np.clip(raw, 1e-6, 1-1e-6)/np.clip(1-raw, 1e-6, 1))
    return expit(ab[0]*x+ab[1])


def keep_top(score, coverage, ids):
    n = min(len(score), max(1, round(coverage*len(score))))
    # Label-independent reproducible tie breaking at the operating point.
    tie = np.array([stable_int(str(i), 20260906) for i in ids], dtype=np.uint64)
    order = np.lexsort((tie, score))
    keep = np.zeros(len(score), bool)
    keep[order[:n]] = True
    return keep


def selective(y, keep):
    known = np.isfinite(y)
    k = int(keep.sum())
    e = int(np.nansum(y[keep]))
    u = int(np.sum(keep & ~known))
    total_errors = float(np.nansum(y))
    return dict(coverage=float(keep.mean()), kept=k, known_errors=e, unresolved=u,
                known_label_risk=float(np.mean(y[keep & known])) if np.any(keep & known) else None,
                risk_lower=e/k if k else None, risk_upper=(e+u)/k if k else None,
                known_error_capture=float(np.nansum(y[~keep])/total_errors) if total_errors else None)


def describe(rows, raw, scores, threshold):
    y = np.array([r['severe_fact_error'] if r['severe_fact_error'] is not None else np.nan for r in rows], float)
    known = np.isfinite(y)
    ids = [r['id'] for r in rows]
    result = dict(n=len(rows), unresolved=int((~known).sum()),
        base_known_risk=float(y[known].mean()),
        aurc_known=aurc(y[known], raw[known]),
        auroc_known=float(roc_auc_score(y[known], raw[known])) if len(np.unique(y[known]))==2 else None,
        brier_raw=float(np.mean((raw[known]-y[known])**2)),
        brier_calibrated=float(np.mean((scores[known]-y[known])**2)),
        ece_raw=ece_score(y[known], raw[known]), ece_calibrated=ece_score(y[known], scores[known]),
        fixed={str(c):selective(y, keep_top(raw, c, ids)) for c in (.8, .9, .95)},
        operational=selective(y, scores < threshold))
    clean_correct = np.array([r['condition']=='clean' for r in rows]) & (y==0)
    result['clean_over_abstention'] = float(np.mean(scores[clean_correct]>=threshold)) if clean_correct.any() else None
    return result


def cluster_comparison(rows, score_a, score_b, ta, tb, probability_a, probability_b, samples=2000):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.get('cluster_id',row['speaker_id'])].append(i)
    keys = sorted(groups)
    y = np.array([r['severe_fact_error'] if r['severe_fact_error'] is not None else np.nan for r in rows], float)
    rng = np.random.default_rng(20260906)
    draws = []
    for _ in range(samples):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        known = idx[np.isfinite(y[idx])]
        if not len(known):
            continue
        a, b = probability_a[idx]<ta, probability_b[idx]<tb
        ya = y[idx][a & np.isfinite(y[idx])]
        yb = y[idx][b & np.isfinite(y[idx])]
        draws.append([aurc(y[known], score_a[known])-aurc(y[known], score_b[known]),
                      ya.mean()-yb.mean() if len(ya) and len(yb) else np.nan,
                      a.mean()-b.mean()])
    d = np.array(draws)
    return dict(speakers=len(keys), bootstrap_replicates=len(draws),
                cluster_unit='speaker_donor_connected_component' if any('cluster_id' in r for r in rows) else 'speaker',
                paired_aurc_difference_ci=np.nanquantile(d[:,0], [.025,.975]).tolist(),
                operational_known_risk_difference_ci=np.nanquantile(d[:,1], [.025,.975]).tolist(),
                operational_coverage_difference_ci=np.nanquantile(d[:,2], [.025,.975]).tolist(),
                limitation='Conditional on fixed trained heads; known-label AURC excludes unresolved cases.')


def run(directory):
    directory = Path(directory)
    all_rows = read_jsonl(directory/'features.jsonl')
    if (directory/'labels_unresolved.jsonl').exists():
        all_rows += read_jsonl(directory/'labels_unresolved.jsonl')
    if len({r['id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Duplicate inputs')
    # Primary number cohort and explicitly exploratory polarity-proxy cohort.
    cohorts = ('number', 'number_negation_proxy') if any(r['fact_type']=='negation' for r in all_rows) else ('number',)
    for cohort in cohorts:
        rows = [r for r in all_rows if cohort != 'number' or r['fact_type']=='number']
        splits = {s:[r for r in rows if r['split']==s] for s in ('train','calibration','test')}
        train = [r for r in splits['train'] if r['severe_fact_error'] is not None]
        cal = [r for r in splits['calibration'] if r['severe_fact_error'] is not None]
        test = splits['test']
        y_train = np.array([r['severe_fact_error'] for r in train])
        y_cal = np.array([r['severe_fact_error'] for r in cal])
        for split, subset in splits.items():
            others = {r['speaker_id'] for s, rr in splits.items() if s != split for r in rr}
            if others & {r['speaker_id'] for r in subset}:
                raise ValueError('Speaker overlap')
        if len(np.unique(y_train))<2 or len(np.unique(y_cal))<2:
            raise ValueError('Training and calibration must have both classes')
        dest = directory/cohort
        dest.mkdir(parents=True, exist_ok=True)
        metrics, predictions, cached = {}, {r['id']:{} for r in test}, {}
        for name, names in FEATURES.items():
            x_train, x_cal, x_test = [matrix(rr, names) for rr in (train,cal,test)]
            if any(not np.isfinite(x).any(axis=0).all() for x in (x_train, x_cal, x_test)):
                metrics[name] = dict(status='unavailable', reason='At least one feature absent in an entire split')
                continue
            start = time.monotonic()
            model = build_model(name)
            model.fit(x_train,y_train)
            raw_cal = model.predict_proba(x_cal)[:,1]
            ab = calibrate(raw_cal,y_cal)
            calibrated_cal = apply_calibration(raw_cal,ab)
            threshold = float(np.quantile(calibrated_cal,.9))
            raw = model.predict_proba(x_test)[:,1]
            score = apply_calibration(raw,ab)
            value = describe(test,raw,score,threshold)
            value.update(status='completed', feature_names=names, threshold=threshold,
                         calibration_parameters=ab.tolist(), fitting_seconds=time.monotonic()-start,
                         by_condition={})
            for condition in sorted({r['condition'] for r in test}):
                idx = np.array([i for i,r in enumerate(test) if r['condition']==condition])
                value['by_condition'][condition] = describe([test[i] for i in idx],raw[idx],score[idx],threshold)
            value['leave_one_speaker_out'] = {}
            for speaker in sorted({r['speaker_id'] for r in test}):
                idx = np.array([i for i,r in enumerate(test) if r['speaker_id']!=speaker and r['severe_fact_error'] is not None])
                value['leave_one_speaker_out'][speaker] = aurc(np.array([test[i]['severe_fact_error'] for i in idx]),raw[idx])
            metrics[name] = value
            cached[name] = (raw,score,threshold)
            for r,a,b in zip(test,raw,score):
                predictions[r['id']][name] = dict(raw=float(a), calibrated=float(b))
            checkpoint_dir = dest/'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True)
            joblib.dump(dict(model=model, calibration=ab, threshold=threshold, features=names,
                             cohort=cohort, label_version='critical_fact_v2_context_20260906'), checkpoint_dir/f'{name}.joblib')
            write_json(dest/'metrics.partial.json',metrics)
            print(f'{cohort} {name}: AURC={value["aurc_known"]:.4f} Brier={value["brier_calibrated"]:.4f}',flush=True)
        primary = 'et_full'
        comparisons = {}
        if primary in cached:
            a,pa,ta = cached[primary]
            for comparator in ('et_evidence','et_asr_evidence','et_without_stability','lr_sequence_probability'):
                if comparator in cached:
                    b,pb,tb = cached[comparator]
                    comparisons[comparator] = cluster_comparison(test,a,b,ta,tb,pa,pb)
        write_json(dest/'metrics.json',dict(status='historical_diagnostic_automatic_labels', cohort=cohort,
            primary=primary, selection='Fixed before revised evaluation; no test-based model selection',
            feature_file_sha256=sha256_file(directory/'features.jsonl'), methods=metrics, comparisons=comparisons))
        write_jsonl(dest/'predictions.jsonl',[dict(id=r['id'],speaker_id=r['speaker_id'],condition=r['condition'],
                    label=r['severe_fact_error'],risks=predictions[r['id']]) for r in test])
        with (dest/'main_results.csv').open('w',newline='') as f:
            fields=['method','aurc_known','brier_raw','brier_calibrated','ece_calibrated','risk90_lower','risk90_upper','operational_coverage']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for name,v in metrics.items():
                if v['status']=='completed':
                    writer.writerow(dict(method=name,**{k:v[k] for k in fields[1:5]},
                        risk90_lower=v['fixed']['0.9']['risk_lower'],risk90_upper=v['fixed']['0.9']['risk_upper'],
                        operational_coverage=v['operational']['coverage']))
