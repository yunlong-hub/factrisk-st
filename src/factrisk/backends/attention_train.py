"""Fit the all-layer/all-head attention baseline on the shared number labels."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from factrisk.core.io import read_jsonl,sha256_file,write_json,write_jsonl
from factrisk.core.models import build_model
from factrisk.eval.evaluation import calibrate,apply_calibration,describe


def run(directory,external):
    directory=Path(directory);external=Path(external)
    rows=read_jsonl(directory/'features.jsonl')+read_jsonl(directory/'labels_unresolved.jsonl')
    rows=[r for r in rows if r['fact_type']=='number']
    att={r['id']:r for r in read_jsonl(directory/'attention.jsonl')}
    if set(att)!={r['id'] for r in rows}: raise ValueError('Attention coverage mismatch')
    if len({tuple(r['feature_shape']) for r in att.values()})!=1: raise ValueError('Mixed attention feature shapes')
    train=[r for r in rows if r['split']=='train' and r['severe_fact_error'] is not None]
    cal=[r for r in rows if r['split']=='calibration' and r['severe_fact_error'] is not None]
    test=[r for r in rows if r['split']=='test']
    def mat(rr,aa): return np.array([aa[r['id']]['features'] for r in rr],float)
    model=build_model('lr_attention')
    model.fit(mat(train,att),np.array([r['severe_fact_error'] for r in train]))
    rawcal=model.predict_proba(mat(cal,att))[:,1]
    ab=calibrate(rawcal,np.array([r['severe_fact_error'] for r in cal]))
    threshold=float(np.quantile(apply_calibration(rawcal,ab),.9))
    dest=directory/'number/attention';dest.mkdir(parents=True,exist_ok=True)
    cp=dest/'model.joblib';joblib.dump(dict(model=model,calibration=ab,threshold=threshold),cp)
    # Checkpoint is saved before opening external labels.
    external_rows=read_jsonl(external/'features.jsonl')
    external_att={r['id']:r for r in read_jsonl(external/'attention.jsonl')}
    metrics={}
    for name,rr,aa in (('controlled',test,att),('natural_clean',external_rows,external_att)):
        if set(r['id'] for r in rr)-set(aa): raise ValueError('Missing external attention')
        raw=model.predict_proba(mat(rr,aa))[:,1];prob=apply_calibration(raw,ab)
        metrics[name]=describe(rr,raw,prob,threshold)
        write_jsonl(dest/f'{name}_predictions.jsonl',[dict(id=r['id'],raw=float(a),calibrated=float(b)) for r,a,b in zip(rr,raw,prob)])
    write_json(dest/'metrics.json',dict(method='attention_lr_all_heads',source='Waldendorf et al. 2026, task adaptation',
        shape=next(iter(att.values()))['feature_shape'],checkpoint_sha256=sha256_file(cp),metrics=metrics,
        preprocessing='training-median imputation, standard scaling, balanced L2 LR C=1; no head selection'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--external',required=True)
    a=p.parse_args();run(a.directory,a.external)
