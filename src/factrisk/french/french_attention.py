"""Descriptive frozen DE attention-LR comparison on the French number cohort.

Uses cached printed hypotheses, not references. No French fitting, head
selection or recalibration; the primary comparison remains ASR+evidence ET.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from factrisk.backends.attention import run as extract
from factrisk.core.contracts import validated_predictions
from factrisk.eval.evaluation import apply_calibration, cluster_comparison, describe, matrix
from factrisk.french.french_likelihood import inputs
from factrisk.core.io import read_jsonl, sha256_file, write_json, write_jsonl
from factrisk.pipeline.workflow import OUT


def feature_matrix(rows, cached, expected_shape):
    values={r['id']:r for r in cached}
    if len(values)!=len(cached) or set(values)!={r['id'] for r in rows}:
        raise ValueError('Attention cache has missing, extra, or duplicate IDs')
    if any(r['feature_shape']!=expected_shape or len(r['features'])!=np.prod(expected_shape)
        for r in cached):raise ValueError('Attention feature shape differs from frozen DE baseline')
    x=np.asarray([values[r['id']]['features'] for r in rows],float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite attention features')
    return x,values


def execute(args):
    destination=Path(args.destination)
    cfg,manifest,selected,identity=inputs(args.config,args.checkpoint,args.full_checkpoint)
    if len(selected)!=943 or any(r['condition']!='clean' for r in selected):
        raise ValueError('Expected the frozen 943-recording French clean number cohort')
    identity.update(protocol='frozen_de_en_attention_lr_no_french_fit_or_recalibration',
        feature_shape=[4,32,32],comparison_role='descriptive_secondary',
        extractor='attention_v1_cached_hypothesis_no_eos')
    freeze=destination/'frozen_protocol.json'
    if args.stage=='freeze':
        cp=joblib.load(args.checkpoint)
        if cp['model'].n_features_in_!=4096:raise ValueError('Unexpected DE attention width')
        if freeze.exists() and json.loads(freeze.read_text())!=identity:
            raise ValueError('Refusing to change frozen attention protocol')
        write_json(freeze,identity);print('Frozen German attention model for 943 French inputs');return
    if json.loads(freeze.read_text())!=identity:raise ValueError('Changed attention protocol inputs')
    predictions=Path(args.predictions or Path(cfg['project']['output_dir'])/'predictions/qwen2_audio.jsonl')
    hypotheses=validated_predictions(predictions,manifest,'direct',cfg['models']['direct_st'])
    cache=destination/'attention.jsonl'
    if args.stage=='extract':
        extract(args.config,predictions,cache,selected_ids=identity['selected_ids']);return
    if not args.features or not args.full_predictions:
        raise ValueError('evaluate requires --features and --full-predictions')
    rows=read_jsonl(args.features);expected=set(identity['selected_ids'])
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=expected:
        raise ValueError('Features differ from frozen numeric cohort')
    x,values=feature_matrix(rows,read_jsonl(cache),identity['feature_shape'])
    by_id={r['id']:r for r in selected}
    for row in rows:
        original=by_id[row['id']];hyp=hypotheses[row['id']]['translation']
        if row['split']!='test' or row['fact_type']!='number' or row['translation']!=hyp:
            raise ValueError('Features do not match cached test hypotheses')
        fingerprint=hashlib.sha256(json.dumps([sha256_file(original['audio']),hyp,
            cfg['models']['direct_st']['model_path'],'attention_v1'],ensure_ascii=False).encode()).hexdigest()
        if values[row['id']]['input_fingerprint']!=fingerprint:
            raise ValueError('Stale French attention cache')
    cp=joblib.load(args.checkpoint)
    raw=cp['model'].predict_proba(x)[:,1];prob=apply_calibration(raw,cp['calibration'])
    metric=describe(rows,raw,prob,cp['threshold'])
    full_rows=read_jsonl(args.full_predictions);full={r['id']:r for r in full_rows}
    if len(full)!=len(full_rows) or set(full)!=expected:
        raise ValueError('Full predictions differ from numeric cohort')
    if any(full[r['id']]['label']!=r['severe_fact_error'] for r in rows):
        raise ValueError('Full predictions use different labels')
    full_cp=joblib.load(args.full_checkpoint)
    a=np.array([full[r['id']]['risks']['et_full']['raw'] for r in rows])
    pa=np.array([full[r['id']]['risks']['et_full']['calibrated'] for r in rows])
    check=full_cp['model'].predict_proba(matrix(rows,full_cp['features']))[:,1]
    if not np.allclose(a,check,atol=1e-12,rtol=0) or not np.allclose(
        pa,apply_calibration(check,full_cp['calibration']),atol=1e-12,rtol=0):
        raise ValueError('Full risk cache differs from frozen checkpoint')
    metric['full_minus_attention']=cluster_comparison(rows,a,raw,full_cp['threshold'],
        cp['threshold'],pa,prob,samples=2000)
    write_jsonl(destination/'predictions.jsonl',[dict(id=r['id'],raw=float(a),calibrated=float(b))
        for r,a,b in zip(rows,raw,prob)])
    write_json(destination/'metrics.json',dict(method='attention_lr_all_heads',**identity,
        feature_sha256=sha256_file(args.features),hypotheses_sha256=sha256_file(predictions),
        cache_sha256=sha256_file(cache),full_predictions_sha256=sha256_file(args.full_predictions),metrics=metric))
    print(f'Evaluated descriptive frozen attention baseline on {len(rows)} French inputs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','extract','evaluate'])
    p.add_argument('--config',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--predictions');p.add_argument('--features');p.add_argument('--full-predictions')
    p.add_argument('--checkpoint',default=str(OUT/'corrected/number/attention/model.joblib'))
    p.add_argument('--full-checkpoint',default=str(OUT/'corrected/number/checkpoints/et_full.joblib'))
    execute(p.parse_args())


if __name__=='__main__':main()
