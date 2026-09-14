"""Frozen German-trained hypothesis-NLL baseline on the French number cohort.

freeze records the original checkpoint before inference; extract scores only
cached hypotheses plus EOS. evaluate never fits a model or a calibrator.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from factrisk.core.contracts import validated_predictions
from factrisk.eval.evaluation import apply_calibration, cluster_comparison, describe, matrix
from factrisk.core.io import read_jsonl, read_yaml, sha256_file, write_json, write_jsonl
from factrisk.pipeline.likelihood import extract
from factrisk.pipeline.workflow import OUT


def inputs(config, checkpoint, full_checkpoint):
    cfg=read_yaml(config)
    if cfg['models']['direct_st'].get('backend')!='qwen2_audio':
        raise ValueError('Existing hypothesis-NLL checkpoint is Qwen-only')
    manifest=Path(cfg['project']['data_dir'])/'manifest.jsonl'
    rows=read_jsonl(manifest)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate French manifest IDs')
    selected=[r for r in rows if r.get('numeric_cohort',False)]
    if not selected or any(r['source_language']!='fr' or r['target_language']!='en'
        or r['split']!='test' or r['fact_type']!='number' for r in selected):
        raise ValueError('Expected nonempty French-English numerical test cohort')
    identity=dict(config_sha256=sha256_file(config),manifest_sha256=sha256_file(manifest),
        checkpoint_sha256=sha256_file(checkpoint),full_checkpoint_sha256=sha256_file(full_checkpoint),
        selected_ids=[r['id'] for r in selected],
        protocol='frozen_de_en_hypothesis_nll_lr_no_french_fit_or_recalibration')
    return cfg,rows,selected,identity


def execute(args):
    destination=Path(args.destination)
    cfg,manifest,selected,identity=inputs(args.config,args.checkpoint,args.full_checkpoint)
    freeze_path=destination/'frozen_protocol.json'
    if args.stage=='freeze':
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != identity:
            raise ValueError('Refusing to overwrite different likelihood freeze')
        write_json(freeze_path,identity)
        print(f'Frozen existing German NLL checkpoint for {len(selected)} French inputs')
        return
    if json.loads(freeze_path.read_text()) != identity:
        raise ValueError('Likelihood inputs differ from frozen protocol')
    predictions=Path(args.predictions or Path(cfg['project']['output_dir'])/'predictions/qwen2_audio.jsonl')
    hypotheses=validated_predictions(predictions,manifest,'direct',cfg['models']['direct_st'])
    cache=destination/'hypothesis_nll.jsonl'
    if args.stage=='extract':
        extract(args.config,predictions,cache,selected_ids=identity['selected_ids'])
        return
    if not args.features or not args.full_predictions:
        raise ValueError('evaluate requires --features and --full-predictions')
    rows=read_jsonl(args.features)
    expected=set(identity['selected_ids'])
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=expected:
        raise ValueError('Numeric feature cohort differs from frozen cohort')
    cached=read_jsonl(cache);values={r['id']:r for r in cached}
    if len(values)!=len(cached) or set(values)!=expected:
        raise ValueError('Incomplete or duplicate likelihood cache')
    by_id={r['id']:r for r in selected}
    for row in rows:
        source=by_id[row['id']]
        if row['split']!='test' or row['fact_type']!='number':
            raise ValueError('Only test number features allowed')
        hyp=hypotheses[row['id']]['translation']
        if row['translation']!=hyp:
            raise ValueError('Feature translation differs from scored hypothesis')
        fingerprint=hashlib.sha256(json.dumps([sha256_file(source['audio']),hyp,
            cfg['models']['direct_st']['model_path'],'hypothesis_eos_v1']).encode()).hexdigest()
        if values[row['id']]['input_fingerprint']!=fingerprint:
            raise ValueError('Stale French likelihood cache')
    cp=joblib.load(args.checkpoint)
    x=np.array([[values[r['id']]['hypothesis_nll']] for r in rows],float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite hypothesis likelihood')
    raw=cp['model'].predict_proba(x)[:,1]
    prob=apply_calibration(raw,cp['calibration'])
    metric=describe(rows,raw,prob,cp['threshold'])
    full_rows=read_jsonl(args.full_predictions);full={r['id']:r for r in full_rows}
    if len(full)!=len(full_rows) or set(full)!=expected:
        raise ValueError('Full predictions differ from numerical cohort')
    for r in rows:
        if full[r['id']]['label']!=r['severe_fact_error']:
            raise ValueError('Full predictions use different evaluation labels')
    full_cp=joblib.load(args.full_checkpoint)
    a=np.array([full[r['id']]['risks']['et_full']['raw'] for r in rows])
    pa=np.array([full[r['id']]['risks']['et_full']['calibrated'] for r in rows])
    recomputed=full_cp['model'].predict_proba(matrix(rows,full_cp['features']))[:,1]
    if not np.allclose(a,recomputed,atol=1e-12,rtol=0) or not np.allclose(
        pa,apply_calibration(recomputed,full_cp['calibration']),atol=1e-12,rtol=0):
        raise ValueError('Full risks differ from frozen checkpoint predictions')
    metric['full_minus_likelihood']=cluster_comparison(rows,a,raw,
        full_cp['threshold'],cp['threshold'],pa,prob)
    write_jsonl(destination/'predictions.jsonl',[dict(id=r['id'],raw=float(a),calibrated=float(b))
        for r,a,b in zip(rows,raw,prob)])
    write_json(destination/'metrics.json',dict(method='hypothesis_nll_lr',**identity,
        feature_sha256=sha256_file(args.features),hypotheses_sha256=sha256_file(predictions),
        cache_sha256=sha256_file(cache),full_predictions_sha256=sha256_file(args.full_predictions),
        metrics=metric))
    print(f'Evaluated frozen NLL baseline on {len(rows)} French number inputs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','extract','evaluate'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--predictions')
    parser.add_argument('--features')
    parser.add_argument('--full-predictions')
    parser.add_argument('--checkpoint',default=str(OUT/'likelihood/model.joblib'))
    parser.add_argument('--full-checkpoint',default=str(OUT/'corrected/number/checkpoints/et_full.joblib'))
    execute(parser.parse_args())


if __name__=='__main__':main()
