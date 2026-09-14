"""Feature construction and frozen, no-recalibration external evaluation."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from factrisk.method.features import _feature_row
from factrisk.core.io import read_jsonl, read_yaml, sha256_file, write_json, write_jsonl
from factrisk.core.labels_v2 import label_fact
from factrisk.core.contracts import validated_predictions, qe_fingerprint
from factrisk.pipeline.workflow import deployment_features
from factrisk.eval.evaluation import matrix, apply_calibration, describe, cluster_comparison


def features(config_path):
    cfg=read_yaml(config_path);out=Path(cfg['project']['output_dir'])
    rows=read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl')
    pred=validated_predictions(out/'predictions/qwen2_audio.jsonl',rows,'direct',cfg['models']['direct_st'])
    ev_path=cfg.get('sources',{}).get('revision_evidence_file',out/'evidence/whisper_nllb.jsonl')
    ev=validated_predictions(ev_path,rows,'evidence',cfg['models']['evidence'])
    qe={r['id']:r for r in read_jsonl(out/'qe/comet_qe.jsonl')}
    if set(qe)!=set(pred):
        raise ValueError('Incomplete real QE scores')
    result=[]
    for row in rows:
        p,e=pred[row['id']],ev[row['id']]
        if qe[row['id']]['input_fingerprint']!=qe_fingerprint(p,e):
            raise ValueError('Stale QE features')
        metadata=dict(row,side=row.get('side','original'),
            evidence_insufficient=row.get('evidence_insufficient',False),tts_engine=row.get('tts_engine','natural_speech'))
        item=_feature_row(metadata,p,e,qe[row['id']]['score'])
        item.update(label_fact(p['translation'],row['expected_slot'],row.get('contrast_slot',''),'number',row['reference']))
        item['features'].update(deployment_features(p['translation'],p.get('sample_translations',[]),e['cascade_translation']))
        for key in ('sample_fact_disagreement','evidence_fact_mismatch'):
            item['features'].pop(key,None)
        # There is no counterfactual value in natural recordings.
        item['unsupported']=item['contrast_value_present'] if row.get('contrast_slot') else None
        if 'cluster_id' in row: item['cluster_id']=row['cluster_id']
        result.append(item)
    write_jsonl(out/'features.jsonl',result)
    return out/'features.jsonl'


def evaluate(feature_path, checkpoint_dir, destination):
    rows=read_jsonl(feature_path);dest=Path(destination);checkpoint_dir=Path(checkpoint_dir)
    if any(r['split']!='test' or r['fact_type']!='number' for r in rows):
        raise ValueError('External transfer requires number-only test rows')
    import json
    freeze_path=Path(feature_path).parent/'frozen_protocol.json'
    frozen=json.loads(freeze_path.read_text())
    metrics,cached,per_row={}, {}, {r['id']:{} for r in rows}
    for path in sorted(checkpoint_dir.glob('*.joblib')):
        if frozen['checkpoints'].get(path.name)!=sha256_file(path):
            raise ValueError('Checkpoint differs from frozen external protocol')
        checkpoint=joblib.load(path)  # only project-produced trusted checkpoints
        x=matrix(rows,checkpoint['features'])
        if not np.isfinite(x).any(axis=0).all():
            metrics[path.stem]=dict(status='unavailable',reason='Entire feature channel missing');continue
        raw=checkpoint['model'].predict_proba(x)[:,1]
        prob=apply_calibration(raw,checkpoint['calibration']);threshold=checkpoint['threshold']
        metrics[path.stem]=dict(status='complete',checkpoint_sha256=sha256_file(path),**describe(rows,raw,prob,threshold))
        metrics[path.stem]['by_condition']={}
        for condition in sorted({r['condition'] for r in rows}):
            idx=np.array([i for i,r in enumerate(rows) if r['condition']==condition])
            metrics[path.stem]['by_condition'][condition]=describe([rows[i] for i in idx],raw[idx],prob[idx],threshold)
        cached[path.stem]=(raw,prob,threshold)
        for row,a,b in zip(rows,raw,prob):
            per_row[row['id']][path.stem]=dict(raw=float(a),calibrated=float(b))
    if 'et_full' not in cached:
        raise ValueError('Missing primary model')
    a,pa,ta=cached['et_full'];comparisons={}
    for method in ('et_evidence','et_asr_evidence','et_without_stability','lr_sequence_probability'):
        if method in cached:
            b,pb,tb=cached[method]
            comparisons[method]=cluster_comparison(rows,a,b,ta,tb,pa,pb)
    write_json(dest/'metrics.json',dict(protocol='frozen_external_transfer_no_recalibration',
        feature_sha256=sha256_file(feature_path),methods=metrics,comparisons=comparisons))
    write_jsonl(dest/'predictions.jsonl',[dict(id=r['id'],speaker_id=r['speaker_id'],label=r['severe_fact_error'],risks=per_row[r['id']]) for r in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--checkpoint-dir');p.add_argument('--destination')
    a=p.parse_args();path=features(a.config)
    if a.checkpoint_dir:
        if not a.destination: p.error('--destination required with --checkpoint-dir')
        evaluate(path,a.checkpoint_dir,a.destination)
