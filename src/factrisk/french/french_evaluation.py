"""Frozen Fr→En transfer and reference-based quality on output-blind cohorts."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sacrebleu.metrics import CHRF

from factrisk.core.io import read_jsonl, read_yaml, write_json, write_jsonl, sha256_file
from factrisk.pipeline.workflow import OUT
from factrisk.eval.evaluation import matrix, keep_top, apply_calibration
from factrisk.pipeline.transfer import features, evaluate


def quality(rows, checkpoint_dir, replicates=2000):
    """Speaker bootstrap conditional on the observed score-based selections."""
    metric=CHRF()
    statistics=np.asarray(metric._extract_corpus_statistics(
        [r['translation'] for r in rows],[[r['reference'] for r in rows]]),float)
    corpus=lambda weights: float(metric._compute_score_from_stats((statistics*weights[:,None]).sum(axis=0)).score)
    cluster=lambda r:r.get('cluster_id',r['speaker_id'])
    speakers=sorted({cluster(r) for r in rows})
    idx={s:i for i,s in enumerate(speakers)}
    row_clusters=np.array([idx[cluster(r)] for r in rows])
    rng=np.random.default_rng(20260906)
    draws=rng.multinomial(len(speakers),np.full(len(speakers),1/len(speakers)),size=replicates)
    base=corpus(np.ones(len(rows)))
    base_boot=np.asarray([corpus(draw[row_clusters]) for draw in draws])
    selected_boot={}
    result=dict(n=len(rows),clusters=len(speakers),base_chrf=base,metric=metric.get_signature().__str__(),
        bootstrap=f'{replicates} paired cluster resamples conditional on fixed selections',
        cluster_unit='speaker_donor_component' if any('cluster_id' in r for r in rows) else 'speaker',methods={})
    for path in sorted(Path(checkpoint_dir).glob('*.joblib')):
        cp=joblib.load(path);raw=cp['model'].predict_proba(matrix(rows,cp['features']))[:,1]
        keep=keep_top(raw,.9,[r['id'] for r in rows]);prob=apply_calibration(raw,cp['calibration'])
        bootstrap_scores=[]
        for draw in draws:
            weights=draw[row_clusters]
            bootstrap_scores.append(corpus(weights*keep) if (weights*keep).sum() else np.nan)
        selected_boot[path.stem]=np.asarray(bootstrap_scores)
        differences=selected_boot[path.stem]-base_boot
        result['methods'][path.stem]=dict(top90_chrf=corpus(keep.astype(float)),
            delta_top90_vs_all=corpus(keep.astype(float))-base,
            delta_ci95=np.nanquantile(differences,[.025,.975]).tolist(),
            retained=int(keep.sum()),frozen_coverage=float(np.mean(prob<cp['threshold'])))
    result['full_minus_asr_evidence']=dict(
        delta=result['methods']['et_full']['top90_chrf']-result['methods']['et_asr_evidence']['top90_chrf'],
        ci95=np.nanquantile(selected_boot['et_full']-selected_boot['et_asr_evidence'],[.025,.975]).tolist())
    rng=np.random.default_rng(20260908);random_scores=[]
    for _ in range(replicates):
        keep=np.zeros(len(rows));keep[rng.choice(len(rows),round(.9*len(rows)),replace=False)]=1
        random_scores.append(corpus(keep))
    result['random90']=dict(mean_chrf=float(np.mean(random_scores)),
        random_selection_interval95=np.quantile(random_scores,[.025,.975]).tolist(),
        note='random-selection variability, not a population confidence interval')
    return result


def run(backend, group='clean', quality_only=False):
    root=OUT/'fr_en'
    if group=='stress':root=root/'stress'
    cfg=read_yaml(root/f'{backend}_config.yaml')
    directory=root/backend
    manifest=read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl')
    frozen=json.loads((directory/'frozen_protocol.json').read_text())
    if frozen['manifest_sha256']!=sha256_file(Path(cfg['project']['data_dir'])/'manifest.jsonl'):
        raise ValueError('French manifest changed after freeze')
    cp=OUT/('corrected' if backend=='qwen' else 'seamless_controlled')/'number/checkpoints'
    for p in cp.glob('*.joblib'):
        if frozen['checkpoints'][p.name]!=sha256_file(p):raise ValueError('Changed checkpoint')
    feature_path=directory/'features.jsonl' if quality_only else features(root/f'{backend}_config.yaml')
    all_features=read_jsonl(feature_path)
    by_id={r['id']:r for r in manifest}
    numerical=[r for r in all_features if by_id[r['id']]['numeric_cohort']]
    general=[r for r in all_features if by_id[r['id']]['general_cohort']]
    if not quality_only:
        write_jsonl(directory/'numeric_features.jsonl',numerical)
        evaluate(directory/'numeric_features.jsonl',cp,directory/f'numeric_{group}')
    for name,rows in [('numeric',numerical),('general',general)]:
        if rows:
            result=quality(rows,cp)
            result.update(source_feature_sha256=sha256_file(feature_path),
                manifest_sha256=frozen['manifest_sha256'],checkpoints=frozen['checkpoints'],
                selected_ids=[r['id'] for r in rows])
            write_json(directory/f'{name}_quality.json',result)
    write_json(directory/'evaluation.status.json',dict(status='complete',numeric=len(numerical),general=len(general)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['qwen','seamless'])
    p.add_argument('--group',choices=['clean','stress'],default='clean')
    p.add_argument('--quality-only',action='store_true')
    a=p.parse_args();run(a.backend,a.group,a.quality_only)
