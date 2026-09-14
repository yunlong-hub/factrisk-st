"""Shared-unresolved-label sensitivity, paired masking and complete timing."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
from factrisk.core.io import read_jsonl,write_json
from factrisk.pipeline.workflow import OUT
from factrisk.eval.evaluation import keep_top,matrix,cluster_comparison


def difference_bounds(y,keep_a,keep_b):
    """One shared unknown label per item, not independent worst cases per method."""
    if not keep_a.any() or not keep_b.any():return None
    weights=keep_a/keep_a.sum()-keep_b/keep_b.sum()
    known=np.isfinite(y)
    fixed=float(np.sum(weights[known]*y[known]));unknown=weights[~known]
    return dict(lower=fixed+float(np.minimum(unknown,0).sum()),
                upper=fixed+float(np.maximum(unknown,0).sum()))


def label_flip_sensitivity(y, keep_a, keep_b):
    """Exact worst-case finite-cohort difference with up to m known-label flips.

    Each item has one shared true binary label across both selections. Unknown
    labels are unrestricted; a budget applies only to currently decidable test
    labels, not to training labels, sampling variation, or score refitting.
    """
    bounds = difference_bounds(y, keep_a, keep_b)
    if bounds is None:
        return None
    known = np.isfinite(y)
    w = keep_a/keep_a.sum()-keep_b/keep_b.sum()
    gains = w[known]*(1-2*y[known])
    gains = np.sort(np.maximum(gains, 0.))[::-1]
    worst = np.r_[bounds['upper'], bounds['upper']+np.cumsum(gains)]
    crossing = np.flatnonzero(worst >= -1e-12)
    first = int(crossing[0]) if len(crossing) else None
    return dict(known_labels=int(known.sum()), unknown_labels=int((~known).sum()),
                shared_unknown_difference_bounds=bounds,
                minimum_known_flips_to_erase_strict_improvement=first,
                max_flips_preserving_strict_improvement=(first-1 if first is not None else len(gains)),
                worst_case_difference_by_flip_budget=worst.tolist(),
                scope='Fixed observed cohort and selections; not label accuracy or population confidence.')


def cohort(directory,prediction_path,metric_path,destination):
    directory=Path(directory);rows=read_jsonl(directory/'features.jsonl')
    if (directory/'labels_unresolved.jsonl').exists():rows+=read_jsonl(directory/'labels_unresolved.jsonl')
    rows=[r for r in rows if r['fact_type']=='number' and r['split']=='test']
    predictions={r['id']:r['risks'] for r in read_jsonl(prediction_path)}
    metrics=json.loads(Path(metric_path).read_text())['methods']
    y=np.array([r['severe_fact_error'] if r['severe_fact_error'] is not None else np.nan for r in rows])
    ids=[r['id'] for r in rows]
    scores={m:np.array([predictions[r['id']][m]['raw'] for r in rows]) for m in predictions[ids[0]]}
    result={}
    for baseline in ('et_evidence','et_asr_evidence','et_without_stability','lr_sequence_probability','lr_qe'):
        result[baseline]={str(c):difference_bounds(y,keep_top(scores['et_full'],c,ids),keep_top(scores[baseline],c,ids)) for c in (.8,.9,.95)}
    attention_name='controlled' if directory.name=='corrected' else 'natural_clean'
    attention_dir=OUT/'corrected/number/attention'
    attention_file=attention_dir/f'{attention_name}_predictions.jsonl'
    attention_comparison=None
    if attention_file.exists():
        att={r['id']:r for r in read_jsonl(attention_file)}
        b=np.array([att[i]['raw'] for i in ids]);pb=np.array([att[i]['calibrated'] for i in ids])
        pa=np.array([predictions[i]['et_full']['calibrated'] for i in ids])
        ta=joblib.load(OUT/'corrected/number/checkpoints/et_full.joblib')['threshold']
        tb=joblib.load(attention_dir/'model.joblib')['threshold']
        attention_comparison=cluster_comparison(rows,scores['et_full'],b,ta,tb,pa,pb)
        result['attention_lr']={str(c):difference_bounds(y,keep_top(scores['et_full'],c,ids),keep_top(b,c,ids)) for c in (.8,.9,.95)}
    paired={}
    if any(r['condition']=='local_mask' for r in rows):
        index={(r['pair_id'],r['side'],r['condition']):i for i,r in enumerate(rows)}
        pairs=[]
        for key,i in index.items():
            if key[2]=='local_mask':
                j=index[(key[0],key[1],'irrelevant_mask')]
                pairs.append((i,j))
        known=[(i,j) for i,j in pairs if np.isfinite(y[[i,j]]).all()]
        paired=dict(n_pairs=len(pairs),both_decidable=len(known),
            mean_known_error_difference=float(np.mean([y[i]-y[j] for i,j in known])),
            mean_full_score_difference=float(np.mean([scores['et_full'][i]-scores['et_full'][j] for i,j in pairs])),
            unit='same_source_and_fact_variant',causal_scope='paired_equal_duration_masks')
    quality={}
    try:
        from sacrebleu.metrics import CHRF
        metric=CHRF()
        for name,score in scores.items():
            keep=keep_top(score,.9,ids)
            kept=[r for r,k in zip(rows,keep) if k]
            quality[name]=metric.corpus_score([r['translation'] for r in kept],[[r['reference'] for r in kept]]).score
        quality['no_abstention']=metric.corpus_score([r['translation'] for r in rows],[[r['reference'] for r in rows]]).score
    except ImportError:
        quality=dict(status='unavailable',reason='sacrebleu not installed')
    flips=label_flip_sensitivity(y,keep_top(scores['et_full'],.9,ids),np.ones(len(y),bool))
    write_json(destination,dict(shared_unresolved_risk90_differences=result,paired_mask=paired,selective_chrf=quality,attention_comparison=attention_comparison,
        number_label_flip_sensitivity90_vs_all=flips,
        label_interpretation='operational_number_preservation',quality_interpretation='selection changes retained subset, not translator weights'))


def costs():
    timing=read_jsonl(OUT/'efficiency/component_timings.jsonl')
    if len(timing)!=64:raise ValueError('Component timing run must complete before aggregation')
    evidence={r['id']:r for r in read_jsonl(OUT/'real_clean/evidence/whisper_nllb.timing.jsonl')}
    rows={r['id']:r for r in read_jsonl(OUT/'real_clean/features.jsonl')}
    checkpoint=joblib.load(OUT/'corrected/number/checkpoints/et_full.joblib')
    cp=checkpoint['model'];head_times=[];single_times=[];differences=[]
    # Time one-example head calls; same trained model, no decoder work included.
    for record in timing:
        x=matrix([rows[record['id']]],checkpoint['features'])
        cp.steps[-1][1].n_jobs=8
        started=time.perf_counter();a=cp.predict_proba(x);head_times.append(time.perf_counter()-started)
        cp.steps[-1][1].n_jobs=1
        started=time.perf_counter();b=cp.predict_proba(x);single_times.append(time.perf_counter()-started)
        differences.append(float(np.max(np.abs(a-b))))
    basic=np.array([r['st_only']['seconds'] for r in timing])
    qwen=np.array([r['full_qwen']['seconds'] for r in timing])
    ev=np.array([evidence[r['id']]['seconds'] for r in timing])
    head=np.array(single_times);total=qwen+ev+head
    bare=np.array([r['translation_only']['seconds'] for r in timing])
    write_json(OUT/'efficiency/end_to_end.json',dict(n=len(timing),device='A100 80GB',batch_size=1,
        translation_only_mean=float(bare.mean()),st_with_decoder_confidence_mean=float(basic.mean()),qwen_full_mean=float(qwen.mean()),
        independent_path_mean=float(ev.mean()),head_mean=float(head.mean()),head_eight_workers_mean=float(np.mean(head_times)),
        head_worker_max_prediction_difference=max(differences),
        serial_full_mean=float(total.mean()),serial_full_over_st=float(total.mean()/bare.mean()),
        peak_memory_gib=max(r['full_qwen']['peak_allocated_bytes'] for r in timing)/2**30,
        method='sum of measured component latencies on same recordings; no overlap or model loading',
        limitation='Sum of separately measured stages; excludes model loading and scheduling overhead',
        comet_qe='baseline only, not used by the 21-feature full estimator'))


def cost_ablation():
    timing=read_jsonl(OUT/'efficiency/component_timings.jsonl')
    if len(timing)!=64:raise ValueError('Paired timing inputs incomplete')
    evidence={r['id']:r for r in read_jsonl(OUT/'real_clean/evidence/whisper_nllb.timing.jsonl')}
    rows={r['id']:r for r in read_jsonl(OUT/'real_clean/features.jsonl')}
    checkpoint=joblib.load(OUT/'corrected/number/checkpoints/et_without_stability.joblib')
    checkpoint['model'].steps[-1][1].n_jobs=1
    head=[]
    for repeat in [timing[0],*timing]:
        x=matrix([rows[repeat['id']]],checkpoint['features'])
        start=time.perf_counter();checkpoint['model'].predict_proba(x);head.append(time.perf_counter()-start)
    head=np.array(head[1:])
    total=np.array([r['st_only']['seconds']+evidence[r['id']]['seconds'] for r in timing])+head
    bare=np.array([r['translation_only']['seconds'] for r in timing])
    write_json(OUT/'efficiency/ablations.json',dict(method='et_without_stability',n=64,
        head_mean=float(head.mean()),serial_mean=float(total.mean()),serial_over_st=float(total.mean()/bare.mean()),
        direct_decoding='base plus decoder confidence; no probes or sampled translations',
        interpretation='Sum of same-recording component measurements; excludes loading and scheduling.'))


def main():
    cohort(OUT/'corrected',OUT/'corrected/number/predictions.jsonl',OUT/'corrected/number/metrics.json',OUT/'corrected/number/additional_analysis.json')
    cohort(OUT/'real_clean',OUT/'real_clean/frozen_transfer/predictions.jsonl',OUT/'real_clean/frozen_transfer/metrics.json',OUT/'real_clean/frozen_transfer/additional_analysis.json')
    costs()


if __name__=='__main__':main()
