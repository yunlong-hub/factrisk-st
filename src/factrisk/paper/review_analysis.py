"""Post-hoc reporting only: retained counts and signal diagnostics, no refitting."""
import csv
import json
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from factrisk.core.io import read_jsonl, write_json
from factrisk.eval.evaluation import keep_top, selective
from factrisk.pipeline.workflow import OUT, ROOT

COHORTS = {
    'Qwen controlled': 'corrected/number',
    'Qwen natural clean': 'real_clean/frozen_transfer',
    'Qwen natural stress': 'real_stress/frozen_transfer',
    'Seamless controlled': 'seamless_controlled/number',
    'Seamless natural clean': 'seamless_real/frozen_transfer',
}


def counts_record(cohort, method, policy, n, value):
    k, e, u = (value[x] for x in ('kept', 'known_errors', 'unresolved'))
    if not 0 <= e + u <= k <= n:
        raise ValueError('Invalid retained-set counts')
    return dict(cohort=cohort, method=method, policy=policy, N=n, K=k, E=e, U=u,
        coverage_pct=100*k/n, lower_pct=100*e/k if k else None,
        upper_pct=100*(e+u)/k if k else None, width_pp=100*u/k if k else None)


def count_tables():
    from factrisk.paper.publication import NAMES
    records=[]
    for cohort, relative in COHORTS.items():
        directory=OUT/relative
        methods=json.loads((directory/'metrics.json').read_text())['methods']
        rows=read_jsonl(directory/'predictions.jsonl')
        ids=[r['id'] for r in rows]
        if len(ids)!=len(set(ids)):
            raise ValueError('Duplicate evaluation IDs')
        y=np.array([r['label'] if r['label'] is not None else np.nan for r in rows])
        records.append(counts_record(cohort,'No selection','all',len(rows),
                                     selective(y,np.ones(len(y),bool))))
        scores={m:np.array([r['risks'][m]['raw'] for r in rows]) for m in methods}
        if cohort in ('Qwen controlled','Qwen natural clean'):
            key='controlled' if cohort.endswith('controlled') else 'natural_clean'
            for directory_name, name in [('corrected/number/attention','attention_lr'),
                                         ('likelihood','hypothesis_nll_lr')]:
                aux=OUT/directory_name
                methods[name]=json.loads((aux/'metrics.json').read_text())['metrics'][key]
                score_rows=read_jsonl(aux/f'{key}_predictions.jsonl')
                index={r['id']:r['raw'] for r in score_rows}
                if len(index)!=len(score_rows) or set(index)!=set(ids):
                    raise ValueError('Auxiliary score IDs differ')
                scores[name]=np.array([index[i] for i in ids])
        for method, value in methods.items():
            v=selective(y,keep_top(scores[method],.9,ids))
            for field in ('kept','known_errors','unresolved'):
                if v[field]!=value['fixed']['0.9'][field]:
                    raise ValueError(f'Frozen top-90% count mismatch: {cohort}/{method}/{field}')
            records.append(counts_record(cohort,NAMES[method],'top90',len(rows),v))
            records.append(counts_record(cohort,NAMES[method],'frozen',len(rows),value['operational']))
    with (OUT/'metrics/review_counts.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    lines=['# Evaluation count appendix', '',
        'Generated from frozen outputs. No model, threshold, or label is refitted.',
        'K: retained; E: decidable number errors; U: unresolved. Width = 100 U/K percentage points.',
        'Bounds are identification bounds, not confidence intervals or human-label accuracy.',
        'This local artifact appendix is separate from the four-page technical manuscript; it is not an assumed conference supplementary entitlement.', '']
    for cohort in COHORTS:
        lines += [f'## {cohort}', '', '| Method | Policy | K | E | U | Coverage (%) | Bounds (%) | Width (pp) |',
                  '|---|---|---:|---:|---:|---:|---|---:|']
        for r in records:
            if r['cohort']!=cohort or (r['policy']=='frozen' and r['method']!='FactRisk-ST'):
                continue
            bound='undefined' if not r['K'] else f'{r["lower_pct"]:.2f}–{r["upper_pct"]:.2f}'
            width='undefined' if not r['K'] else f'{r["width_pp"]:.2f}'
            lines.append(f'| {r["method"]} | {r["policy"]} | {r["K"]} | {r["E"]} | {r["U"]} | '
                         f'{r["coverage_pct"]:.2f} | {bound} | {width} |')
        lines.append('')
    (ROOT/'papers/factriskst/metadata/evaluation_details.md').write_text('\n'.join(lines)+'\n')
    return records


def signal_diagnostics():
    sources={
        'controlled_train':[OUT/'corrected/features.jsonl',OUT/'corrected/labels_unresolved.jsonl'],
        'natural_clean':[OUT/'real_clean/features.jsonl'],
        'natural_stress':[OUT/'real_stress/features.jsonl'],
    }
    names=['perturb_mean_distance','sample_mean_distance','evidence_text_distance']
    result={}
    for cohort,paths in sources.items():
        rows=[r for p in paths for r in read_jsonl(p) if r['fact_type']=='number'
              and (cohort!='controlled_train' or r['split']=='train')]
        groups={'pooled':rows,**{c:[r for r in rows if r['condition']==c]
                              for c in sorted({r['condition'] for r in rows})}}
        result[cohort]={}
        for condition,subset in groups.items():
            correlations=[]
            for x,y in combinations(names,2):
                a=np.array([[r['features'][x],r['features'][y]] for r in subset])
                a=a[np.isfinite(a).all(axis=1)]
                rho=float(spearmanr(a[:,0],a[:,1]).statistic) if len(a)>1 and all(np.ptp(a,axis=0)) else None
                correlations.append(dict(x=x,y=y,n=len(a),spearman=rho))
            correct=[r for r in subset if r['severe_fact_error']==0]
            count=sum(r['features']['sample_number_disagreement']>0 for r in correct)
            result[cohort][condition]=dict(n=len(subset),correlations=correlations,
                correct_outputs=len(correct),correct_with_sample_number_disagreement=count)
    write_json(OUT/'metrics/review_diagnostics.json',dict(
        scope='Post-hoc descriptive analysis of cached signals; not a causal explanation or model-selection criterion.',
        groups=result))
    return result


def main():
    records=count_tables();diagnostics=signal_diagnostics()
    print(f'Validated and exported {len(records)} count rows; descriptive diagnostics for {len(diagnostics)} cohorts.')


if __name__=='__main__':main()
