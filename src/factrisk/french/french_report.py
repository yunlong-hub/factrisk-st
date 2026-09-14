"""Publish French evidence only after all required measured outputs exist."""
import csv
import json
from pathlib import Path

from factrisk.pipeline.workflow import ROOT, OUT
from factrisk.core.io import read_jsonl, sha256_file, write_json


def read(path):
    return json.loads(Path(path).read_text())


def build():
    from factrisk.eval.readiness import require_experiments
    require_experiments()
    root=OUT/'fr_en';paper=ROOT/'papers/factriskst';records=[]
    table=[r'\begin{table}[t]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',
        r'\caption{Frozen French transfer. Q/S: Qwen/Seamless; F/B: \factrisk{}/ASR+evidence. Intervals are paired 95\% CIs for AURC F minus B, using speaker--donor components under stress.}',
        r'\label{tab:french}',r'\vspace{6pt}',r'\begin{tabular}{@{\hspace{4pt}}lrr@{\hspace{4pt}}}',r'\toprule',
        r'Setting & AURC F/B & $\Delta$ 95\% CI \\',r'\midrule']
    lines=['# French-to-English frozen transfer','',
        'All labels are reference-number preservation labels; bounds are identification bounds, not confidence intervals.',
        'Heads, calibration and thresholds are transferred from German. Main comparator is ASR+evidence.',
        '', '| Cohort | Method | AURC | Brier | K/E/U at 90% | Risk bounds (%) | Frozen coverage (%) |',
        '|---|---|---:|---:|---|---|---:|']
    provenance={}
    for group in ('clean','stress'):
        base=root if group=='clean' else root/'stress'
        for backend in ('qwen','seamless'):
            directory=base/backend
            if read(directory/'finish.status.json')['status']!='complete':
                raise RuntimeError(f'French {group}/{backend} incomplete')
            path=directory/f'numeric_{group}/metrics.json';d=read(path)
            if d['feature_sha256']!=sha256_file(directory/'numeric_features.jsonl'):
                raise ValueError('Stale French metrics')
            provenance[str(path.relative_to(ROOT))]=sha256_file(path)
            frozen=read(directory/'frozen_protocol.json')
            for name in (('numeric','general') if group=='clean' else ('numeric',)):
                quality_path=directory/f'{name}_quality.json'
                quality=read(quality_path)
                if (quality['source_feature_sha256']!=sha256_file(directory/'features.jsonl')
                    or quality['manifest_sha256']!=frozen['manifest_sha256']
                    or quality['checkpoints']!=frozen['checkpoints']):
                    raise ValueError(f'Stale French quality: {group}/{backend}/{name}')
                expected=1000 if name=='general' else (943 if group=='clean' else 1886)
                if quality['n']!=expected or len(set(quality['selected_ids']))!=expected:
                    raise ValueError('French quality cohort count mismatch')
                provenance[str(quality_path.relative_to(ROOT))]=sha256_file(quality_path)
            if group=='clean' and backend=='qwen':
                for folder,method in (('likelihood','hypothesis_nll_lr'),('attention','attention_lr_all_heads')):
                    auxiliary=directory/folder/'metrics.json'
                    evidence=read(auxiliary)
                    if evidence['feature_sha256']!=sha256_file(directory/'numeric_features.jsonl'):
                        raise ValueError(f'Stale French {folder} metrics')
                    d['methods'][method]=evidence['metrics']
                    provenance[str(auxiliary.relative_to(ROOT))]=sha256_file(auxiliary)
            a=d['methods']['et_full'];b=d['methods']['et_asr_evidence']
            rows=read_jsonl(directory/'numeric_features.jsonl')
            all_errors=sum(r['severe_fact_error']==1 for r in rows)
            all_unresolved=sum(r['severe_fact_error'] is None for r in rows)
            ci=d['comparisons']['et_asr_evidence']['paired_aurc_difference_ci']
            label=('Qwen' if backend=='qwen' else 'Seamless')+' '+group
            lines.append(f'| {label} | No selection | — | — | {len(rows)}/{all_errors}/{all_unresolved} | '
                f'{100*all_errors/len(rows):.2f}–{100*(all_errors+all_unresolved)/len(rows):.2f} | 100.00 |')
            compact_label=('Q' if backend=='qwen' else 'S')+' '+group
            table.append(f'{compact_label} & {a["aurc_known"]:.4f}/{b["aurc_known"]:.4f} & '
                f'$[{ci[0]:.4f},{ci[1]:.4f}]$ '+r'\\')
            for method,m in d['methods'].items():
                op=m['fixed']['0.9'];frozen=m['operational']
                record=dict(cohort=f'{backend}_{group}',method=method,n=m['n'],aurc=m['aurc_known'],
                    brier=m['brier_calibrated'],K=op['kept'],E=op['known_errors'],U=op['unresolved'],
                    all_E=all_errors,all_U=all_unresolved,width_pp=100*op['unresolved']/op['kept'],
                    lower_pct=100*op['risk_lower'],upper_pct=100*op['risk_upper'],
                    frozen_coverage_pct=100*frozen['coverage'],frozen_K=frozen['kept'],
                    frozen_E=frozen['known_errors'],frozen_U=frozen['unresolved'])
                records.append(record)
                lines.append(f'| {label} | {method} | {record["aurc"]:.4f} | {record["brier"]:.4f} | '
                    f'{record["K"]}/{record["E"]}/{record["U"]} | {record["lower_pct"]:.2f}–{record["upper_pct"]:.2f} | '
                    f'{record["frozen_coverage_pct"]:.2f} |')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    quality_lines=['','## General translation quality','',
        '| Backend | n | All chrF | Random 90% mean | Full top90 chrF | Full−all 95% CI | Full−ASR+evidence 95% CI |',
        '|---|---:|---:|---:|---:|---|---|']
    for backend in ('qwen','seamless'):
        q=read(root/backend/'general_quality.json');m=q['methods']['et_full']
        quality_lines.append(f'| {backend} | {q["n"]} | {q["base_chrf"]:.2f} | {q["random90"]["mean_chrf"]:.2f} | '
            f'{m["top90_chrf"]:.2f} | {m["delta_ci95"]} | {q["full_minus_asr_evidence"]["ci95"]} |')
    lines += quality_lines
    lines += ['','Intervals are paired cluster-bootstrap percentile intervals; selections are fixed for the quality bootstrap.',
        'Random-selection variability is not a population confidence interval. No equivalence or noninferiority is inferred from an interval spanning zero.']
    # All required inputs have been validated before updating publication files.
    (paper/'tables/french.tex').write_text('\n'.join(table)+'\n')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_json(root/'publication_provenance.json',provenance)
    print(f'French publication: {len(records)} method/cohort rows')


if __name__=='__main__':build()
