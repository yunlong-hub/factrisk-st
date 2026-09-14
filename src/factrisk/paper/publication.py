"""Render publication tables/curves directly from immutable experiment outputs.

The CCFA small-multiple recipe informs the panel layout; Matplotlib is used
for multiple series and identification-bound ribbons, absent from that recipe.
No reference labels, predictions, or fitted models are changed here.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from factrisk.core.io import read_jsonl, sha256_file, write_json
from factrisk.eval.evaluation import keep_top, selective
from factrisk.pipeline.workflow import OUT, ROOT

PAPER = ROOT / 'papers/factriskst'
NAMES = {
    'lr_sequence_probability': 'Beam NLL summary', 'lr_token_entropy': 'Token entropy',
    'lr_asr': 'ASR confidence', 'lr_qe': 'COMET-QE',
    'lr_self_consistency': 'Self-consistency', 'lr_perturbation': 'Perturbation',
    'et_evidence': 'Evidence ET', 'et_asr_evidence': 'ASR+evidence ET',
    'attention_lr': 'Attention LR', 'et_full': 'FactRisk-ST',
    'et_without_acoustic': 'Without acoustics', 'et_without_stability': 'Without stability',
    'et_without_evidence': 'Without evidence', 'et_without_waveform': 'Without waveform',
    'et_without_sampling': 'Without sampling', 'et_without_probes': 'Without probes',
    'lr_full': 'Full logistic', 'hgb_full': 'Full boosted trees', 'lr_evidence': 'Evidence logistic',
    'hypothesis_nll_lr': 'Hypothesis NLL',
}


def read(path):
    return json.loads(Path(path).read_text())


def main_table():
    controlled = read(OUT/'corrected/number/metrics.json')['methods']
    natural = read(OUT/'real_clean/frozen_transfer/metrics.json')['methods']
    att = read(OUT/'corrected/number/attention/metrics.json')['metrics']
    controlled['attention_lr'] = att['controlled']
    natural['attention_lr'] = att['natural_clean']
    likelihood = OUT/'likelihood/metrics.json'
    if likelihood.exists():
        m = read(likelihood)['metrics']
        controlled['hypothesis_nll_lr'] = m['controlled']
        natural['hypothesis_nll_lr'] = m['natural_clean']
    qwen = ('lr_sequence_probability', 'lr_token_entropy', 'lr_qe',
            'hypothesis_nll_lr', 'et_evidence', 'et_asr_evidence',
            'attention_lr', 'et_full')
    seamless = ('lr_sequence_probability', 'et_evidence', 'et_asr_evidence', 'et_full')
    write_table(
        r'\caption{Number-preservation ranking for both backends; lower AURC is '
        r'better. $R_{90}$ is the identification bound $[E/K,(E+U)/K]$ on retained '
        r'error at top-90\% selection ($K=643$ controlled, $488$ natural), not a '
        r'confidence interval. Rows follow the controlled-AURC order within each '
        r'backend and bold marks the best AURC per cohort. '
        r'Attention LR is analyzed in Section~\ref{sec:attribution}.}',
        r'\label{tab:main}',
        [('Qwen2-Audio', qwen, controlled, natural),
         ('SeamlessM4T-v2', seamless,
          read(OUT/'seamless_controlled/number/metrics.json')['methods'],
          read(OUT/'seamless_real/frozen_transfer/metrics.json')['methods'])])
    return controlled, natural


def write_table(caption, label, backends):
    """One cross-column float holding every backend, so panels share a header."""
    lines = [r'\begin{table}[t]', r'\centering\footnotesize',
             r'\setlength{\tabcolsep}{4pt}', caption, label, r'\vspace{4pt}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l rr rr@{}}', r'\toprule',
             r'& \multicolumn{2}{c}{Controlled (C)} & \multicolumn{2}{c}{Natural clean (N)} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Method & AURC & $R_{90}$ (\%) & AURC & $R_{90}$ (\%) \\', r'\midrule']
    for position, (backend, names, c, n) in enumerate(backends):
        if position:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{5}{@{}l}{\textit{' + backend + r'}} \\')
        lines += backend_body(names, c, n)
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    (PAPER/'tables/merged_results.tex').write_text('\n'.join(lines)+'\n')
    for stale in ('qwen_results.tex', 'seamless_results.tex'):
        (PAPER/'tables'/stale).unlink(missing_ok=True)


def backend_body(names, c, n):
    """Data rows for one backend over both cohorts, without the float wrapper."""
    best = {'c': min(names, key=lambda x: c[x]['aurc_known']),
            'n': min(names, key=lambda x: n[x]['aurc_known'])}
    rows = []
    for name in sorted(names, key=lambda x: -c[x]['aurc_known']):
        row = [r'\factrisk{}' if name == 'et_full' else NAMES[name]]
        for key, v in (('c', c[name]), ('n', n[name])):
            op = v['fixed']['0.9']
            aurc = f'{v["aurc_known"]:.4f}'
            bound = (f'{100*op["known_errors"]/op["kept"]:.2f}'
                     f'--{100*(op["known_errors"]+op["unresolved"])/op["kept"]:.2f}')
            row += [r'\textbf{' + aurc + r'}' if name == best[key] else aurc, bound]
        rows.append(' & '.join(row) + r' \\')
    return rows


def curves():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':9,
        'axes.labelsize':9, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'legend.fontsize':9, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(6.98, 2.1))
    sources = [OUT/'corrected/number/predictions.jsonl', OUT/'real_clean/frozen_transfer/predictions.jsonl']
    styles = [('et_full','#0072B2','-'), ('et_asr_evidence','#D55E00','--'), ('attention_lr','#666666',':')]
    exports = []
    for ax, source, cohort, title in zip(axes, sources, ('controlled','natural_clean'),
            ('(a) German controlled: 714 inputs', '(b) German natural: 542 inputs')):
        rows = read_jsonl(source)
        ids = [r['id'] for r in rows]
        y = np.array([r['label'] if r['label'] is not None else np.nan for r in rows])
        attention = {r['id']:r['raw'] for r in read_jsonl(OUT/f'corrected/number/attention/{cohort}_predictions.jsonl')}
        for method, color, style in styles:
            score = np.array([attention[i] for i in ids]) if method=='attention_lr' else np.array([r['risks'][method]['raw'] for r in rows])
            values = [selective(y, keep_top(score, c, ids)) for c in np.linspace(.05,1.,191)]
            x = [100*v['coverage'] for v in values]
            lo = [100*v['risk_lower'] for v in values]
            hi = [100*v['risk_upper'] for v in values]
            ax.plot(x, lo, color=color, ls=style, lw=1.3, label=NAMES[method])
            ax.plot(x, hi, color=color, ls=style, lw=.8, alpha=.8)
            ax.fill_between(x, lo, hi, color=color, alpha=.07)
            for value in values:
                exports.append(dict(cohort=cohort, method=method, **value))
        ax.set(xlim=(5,100), ylim=(0,45 if cohort=='controlled' else 9),
               xlabel='Coverage (%)', ylabel='Retained error (%)', title=title)
        metric_path=OUT/('corrected/number/metrics.json' if cohort=='controlled' else 'real_clean/frozen_transfer/metrics.json')
        op=read(metric_path)['methods']['et_full']['operational']
        frozen_x=100*op['coverage']; top=ax.get_ylim()[1]
        ax.axvline(90,color='#999999',ls=':',lw=.7,zorder=0)
        ax.axvline(frozen_x,color='#7B3294',ls='-.',lw=.9)
        ax.plot([frozen_x,frozen_x],[100*op['risk_lower'],100*op['risk_upper']],
                color='#7B3294',marker='o',ms=3,lw=2,zorder=6)
        ax.annotate(f'Frozen: {frozen_x:.1f}%',xy=(frozen_x,.9*top),xytext=(39,.9*top),
                    color='#7B3294',fontsize=9,arrowprops=dict(arrowstyle='->',color='#7B3294',lw=.7))
        exports.append(dict(cohort=cohort,method='et_full',policy='frozen',**op))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', lw=.4, alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.53,-.015))
    fig.subplots_adjust(left=.07,right=.97,top=.88,bottom=.33,wspace=.32)
    destination = PAPER/'figures'
    metadata = PAPER/'metadata'
    destination.mkdir(exist_ok=True)
    metadata.mkdir(exist_ok=True)
    for suffix in ('pdf','svg','png'):
        fig.savefig(destination/f'risk_coverage.{suffix}', dpi=200)
    plt.close(fig)
    write_json(metadata/'risk_coverage.data.json', exports)
    write_json(metadata/'risk_coverage.provenance.json', dict(sources={str(p):sha256_file(p) for p in sources},
        interpretation='Ribbons bound unresolved labels; they are not sampling confidence intervals.',
        selection='Curves: retrospective all-input raw-score ordering; purple markers/lines: unchanged calibration threshold.'))


def cost_text():
    """Single-sentence cost note; the float is dropped so the page budget goes to results."""
    c = read(OUT/'efficiency/end_to_end.json')
    text = (f'A100 80\\,GB timing at batch size one over 64 recordings with GPU '
        f'synchronization: full extraction costs {c["serial_full_over_st"]:.2f}$\\times$ '
        f'ST-only decoding, of which the head takes {c["head_mean"]:.3f} s, excluding '
        'loading and scheduling.\n')
    (PAPER/'tables/cost.tex').write_text(text)


def extension_tables():
    """Stress cohort row plus the two conditions the text argues from."""
    stress_path=OUT/'real_stress/frozen_transfer/metrics.json'
    if stress_path.exists():
        entry=read(stress_path)['methods']['et_full']
        by_condition=entry['by_condition']
        m={'all':entry}
        for key in ('global_overlap_0db','tail_truncation_30pct'):
            if key in by_condition:
                m[key]=by_condition[key]
        names={'all':'All stress','tail_truncation_30pct':'Tail truncation',
               'global_overlap_0db':'Global overlap'}
        lines=[r'\begin{table}[t]',r'\centering\small\setlength{\tabcolsep}{3pt}',
            r'\caption{Frozen Qwen policy on the stress cohort and the two conditions the text argues from. $n_d$: decidable labels; $C_\tau$: coverage (\%); $R_\tau$: identification bounds (\%), not confidence intervals.}',
            r'\label{tab:stress}',r'\vspace{6pt}',r'\begin{tabular}{@{\hspace{4pt}}lrrrr@{\hspace{4pt}}}',r'\toprule',
            r'Condition & $n_d$ & AURC & $C_\tau$ & $R_\tau$ \\',r'\midrule']
        for key,v in m.items():
            op=v['operational']
            bound='--' if not op['kept'] else f'{100*op["risk_lower"]:.1f}--{100*op["risk_upper"]:.1f}'
            lines.append(f'{names[key]} & {v["n"]-v["unresolved"]} & {v["aurc_known"]:.3f} & '
                f'{100*op["coverage"]:.1f} & {bound} '+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        (PAPER/'tables/stress.tex').write_text('\n'.join(lines)+'\n')


def report(controlled, natural):
    lines = ['# 修订实验结果（自动生成）', '',
        '所有数值由逐样本预测或冻结评测输出生成。数值保持标签是操作性指标；未进行新增人工评测。', '',
        '## Qwen2-Audio 主实验', '',
        '| 方法 | 受控 AURC | 真人 AURC | 受控 Brier（校准前/后） | 真人 Brier（前/后） |',
        '|---|---:|---:|---:|---:|']
    for method, c in controlled.items():
        n = natural[method]
        lines.append(f'| {NAMES[method]} | {c["aurc_known"]:.6f} | {n["aurc_known"]:.6f} | '
                     f'{c["brier_raw"]:.4f}/{c["brier_calibrated"]:.4f} | {n["brier_raw"]:.4f}/{n["brier_calibrated"]:.4f} |')
    for title, metrics in [('受控',controlled),('真人 clean',natural)]:
        lines += ['', f'## {title}：选择性数值保持', '',
                  '| 方法 | 80% 风险界 | 90% 风险界 | 95% 风险界 | 冻结阈值实际覆盖率 |',
                  '|---|---:|---:|---:|---:|']
        for method, m in metrics.items():
            bounds = ['{:.2f}–{:.2f}%'.format(100*m['fixed'][c]['risk_lower'],100*m['fixed'][c]['risk_upper']) for c in ('0.8','0.9','0.95')]
            lines.append(f'| {NAMES[method]} | '+ ' | '.join(bounds)+f' | {100*m["operational"]["coverage"]:.2f}% |')
    for title, path in [('真人退化',OUT/'real_stress/frozen_transfer/metrics.json'),
                        ('Seamless 受控',OUT/'seamless_controlled/number/metrics.json'),
                        ('Seamless 真人',OUT/'seamless_real/frozen_transfer/metrics.json')]:
        lines += ['',f'## {title}','']
        if path.exists():
            d = read(path)
            lines += ['| 方法 | AURC | 90% 风险下界 | 90% 风险上界 | 实际覆盖率 |', '|---|---:|---:|---:|---:|']
            for method, m in d['methods'].items():
                lines.append(f'| {NAMES[method]} | {m["aurc_known"]:.4f} | {100*m["fixed"]["0.9"]["risk_lower"]:.2f}% | '
                    f'{100*m["fixed"]["0.9"]["risk_upper"]:.2f}% | {100*m["operational"]["coverage"]:.2f}% |')
        else:
            lines.append('实验运行中，尚无完整指标，未填充估算值。')
    lines += ['', '## 延迟（64 条配对录音）', '', '```json', json.dumps(read(OUT/'efficiency/end_to_end.json'), indent=2), '```']
    if (OUT/'fr_en/protocol.json').exists():
        lines += ['', '## French source-language transfer', '',
            'See [French results](fr_en/RESULTS.md) for both backends, clean/stress, '
            'retained counts, general chrF, and frozen-threshold coverage.',
            'See [all paired comparisons](significance_summary.md) for the unified interval summary.']
    if (OUT/'robustness/seeds/summary.json').exists():
        lines += ['', '## Training and implementation stability', '',
            'See [five-seed results](robustness/seeds/README.md) for all 30 fitted heads and 135 measurements, '
            '[empty-output sensitivity](robustness/empty/README.md) for frozen-head implementation differences, '
            'and [reproduction checks](robustness/README.md) for isolated CPU replay and COMET cache replay.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    (PAPER/'tables').mkdir(exist_ok=True)
    (PAPER/'metadata').mkdir(exist_ok=True)
    controlled, natural = main_table()
    curves()
    cost_text()
    extension_tables()
    report(controlled, natural)
    from factrisk.paper.review_analysis import main as review_analysis
    review_analysis()
    if (OUT/'fr_en/protocol.json').exists():
        from factrisk.french.french_report import build as french_report
        french_report()


if __name__ == '__main__':
    main()
