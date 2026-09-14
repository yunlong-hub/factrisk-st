"""Build/evidence gates separate scientific completion from author metadata."""
import json
import re
from pathlib import Path

from factrisk.core.io import read_yaml, read_jsonl, sha256_file, write_json
from factrisk.pipeline.workflow import OUT, ROOT, DATA


def unique_rows(path):
    rows=read_jsonl(path)
    indexed={r['id']:r for r in rows}
    if len(indexed)!=len(rows):
        raise RuntimeError(f'Duplicate IDs: {path}')
    return indexed


def require_french_audit():
    audit=json.loads((OUT/'fr_en/reference_property_audit.json').read_text())
    manifest=DATA/'fr_en/clean/manifest.jsonl'
    rows=unique_rows(manifest)
    n=sum(bool(r.get('numeric_cohort')) for r in rows.values())
    if audit.get('status')!='passed' or audit.get('failed') or audit.get('failures'):
        raise RuntimeError('French reference property audit has not passed')
    if audit.get('manifest_sha256')!=sha256_file(manifest) or audit.get('numeric_rows')!=n:
        raise RuntimeError('French reference property audit does not match clean cohort')
    required={'reference_identity','digit_equivalence','word_equivalence','replacement_negative','deletion_negative'}
    if n==0 or any(audit.get('passed',{}).get(key)!=n for key in required):
        raise RuntimeError('French reference property audit checks incomplete')


def require_french_numeric(directory, manifest, group, metrics):
    manifest_rows=unique_rows(manifest)
    expected={key for key,row in manifest_rows.items() if row.get('numeric_cohort')}
    features=unique_rows(directory/'numeric_features.jsonl')
    predictions=unique_rows(directory/f'numeric_{group}/predictions.jsonl')
    if not expected or set(features)!=expected or set(predictions)!=expected:
        raise RuntimeError(f'French numeric cohort IDs differ: {directory}')
    if any(predictions[key]['label']!=row['severe_fact_error'] for key,row in features.items()):
        raise RuntimeError(f'French numeric labels differ: {directory}')
    for name,metric in metrics['methods'].items():
        if metric.get('status')=='unavailable':
            continue
        if metric.get('n')!=len(expected) or any(name not in row['risks'] for row in predictions.values()):
            raise RuntimeError(f'French method count or risks differ: {directory}/{name}')
    return expected


def require_french_auxiliary(folder):
    directory=OUT/'fr_en/qwen'
    auxiliary=directory/folder
    metrics=json.loads((auxiliary/'metrics.json').read_text())
    frozen=json.loads((auxiliary/'frozen_protocol.json').read_text())
    if any(metrics.get(key)!=value for key,value in frozen.items()):
        raise RuntimeError(f'French auxiliary protocol differs: {folder}')
    config=OUT/'fr_en/qwen_config.yaml'
    cfg=read_yaml(config)
    paths=dict(config_sha256=config,
        manifest_sha256=Path(cfg['project']['data_dir'])/'manifest.jsonl',
        feature_sha256=directory/'numeric_features.jsonl',
        checkpoint_sha256=OUT/('corrected/number/attention/model.joblib' if folder=='attention' else 'likelihood/model.joblib'),
        full_checkpoint_sha256=OUT/'corrected/number/checkpoints/et_full.joblib',
        hypotheses_sha256=directory/'predictions/qwen2_audio.jsonl',
        cache_sha256=auxiliary/('attention.jsonl' if folder=='attention' else 'hypothesis_nll.jsonl'),
        full_predictions_sha256=directory/'numeric_clean/predictions.jsonl')
    for key,path in paths.items():
        if metrics.get(key)!=sha256_file(path):
            raise RuntimeError(f'French auxiliary asset changed: {folder}/{key}')
    expected=set(unique_rows(directory/'numeric_features.jsonl'))
    selected=metrics.get('selected_ids',[])
    if len(selected)!=len(expected) or set(selected)!=expected or metrics['metrics'].get('n')!=len(expected):
        raise RuntimeError(f'French auxiliary cohort changed: {folder}')
    if set(unique_rows(auxiliary/'predictions.jsonl'))!=expected:
        raise RuntimeError(f'French auxiliary predictions incomplete: {folder}')


def require_french_quality(directory, manifest, group, frozen):
    rows=unique_rows(manifest)
    all_features=unique_rows(directory/'features.jsonl')
    if set(all_features)!=set(rows):
        raise RuntimeError(f'French quality feature IDs differ: {directory}')
    feature_digest=sha256_file(directory/'features.jsonl')
    for cohort in ('numeric','general') if group=='clean' else ('numeric',):
        metric=json.loads((directory/f'{cohort}_quality.json').read_text())
        expected={key for key,row in rows.items() if row.get(f'{cohort}_cohort')}
        selected=metric.get('selected_ids',[])
        if not expected or len(selected)!=len(expected) or set(selected)!=expected or metric.get('n')!=len(expected):
            raise RuntimeError(f'French quality selected IDs differ: {directory}/{cohort}')
        if (metric.get('source_feature_sha256')!=feature_digest or
            metric.get('manifest_sha256')!=frozen['manifest_sha256'] or
            metric.get('checkpoints')!=frozen['checkpoints']):
            raise RuntimeError(f'French quality assets changed: {directory}/{cohort}')
        if not {'et_full','et_asr_evidence'}<=set(metric['methods']):
            raise RuntimeError(f'French quality main methods missing: {directory}/{cohort}')
        if any(value.get('retained')!=round(.9*len(expected)) for value in metric['methods'].values()):
            raise RuntimeError(f'French quality retained count differs: {directory}/{cohort}')


def require_experiments():
    from factrisk.eval.robustness_gate import require_robustness
    require_robustness(ROOT)
    if (OUT/'fr_en/protocol.json').exists():
        require_french_audit()
        for group in ('clean','stress'):
            root=OUT/'fr_en' if group=='clean' else OUT/'fr_en/stress'
            for backend in ('qwen','seamless'):
                state=root/backend/'finish.status.json'
                if not state.exists() or json.loads(state.read_text()).get('status')!='complete':
                    raise RuntimeError(f'French experiment incomplete: {group}/{backend}')
                directory=root/backend
                frozen=json.loads((directory/'frozen_protocol.json').read_text())
                cfg=read_yaml(root/f'{backend}_config.yaml')
                if sha256_file(Path(cfg['project']['data_dir'])/'manifest.jsonl')!=frozen['manifest_sha256']:
                    raise RuntimeError(f'French manifest changed: {group}/{backend}')
                training='corrected' if backend=='qwen' else 'seamless_controlled'
                for name,digest in frozen['checkpoints'].items():
                    if sha256_file(OUT/training/'number/checkpoints'/name)!=digest:
                        raise RuntimeError(f'French checkpoint changed: {group}/{backend}/{name}')
                metrics=json.loads((directory/f'numeric_{group}/metrics.json').read_text())
                if sha256_file(directory/'numeric_features.jsonl')!=metrics['feature_sha256']:
                    raise RuntimeError(f'French features changed: {group}/{backend}')
                require_french_numeric(directory,Path(cfg['project']['data_dir'])/'manifest.jsonl',group,metrics)
                require_french_quality(directory,Path(cfg['project']['data_dir'])/'manifest.jsonl',group,frozen)
        for folder in ('attention','likelihood'):
            require_french_auxiliary(folder)
    for relative in ('logs/finish_experiments.status.json','likelihood/status.json'):
        path=OUT/relative
        if not path.exists() or json.loads(path.read_text()).get('status')!='complete':
            raise RuntimeError(f'Experiments incomplete: {path}')
    for relative in ('corrected/number/metrics.json','corrected/number/attention/metrics.json',
        'real_clean/frozen_transfer/metrics.json','real_stress/frozen_transfer/metrics.json',
        'seamless_controlled/number/metrics.json','seamless_real/frozen_transfer/metrics.json',
        'likelihood/metrics.json','efficiency/end_to_end.json','audit/data_isolation.json'):
        if not (OUT/relative).is_file():raise FileNotFoundError(relative)
    for relative in ('audit/data_isolation.json','audit/representation_tests.json'):
        if json.loads((OUT/relative).read_text()).get('status')!='passed':
            raise RuntimeError(f'Audit has not passed: {relative}')
    for cohort, config_name, training in (
        ('real_clean','real_full','corrected'), ('real_stress','real_stress','corrected'),
        ('seamless_real','seamless_real','seamless_controlled')):
        frozen=json.loads((OUT/cohort/'frozen_protocol.json').read_text())
        cfg=read_yaml(OUT/f'configs/{config_name}_config.yaml')
        manifest=Path(cfg['project']['data_dir'])/'manifest.jsonl'
        if sha256_file(manifest)!=frozen['manifest_sha256']:
            raise RuntimeError(f'External manifest changed after freeze: {cohort}')
        for name,digest in frozen['checkpoints'].items():
            if sha256_file(OUT/training/'number/checkpoints'/name)!=digest:
                raise RuntimeError(f'Frozen checkpoint changed: {cohort}/{name}')
        m=json.loads((OUT/cohort/'frozen_transfer/metrics.json').read_text())
        if sha256_file(OUT/cohort/'features.jsonl')!=m['feature_sha256']:
            raise RuntimeError(f'External features changed after scoring: {cohort}')


def gate_failures(checks,mode='technical'):
    """Only author metadata is outside the technical-draft gate."""
    if mode not in ('technical','submission'):
        raise ValueError(f'Unknown readiness mode: {mode}')
    return [key for key,value in checks.items() if not value and
            (mode=='submission' or key!='author_information_supplied')]


def check(mode='technical'):
    gate_failures({},mode)
    import fitz
    paper=ROOT/'papers/factriskst'
    tex='\n'.join(p.read_text() for p in [paper/'main.tex',*sorted((paper/'tables').glob('*.tex'))])
    visible='\n'.join(line.split('%')[0] if not r'\%' in line else line for line in tex.splitlines())
    missing=re.findall(r'\b(?:TBD|TODO)\b',visible)
    pdf=paper/'build/main.pdf';document=fitz.open(pdf)
    fonts={font[0]:font for page in document for font in page.get_fonts(full=True)}
    unembedded=[font[3] for xref,font in fonts.items() if not document.extract_font(xref)[3]]
    type3=[font[3] for font in fonts.values() if font[2]=='Type3']
    alltext='\n'.join(p.get_text() for p in document)
    log=(paper/'build/compile.log').read_text()
    citekeys={k for match in re.findall(r'\\cite\{([^}]+)\}',tex) for k in match.split(',')}
    bibkeys=set(re.findall(r'@\w+\{([^,]+),',(paper/'references.bib').read_text()))
    try:require_experiments();experiments=True
    except (RuntimeError,FileNotFoundError):experiments=False
    conclusion_pages=[i+1 for i,p in enumerate(document) if re.search(r'\bCONCLUSION\b',p.get_text())]
    acknowledgments_pages=[i+1 for i,p in enumerate(document) if 'ACKNOWLEDGMENTS' in p.get_text().upper()]
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S).group(1)
    abstract_words=len(abstract.split())
    late_float_pages=[i+1 for i,p in enumerate(document) if i>=4 and
        re.search(r'\b(?:Fig\.|Table)\s*\d+\s*[:.]',p.get_text())]
    checks=dict(experiments_complete=experiments,no_result_placeholders=not missing,
        no_technical_float_captions_after_page_four=not late_float_pages,
        at_most_five_pages=len(document)<=5,conclusion_in_technical_budget=bool(conclusion_pages) and max(conclusion_pages)<=4,
        fonts_embedded=not unembedded,no_type3_fonts=not type3,
        below_five_mb=pdf.stat().st_size<5_000_000,
        citations_resolve=not(citekeys-bibkeys),no_overfull_boxes='Overfull' not in log,
        abstract_100_to_150_words=100<=abstract_words<=150,
        nonfunding_acknowledgments_in_first_four_pages=bool(acknowledgments_pages) and max(acknowledgments_pages)<=4,
        author_information_supplied='to be supplied' not in alltext)
    failed=gate_failures(checks,mode)
    report=dict(checks=checks,mode=mode,passed=not failed,
        technical_passed=not gate_failures(checks,'technical'),
        submission_checks_passed=not gate_failures(checks,'submission'),
        failed_checks=failed,
        pages=len(document),pdf_sha256=sha256_file(pdf),
        manuscript_sha256=sha256_file(paper/'main.tex'),abstract_words=abstract_words,
        unembedded_fonts=unembedded,type3_fonts=type3,
        rule_source='https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php',rule_checked='2026-09-10',
        not_automatically_verified=['fifth-page content and complete visual layout',
            'author identity/order/ORCID, funding/conflicts, institutional ethics applicability',
            'submission portal metadata and copyright forms'],
        submission_ready=False)  # An automatic build check cannot certify author declarations.
    write_json(paper/'build/readiness.json',report)
    print(json.dumps(report,indent=2))
    return report


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['technical','submission'],default='technical',
                        help='Submission additionally requires supplied author information')
    args=parser.parse_args(argv)
    report=check(args.mode)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
