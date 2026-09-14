"""Freeze French-to-English cohorts and existing German risk heads before inference."""
import argparse
import copy
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from factrisk.french.mentions import french_mentions
from factrisk.core.numbers import number_mentions
from factrisk.core.io import read_jsonl, read_yaml, write_json, write_jsonl, write_yaml, sha256_file, stable_int
from factrisk.pipeline.workflow import ROOT, OUT, DATA
from factrisk.core.freeze import seal

SEED=20260908
FR_DATA=DATA/'fr_en'
FR_OUT=OUT/'fr_en'


def select():
    if (FR_OUT/'selection.json').exists():
        raise RuntimeError('French cohort already selected; use existing frozen manifest')
    cv=Path('/workspace/dataset/st/cv/fr')
    with (cv/'covost_v2.fr_en.tsv').open() as f:
        translations={r['path']:r['translation'] for r in csv.DictReader(f,delimiter='\t') if r['split']=='test'}
    metadata={}
    for name in ('test.tsv','dev.tsv','train.tsv','validated.tsv'):
        with (cv/name).open() as f:
            for r in csv.DictReader(f,delimiter='\t'):
                if r['path'] in translations and r['path'] not in metadata: metadata[r['path']]=r
        if len(metadata)==len(translations):break
    old=read_jsonl(DATA/'baseline/manifest.jsonl')
    normalize=lambda s: re.sub(r'\s+',' ',s.lower()).strip()
    old_speakers={r['speaker_id'] for r in old}
    old_texts={normalize(r['source_text']) for r in old}
    old_refs={normalize(r['reference']) for r in old}
    old_hashes={sha256_file(r['audio']) for r in read_jsonl(DATA/'manifest.jsonl') if Path(r['audio']).exists()}
    counts=Counter(official_test=len(translations));rows=[];seen_text=set();seen_audio=set()
    for path,ref in sorted(translations.items()):
        r=metadata.get(path)
        if not r or not r['client_id']:
            counts['missing_metadata_or_speaker']+=1;continue
        source=r['sentence'];speaker=r['client_id'];audio=cv/'clips'/path
        if speaker in old_speakers or normalize(source) in old_texts or normalize(ref) in old_refs:
            counts['historical_overlap']+=1;continue
        if not audio.is_file():counts['missing_audio']+=1;continue
        key=(normalize(source),normalize(ref));digest=sha256_file(audio)
        if key in seen_text or digest in seen_audio or digest in old_hashes:
            counts['duplicate_text_or_audio']+=1;continue
        seen_text.add(key);seen_audio.add(digest)
        item=dict(id='covost2-fr-en-test::'+path,pair_id=path,split='test',condition='clean',
            speaker_id=speaker,source_language='fr',target_language='en',audio=str(audio),clean_audio=str(audio),
            source_text=source,reference=ref,expected_slot='',contrast_slot='',fact_type='number',
            source_dataset='CoVoST2',source_split='test',audio_sha256=digest,probes=[],numeric_cohort=False,general_cohort=False)
        if re.search(r'\d\s*[-–/:.,]\s*\d|[%€$£]|\b(?:entre|between|pourcent|percent)\b',source+' '+ref,re.I):
            counts['numeric_format_excluded']+=1
        else:
            fr=french_mentions(source);en=number_mentions(ref)
            if fr is not None and len(fr)==len(en)==1 and fr[0]['value']==en[0]['value'] and en[0]['value'].isdigit():
                item.update(numeric_cohort=True,expected_slot=fr[0]['value'],source_number_span=[fr[0]['start'],fr[0]['end']],
                    reference_number_span=[en[0]['start'],en[0]['end']])
        rows.append(item)
    # Round-robin speakers; fixed ordering, with no access to ST predictions.
    groups=defaultdict(list)
    for r in rows:groups[r['speaker_id']].append(r)
    for group in groups.values():group.sort(key=lambda r:stable_int(r['id'],SEED))
    speakers=sorted(groups,key=lambda s:stable_int(s,SEED));general=[];depth=0
    while len(general)<min(1000,len(rows)):
        for speaker in speakers:
            if len(groups[speaker])>depth:
                general.append(groups[speaker][depth])
                if len(general)==min(1000,len(rows)):break
        depth+=1
    for r in general:r['general_cohort']=True
    selected=[r for r in rows if r['numeric_cohort'] or r['general_cohort']]
    selected.sort(key=lambda r:stable_int(r['id'],SEED))
    write_jsonl(FR_DATA/'clean/manifest.jsonl',selected)
    counts.update(eligible_general=len(rows),numeric=sum(r['numeric_cohort'] for r in selected),general=len(general),union=len(selected))
    write_json(FR_OUT/'selection.json',dict(seed=SEED,funnel=dict(counts),
        numeric_speakers=len({r['speaker_id'] for r in selected if r['numeric_cohort']}),
        official_reference_sha256=sha256_file(cv/'covost_v2.fr_en.tsv'),
        manifest_sha256=sha256_file(FR_DATA/'clean/manifest.jsonl'),
        eligibility='unique matching source/reference integer; deterministic unrestricted cohort',
        source='official existing CoVoST2 translations',evaluated_before_outputs=True))
    print(dict(counts),flush=True)


def prepare():
    import numpy as np
    from factrisk.backends.audio import materialize_transform
    rows=read_jsonl(FR_DATA/'clean/manifest.jsonl')
    base=read_yaml(OUT/'configs/real_full_config.yaml')
    for row in rows:
        row['probes']=[]
        for probe in base['data']['probes']:
            key=f"{stable_int(row['id']+probe['name'],SEED):016x}"
            path=FR_DATA/f"probes/{key}.wav"
            materialize_transform(row['audio'],path,probe,np.random.default_rng(stable_int(row['id']+probe['name'],SEED)))
            row['probes'].append(dict(name=probe['name'],audio=str(path),audio_sha256=sha256_file(path)))
    write_jsonl(FR_DATA/'full/manifest.jsonl',rows)
    for backend,training in (('qwen','corrected'),('seamless','seamless_controlled')):
        cfg=copy.deepcopy(base)
        if backend=='seamless':cfg['models']['direct_st']=read_yaml(OUT/'configs/seamless_real_config.yaml')['models']['direct_st']
        destination=FR_OUT/backend
        cfg['project'].update(data_dir=str(FR_DATA/'full'),output_dir=str(destination),protocol='frozen_de_en_to_fr_en_v1')
        cfg.setdefault('sources',{})['revision_evidence_file']=str(FR_OUT/'qwen/evidence/whisper_nllb.jsonl')
        cfg['models']['optional_qe']['score_file']=str(destination/'qe/comet_qe.jsonl')
        write_yaml(FR_OUT/f'{backend}_config.yaml',cfg)
        seal(OUT/training/'number/checkpoints',FR_DATA/'full/manifest.jsonl',destination/'frozen_protocol.json')
    write_json(FR_OUT/'protocol.json',dict(seed=SEED,adaptation='none',primary_comparator='et_asr_evidence',
        primary_endpoint='numeric_clean_aurc',secondary=['numeric_top90_bounds','frozen_coverage','general_chrf'],
        model_selection='German checkpoints unchanged',feature_budget='full existing 21 features; two probes, three samples',
        bootstrap_replicates=2000,bootstrap_unit='speaker',general_sample_size=1000))


def refine():
    """Apply pre-inference lexical audit; never use predictions to revise eligibility."""
    import json
    if list(FR_OUT.glob('*/predictions/*.jsonl')) or list(FR_OUT.glob('*/frozen_protocol.json')):
        raise RuntimeError('Cannot refine after inference/freeze')
    rows=read_jsonl(FR_DATA/'clean/manifest.jsonl');removed=[];kept=[]
    pattern=r'\b(?:demi|demie|demies|tiers|quart|quarts|moitié|virgule|half|quarter|third|percent|pourcent)\b|(?<!\w)[+−-]\s*\d|\d+(?:er|ère|ème|ième|e)\b'
    for r in rows:
        if r['numeric_cohort'] and re.search(pattern,r['source_text']+' '+r['reference'],re.I):
            removed.append(dict(id=r['id'],reason='fraction_time_or_noncardinal_expression'))
            r.update(numeric_cohort=False,expected_slot='')
        if r['numeric_cohort'] or r['general_cohort']:kept.append(r)
    report=json.loads((FR_OUT/'selection.json').read_text())
    report['pre_inference_refinement']=dict(previous_funnel=report['funnel'],removed=removed,
        reason='lexical audit before any model inference; integer-only cohort')
    report['funnel']=dict(report['funnel'],numeric=sum(r['numeric_cohort'] for r in kept),union=len(kept))
    report['numeric_speakers']=len({r['speaker_id'] for r in kept if r['numeric_cohort']})
    write_jsonl(FR_DATA/'clean/manifest.jsonl',kept)
    report['manifest_sha256']=sha256_file(FR_DATA/'clean/manifest.jsonl')
    write_json(FR_OUT/'selection.json',report);print(report['funnel'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','prepare','refine']);a=p.parse_args()
    {'select':select,'prepare':prepare,'refine':refine}[a.stage]()
