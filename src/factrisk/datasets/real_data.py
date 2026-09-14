"""Output-blind selection of public CoVoST2 German number-preservation inputs."""
from __future__ import annotations
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from factrisk.core.io import read_jsonl, read_yaml, sha256_file, stable_int, write_json, write_jsonl, write_yaml
from factrisk.core.numbers import number_mentions
from factrisk.pipeline.workflow import ROOT, OUT, migrated

DE_SMALL = dict(zip('null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn siebzehn achtzehn neunzehn'.split(), range(20)))
DE_TENS = dict(zip('zwanzig dreißig vierzig fünfzig sechzig siebzig achtzig neunzig'.split(),range(20,100,10)))


def german_integer(word):
    word = word.lower()
    if word in DE_SMALL | DE_TENS:
        return (DE_SMALL | DE_TENS)[word]
    if word == 'ein':
        return 1
    for scale, n in (('tausend',1000),('hundert',100)):
        if word.count(scale)==1:
            left,right=word.split(scale)
            a=german_integer(left) if left else 1
            b=german_integer(right) if right else 0
            if a is not None and b is not None and 0<a<1000 and 0<=b<n:
                return a*n+b
    if word.count('und')==1:
        left,right=word.split('und')
        a=german_integer(left)
        if a is not None and 0<a<10 and right in DE_TENS:
            return a+DE_TENS[right]
    return None


def german_values(text):
    values=[]
    for match in re.finditer(r'\d+|[a-zäöüß]+',text.lower()):
        word=match.group()
        if word.isdigit():
            values.append(str(int(word)))
        elif word!='ein':  # bare indefinite articles are intentionally excluded
            n=german_integer(word)
            if n is not None:
                values.append(str(n))
    return values


def select():
    cv=Path('/workspace/dataset/st/cv/de')
    hist=migrated(read_jsonl(ROOT/'data/derived/factrisk/baseline/manifest.jsonl'))
    old_speakers={r['speaker_id'] for r in hist}
    old_texts={r['source_text'].lower().strip() for r in hist}
    old_refs={r['reference'].lower().strip() for r in hist}
    translations={}
    counts=Counter()
    with (cv/'covost_v2.de_en.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['split']=='test':
                counts['official_test_rows']+=1
                translations[r['path']]=r['translation']
    records={}
    for filename in ('test.tsv','dev.tsv','train.tsv','validated.tsv'):
        with (cv/filename).open() as f:
            for r in csv.DictReader(f,delimiter='\t'):
                if r['path'] in translations and r['path'] not in records:
                    records[r['path']]=r
        if len(records)==len(translations):
            break
    selected=[]
    seen=set()
    for path in sorted(translations):
        counts['considered']+=1
        if path not in records:
            counts['metadata_missing']+=1;continue
        record=records[path]; source=record['sentence'];ref=translations[path]
        speaker=record['client_id']
        if not speaker or speaker in old_speakers or source.lower().strip() in old_texts or ref.lower().strip() in old_refs:
            counts['historical_overlap_or_unknown_speaker']+=1;continue
        # Output-blind exclusions for ranges, decimals, ratios, dates, units and
        # ambiguous numeric formatting. This defines a restricted integer cohort.
        if re.search(r'\d\s*[-–/:.,]\s*\d|[%€$£]|\b(?:between|bis|zwischen|percent|prozent)\b',source+' '+ref,re.I):
            counts['format_or_range_ambiguity']+=1;continue
        de=german_values(source);en=number_mentions(ref)
        if len(de)!=1 or len(en)!=1 or de[0]!=en[0]['value']:
            counts['not_unique_matching_integer']+=1;continue
        audio=cv/'clips'/path
        if not audio.is_file():
            counts['audio_missing']+=1;continue
        key=(source.lower().strip(),ref.lower().strip())
        if key in seen:
            counts['duplicate_text']+=1;continue
        seen.add(key)
        selected.append(dict(id='covost2-official-test::'+path,pair_id=path,
            split='test',condition='clean',fact_type='number',speaker_id=speaker,
            source_language='de',target_language='en',audio=str(audio),clean_audio=str(audio),
            source_text=source,reference=ref,expected_slot=de[0],contrast_slot='',
            source_dataset='CoVoST2',source_split='test',selection='unique_source_reference_integer_v1',
            audio_sha256=sha256_file(audio),reference_number_span=[en[0]['start'],en[0]['end']],probes=[]))
    # A fixed ordering avoids source-file order driving later sharding or ties.
    selected.sort(key=lambda r:stable_int(r['id'],20260906))
    directory=ROOT/'data/derived/factrisk/real_clean'
    counts['selected']=len(selected)
    # Same waveforms cannot appear as distinct independent evaluation examples.
    hashes=[r['audio_sha256'] for r in selected]
    if len(hashes)!=len(set(hashes)):
        raise ValueError('Duplicate real waveforms: require provenance review')
    write_jsonl(directory/'manifest.jsonl',selected)
    cfg=migrated(read_yaml(ROOT/'configs/paper.yaml'))
    cfg['project'].update(protocol='external_real_number_preservation_v1',
        data_dir=str(directory),output_dir=str(OUT/'real_clean'))
    cfg['models']['direct_st']['batch_size']=1
    cfg['models']['evidence']['batch_size']=8
    # Samples needed for the full fusion comparison. Mild probes are added in a
    # separate preparation stage; no prediction may run without that completed.
    write_yaml(OUT/'configs/real_clean_config.yaml',cfg)
    report=dict(status='selected_before_inference',selection_version='unique_source_reference_integer_v1',
        funnel=dict(counts),speakers=len({r['speaker_id'] for r in selected}),
        covost_reference_sha256=sha256_file(cv/'covost_v2.de_en.tsv'),
        manifest_sha256=sha256_file(directory/'manifest.jsonl'),
        new_human_annotations=0,reference_type='existing_professional_translations',
        target='operational_integer_preservation_not_all_semantic_errors',
        limits='Restricted read-speech cohort; source-reference agreement is not new human validation.')
    write_json(OUT/'configs/real_selection.json',report)
    print(report,flush=True)


def prepare_probes():
    from factrisk.backends.audio import materialize_transform
    import numpy as np
    cfg=read_yaml(OUT/'configs/real_clean_config.yaml')
    rows=read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl')
    directory=ROOT/'data/derived/factrisk/real_full'
    for i,row in enumerate(rows):
        key=f"{stable_int(row['id'],20260906):016x}"
        row['probes']=[]
        for probe in cfg['data']['probes']:
            path=directory/f"audio/{probe['name']}/{key}.wav"
            materialize_transform(row['audio'],path,probe,np.random.default_rng(stable_int(row['id']+probe['name'],20260906)))
            row['probes'].append(dict(name=probe['name'],audio=str(path),audio_sha256=sha256_file(path)))
        if (i+1)%100==0:
            print(f'Prepared real probes {i+1}/{len(rows)}',flush=True)
    write_jsonl(directory/'manifest.jsonl',rows)
    cfg['project']['data_dir']=str(directory)
    write_yaml(OUT/'configs/real_full_config.yaml',cfg)
    print('Real full manifest ready',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--probes',action='store_true')
    args=parser.parse_args()
    prepare_probes() if args.probes else select()
