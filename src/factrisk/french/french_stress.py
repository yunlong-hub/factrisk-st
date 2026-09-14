"""Output-blind French numerical stress data; no training or GPU inference."""
import copy
import heapq
from collections import defaultdict

import numpy as np

from factrisk.backends.audio import read_audio, write_audio, add_noise, local_overlap, materialize_transform
from factrisk.french.french_data import FR_DATA, FR_OUT, SEED
from factrisk.core.freeze import seal
from factrisk.core.io import read_jsonl, read_yaml, sha256_file, stable_int, write_json, write_jsonl, write_yaml
from factrisk.pipeline.workflow import OUT

CONDITIONS=('noise_0db','global_overlap_0db')


def donor_design(rows, seed=SEED):
    """Disjoint recording pairs, with one distinct-speaker 3-cycle if odd.

    Pair largest remaining speaker groups first, so rare speakers are not
    exhausted prematurely. Speaker reuse connects recording pairs for inference.
    """
    groups=defaultdict(list)
    for row in rows:groups[row['speaker_id']].append(row)
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate source IDs')
    if len(rows)<2 or max(map(len,groups.values()))>len(rows)//2:
        raise ValueError('Distinct-speaker pair/triangle design is infeasible')
    for group in groups.values():group.sort(key=lambda r:stable_int(r['id'],seed))
    heap=[(-len(group),stable_int(s,seed),s) for s,group in groups.items()]
    heapq.heapify(heap);donors={};parent={s:s for s in groups}
    def root(s):
        while parent[s]!=s:
            parent[s]=parent[parent[s]];s=parent[s]
        return s
    def connect(a,b):
        if a['speaker_id']==b['speaker_id']:raise ValueError('Same-speaker donor')
        donors[a['id']]=b
        ra,rb=root(a['speaker_id']),root(b['speaker_id'])
        parent[max(ra,rb)]=min(ra,rb)
    remaining=len(rows)
    while remaining>3 or remaining==2:
        selected=[heapq.heappop(heap) for _ in range(2)]
        a,b=[groups[s].pop() for _,_,s in selected]
        connect(a,b);connect(b,a)
        for count,tie,s in selected:
            if count+1<0:heapq.heappush(heap,(count+1,tie,s))
        remaining-=2
    if remaining:
        if len(heap)!=3 or any(count!=-1 for count,_,_ in heap):
            raise ValueError('Odd remainder must contain three distinct speakers')
        triple=[groups[heapq.heappop(heap)[2]].pop() for _ in range(3)]
        for a,b in zip(triple,triple[1:]+triple[:1]):connect(a,b)
    clusters={r['id']:root(r['speaker_id']) for r in rows}
    return donors,clusters


def prepare():
    directory=FR_DATA/'stress';destination=FR_OUT/'stress'
    if (directory/'manifest.jsonl').exists():
        raise RuntimeError('Stress cohort already prepared; refusing to overwrite frozen inputs')
    if list(destination.glob('*/predictions/*.jsonl')):
        raise RuntimeError('Stress inference already exists')
    source=FR_DATA/'full/manifest.jsonl'
    source_hash=sha256_file(source)
    rows=[r for r in read_jsonl(source) if r['numeric_cohort']]
    if len(rows)!=943:raise ValueError(f'Expected final 943-number cohort, got {len(rows)}')
    if any(r['split']!='test' or r['condition']!='clean' for r in rows):
        raise ValueError('Stress inputs must be clean test recordings')
    cfg=read_yaml(FR_OUT/'qwen_config.yaml')
    donors,clusters=donor_design(rows)
    result=[]
    for i,row in enumerate(rows):
        x,sr=read_audio(row['audio']);donor=donors[row['id']]
        dx,dsr=read_audio(donor['audio'])
        for condition in CONDITIONS:
            item=copy.deepcopy(row)
            item.update(id=row['id']+'::'+condition,condition=condition,original_id=row['id'],
                cluster_id=clusters[row['id']],general_cohort=False,probes=[])
            key=f"{stable_int(item['id'],SEED):016x}"
            if condition=='noise_0db':
                waveform=add_noise(x,0.,np.random.default_rng(stable_int(item['id'],SEED)))
            else:
                waveform=local_overlap(x,sr,dx,dsr,[0.,len(x)/sr],0.)
                item['overlap_donor']=dict(id=donor['id'],speaker_id=donor['speaker_id'],split='test')
            path=directory/f'audio/{condition}/{key}.wav'
            write_audio(path,waveform,sr)
            item.update(audio=str(path),audio_sha256=sha256_file(path))
            for probe in cfg['data']['probes']:
                pp=directory/f"audio/probes/{probe['name']}/{key}.wav"
                materialize_transform(path,pp,probe,np.random.default_rng(stable_int(item['id']+probe['name'],SEED)))
                item['probes'].append(dict(name=probe['name'],audio=str(pp),audio_sha256=sha256_file(pp)))
            result.append(item)
        if (i+1)%100==0:print(f'French stress: {i+1}/{len(rows)} source recordings',flush=True)
    if sha256_file(source)!=source_hash:raise ValueError('Clean manifest changed during preparation')
    write_jsonl(directory/'manifest.jsonl',result)
    for backend,training in [('qwen','corrected'),('seamless','seamless_controlled')]:
        backend_cfg=read_yaml(FR_OUT/f'{backend}_config.yaml')
        output=destination/backend
        backend_cfg['project'].update(data_dir=str(directory),output_dir=str(output),
            protocol='frozen_de_en_to_fr_en_stress_v1')
        backend_cfg.setdefault('sources',{})['revision_evidence_file']=str(destination/'qwen/evidence/whisper_nllb.jsonl')
        backend_cfg['models']['optional_qe']['score_file']=str(output/'qe/comet_qe.jsonl')
        write_yaml(destination/f'{backend}_config.yaml',backend_cfg)
        seal(OUT/training/'number/checkpoints',directory/'manifest.jsonl',output/'frozen_protocol.json')
    write_json(destination/'selection.json',dict(seed=SEED,recordings=len(rows),inputs=len(result),
        speakers=len({r['speaker_id'] for r in rows}),connected_speaker_donor_clusters=len(set(clusters.values())),
        conditions=list(CONDITIONS),donor_design='470 reciprocal pairs and one distinct-speaker 3-cycle',
        source_manifest_sha256=source_hash,manifest_sha256=sha256_file(directory/'manifest.jsonl'),
        source_selection='all frozen French numerical recordings; same bases across both conditions',
        adaptation='none',metric='reference_integer_preservation',alignment_used=False,
        bootstrap_unit='speaker_donor_connected_component',bootstrap_seed=20260906,
        new_tts=0,probes_per_input=len(cfg['data']['probes']),samples_per_input=cfg['models']['direct_st']['num_samples']))
    print(f'French stress ready: {len(result)} inputs; {len(set(clusters.values()))} clusters',flush=True)


if __name__=='__main__':prepare()
