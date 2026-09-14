"""Output-blind natural-speech corruptions, reusing the frozen real cohort."""
import copy
from pathlib import Path
import numpy as np
from factrisk.backends.audio import read_audio,write_audio,add_noise,local_overlap,materialize_transform
from factrisk.core.io import read_jsonl,read_yaml,sha256_file,stable_int,write_json,write_jsonl,write_yaml
from factrisk.pipeline.workflow import ROOT,OUT


def prepare():
    cfg=read_yaml(OUT/'configs/real_full_config.yaml')
    rows=read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl')
    directory=ROOT/'data/derived/factrisk/real_stress'
    # Pair neighboring recordings with distinct source speakers. Shared donors
    # induce dependence; connected speaker components are the resampling units.
    ordered=list(rows)
    if len(ordered)%2: raise ValueError('Pairwise donor design requires an even cohort')
    donors={};parent={r['speaker_id']:r['speaker_id'] for r in rows}
    def root(a):
        while parent[a]!=a: a=parent[a]
        return a
    for i in range(0,len(ordered),2):
        if ordered[i]['speaker_id']==ordered[i+1]['speaker_id']:
            candidate=next((j for j in range(i+2,len(ordered)) if ordered[j]['speaker_id']!=ordered[i]['speaker_id']),None)
            if candidate is None: raise ValueError('No valid paired donor')
            ordered[i+1],ordered[candidate]=ordered[candidate],ordered[i+1]
        a,b=ordered[i:i+2];donors[a['id']]=b;donors[b['id']]=a
        parent[root(b['speaker_id'])]=root(a['speaker_id'])
    result=[]
    for i,row in enumerate(rows):
        x,sr=read_audio(row['audio']);d=donors[row['id']];dx,dsr=read_audio(d['audio'])
        for condition in ('noise_0db','tail_truncation_30pct','global_overlap_0db'):
            item=copy.deepcopy(row);item.update(id=row['id']+'::'+condition,condition=condition,
                original_id=row['id'],cluster_id=root(row['speaker_id']))
            key=f"{stable_int(item['id'],20260906):016x}"
            if condition=='noise_0db':
                waveform=add_noise(x,0.,np.random.default_rng(stable_int(item['id'],20260906)))
            elif condition=='tail_truncation_30pct':
                waveform=x[:max(1,round(.7*len(x)))]
            else:
                waveform=local_overlap(x,sr,dx,dsr,[0.,len(x)/sr],0.)
                item['overlap_donor']=dict(id=d['id'],speaker_id=d['speaker_id'],split='test')
            path=directory/f'audio/{condition}/{key}.wav';write_audio(path,waveform,sr)
            item.update(audio=str(path),audio_sha256=sha256_file(path),probes=[])
            for probe in cfg['data']['probes']:
                pp=directory/f"audio/probes/{probe['name']}/{key}.wav"
                materialize_transform(path,pp,probe,np.random.default_rng(stable_int(item['id']+probe['name'],20260906)))
                item['probes'].append(dict(name=probe['name'],audio=str(pp),audio_sha256=sha256_file(pp)))
            result.append(item)
        if (i+1)%100==0: print(f'Real stress preparation {i+1}/{len(rows)}',flush=True)
    write_jsonl(directory/'manifest.jsonl',result)
    cfg['project'].update(data_dir=str(directory),output_dir=str(OUT/'real_stress'),protocol='external_real_number_stress_v1')
    write_yaml(OUT/'configs/real_stress_config.yaml',cfg)
    write_json(OUT/'configs/real_stress_selection.json',dict(recordings=len(rows),inputs=len(result),
        speakers=len(parent),connected_speaker_donor_clusters=len({root(s) for s in parent}),
        source_selection='same_output_blind_542_recordings',conditions=['noise_0db','tail_truncation_30pct','global_overlap_0db'],
        new_tts=0,new_human_labels=0,alignment_used=False,
        metric='reference_integer_preservation',manifest_sha256=sha256_file(directory/'manifest.jsonl')))
    print(f'Real stress ready: {len(result)} inputs',flush=True)


if __name__=='__main__': prepare()
