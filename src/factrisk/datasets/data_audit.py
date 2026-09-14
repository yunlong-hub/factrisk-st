"""Deterministic data-isolation and artifact checks; not human validation."""
from collections import Counter, defaultdict
from factrisk.core.io import read_jsonl, sha256_file, write_json
from factrisk.pipeline.workflow import OUT, DATA


def main():
    controlled=read_jsonl(DATA/'manifest.jsonl')
    real=read_jsonl(DATA/'real_full/manifest.jsonl')
    stress=read_jsonl(DATA/'real_stress/manifest.jsonl')
    report={}
    clean=[r for r in controlled if r['condition']=='clean']
    for field in ('speaker_id','pair_id','source_text','reference','audio_sha256'):
        partitions=defaultdict(set)
        for row in clean:partitions[row[field]].add(row['split'])
        count=sum(len(s)>1 for s in partitions.values())
        if count:raise ValueError(f'Cross-split leakage in {field}: {count}')
        report[field+'_cross_partition_unique']=count
    for field in ('speaker_id','source_text','reference'):
        normalize=lambda x:str(x).lower().strip()
        c={normalize(r[field]) for r in controlled};n={normalize(r[field]) for r in real}
        if c&n:raise ValueError(f'External overlap in {field}')
        report['external_'+field+'_overlap']=len(c&n)
    for group,rows in [('controlled',controlled),('natural_clean',real),('natural_stress',stress)]:
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
        for row in rows:
            if sha256_file(row['audio'])!=row['audio_sha256']:raise ValueError(f'Changed audio {row["id"]}')
            donor=row.get('overlap_donor')
            if donor and (donor['split']!=row['split'] or donor['speaker_id']==row['speaker_id']):
                raise ValueError('Invalid overlap donor')
        report[group]=dict(inputs=len(rows),speakers=len({r['speaker_id'] for r in rows}),
            unique_audio_hashes=len({r['audio_sha256'] for r in rows}),
            conditions=dict(Counter(r['condition'] for r in rows)))
    real_by_id={r['id']:r for r in real}
    stress_by_source={r['original_id']:r['cluster_id'] for r in stress}
    for row in stress:
        if row['condition']=='global_overlap_0db':
            donor=row['overlap_donor']
            if donor['id'] not in real_by_id or stress_by_source[donor['id']]!=row['cluster_id']:
                raise ValueError('Mixing dependency crosses resampling clusters')
    report['stress_connected_clusters']=len(set(stress_by_source.values()))
    report.update(status='passed',human_validation=False,
        scope='Manifest provenance, hashes and isolation only; no semantic or perceptual accuracy estimate.')
    write_json(OUT/'audit/data_isolation.json',report)


if __name__=='__main__':main()
