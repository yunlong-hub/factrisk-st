"""Paired, synchronized ST-only versus full-Qwen confidence timing."""
import argparse
import copy
import time
from pathlib import Path
import numpy as np
from factrisk.core.io import read_jsonl,read_yaml,stable_int,write_json,append_jsonl
from factrisk.backends.qwen import Qwen2AudioScorer


def run(config_path,destination,n=64):
    import torch
    cfg=read_yaml(config_path);out=Path(destination)
    if (out/'component_timings.jsonl').exists():
        raise FileExistsError('Use a fresh timing destination; existing measurements must not be appended to')
    rows=read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl')
    rows=sorted(rows,key=lambda r:stable_int(r['id'],60906))[:n]
    scorer=Qwen2AudioScorer(copy.deepcopy(cfg['models']['direct_st']));scorer.load()
    original=copy.deepcopy(scorer.config);baseline=dict(original,num_samples=0)
    bare=dict(baseline,collect_scores=False)
    records=[]
    # Exclude one warmup from the measured set.
    scorer.config=baseline;scorer.infer_rows([dict(rows[0],probes=[])])
    for i,row in enumerate(rows):
        values={}
        # Alternate order to reduce systematic warm-cache/order advantage.
        for mode in (('translation_only','st_only','full_qwen') if i%2==0 else ('full_qwen','st_only','translation_only')):
            scorer.config={'st_only':baseline,'translation_only':bare,'full_qwen':original}[mode]
            item=row if mode=='full_qwen' else dict(row,probes=[])
            seed=stable_int(row['id'],20260906)%(2**32);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            scorer.infer_rows([item]);torch.cuda.synchronize()
            values[mode]=dict(seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        record=dict(id=row['id'],gpu=torch.cuda.get_device_name(),batch_size=1,**values)
        append_jsonl(out/'component_timings.jsonl',[record]);records.append(record)
        if (i+1)%16==0: print(f'Efficiency {i+1}/{len(rows)}',flush=True)
    base=np.array([r['st_only']['seconds'] for r in records]);full=np.array([r['full_qwen']['seconds'] for r in records])
    write_json(out/'component_summary.json',dict(n=len(rows),selection='fixed_hash_sample_from_real_cohort',
        translation_only_mean=float(np.mean([r['translation_only']['seconds'] for r in records])),
        st_only_mean=float(base.mean()),full_qwen_mean=float(full.mean()),
        full_qwen_over_st_ratio=float(full.mean()/base.mean()),
        includes=['audio_io','ST','decoder_confidence','two_audio_probes','three_samples'],
        separately_accounted=['Whisper_NLLB','COMET_QE_baseline','risk_head'],model_loading_excluded=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();run(a.config,a.destination)
