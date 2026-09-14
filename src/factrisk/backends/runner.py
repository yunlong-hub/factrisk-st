"""Single-GPU inference with input fingerprints, resumable outputs and timings."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from factrisk.core.io import append_jsonl, read_jsonl, read_yaml, sha256_file, stable_int, write_json


def fingerprint(row, role):
    payload={k:row[k] for k in ('id','source_language','target_language')}
    payload['audio_sha256']=sha256_file(row['audio'])
    if role=='direct':
        payload['probes']=[(p['name'],sha256_file(p['audio'])) for p in row.get('probes',[])]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def run(config_path,role,shard=0,shards=1):
    import torch
    from factrisk.backends.qwen import Qwen2AudioScorer
    from factrisk.backends.whisper import WhisperNLLBEvidence
    cfg=read_yaml(config_path)
    rows=read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl')
    if shards<1 or not 0<=shard<shards:
        raise ValueError('Invalid shard')
    rows=[r for i,r in enumerate(rows) if i%shards==shard]
    directory=Path(cfg['project']['output_dir'])/('predictions' if role=='direct' else 'evidence')
    stem='qwen2_audio' if role=='direct' else 'whisper_nllb'
    suffix='' if shards==1 else f'.shard-{shard:05d}-of-{shards:05d}'
    output=directory/f'{stem}{suffix}.jsonl'
    status=directory/f'{stem}{suffix}.status.json'
    model_cfg=cfg['models']['direct_st' if role=='direct' else 'evidence']
    model_hash=hashlib.sha256(json.dumps(model_cfg,sort_keys=True).encode()).hexdigest()
    existing=read_jsonl(output) if output.exists() else []
    done={r['id']:r for r in existing}
    if len(done)!=len(existing) or set(done)-{r['id'] for r in rows}:
        raise ValueError('Invalid existing prediction IDs')
    pending=[]
    for row in rows:
        fp=fingerprint(row,role)
        if row['id'] in done:
            previous=done[row['id']]
            if previous.get('input_fingerprint')!=fp or previous.get('model_config_sha256')!=model_hash:
                raise ValueError(f"Stale prediction: {row['id']}")
        else:
            pending.append((row,fp))
    base=dict(pid=os.getpid(),hostname=socket.gethostname(),role=role,shard=shard,shards=shards,
              expected=len(rows),model_config_sha256=model_hash,config_path=str(config_path))
    write_json(status,dict(**base,status='loading',completed=len(done)))
    start=time.monotonic()
    if role=='direct' and model_cfg.get('backend')=='seamless_m4t':
        from factrisk.backends.seamless import SeamlessScorer
        scorer=SeamlessScorer(model_cfg)
    else:
        scorer=Qwen2AudioScorer(model_cfg) if role=='direct' else WhisperNLLBEvidence(model_cfg)
    try:
        scorer.load()
        load_seconds=time.monotonic()-start
        for offset,(row,fp) in enumerate(pending):
            seed=stable_int(row['id'],20260906)%(2**32)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            t=time.monotonic()
            result=(scorer.infer_rows([row]) if role=='direct' else scorer.infer([row]))[0]
            torch.cuda.synchronize()
            seconds=time.monotonic()-t
            result.update(input_fingerprint=fp,model_config_sha256=model_hash,
                          sampling_seed=seed,protocol=cfg['project']['protocol'])
            if role=='direct':result['backend']=model_cfg.get('backend','qwen2_audio')
            append_jsonl(output,[result])
            append_jsonl(directory/f'{stem}{suffix}.timing.jsonl',[
                dict(id=row['id'],seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     batch_size=1,model_load_seconds=load_seconds if offset==0 else None,
                     gpu=torch.cuda.get_device_name(),role=role,cached=False)])
            completed=len(done)+offset+1
            write_json(status,dict(**base,status='running',completed=completed,
                                   elapsed_seconds=time.monotonic()-start))
            if completed%10==0 or offset==0:
                print(f'{role} shard {shard}: {completed}/{len(rows)}, last={seconds:.2f}s',flush=True)
        write_json(status,dict(**base,status='complete',completed=len(rows),elapsed_seconds=time.monotonic()-start))
    except BaseException as exc:
        write_json(status,dict(**base,status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--role',choices=['direct','evidence'],required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();run(a.config,a.role,a.shard,a.shards)
