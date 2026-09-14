"""Task-adapted attention-map baseline (Waldendorf et al., ACL Findings 2026).

Teacher-force the cached hypothesis, never a reference. Reduce each layer's
weights to four per-head summaries immediately. This is a second-pass baseline,
not a claim to reproduce the authors' data, task labels or head selection.
"""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path
import numpy as np
from factrisk.core.io import append_jsonl, read_jsonl, read_yaml, sha256_file, write_json
from factrisk.backends.qwen import LANGUAGE_NAMES


def summarize(weights, audio_positions, prompt_length, output_length):
    """weights [heads, query, key]; queries predict each hypothesis token."""
    import torch
    # The last prompt token predicts y_1; y_(t-1) predicts y_t.
    a=weights[:,prompt_length-1:prompt_length+output_length-1,:].float()
    audio=a[:,:,audio_positions]
    prompt_positions=torch.arange(prompt_length,device=a.device)
    is_audio=torch.zeros(prompt_length,dtype=torch.bool,device=a.device)
    is_audio[audio_positions]=True
    text=a[:,:,prompt_positions[~is_audio]]
    audio_mass=audio.sum(-1)
    art_mass=a[:,:,prompt_length:].sum(-1)  # future tokens have causal zero weights
    ratio=(audio_mass/(audio_mass+art_mass).clamp_min(1e-12)).mean(-1)
    def entropy(v):
        p=v/v.sum(-1,keepdim=True).clamp_min(1e-12)
        return torch.special.entr(p).sum(-1).mean(-1)
    centered=audio-audio.mean(-1,keepdim=True)
    if output_length>1:
        u,v=centered[:,:-1],centered[:,1:]
        norm=(u.square().sum(-1)*v.square().sum(-1)).sqrt()
        # Constant distributions have undefined Pearson correlation; record 0.
        corr=torch.where(norm>1e-12,(u*v).sum(-1)/norm.clamp_min(1e-12),0).mean(-1)
    else:
        corr=torch.zeros_like(ratio)
    return torch.stack([ratio,corr,entropy(audio),entropy(text)]).cpu().numpy()


def run(config_path,predictions_path,output_path,limit=None,*,selected_ids=None):
    import librosa
    import torch
    from transformers import AutoProcessor,Qwen2AudioForConditionalGeneration
    cfg=read_yaml(config_path);mcfg=cfg['models']['direct_st']
    rows=read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl')
    # Primary scope is predeclared numbers, not a post-output success subset.
    rows=[r for r in rows if r['fact_type']=='number']
    if selected_ids is not None:
        selected_ids=set(selected_ids)
        rows=[r for r in rows if r['id'] in selected_ids]
        if {r['id'] for r in rows}!=selected_ids:
            raise ValueError('Attention selection contains missing/non-number inputs')
    if limit is not None: rows=rows[:limit]
    predictions={r['id']:r for r in read_jsonl(predictions_path)}
    if any(r['id'] not in predictions for r in rows): raise ValueError('Missing hypotheses')
    out=Path(output_path);previous=read_jsonl(out) if out.exists() else []
    done={r['id']:r for r in previous}
    if len(done)!=len(previous): raise ValueError('Duplicate attention rows')
    model_path=mcfg['model_path']
    processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    model=Qwen2AudioForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,
        device_map='auto',attn_implementation='eager').eval()
    device=next(model.parameters()).device
    sr=int(processor.feature_extractor.sampling_rate)
    # Transformers 5 moved the text backbone under the multimodal base model.
    layers=model.model.language_model.layers
    status=out.with_suffix('.status.json')
    try:
        for i,row in enumerate(rows):
            hyp=predictions[row['id']]['translation']
            fp=hashlib.sha256(json.dumps([sha256_file(row['audio']),hyp,model_path,'attention_v1'],ensure_ascii=False).encode()).hexdigest()
            if row['id'] in done:
                if done[row['id']]['input_fingerprint']!=fp: raise ValueError('Stale attention cache')
                continue
            source=LANGUAGE_NAMES[row['source_language']];target=LANGUAGE_NAMES[row['target_language']]
            conversation=[dict(role='user',content=[dict(type='audio',audio_url=row['audio']),
                dict(type='text',text=f'Translate the speech from {source} to {target}. Output only the translation.')])]
            prompt=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
            audio,_=librosa.load(row['audio'],sr=sr,mono=True)
            audio_key='audio' if 'audio' in inspect.signature(processor.__call__).parameters else 'audios'
            inputs=processor(text=[prompt],**{audio_key:[audio]},sampling_rate=sr,return_tensors='pt')
            inputs={k:v.to(device) for k,v in inputs.items()}
            prompt_len=inputs['input_ids'].shape[1]
            audio_pos=torch.where(inputs['input_ids'][0]==model.config.audio_token_index)[0]
            hyp_ids=processor.tokenizer(hyp,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)
            if not len(audio_pos) or hyp_ids.shape[1]==0: raise ValueError('Empty audio or hypothesis tokens')
            inputs['input_ids']=torch.cat([inputs['input_ids'],hyp_ids],dim=1)
            inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
            summaries=[]
            def hook(module,args,result):
                if len(result)<2 or result[1] is None: raise RuntimeError('Attention unavailable in eager mode')
                summaries.append(summarize(result[1][0],audio_pos,prompt_len,hyp_ids.shape[1]))
            handles=[layer.self_attn.register_forward_hook(hook) for layer in layers]
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            try:
                with torch.inference_mode(): model(**inputs,use_cache=False,output_attentions=False)
            finally:
                for handle in handles: handle.remove()
            torch.cuda.synchronize()
            if len(summaries)!=len(layers): raise RuntimeError('Missing layer summaries')
            values=np.stack(summaries,axis=1)  # metric, layer, head
            append_jsonl(out,[dict(id=row['id'],input_fingerprint=fp,features=values.reshape(-1).tolist(),
                feature_shape=list(values.shape),seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),hypothesis_source=str(predictions_path),
                label_source='none',version='attention_v1')])
            write_json(status,dict(status='running',completed=i+1,expected=len(rows)))
            if (i+1)%20==0 or i==0: print(f'Attention {i+1}/{len(rows)}',flush=True)
        write_json(status,dict(status='complete',completed=len(rows),expected=len(rows)))
    except BaseException as exc:
        write_json(status,dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.config,a.predictions,a.output,a.limit)
