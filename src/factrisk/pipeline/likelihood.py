"""Padding-free hypothesis likelihood baseline; never supplies gold text to ST.

Legacy beam transition summaries average the returned score tensor, whose
width can reflect other beam/request lengths. This independent baseline scores
only each printed hypothesis and EOS, without changing the frozen fusion head.
"""
import hashlib
import inspect
import json
import time
from pathlib import Path

import joblib
import numpy as np
from factrisk.core.io import read_jsonl, read_yaml, sha256_file, append_jsonl, write_json, write_jsonl
from factrisk.backends.qwen import LANGUAGE_NAMES
from factrisk.pipeline.workflow import OUT
from factrisk.eval.evaluation import calibrate, apply_calibration, describe, cluster_comparison
from factrisk.core.models import build_model
from factrisk.datasets.postprocess import wait_status


def extract(config, predictions, output, *, selected_ids=None):
    import torch
    import librosa
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    cfg=read_yaml(config);mc=cfg['models']['direct_st'];output=Path(output)
    rows=[r for r in read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl') if r['fact_type']=='number']
    if selected_ids is not None:
        selected_ids=set(selected_ids)
        rows=[r for r in rows if r['id'] in selected_ids]
        if {r['id'] for r in rows} != selected_ids:
            raise ValueError('Likelihood selection contains missing/non-number inputs')
    hypotheses={r['id']:r['translation'] for r in read_jsonl(predictions)}
    done={r['id']:r for r in read_jsonl(output)} if output.exists() else {}
    processor=AutoProcessor.from_pretrained(mc['model_path'])
    model=Qwen2AudioForConditionalGeneration.from_pretrained(mc['model_path'],dtype=torch.bfloat16,device_map='auto').eval()
    device=next(model.parameters()).device;sr=processor.feature_extractor.sampling_rate
    for i,row in enumerate(rows):
        hyp=hypotheses[row['id']]
        fp=hashlib.sha256(json.dumps([sha256_file(row['audio']),hyp,mc['model_path'],'hypothesis_eos_v1']).encode()).hexdigest()
        if row['id'] in done:
            if done[row['id']]['input_fingerprint']!=fp:raise ValueError('Stale hypothesis likelihood')
            continue
        source=LANGUAGE_NAMES[row['source_language']];target=LANGUAGE_NAMES[row['target_language']]
        conversation=[dict(role='user',content=[dict(type='audio',audio_url=row['audio']),
            dict(type='text',text=f'Translate the speech from {source} to {target}. Output only the translation.')])]
        prompt=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
        audio,_=librosa.load(row['audio'],sr=sr,mono=True)
        key='audio' if 'audio' in inspect.signature(processor.__call__).parameters else 'audios'
        inputs=processor(text=[prompt],**{key:[audio]},sampling_rate=sr,return_tensors='pt')
        inputs={k:v.to(device) for k,v in inputs.items()}
        p=inputs['input_ids'].shape[1]
        ids=processor.tokenizer(hyp,add_special_tokens=False)['input_ids']+[processor.tokenizer.eos_token_id]
        target_ids=torch.tensor([ids],device=device)
        inputs['input_ids']=torch.cat([inputs['input_ids'],target_ids],dim=1)
        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
        torch.cuda.synchronize();started=time.monotonic()
        with torch.inference_mode():
            logits=model(**inputs,use_cache=False).logits[:,p-1:p+len(ids)-1].float()
            nll=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target_ids.reshape(-1)).item()
        torch.cuda.synchronize()
        append_jsonl(output,[dict(id=row['id'],hypothesis_nll=nll,tokens_with_eos=len(ids),
            input_fingerprint=fp,seconds=time.monotonic()-started,label_source='none')])
        write_json(output.with_suffix('.status.json'),dict(status='running',completed=i+1,expected=len(rows)))
        if i%50==0:print(f'Likelihood {output.name}: {i+1}/{len(rows)}',flush=True)
    write_json(output.with_suffix('.status.json'),dict(status='complete',completed=len(rows),expected=len(rows)))
    del model
    torch.cuda.empty_cache()


def fit():
    directory=OUT/'likelihood'
    values={r['id']:r['hypothesis_nll'] for r in read_jsonl(directory/'controlled.jsonl')}
    rows=read_jsonl(OUT/'corrected/features.jsonl')+read_jsonl(OUT/'corrected/labels_unresolved.jsonl')
    rows=[r for r in rows if r['fact_type']=='number']
    train=[r for r in rows if r['split']=='train' and r['severe_fact_error'] is not None]
    cal=[r for r in rows if r['split']=='calibration' and r['severe_fact_error'] is not None]
    X=lambda rr, vv:np.array([[vv[r['id']]] for r in rr])
    model=build_model('lr_sequence_probability')
    model.fit(X(train,values),np.array([r['severe_fact_error'] for r in train]))
    raw_cal=model.predict_proba(X(cal,values))[:,1]
    ab=calibrate(raw_cal,np.array([r['severe_fact_error'] for r in cal]))
    threshold=float(np.quantile(apply_calibration(raw_cal,ab),.9))
    checkpoint=directory/'model.joblib'
    joblib.dump(dict(model=model,calibration=ab,threshold=threshold),checkpoint)
    # No test labels have been used to fit/calibrate/select this baseline.
    metrics={}
    for name,rr,vv,pred_file in [
        ('controlled',[r for r in rows if r['split']=='test'],values,OUT/'corrected/number/predictions.jsonl'),
        ('natural_clean',read_jsonl(OUT/'real_clean/features.jsonl'),
            {r['id']:r['hypothesis_nll'] for r in read_jsonl(directory/'natural_clean.jsonl')},
            OUT/'real_clean/frozen_transfer/predictions.jsonl')]:
        raw=model.predict_proba(X(rr,vv))[:,1];prob=apply_calibration(raw,ab)
        m=describe(rr,raw,prob,threshold)
        full={r['id']:r['risks']['et_full'] for r in read_jsonl(pred_file)}
        ta=joblib.load(OUT/'corrected/number/checkpoints/et_full.joblib')['threshold']
        m['full_minus_likelihood']=cluster_comparison(rr,np.array([full[r['id']]['raw'] for r in rr]),raw,
            ta,threshold,np.array([full[r['id']]['calibrated'] for r in rr]),prob)
        metrics[name]=m
        write_jsonl(directory/f'{name}_predictions.jsonl',[dict(id=r['id'],raw=float(a),calibrated=float(b)) for r,a,b in zip(rr,raw,prob)])
    write_json(directory/'metrics.json',dict(method='hypothesis_nll_lr',checkpoint_sha256=sha256_file(checkpoint),
        protocol='Single-hypothesis teacher-forcing including EOS; train/cal only; no tuning.',metrics=metrics))


def main():
    status=OUT/'likelihood/status.json'
    try:
        write_json(status,dict(status='waiting',dependency='seamless_controlled_evidence_GPU2'))
        wait_status(OUT/'seamless_controlled/evidence/whisper_nllb.status.json')
        write_json(status,dict(status='running',stage='hypothesis_likelihood_extraction'))
        extract(OUT/'configs/resolved_config.yaml',OUT/'corrected/predictions/qwen2_audio.jsonl',OUT/'likelihood/controlled.jsonl')
        extract(OUT/'configs/real_full_config.yaml',OUT/'real_clean/predictions/qwen2_audio.jsonl',OUT/'likelihood/natural_clean.jsonl')
        fit();write_json(status,dict(status='complete'))
    except BaseException as exc:
        write_json(status,dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':main()
