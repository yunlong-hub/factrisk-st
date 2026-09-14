"""Complete dependency-gated revision stages, preserving raw prediction files."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from factrisk.core.io import read_yaml,write_json
from factrisk.core.sharding import merge_inference_shards
from factrisk.pipeline.workflow import ROOT,OUT
from factrisk.core.freeze import seal


def wait_status(path,hours=24):
    deadline=time.monotonic()+hours*3600
    while time.monotonic()<deadline:
        if path.is_file():
            record=json.loads(path.read_text())
            if record['status']=='complete': return
            if record['status']=='failed': raise RuntimeError(f'Upstream failed: {path}')
        time.sleep(30)
    raise TimeoutError(f'Incomplete dependency: {path}')


def run_module(module,*args,python=None):
    command=[str(python or sys.executable),'-u','-m',module,*map(str,args)]
    print('RUN',command,flush=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    subprocess.run(command,check=True,env=env,timeout=24*3600)


def main():
    status=OUT/'logs/postprocess.status.json'
    def stage(name): write_json(status,dict(status='running',stage=name,pid=os.getpid()))
    try:
        stage('waiting_real_inference')
        wait_status(OUT/'real_clean/predictions/qwen2_audio.status.json')
        wait_status(OUT/'real_clean/evidence/whisper_nllb.status.json')
        stage('real_qe')
        run_module('factrisk.method.qe','--config',OUT/'configs/real_full_config.yaml',python=ROOT/'tools/comet-env/bin/python')
        stage('real_attention')
        run_module('factrisk.backends.attention','--config',OUT/'configs/real_full_config.yaml',
            '--predictions',OUT/'real_clean/predictions/qwen2_audio.jsonl','--output',OUT/'real_clean/attention.jsonl')
        stage('waiting_repaired_inference')
        for shard in range(2):
            wait_status(OUT/f'logs/repaired_direct_{shard}.queue.json')
        wait_status(OUT/'logs/repaired_evidence.queue.json')
        cfg=read_yaml(OUT/'configs/repaired_config.yaml')
        merge_inference_shards(cfg,kind='direct',num_shards=2)
        stage('repaired_qe')
        run_module('factrisk.method.qe','--config',OUT/'configs/repaired_config.yaml',python=ROOT/'tools/comet-env/bin/python')
        stage('corrected_features_and_training')
        run_module('factrisk.pipeline.workflow','features')
        run_module('factrisk.pipeline.workflow','train')
        stage('frozen_external_evaluation')
        manifest=ROOT/'data/derived/factrisk/real_full/manifest.jsonl'
        seal(OUT/'corrected/number/checkpoints',manifest,OUT/'real_clean/frozen_protocol.json')
        run_module('factrisk.pipeline.transfer','--config',OUT/'configs/real_full_config.yaml',
            '--checkpoint-dir',OUT/'corrected/number/checkpoints','--destination',OUT/'real_clean/frozen_transfer')
        write_json(status,dict(status='complete',stage='main_experiments_complete_attention_head_pending',pid=os.getpid()))
    except BaseException as exc:
        write_json(status,dict(status='failed',error=repr(exc),pid=os.getpid()));raise


if __name__=='__main__': main()
