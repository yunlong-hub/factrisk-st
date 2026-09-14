"""Dependency-gated completion for one French backend, on one reserved GPU."""
import argparse
import os
from factrisk.french.french_data import FR_OUT
from factrisk.pipeline.workflow import ROOT
from factrisk.core.io import read_yaml, write_json
from factrisk.datasets.postprocess import wait_status, run_module
from factrisk.core.sharding import merge_inference_shards
from factrisk.french.french_pipeline import shard_count
from factrisk.core.io import read_jsonl


def run(backend, group='clean', shards=None):
    root=FR_OUT if group=='clean' else FR_OUT/'stress'
    directory=root/backend;config=root/f'{backend}_config.yaml'
    status=directory/'finish.status.json'
    def stage(name):
        write_json(status,dict(status='running',stage=name,pid=os.getpid()))
        print(name,flush=True)
    try:
        stage('waiting_for_inference')
        cfg=read_yaml(config)
        shards=shard_count(backend,cfg,shards)
        for i in range(shards):
            suffix=f'.shard-{i:05d}-of-{shards:05d}' if shards>1 else ''
            wait_status(directory/f'predictions/qwen2_audio{suffix}.status.json')
        wait_status(root/'qwen/evidence/whisper_nllb.status.json')
        stage('merge_and_validate')
        if shards>1:
            merge_inference_shards(cfg,kind='direct',num_shards=shards)
        else:
            from pathlib import Path
            expected=[r['id'] for r in read_jsonl(Path(cfg['project']['data_dir'])/'manifest.jsonl')]
            actual=[r['id'] for r in read_jsonl(directory/'predictions/qwen2_audio.jsonl')]
            if len(actual)!=len(set(actual)) or set(actual)!=set(expected):
                raise RuntimeError('Single-shard predictions do not match manifest IDs')
        stage('comet_qe')
        run_module('factrisk.method.qe','--config',config,python=ROOT/'tools/comet-env/bin/python')
        stage('frozen_evaluation_and_quality')
        run_module('factrisk.french.french_evaluation',backend,'--group',group)
        if backend=='qwen' and group=='clean':
            stage('hypothesis_nll_extraction')
            run_module('factrisk.french.french_likelihood','extract','--config',config,'--destination',directory/'likelihood')
            stage('hypothesis_nll_evaluation')
            run_module('factrisk.french.french_likelihood','evaluate','--config',config,'--destination',directory/'likelihood',
                '--features',directory/'numeric_features.jsonl','--full-predictions',directory/'numeric_clean/predictions.jsonl')
        write_json(status,dict(status='complete',stage='all_metrics_ready',pid=os.getpid()))
    except BaseException as exc:
        write_json(status,dict(status='failed',error=repr(exc),pid=os.getpid()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['qwen','seamless'])
    p.add_argument('--group',choices=['clean','stress'],default='clean')
    p.add_argument('--shards',type=int,help='Must match inference; default from shared backend/config policy')
    a=p.parse_args();run(a.backend,a.group,a.shards)
