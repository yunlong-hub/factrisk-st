"""Resume only unfinished experiment stages after their concrete dependencies."""
import argparse
import json
import os
from pathlib import Path
from factrisk.core.io import write_json
from factrisk.pipeline.workflow import ROOT,OUT,DATA
from factrisk.datasets.postprocess import wait_status,run_module
from factrisk.core.freeze import seal


def main(start_at='attention'):
    status=OUT/'logs/finish_experiments.status.json'
    def stage(name):
        print(name,flush=True);write_json(status,dict(status='running',stage=name,pid=os.getpid()))
    try:
        if start_at=='attention':
            stage('attention_training')
            wait_status(OUT/'corrected/attention.status.json')
            run_module('factrisk.backends.attention_train','--directory',OUT/'corrected','--external',OUT/'real_clean')
        stage('real_stress_scoring')
        wait_status(OUT/'logs/real_stress_direct.queue.json')
        wait_status(OUT/'logs/real_stress_evidence.queue.json')
        run_module('factrisk.method.qe','--config',OUT/'configs/real_stress_config.yaml',python=ROOT/'tools/comet-env/bin/python')
        seal(OUT/'corrected/number/checkpoints',DATA/'real_stress/manifest.jsonl',OUT/'real_stress/frozen_protocol.json')
        run_module('factrisk.pipeline.transfer','--config',OUT/'configs/real_stress_config.yaml',
            '--checkpoint-dir',OUT/'corrected/number/checkpoints','--destination',OUT/'real_stress/frozen_transfer')
        stage('seamless_wait')
        for cohort in ('controlled','real'):
            wait_status(OUT/f'seamless_{cohort}/predictions/qwen2_audio.status.json')
        wait_status(OUT/'seamless_controlled/evidence/whisper_nllb.status.json')
        for cohort in ('controlled','real'):
            stage('seamless_'+cohort+'_qe')
            run_module('factrisk.method.qe','--config',OUT/f'configs/seamless_{cohort}_config.yaml',python=ROOT/'tools/comet-env/bin/python')
        stage('seamless_train')
        from factrisk.pipeline.transfer import features,evaluate
        from factrisk.eval.evaluation import run
        features(OUT/'configs/seamless_controlled_config.yaml')
        run(OUT/'seamless_controlled')
        seal(OUT/'seamless_controlled/number/checkpoints',DATA/'seamless_real/manifest.jsonl',OUT/'seamless_real/frozen_protocol.json')
        features(OUT/'configs/seamless_real_config.yaml')
        evaluate(OUT/'seamless_real/features.jsonl',OUT/'seamless_controlled/number/checkpoints',OUT/'seamless_real/frozen_transfer')
        write_json(status,dict(status='complete',stage='experiment_metrics_complete',pid=os.getpid()))
    except BaseException as exc:
        write_json(status,dict(status='failed',error=repr(exc),pid=os.getpid()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start-at',choices=['attention','real_stress'],default='attention')
    main(parser.parse_args().start_at)
