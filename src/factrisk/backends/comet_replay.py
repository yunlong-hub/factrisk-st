"""Read-only small-batch COMET cache replay in the historical scoring environment."""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from factrisk.core.contracts import qe_fingerprint
from factrisk.core.io import read_jsonl, read_yaml, sha256_file, write_json
from factrisk.pipeline.workflow import ROOT, OUT


def run(config_path, count=16, tolerance=1e-5):
    destination=OUT/'robustness/comet'
    destination.mkdir(parents=True,exist_ok=True)
    report={'status':'running','config':str(config_path),'count':count,
            'absolute_tolerance':tolerance,'python':platform.python_version(),
            'executable':sys.executable,'cache_modified':False}
    started=time.monotonic()
    try:
        cfg=read_yaml(config_path)
        output=Path(cfg['project']['output_dir'])
        assets={'replay_code':Path(__file__),'config':Path(config_path),'predictions':output/'predictions/qwen2_audio.jsonl',
                'evidence':Path(cfg['sources']['revision_evidence_file']),
                'scores':output/'qe/comet_qe.jsonl',
                'checkpoint':Path(cfg['models']['optional_qe']['model_path']),
                'environment_manifest':ROOT/'configs/environments/comet.json',
                'environment_lock':ROOT/'configs/environments/comet.lock.txt'}
        before={name:sha256_file(path) for name,path in assets.items()}
        report['assets']={name:{'path':str(path),'sha256':before[name]} for name,path in assets.items()}
        declared=json.loads(assets['environment_manifest'].read_text())
        versions={name:importlib.metadata.version(name) for name in declared['packages']}
        report['versions']=versions
        report['version_matches_export']=versions==declared['packages']
        report['declared_dependency_conflicts']=declared['dependency_conflicts']
        scores=read_jsonl(assets['scores'])[:count]
        if len(scores)!=count:
            raise ValueError(f'Expected {count} cached rows, found {len(scores)}')
        predictions={row['id']:row for row in read_jsonl(assets['predictions'])}
        evidence={row['id']:row for row in read_jsonl(assets['evidence'])}
        samples=[]
        for row in scores:
            key=row['id']
            if row['input_fingerprint']!=qe_fingerprint(predictions[key],evidence[key]):
                raise ValueError(f'Cached input fingerprint differs: {key}')
            if row['model_sha256']!=before['checkpoint']:
                raise ValueError(f'Cached model fingerprint differs: {key}')
            samples.append({'src':evidence[key]['asr_transcript'],'mt':predictions[key]['translation']})
        write_json(destination/'inputs.json',samples)
        report['replay_input_sha256']=sha256_file(destination/'inputs.json')
        import torch
        from comet import load_from_checkpoint
        report['gpu']=torch.cuda.get_device_name(0)
        report['torch_cuda']=torch.version.cuda
        print(f'Loading historical COMET checkpoint for {count}-row replay',flush=True)
        model=load_from_checkpoint(str(assets['checkpoint']),local_files_only=True)
        values=model.predict(samples,batch_size=16,gpus=1,accelerator='gpu',num_workers=0,progress_bar=False).scores
        torch.cuda.synchronize()
        comparisons=[{'id':old['id'],'cached':old['score'],'replayed':float(new),
                      'absolute_error':abs(float(new)-old['score'])}
                     for old,new in zip(scores,values,strict=True)]
        report['comparisons']=comparisons
        report['maximum_absolute_error']=max(row['absolute_error'] for row in comparisons)
        report['mean_absolute_error']=sum(row['absolute_error'] for row in comparisons)/count
        report['assets_unchanged']=all(sha256_file(path)==before[name] for name,path in assets.items())
        report['status']='passed' if report['assets_unchanged'] and report['maximum_absolute_error']<=tolerance else 'mismatch'
        report['scope']='Historical-environment small-batch numerical replay, not a clean-install dependency compatibility guarantee.'
    except Exception as exc:
        report['status']='failed'
        report['error']=repr(exc)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        write_json(destination/'report.json',report)
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(OUT/'fr_en/qwen_config.yaml'))
    parser.add_argument('--count',type=int,default=16)
    args=parser.parse_args()
    if not 1<=args.count<=32:
        parser.error('--count must be 1..32; this tool is deliberately bounded')
    run(Path(args.config),args.count)
