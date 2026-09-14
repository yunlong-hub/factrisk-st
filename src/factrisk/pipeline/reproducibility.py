"""Local-only provenance for exact model assets and source code (no upload)."""
import argparse
from importlib.metadata import version
import platform
from pathlib import Path

from factrisk.core.io import read_yaml, sha256_file, write_json
from factrisk.pipeline.workflow import OUT, ROOT


def assets():
    cfg=read_yaml(OUT/'configs/resolved_config.yaml')['models']
    paths=[cfg['direct_st']['model_path'],cfg['evidence']['whisper_path'],cfg['evidence']['nllb_path'],
           cfg['optional_qe']['model_path'],read_yaml(OUT/'configs/seamless_controlled_config.yaml')['models']['direct_st']['model_path']]
    result={}
    for name in paths:
        root=Path(name)
        if root.is_file():files=[root]
        else:
            weights=sorted(root.glob('*.safetensors')) or sorted(root.glob('pytorch_model*.bin'))
            files=sorted(set(weights+list(root.glob('*.json'))+list(root.glob('*.model'))+list(root.glob('*.txt'))))
        if not files:raise FileNotFoundError(name)
        for p in files:
            print('Fingerprint',p,flush=True)
            result[str(p)]=dict(bytes=p.stat().st_size,sha256=sha256_file(p))
        write_json(OUT/'reproducibility/assets.partial.json',result)
    libraries={name:version(name) for name in ['torch','transformers','numpy','scipy','scikit-learn','librosa','sacrebleu']}
    write_json(OUT/'reproducibility/assets.json',dict(assets=result,python=platform.python_version(),
        libraries=libraries,scope='Local inference assets only; no credentials or remote upload.'))


def source():
    files=[]
    for folder,pattern in [('src/factrisk','*.py'),('tests','*.py'),('scripts','*.sh')]:
        files.extend((ROOT/folder).rglob(pattern))
    files += [ROOT/'pyproject.toml']
    files += list(OUT.glob('configs/*config.yaml'))
    files += list((ROOT/'configs').glob('*.yaml'))
    files += list((ROOT/'configs/environments').glob('*'))
    robustness = OUT/'robustness'
    if robustness.exists():
        files += list(robustness.glob('*.md'))
        for pattern in ('*.json', '*.csv', '*.md'):
            files += list(robustness.glob('*/'+pattern))
    if (OUT/'fr_en').exists():
        files += list((OUT/'fr_en').rglob('*config.yaml'))
        files += list((OUT/'fr_en').rglob('frozen_protocol.json'))
        for pattern in ('*audit.json','protocol.json','selection.json','*quality.json',
                        'metrics.json','publication_provenance.json','RESULTS.md','results.csv'):
            files += list((OUT/'fr_en').rglob(pattern))
    for folder in ('real_clean','real_stress','seamless_real'):
        files += list((OUT/folder).glob('frozen_protocol.json'))
    paper=ROOT/'papers/factriskst'
    for pattern in ('*.tex','*.md','*.bib','*.bst','*.sty','tables/*.tex',
                    'figures/*.pdf','figures/*.svg',
                    'metadata/*.md','metadata/*.json'):
        files += list(paper.glob(pattern))
    files += [ROOT/'docs/experiment_protocol.md',
              ROOT/'docs/runbook.md',
              ROOT/'docs/automatic_evaluation.md',
              OUT/'README.md',OUT/'RESULTS.md']
    files=sorted(set(files));destination=OUT/'reproducibility'
    destination.mkdir(exist_ok=True)
    write_json(destination/'source_hashes.json',{str(p.relative_to(ROOT)):sha256_file(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['assets','source'])
    (assets if parser.parse_args().stage=='assets' else source)()
