"""Export installed dependency closures without inheriting unrelated packages."""
import argparse
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOTS = {
    'evaluation': ['numpy','scipy','scikit-learn','soundfile','PyYAML','sacrebleu',
                   'matplotlib','pymupdf','tensorboard','pytest','setuptools','wheel'],
    'inference': ['torch','transformers','accelerate','librosa','sentencepiece',
                  'soundfile','PyYAML','safetensors'],
    'comet': ['unbabel-comet'],
}


def installed_closure(roots):
    pending=[(name, set()) for name in roots]
    active={}; versions={}; conflicts=[]
    while pending:
        name,extras=pending.pop()
        name=canonicalize_name(name)
        if name in active and extras<=active[name]:
            continue
        extras=extras|active.get(name,set());active[name]=extras
        dist=metadata.distribution(name);versions[name]=dist.version
        for text in dist.requires or []:
            req=Requirement(text)
            if req.marker and not any(req.marker.evaluate({'extra':extra}) for extra in {''}|extras):
                continue
            actual=metadata.version(req.name)
            if req.specifier and not req.specifier.contains(actual, prereleases=True):
                conflicts.append(dict(parent=name,requirement=text,installed=actual))
            pending.append((req.name,set(req.extras)))
    return dict(sorted(versions.items())),conflicts


def export(kind,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    versions,conflicts=installed_closure(ROOTS[kind])
    header=f'# Installed closure: {kind}; Python {platform.python_version()}; {platform.system()} {platform.machine()}\n'
    (destination/f'{kind}.lock.txt').write_text(header+'\n'.join(f'{n}=={v}' for n,v in versions.items())+'\n')
    report=dict(kind=kind,python=platform.python_version(),system=platform.system(),
        architecture=platform.machine(),roots=ROOTS[kind],packages=versions,
        dependency_conflicts=conflicts,installation_test='not_performed_by_export')
    (destination/f'{kind}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{kind}: {len(versions)} packages; {len(conflicts)} installed dependency conflicts')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=ROOTS)
    parser.add_argument('--destination',default='configs/environments')
    args=parser.parse_args();export(args.kind,args.destination)
