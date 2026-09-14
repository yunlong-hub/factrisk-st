"""Seal locally trained heads before scoring the external evaluation set."""
import json
from datetime import datetime,timezone
from pathlib import Path
from factrisk.core.io import read_jsonl,sha256_file,write_json


def seal(checkpoint_dir,manifest,output):
    directory=Path(checkpoint_dir)
    checkpoints={p.name:sha256_file(p) for p in sorted(directory.glob('*.joblib'))}
    if 'et_full.joblib' not in checkpoints: raise ValueError('Cannot freeze missing primary model')
    rows=read_jsonl(manifest)
    clustering='speaker_donor_connected_component' if any('cluster_id' in r for r in rows) else 'speaker_id'
    payload=dict(checkpoints=checkpoints,manifest_sha256=sha256_file(manifest),
        adaptation='none',primary='et_full',coverage_targets=[.8,.9,.95],
        calibration='synthetic_calibration_only',bootstrap_cluster=clustering,bootstrap_seed=20260906,
        methods_fixed_before_external_risk_evaluation=True)
    output=Path(output)
    if output.exists():
        previous=json.loads(output.read_text())
        if any(previous.get(k)!=v for k,v in payload.items()):
            raise ValueError('Frozen external protocol changed')
    else:
        write_json(output,dict(**payload,sealed_at=datetime.now(timezone.utc).isoformat()))
    return payload
