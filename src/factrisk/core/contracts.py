"""Artifact joins fail closed on missing, duplicate or stale inference rows."""
import hashlib
import json
from factrisk.core.io import read_jsonl
from factrisk.backends.runner import fingerprint


def validated_predictions(path, rows, role, model_config):
    records = read_jsonl(path)
    index = {r['id']: r for r in records}
    if len(index) != len(records) or set(index) != {r['id'] for r in rows}:
        raise ValueError(f'Incomplete or duplicate {role} outputs: {path}')
    expected = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest()
    for row in rows:
        pred = index[row['id']]
        if pred.get('input_fingerprint') != fingerprint(row, role):
            raise ValueError(f"Changed audio/probes for {row['id']}")
        if pred.get('model_config_sha256') != expected:
            raise ValueError(f"Changed inference configuration for {row['id']}")
    return index


def qe_fingerprint(prediction, evidence):
    payload = [prediction['translation'], evidence['asr_transcript']]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
