from types import SimpleNamespace

import pytest

from factrisk.french.french_likelihood import execute, inputs
from factrisk.core.io import write_json, write_jsonl, write_yaml


def fixture_inputs(tmp_path):
    config=tmp_path/'config.yaml'
    write_yaml(config,dict(project=dict(data_dir=str(tmp_path)),
        models=dict(direct_st=dict(backend='qwen2_audio'))))
    rows=[dict(id='fr1',numeric_cohort=True,source_language='fr',target_language='en',
        split='test',fact_type='number'),dict(id='general',numeric_cohort=False)]
    write_jsonl(tmp_path/'manifest.jsonl',rows)
    checkpoint=tmp_path/'checkpoint.json'
    write_json(checkpoint,dict(dummy='hash-only test fixture'))
    return config,checkpoint,rows


def test_freeze_uses_only_numeric_manifest_and_rejects_changes(tmp_path):
    config,checkpoint,rows=fixture_inputs(tmp_path)
    args=SimpleNamespace(config=config,checkpoint=checkpoint,full_checkpoint=checkpoint,
        destination=tmp_path/'output',stage='freeze')
    execute(args)
    _,_,selected,identity=inputs(config,checkpoint,checkpoint)
    assert identity['selected_ids']==['fr1'] and len(selected)==1
    rows[0]['id']='fr2'
    write_jsonl(tmp_path/'manifest.jsonl',rows)
    with pytest.raises(ValueError,match='different likelihood freeze'):
        execute(args)


def test_rejects_target_training_rows(tmp_path):
    config,checkpoint,rows=fixture_inputs(tmp_path)
    rows[0]['split']='train'
    write_jsonl(tmp_path/'manifest.jsonl',rows)
    with pytest.raises(ValueError,match='numerical test cohort'):
        inputs(config,checkpoint,checkpoint)


def test_rejects_duplicate_manifest_ids(tmp_path):
    config,checkpoint,rows=fixture_inputs(tmp_path)
    write_jsonl(tmp_path/'manifest.jsonl',rows+[rows[0]])
    with pytest.raises(ValueError,match='Duplicate French'):
        inputs(config,checkpoint,checkpoint)
