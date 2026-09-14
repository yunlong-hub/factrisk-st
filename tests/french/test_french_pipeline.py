from argparse import Namespace

import pytest

from factrisk.french import french_pipeline as pipeline
from factrisk.eval import readiness


def args(**updates):
    values=dict(stage='inference',backend='qwen',group='clean',shard=0,shards=None,mode='technical')
    values.update(updates)
    return Namespace(**values)


def test_shared_shard_resolution():
    assert pipeline.shard_count('qwen',{})==3
    assert pipeline.shard_count('seamless',{})==2
    cfg={'execution':{'direct_shards':4}}
    assert pipeline.shard_count('qwen',cfg)==4
    assert pipeline.shard_count('qwen',cfg,1)==1
    for value in (0,-1,True,1.5,'3'):
        with pytest.raises(ValueError):
            pipeline.shard_count('qwen',{},value)


def test_inference_finish_match_and_stress_path(monkeypatch):
    monkeypatch.setattr(pipeline,'read_yaml',lambda _: {})
    for backend,count in [('qwen','3'),('seamless','2')]:
        inference=pipeline.commands(args(backend=backend,group='stress'))[0]
        finish=pipeline.commands(args(stage='finish',backend=backend,group='stress'))[0]
        assert inference[-1]==finish[-1]==count
        assert '/fr_en/stress/' in inference[inference.index('--config')+1]
    with pytest.raises(ValueError):
        pipeline.commands(args(shard=3))


def test_evidence_shared_and_auxiliary_scope(monkeypatch):
    monkeypatch.setattr(pipeline,'read_yaml',lambda _: {})
    assert pipeline.commands(args(stage='evidence'))[0][-1]=='1'
    with pytest.raises(ValueError):
        pipeline.commands(args(stage='evidence',backend='seamless'))
    with pytest.raises(ValueError):
        pipeline.commands(args(stage='attention',group='stress'))
    for stage in ('attention','nll'):
        commands=pipeline.commands(args(stage=stage))
        assert len(commands)==2
        assert 'extract' in commands[0] and 'evaluate' in commands[1]


def test_plan_does_not_execute(monkeypatch,capsys):
    monkeypatch.setattr(pipeline,'read_yaml',lambda _: {})
    def forbidden(*args,**kwargs):
        raise AssertionError('dry-run launched a process')
    monkeypatch.setattr(pipeline.subprocess,'run',forbidden)
    pipeline.main(['inference'])
    assert '--shards 3' in capsys.readouterr().out


def test_cli_readiness_exit_code(monkeypatch):
    modes=[]
    def fake_check(mode):
        modes.append(mode)
        return {'passed':mode=='technical'}
    monkeypatch.setattr(readiness,'check',fake_check)
    assert readiness.main([])==0
    assert readiness.main(['--mode','submission'])==1
    assert modes==['technical','submission']


def test_only_metadata_is_excluded_from_technical_gate():
    checks={'author_information_supplied':False,'experiments_complete':False,
            'no_overfull_boxes':False,'fonts_embedded':True}
    assert readiness.gate_failures(checks)==['experiments_complete','no_overfull_boxes']
    assert readiness.gate_failures(checks,'submission')==[
        'author_information_supplied','experiments_complete','no_overfull_boxes']
    assert readiness.gate_failures({'author_information_supplied':False})==[]
    with pytest.raises(ValueError):
        readiness.gate_failures(checks,'unsupported')


def test_single_shard_finish_validates_ids(monkeypatch,tmp_path):
    from factrisk.french import french_finish as finish
    monkeypatch.setattr(finish,'FR_OUT',tmp_path)
    monkeypatch.setattr(finish,'read_yaml',lambda _: {'project':{'data_dir':str(tmp_path)}})
    monkeypatch.setattr(finish,'read_jsonl',lambda _: [{'id':'sample'}])
    statuses=[]
    monkeypatch.setattr(finish,'wait_status',lambda path: statuses.append(str(path)))
    monkeypatch.setattr(finish,'run_module',lambda *a,**k: None)
    monkeypatch.setattr(finish,'write_json',lambda *a: None)
    def no_merge(*a,**k):
        raise AssertionError('Single shard must not use multi-shard merge')
    monkeypatch.setattr(finish,'merge_inference_shards',no_merge)
    finish.run('seamless','stress',1)
    assert statuses[0].endswith('predictions/qwen2_audio.status.json')
    assert statuses[1].endswith('qwen/evidence/whisper_nllb.status.json')
    monkeypatch.setattr(finish,'read_jsonl',lambda path: [{'id':'sample'}] if path.name=='manifest.jsonl' else [])
    with pytest.raises(RuntimeError,match='manifest IDs'):
        finish.run('seamless','stress',1)
