import pytest

from factrisk.eval import readiness
from factrisk.core.io import write_json, write_jsonl, sha256_file


def test_numeric_gate_uses_manifest_and_rejects_duplicate_or_missing_outputs(tmp_path):
    manifest=tmp_path/'manifest.jsonl'
    write_jsonl(manifest,[{'id':'a','numeric_cohort':True},{'id':'b','numeric_cohort':False}])
    feature={'id':'a','severe_fact_error':0}
    prediction={'id':'a','label':0,'risks':{'et_full':{'raw':0.1}}}
    write_jsonl(tmp_path/'numeric_features.jsonl',[feature])
    write_jsonl(tmp_path/'numeric_stress/predictions.jsonl',[prediction])
    metrics={'methods':{'et_full':{'n':1}}}
    assert readiness.require_french_numeric(tmp_path,manifest,'stress',metrics)=={'a'}
    write_jsonl(tmp_path/'numeric_stress/predictions.jsonl',[prediction,prediction])
    with pytest.raises(RuntimeError,match='Duplicate'):
        readiness.require_french_numeric(tmp_path,manifest,'stress',metrics)
    write_jsonl(tmp_path/'numeric_stress/predictions.jsonl',[])
    with pytest.raises(RuntimeError,match='IDs differ'):
        readiness.require_french_numeric(tmp_path,manifest,'stress',metrics)


def test_numeric_gate_rejects_stale_label_and_count(tmp_path):
    manifest=tmp_path/'manifest.jsonl'
    write_jsonl(manifest,[{'id':'a','numeric_cohort':True}])
    write_jsonl(tmp_path/'numeric_features.jsonl',[{'id':'a','severe_fact_error':0}])
    path=tmp_path/'numeric_clean/predictions.jsonl'
    write_jsonl(path,[{'id':'a','label':1,'risks':{'et_full':{}}}])
    with pytest.raises(RuntimeError,match='labels differ'):
        readiness.require_french_numeric(tmp_path,manifest,'clean',{'methods':{'et_full':{'n':1}}})
    write_jsonl(path,[{'id':'a','label':0,'risks':{'et_full':{}}}])
    with pytest.raises(RuntimeError,match='count or risks'):
        readiness.require_french_numeric(tmp_path,manifest,'clean',{'methods':{'et_full':{'n':2}}})


def test_reference_audit_bound_to_clean_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr(readiness,'DATA',tmp_path/'data')
    monkeypatch.setattr(readiness,'OUT',tmp_path/'exp')
    manifest=readiness.DATA/'fr_en/clean/manifest.jsonl'
    write_jsonl(manifest,[{'id':'a','numeric_cohort':True}])
    audit={'status':'passed','numeric_rows':1,'manifest_sha256':sha256_file(manifest),
           'passed':dict.fromkeys(['reference_identity','digit_equivalence','word_equivalence',
                                   'replacement_negative','deletion_negative'],1)}
    path=readiness.OUT/'fr_en/reference_property_audit.json'
    write_json(path,audit)
    readiness.require_french_audit()
    write_jsonl(manifest,[{'id':'b','numeric_cohort':True}])
    with pytest.raises(RuntimeError,match='does not match'):
        readiness.require_french_audit()


def test_quality_gate_checks_cohort_and_frozen_assets(tmp_path):
    manifest=tmp_path/'manifest.jsonl'
    rows=[{'id':'a','numeric_cohort':True,'general_cohort':False},
          {'id':'b','numeric_cohort':False,'general_cohort':True}]
    write_jsonl(manifest,rows)
    write_jsonl(tmp_path/'features.jsonl',rows)
    frozen={'manifest_sha256':sha256_file(manifest),'checkpoints':{'et_full.joblib':'fixed'}}
    for cohort,identifier in [('numeric','a'),('general','b')]:
        metric={'n':1,'selected_ids':[identifier],'source_feature_sha256':sha256_file(tmp_path/'features.jsonl'),
                **frozen,'methods':{'et_full':{'retained':1},'et_asr_evidence':{'retained':1}}}
        write_json(tmp_path/f'{cohort}_quality.json',metric)
    readiness.require_french_quality(tmp_path,manifest,'clean',frozen)
    metric['selected_ids']=['a']
    write_json(tmp_path/'general_quality.json',metric)
    with pytest.raises(RuntimeError,match='selected IDs'):
        readiness.require_french_quality(tmp_path,manifest,'clean',frozen)
    metric['selected_ids']=['b'];metric['checkpoints']={}
    write_json(tmp_path/'general_quality.json',metric)
    with pytest.raises(RuntimeError,match='assets changed'):
        readiness.require_french_quality(tmp_path,manifest,'clean',frozen)
