import json

import pytest

from factrisk.eval import robustness_gate as gate
from factrisk.core.io import sha256_file


def test_hash_gate_rejects_modified_or_missing_input(tmp_path):
    path = tmp_path/'input'
    path.write_text('frozen')
    inputs = {'input':sha256_file(path)}
    gate.verify_hashes(tmp_path, inputs)
    path.write_text('changed')
    with pytest.raises(RuntimeError, match='Changed robustness input'):
        gate.verify_hashes(tmp_path, inputs)
    path.unlink()
    with pytest.raises(RuntimeError):
        gate.verify_hashes(tmp_path, inputs)


def test_empty_provenance_and_nonfinite_measured_values_fail(tmp_path):
    with pytest.raises(RuntimeError):
        gate.verify_hashes(tmp_path, {})
    with pytest.raises(RuntimeError):
        gate.same(float('nan'), .1, 'numerical evidence')
    with pytest.raises(RuntimeError):
        gate.same({'complete':True}, {'complete':True,'n':27}, 'missing evidence')


def test_seed_complete_flag_cannot_replace_design(tmp_path):
    folder = tmp_path/'seeds'
    folder.mkdir()
    (folder/'protocol.json').write_text(json.dumps({'design':{'seeds':[], 'methods':[], 'cohorts':[]}}))
    (folder/'status.json').write_text('{"stage":"complete"}')
    with pytest.raises(RuntimeError, match='Seed design incomplete'):
        gate.require_seeds(tmp_path, tmp_path)


def test_all_four_evidence_groups_are_required(monkeypatch, tmp_path):
    visited = []
    for name in ('seeds','empty','environment','comet'):
        monkeypatch.setattr(gate, 'require_'+name, lambda r,b,n=name:visited.append(n))
    gate.require_robustness(tmp_path)
    assert visited == ['seeds','empty','environment','comet']


def test_comet_complete_flag_without_assets_fails(tmp_path):
    folder = tmp_path/'comet'
    folder.mkdir()
    (folder/'report.json').write_text('{"status":"passed","assets":{}}')
    with pytest.raises(RuntimeError, match='asset provenance incomplete'):
        gate.require_comet(tmp_path, tmp_path)
