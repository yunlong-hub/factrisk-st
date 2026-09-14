import pytest

from factrisk.core.source_snapshot import pin
from factrisk.core.io import read_jsonl, sha256_file


def test_snapshot_is_immutable_and_removes_mutable_upstream(tmp_path):
    report = tmp_path / 'historical_report.json'
    report.write_text('{}')
    cfg = {'project': {'data_dir': str(tmp_path)},
           'sources': {'pair_manifest': '/mutable/upstream.jsonl'}}
    rows = [{'id': 'saved', 'pair_id': 'historical'}]
    result = pin(cfg, rows, report)
    path = tmp_path / 'frozen_inputs/historical_manifest.jsonl'
    assert read_jsonl(path) == rows
    assert 'pair_manifest' not in result['sources']
    assert result['sources']['historical_manifest_sha256'] == sha256_file(path)
    assert pin(result, rows, report) == result
    with pytest.raises(ValueError, match='Frozen historical input manifest changed'):
        pin(result, [{'id': 'replacement'}], report)
    assert read_jsonl(path) == rows
