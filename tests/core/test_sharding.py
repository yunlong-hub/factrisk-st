from pathlib import Path

import pytest

from factrisk.core.io import read_jsonl, write_jsonl
from factrisk.core.sharding import merge_inference_shards


def _config(root: Path) -> dict:
    return {
        "project": {
            "protocol": "formal",
            "data_dir": str(root / "data"),
            "output_dir": str(root / "exp"),
        }
    }


def test_merge_shards_preserves_manifest_order(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = [{"id": f"row-{index}"} for index in range(5)]
    write_jsonl(tmp_path / "data" / "manifest.jsonl", manifest)
    write_jsonl(
        tmp_path / "exp" / "predictions" / "qwen2_audio.shard-00000-of-00002.jsonl",
        [{"id": "row-0"}, {"id": "row-2"}, {"id": "row-4"}],
    )
    write_jsonl(
        tmp_path / "exp" / "predictions" / "qwen2_audio.shard-00001-of-00002.jsonl",
        [{"id": "row-1"}, {"id": "row-3"}],
    )
    report = merge_inference_shards(config, kind="direct", num_shards=2)
    assert report["num_predictions"] == 5
    assert [row["id"] for row in read_jsonl(report["output"])] == [
        row["id"] for row in manifest
    ]


def test_merge_shards_rejects_incomplete_shard(tmp_path: Path) -> None:
    config = _config(tmp_path)
    write_jsonl(tmp_path / "data" / "manifest.jsonl", [{"id": "a"}, {"id": "b"}])
    write_jsonl(
        tmp_path / "exp" / "evidence" / "whisper_nllb.shard-00000-of-00002.jsonl",
        [{"id": "a"}],
    )
    write_jsonl(
        tmp_path / "exp" / "evidence" / "whisper_nllb.shard-00001-of-00002.jsonl",
        [],
    )
    with pytest.raises(ValueError, match="Incomplete shard"):
        merge_inference_shards(config, kind="evidence", num_shards=2)
