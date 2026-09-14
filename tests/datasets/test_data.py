from factrisk.datasets.data import grouped_split, validate_speaker_splits


def _pair(index: int, speaker: str, fact_type: str) -> dict:
    return {
        "id": f"p{index}",
        "fact_type": fact_type,
        "meta": {"speaker_id": speaker, "tts_engine": f"e{index % 2}"},
    }


def test_grouped_split_is_deterministic_and_disjoint() -> None:
    pairs = [
        _pair(index, f"s{index % 12}", "number" if index % 2 else "negation")
        for index in range(48)
    ]
    config = {"train": 0.6, "calibration": 0.2, "test": 0.2}
    first = grouped_split(pairs, config, seed=9)
    second = grouped_split(pairs, config, seed=9)
    assert first == second
    rows = [
        {
            "split": first[pair["id"]],
            "speaker_id": pair["meta"]["speaker_id"],
            "pair_id": pair["id"],
        }
        for pair in pairs
    ]
    report = validate_speaker_splits(rows)
    assert all(not values for values in report["speaker_overlaps"].values())

