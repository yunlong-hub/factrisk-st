from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from factrisk.backends.audio import materialize_transform
from factrisk.core.io import (
    read_jsonl,
    resolve_existing_path,
    sha256_file,
    stable_int,
    write_json,
    write_jsonl,
)


def prepare_factst_data(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    project = config["project"]
    sources = config["sources"]
    data_config = config["data"]
    seed = int(project["seed"])
    data_dir = Path(project["data_dir"])
    audio_dir = data_dir / "audio"
    roots = sources.get("audio_roots", [])
    rewrites = sources.get("audio_path_rewrites", {})

    pair_path = Path(sources["pair_manifest"])
    intervention_path = Path(sources["interventions"])
    pairs = [
        row
        for row in read_jsonl(pair_path)
        if str(row.get("fact_type", "")).lower() in set(data_config["fact_types"])
    ]
    interventions = {str(row["id"]): row for row in read_jsonl(intervention_path)}
    unaligned_pair_ids = [
        str(row["id"]) for row in pairs if str(row["id"]) not in interventions
    ]
    if data_config.get("only_forced_aligned", False):
        pairs = [row for row in pairs if str(row["id"]) in interventions]
    limit = data_config.get("limit_pairs")
    if limit is not None:
        pairs = pairs[: int(limit)]
    if not pairs:
        raise ValueError("No Fact-ST pairs remain after fact-type filtering")
    missing_interventions = [str(row["id"]) for row in pairs if str(row["id"]) not in interventions]
    if missing_interventions:
        raise ValueError(f"Missing interventions for {len(missing_interventions)} pairs")

    pair_splits = grouped_split(pairs, data_config["split"], seed=seed)
    donors = _donor_table(pairs, roots, rewrites, seed, pair_splits=pair_splits)
    rows: list[dict[str, Any]] = []
    condition_counts: Counter[str] = Counter()
    for pair in pairs:
        pair_id = str(pair["id"])
        intervention = interventions[pair_id]["interventions"]
        for side in ("original", "counterfactual"):
            side_data = pair[side]
            other_side = "counterfactual" if side == "original" else "original"
            clean_audio = resolve_existing_path(side_data["audio"], roots, rewrites)
            span = intervention[f"{side}_key_mask"].get("span")
            if not span:
                raise ValueError(f"Missing aligned key span: {pair_id}/{side}")
            for condition in data_config["conditions"]:
                name = str(condition["name"])
                kind = str(condition["kind"])
                row_id = f"{pair_id}::{side}::{name}"
                if kind == "identity":
                    audio_path = clean_audio
                elif kind in {"factst_key_mask", "factst_irrelevant_mask"}:
                    suffix = "key_mask" if kind == "factst_key_mask" else "irrelevant_mask"
                    audio_path = resolve_existing_path(
                        intervention[f"{side}_{suffix}"]["audio"], roots, rewrites
                    )
                else:
                    audio_path = audio_dir / name / f"{_safe_id(pair_id)}__{side}.wav"
                    if not dry_run:
                        rng = np.random.default_rng(stable_int(row_id, seed))
                        donor_path = donors[(pair_id, side)] if kind == "local_overlap" else None
                        audio_path = materialize_transform(
                            clean_audio,
                            audio_path,
                            condition,
                            rng,
                            span=span,
                            donor_path=donor_path,
                        )
                probes = []
                for probe in data_config.get("probes", []):
                    probe_name = str(probe["name"])
                    probe_path = (
                        audio_dir
                        / "probes"
                        / name
                        / probe_name
                        / f"{_safe_id(pair_id)}__{side}.wav"
                    )
                    if not dry_run:
                        rng = np.random.default_rng(stable_int(f"{row_id}:{probe_name}", seed))
                        probe_path = materialize_transform(
                            audio_path,
                            probe_path,
                            probe,
                            rng,
                            span=span,
                        )
                    probes.append({"name": probe_name, "audio": str(probe_path)})
                rows.append(
                    {
                        "id": row_id,
                        "pair_id": pair_id,
                        "side": side,
                        "condition": name,
                        "condition_kind": kind,
                        "evidence_insufficient": bool(
                            condition.get("evidence_insufficient", False)
                        ),
                        "split": pair_splits[pair_id],
                        "speaker_id": str(pair.get("meta", {}).get("speaker_id", "unknown")),
                        "fact_type": str(pair["fact_type"]).lower(),
                        "tts_engine": str(pair.get("meta", {}).get("tts_engine", "unknown")),
                        "source_language": str(pair.get("src_lang", data_config["source_language"])),
                        "target_language": str(pair.get("tgt_lang", data_config["target_language"])),
                        "audio": str(audio_path),
                        "clean_audio": str(clean_audio),
                        "source_text": str(side_data.get("text", "")),
                        "reference": str(side_data.get("reference", "")),
                        "expected_slot": side_data.get(
                            "slot_translation", side_data.get("slot", "")
                        ),
                        "contrast_slot": pair[other_side].get(
                            "slot_translation", pair[other_side].get("slot", "")
                        ),
                        "aligned_fact_span": [float(span[0]), float(span[1])],
                        "probes": probes,
                        "label_only_fields": [
                            "source_text",
                            "reference",
                            "expected_slot",
                            "contrast_slot",
                            "condition",
                            "evidence_insufficient",
                            "aligned_fact_span",
                        ],
                    }
                )
                condition_counts[name] += 1

    split_report = validate_speaker_splits(rows)
    report = {
        "protocol": project["protocol"],
        "seed": seed,
        "dry_run": dry_run,
        "source": {
            "pair_manifest": str(pair_path),
            "pair_manifest_sha256": sha256_file(pair_path),
            "interventions": str(intervention_path),
            "interventions_sha256": sha256_file(intervention_path),
        },
        "num_pairs": len(pairs),
        "num_unaligned_pairs_excluded": (
            len(unaligned_pair_ids) if data_config.get("only_forced_aligned", False) else 0
        ),
        "num_utterances": len(rows),
        "condition_counts": dict(sorted(condition_counts.items())),
        "fact_type_counts": dict(sorted(Counter(row["fact_type"] for row in rows).items())),
        "split_report": split_report,
    }
    if not dry_run:
        write_jsonl(data_dir / "manifest.jsonl", rows)
        write_json(data_dir / "split_report.json", split_report)
        write_json(data_dir / "preparation_report.json", report)
    return report


def grouped_split(
    pairs: list[dict[str, Any]],
    split_config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, str]:
    fractions = {
        "train": float(split_config["train"]),
        "calibration": float(split_config["calibration"]),
        "test": float(split_config["test"]),
    }
    if abs(sum(fractions.values()) - 1.0) > 1e-8 or any(value <= 0 for value in fractions.values()):
        raise ValueError("Split fractions must be positive and sum to one")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        speaker = str(pair.get("meta", {}).get("speaker_id", "unknown"))
        groups[speaker].append(pair)
    if len(groups) < 3:
        raise ValueError("At least three speaker groups are required")
    group_sizes = {speaker: len(group) for speaker, group in groups.items()}
    group_strata = {
        speaker: Counter(
            (
                str(row["fact_type"]),
                str(row.get("meta", {}).get("tts_engine", "unknown")),
            )
            for row in group
        )
        for speaker, group in groups.items()
    }
    stratum_totals = Counter(
        key for counter in group_strata.values() for key, count in counter.items() for _ in range(count)
    )
    target_size = {split: len(pairs) * fraction for split, fraction in fractions.items()}
    target_strata = {
        split: {key: total * fractions[split] for key, total in stratum_totals.items()}
        for split in fractions
    }
    split_names = list(fractions)
    exact_group_targets = _integer_targets(len(groups), fractions)
    rng = np.random.default_rng(seed)
    speaker_names = np.asarray(sorted(groups), dtype=object)
    best_assignment: dict[str, str] | None = None
    best_cost = float("inf")
    # Exact speaker-count allocation avoids a nominal 20% partition collapsing
    # to only the three largest Common Voice speakers.
    for iteration in range(20000):
        permutation = (
            speaker_names
            if iteration == 0
            else speaker_names[rng.permutation(len(speaker_names))]
        )
        candidate: dict[str, str] = {}
        cursor = 0
        for split in split_names:
            next_cursor = cursor + exact_group_targets[split]
            candidate.update(
                {str(speaker): split for speaker in permutation[cursor:next_cursor]}
            )
            cursor = next_cursor
        cost = _split_cost(
            candidate,
            split_names,
            group_sizes,
            group_strata,
            target_size,
            target_strata,
        )
        if cost < best_cost:
            best_cost = cost
            best_assignment = candidate
    if best_assignment is None:
        raise RuntimeError("Failed to construct grouped split")
    return {
        str(pair["id"]): best_assignment[
            str(pair.get("meta", {}).get("speaker_id", "unknown"))
        ]
        for pair in pairs
    }


def validate_speaker_splits(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    speakers: dict[str, set[str]] = defaultdict(set)
    pair_ids: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        split = str(row["split"])
        speakers[split].add(str(row["speaker_id"]))
        pair_ids[split].add(str(row["pair_id"]))
        counts[split] += 1
    overlaps: dict[str, list[str]] = {}
    split_names = sorted(speakers)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = speakers[left] & speakers[right]
            overlaps[f"{left}__{right}"] = sorted(overlap)
            if overlap:
                raise ValueError(f"Speaker leakage between {left} and {right}: {len(overlap)}")
    return {
        "utterance_counts": dict(sorted(counts.items())),
        "pair_counts": {split: len(values) for split, values in sorted(pair_ids.items())},
        "speaker_counts": {split: len(values) for split, values in sorted(speakers.items())},
        "speaker_overlaps": overlaps,
    }


def _integer_targets(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {split: total * fraction for split, fraction in fractions.items()}
    targets = {split: int(value) for split, value in raw.items()}
    remaining = total - sum(targets.values())
    order = sorted(fractions, key=lambda split: (raw[split] - targets[split], split), reverse=True)
    for split in order[:remaining]:
        targets[split] += 1
    return targets


def _split_cost(
    assignment: dict[str, str],
    split_names: list[str],
    group_sizes: dict[str, int],
    group_strata: dict[str, Counter[tuple[str, str]]],
    target_size: dict[str, float],
    target_strata: dict[str, dict[tuple[str, str], float]],
) -> float:
    sizes = Counter()
    strata = {split: Counter() for split in split_names}
    for speaker, split in assignment.items():
        sizes[split] += group_sizes[speaker]
        strata[split].update(group_strata[speaker])
    cost = 0.0
    for split in split_names:
        cost += 4.0 * (
            (sizes[split] - target_size[split]) / max(target_size[split], 1.0)
        ) ** 2
        for key, target in target_strata[split].items():
            cost += 0.1 * (
                (strata[split][key] - target) / max(target, 1.0)
            ) ** 2
    return cost


def _donor_table(
    pairs: list[dict[str, Any]],
    roots: list[str | Path],
    rewrites: dict[str, str],
    seed: int,
    *,
    pair_splits: dict[str, str],
) -> dict[tuple[str, str], Path]:
    candidates: list[tuple[str, str, str, Path]] = []
    for pair in pairs:
        pair_id = str(pair["id"])
        speaker = str(pair.get("meta", {}).get("speaker_id", "unknown"))
        for side in ("original", "counterfactual"):
            candidates.append(
                (
                    pair_id,
                    side,
                    speaker,
                    resolve_existing_path(pair[side]["audio"], roots, rewrites),
                )
            )
    table: dict[tuple[str, str], Path] = {}
    for pair in pairs:
        pair_id = str(pair["id"])
        speaker = str(pair.get("meta", {}).get("speaker_id", "unknown"))
        for side in ("original", "counterfactual"):
            valid = [item for item in candidates if item[2] != speaker
                     and pair_splits[item[0]] == pair_splits[pair_id]]
            if not valid:
                raise ValueError("Cannot construct overlap: no different-speaker donor")
            index = stable_int(f"{pair_id}:{side}:donor", seed) % len(valid)
            table[(pair_id, side)] = valid[index][3]
    return table


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
