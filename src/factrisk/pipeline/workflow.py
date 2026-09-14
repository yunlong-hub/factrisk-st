"""ICASSP experiment stages with explicit historical/provisional provenance.

Run with ``factrisk-run audit|prepare|features|train``.
Baseline inputs remain read-only under the canonical experiment directory.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

from factrisk.core.io import read_jsonl, read_yaml, sha256_file, stable_int, write_json, write_jsonl, write_yaml
from factrisk.core.labels_v2 import VERSION, label_fact
from factrisk.core.text import fact_signature

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'exp/factrisk'
DATA = ROOT / 'data/derived/factrisk'
FORMAL = OUT / 'baseline'
SOURCE = Path('/workspace/dataset/st/fact-st/de-en/qwen36-35b-a3b_qwen35-35b-a3b')
PROJECTS = ('factrisk-st', 'bind-st', 'fact-st', 'fovea-st', 'latent-st')
SEED = 20260729


def migrated(value):
    if isinstance(value, dict):
        return {k: migrated(v) for k, v in value.items()}
    if isinstance(value, list):
        return [migrated(v) for v in value]
    if isinstance(value, str):
        for name in PROJECTS:
            old = f'/workspace/yunlong/LLM/{name}'
            if value == old or value.startswith(old + '/'):
                return value.replace(old, f'/workspace/yunlong/ST/{name}', 1)
        if value.startswith('dataset/'):
            return '/workspace/' + value
        # This dataset alias is resolved only if its concrete replacement exists.
        if value.startswith('/workspace/dataset/st/fact-st-old/'):
            candidate = value.replace('/fact-st-old/', '/fact-st/', 1)
            if Path(candidate).exists():
                return candidate
    return value


def unique(rows):
    indexed = {r['id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate IDs')
    return indexed


def donor_selections(rows, same_split):
    clean = [r for r in rows if r['condition'] == 'clean']
    result = {}
    for row in clean:
        valid = [d for d in clean if d['speaker_id'] != row['speaker_id']
                 and (not same_split or d['split'] == row['split'])]
        if not valid:
            raise ValueError('No eligible donor')
        i = stable_int(f"{row['pair_id']}:{row['side']}:donor", SEED) % len(valid)
        result[(row['pair_id'], row['side'])] = valid[i]
    return result


def audit():
    from factrisk.backends.audio import local_overlap, read_audio
    import soundfile as sf

    snapshot = DATA / 'frozen_inputs/historical_manifest.jsonl'
    rows = migrated(read_jsonl(snapshot if snapshot.exists() else DATA / 'baseline/manifest.jsonl'))
    old_features = unique(read_jsonl(FORMAL / 'features.jsonl'))
    old_donors = donor_selections(rows, False)
    donors = []
    for row in rows:
        if row['condition'] != 'overlap':
            continue
        d = old_donors[(row['pair_id'], row['side'])]
        item = dict(id=row['id'], split=row['split'], donor_id=d['id'],
                    donor_split=d['split'], crosses_split=d['split'] != row['split'])
        try:
            x, sr = read_audio(row['clean_audio'])
            y, yr = read_audio(d['clean_audio'])
            expected = local_overlap(x, sr, y, yr, row['aligned_fact_span'], 0.)
            buffer = io.BytesIO()
            sf.write(buffer, np.clip(expected, -1, 1), sr, format='WAV', subtype='PCM_16')
            import hashlib
            item['reconstructed_sha256'] = hashlib.sha256(buffer.getvalue()).hexdigest()
            item['existing_sha256'] = sha256_file(row['audio'])
            item['waveform_match'] = item['existing_sha256'] == item['reconstructed_sha256']
        except (OSError, RuntimeError) as exc:
            item['waveform_match'] = None
            item['error'] = str(exc)
        donors.append(item)
    changes = []
    for row in rows:
        old = old_features[row['id']]
        new = label_fact(old['translation'], row['expected_slot'], row['contrast_slot'], row['fact_type'], row['reference'])
        changes.append(dict(id=row['id'], split=row['split'], fact_type=row['fact_type'],
                            condition=row['condition'], previous_error=old['severe_fact_error'], **new))
    write_jsonl(OUT / 'audit/donor_reconstruction.jsonl', donors)
    write_jsonl(OUT / 'audit/label_changes.jsonl', changes)
    report = dict(protocol='revision_20260906', status='automatic_audit_not_human_verified',
                  n=len(rows), historical_manifest_sha256=sha256_file(DATA / 'baseline/manifest.jsonl'),
                  historical_features_sha256=sha256_file(FORMAL / 'features.jsonl'),
                  donor_cross_split=sum(d['crosses_split'] for d in donors),
                  train_with_test_donor=sum(d['split']=='train' and d['donor_split']=='test' for d in donors),
                  donor_waveforms_matching=sum(d['waveform_match'] is True for d in donors),
                  donor_waveforms_unverified=sum(d['waveform_match'] is None for d in donors),
                  label_changed=sum(r['severe_fact_error'] is not None and r['severe_fact_error'] != r['previous_error'] for r in changes),
                  label_changed_by_type=dict(Counter(r['fact_type'] for r in changes if r['severe_fact_error'] is not None and r['severe_fact_error'] != r['previous_error'])),
                  label_status=dict(Counter(r['label_status'] for r in changes)))
    # New-real candidate audit is metadata-only and never reads model outputs.
    real_path = SOURCE / 'research/paired_conditions/manifests/real_tts.jsonl'
    real = migrated(read_jsonl(real_path))
    speakers = {r['speaker_id'] for r in rows}
    texts = {r['source_text'].strip().lower() for r in rows}
    eligible = [p for p in real if p.get('meta', {}).get('speaker_id') not in speakers
                and p['original']['text'].strip().lower() not in texts]
    eligible = list({p['original']['audio']: p for p in eligible}.values())
    write_jsonl(OUT / 'audit/real_candidates.jsonl', eligible)
    report['real_candidates'] = dict(raw_pairs=len(real), speaker_and_text_disjoint_recordings=len(eligible),
                                   speakers=len({p['meta']['speaker_id'] for p in eligible}),
                                   historical_use_unknown=True)
    write_json(OUT / 'audit/summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    write_json(OUT / 'audit/human_status.json', dict(status='not_planned_by_user',
               replacement='public_existing_annotations_and_explicit_automatic_metrics',
               new_human_annotations=0))


def prepare():
    from factrisk.backends.audio import materialize_transform, read_audio, write_audio
    from factrisk.datasets.data import validate_speaker_splits
    snapshot = DATA / 'frozen_inputs/historical_manifest.jsonl'
    rows = migrated(read_jsonl(snapshot if snapshot.exists() else DATA / 'baseline/manifest.jsonl'))
    donors = donor_selections(rows, True)
    config = migrated(read_yaml(ROOT / 'configs/paper.yaml'))
    config['project'].update(name='factrisk-st-revision', protocol='revised_historical_diagnostic',
                             output_dir=str(OUT), data_dir=str(DATA))
    from factrisk.core.source_snapshot import pin
    config = pin(config, rows, DATA / 'baseline/preparation_report.json')
    config['models']['direct_st']['batch_size'] = 1
    config['models']['evidence']['batch_size'] = 8
    repaired = []
    # Missing historical mask assets cannot be reconstructed byte-for-byte.
    # Regenerate both paired mask conditions, declare their construction and
    # require fresh predictions. No revised mask inherits an old prediction.
    for row in rows:
        changed = row['condition'] in {'overlap', 'local_mask', 'irrelevant_mask'}
        key = f"{stable_int(row['id'], SEED):016x}"
        if row['condition'] == 'overlap':
            donor = donors[(row['pair_id'], row['side'])]
            key = f"{stable_int(row['id'], SEED):016x}"
            destination = DATA / f'audio/overlap/{key}.wav'
            row['audio'] = str(materialize_transform(row['clean_audio'], destination,
                               dict(kind='local_overlap', snr_db=0.), np.random.default_rng(stable_int(row['id'], SEED)),
                               span=row['aligned_fact_span'], donor_path=donor['clean_audio']))
            row['overlap_donor'] = dict(id=donor['id'], speaker_id=donor['speaker_id'], split=donor['split'],
                                       audio=donor['clean_audio'], sha256=sha256_file(donor['clean_audio']))
        elif row['condition'] in {'local_mask', 'irrelevant_mask'}:
            waveform, sr = read_audio(row['clean_audio'])
            start, end = [round(t * sr) for t in row['aligned_fact_span']]
            start, end = max(0, start), min(len(waveform), end)
            length = end - start
            if length <= 0:
                raise ValueError(f"Invalid mask span: {row['id']}")
            if row['condition'] == 'irrelevant_mask':
                if start >= length:
                    start, end = 0, length
                elif len(waveform) - end >= length:
                    start, end = len(waveform) - length, len(waveform)
                else:
                    raise ValueError(f"No equal-length disjoint mask: {row['id']}")
            waveform[start:end] = 0
            destination = DATA / f"audio/{row['condition']}/{key}.wav"
            write_audio(destination, waveform, sr)
            row['audio'] = str(destination)
            row['revised_mask_span'] = [start/sr, end/sr]
            row['mask_construction'] = 'zero_samples_equal_duration_disjoint_v2'
        if changed:
            row['requires_fresh_prediction'] = True
            row['probes'] = []
            for probe in config['data']['probes']:
                path = DATA / f"audio/probes/{probe['name']}/{key}.wav"
                materialize_transform(row['audio'], path, probe,
                                      np.random.default_rng(stable_int(f"{row['id']}:{probe['name']}", SEED)))
                row['probes'].append(dict(name=probe['name'], audio=str(path)))
            repaired.append(row)
        row['audio_sha256'] = sha256_file(row['audio'])
        for probe in row['probes']:
            probe['audio_sha256'] = sha256_file(probe['audio'])
    write_jsonl(DATA / 'manifest.jsonl', rows)
    write_jsonl(DATA / 'repaired/manifest.jsonl', repaired)
    write_json(DATA / 'split_report.json', validate_speaker_splits(rows))
    write_yaml(OUT / 'configs/resolved_config.yaml', config)
    overlap_config = copy.deepcopy(config)
    overlap_config['project']['data_dir'] = str(DATA / 'repaired')
    overlap_config['project']['output_dir'] = str(OUT / 'repaired')
    write_yaml(OUT / 'configs/repaired_config.yaml', overlap_config)
    write_json(OUT / 'metrics/preparation.json', dict(rows=len(rows), rebuilt_by_condition=dict(Counter(r['condition'] for r in repaired)),
               manifest_sha256=sha256_file(DATA / 'manifest.jsonl'),
               donor_cross_split=sum(r['overlap_donor']['split'] != r['split'] for r in repaired if 'overlap_donor' in r)))
    print(f'Prepared {len(rows)} inputs, {len(repaired)} require fresh prediction', flush=True)


def deployment_features(translation, samples, evidence):
    features = {}
    for kind in ('number', 'negation'):
        signature = fact_signature(translation, kind)
        signatures = [signature] + [fact_signature(x, kind) for x in samples]
        features[f'sample_{kind}_disagreement'] = (len(set(signatures))-1)/max(len(signatures)-1, 1) if samples else float('nan')
        features[f'evidence_{kind}_mismatch'] = float(signature != fact_signature(evidence, kind)) if evidence else float('nan')
    return features


def features(*, interim=False):
    from factrisk.method.features import _feature_row
    from factrisk.core.contracts import validated_predictions, qe_fingerprint
    manifest = migrated(read_jsonl((DATA / 'baseline' if interim else DATA) / 'manifest.jsonl'))
    predictions = unique(read_jsonl(FORMAL / 'predictions/qwen2_audio.jsonl'))
    evidence = unique(read_jsonl(FORMAL / 'evidence/whisper_nllb.jsonl'))
    old = unique(read_jsonl(FORMAL / 'features.jsonl'))
    qe = {}
    if not interim:
        repaired_rows = [r for r in manifest if r.get('requires_fresh_prediction')]
        cfg = read_yaml(OUT / 'configs/repaired_config.yaml')
        # Fail closed until every repaired input has a new prediction.
        for category, filename, destination in (
            ('predictions', 'qwen2_audio.jsonl', predictions),
            ('evidence', 'whisper_nllb.jsonl', evidence),
        ):
            role = 'direct' if category == 'predictions' else 'evidence'
            model_key = 'direct_st' if role == 'direct' else 'evidence'
            updated = validated_predictions(OUT / 'repaired' / category / filename,
                repaired_rows, role, cfg['models'][model_key])
            destination.update(updated)
        qe = unique(read_jsonl(OUT / 'repaired/qe/comet_qe.jsonl'))
        if set(qe) != {r['id'] for r in repaired_rows}:
            raise ValueError('Incomplete corrected QE scores')
    result, excluded = [], []
    for row in manifest:
        if interim and row['condition']=='overlap':
            continue
        pred, ev = predictions[row['id']], evidence[row['id']]
        item = copy.deepcopy(old[row['id']])
        if row.get('requires_fresh_prediction'):
            value = qe[row['id']]
            if value['input_fingerprint'] != qe_fingerprint(pred, ev):
                raise ValueError(f"Stale QE score: {row['id']}")
            item = _feature_row(row, pred, ev, value['score'])
        new_label = label_fact(pred['translation'], row['expected_slot'], row['contrast_slot'], row['fact_type'], row['reference'])
        item.update(new_label)
        item['translation'] = pred['translation']
        item['evidence_translation'] = ev['cascade_translation']
        item['features'].update(deployment_features(pred['translation'], pred.get('sample_translations', []), ev['cascade_translation']))
        item['features'].pop('sample_fact_disagreement', None)
        item['features'].pop('evidence_fact_mismatch', None)
        item['unsupported'] = new_label['contrast_value_present']
        if new_label['severe_fact_error'] is None:
            excluded.append(item)
        else:
            result.append(item)
    destination = OUT / ('interim_no_overlap' if interim else 'corrected')
    write_jsonl(destination / 'features.jsonl', result)
    write_jsonl(destination / 'labels_unresolved.jsonl', excluded)
    write_json(destination / 'label_coverage.json', dict(included=len(result), excluded=len(excluded),
               included_by_status=dict(Counter(r['label_status'] for r in result)),
               excluded_by_split=dict(Counter(r['split'] for r in excluded)),
               status='provisional_automatic_labels_not_human_verified',
               selection_warning='Unresolved numeric contexts excluded; polarity proxies require review.'))
    print(f'Features: {len(result)} included, {len(excluded)} unresolved; provisional only', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'prepare', 'features', 'train'])
    parser.add_argument('--interim', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'audit':
        audit()
    elif args.stage == 'prepare':
        prepare()
    elif args.stage == 'features':
        features(interim=args.interim)
    elif args.stage == 'train':
        from factrisk.eval.evaluation import run
        run(OUT / ('interim_no_overlap' if args.interim else 'corrected'))


if __name__ == '__main__':
    main()
