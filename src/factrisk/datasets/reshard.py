"""Resume a stopped single-GPU lane as independent, fingerprint-checked shards.

The caller must stop the old worker first. Existing completed rows are archived
and reused, not recomputed; per-record sampling seeds remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path

from factrisk.core.io import read_jsonl, read_yaml, write_json, write_jsonl
from factrisk.core.contracts import validated_predictions
from factrisk.datasets.postprocess import wait_status


def paths(config):
    cfg = read_yaml(config)
    directory = Path(cfg['project']['output_dir']) / 'predictions'
    rows = read_jsonl(Path(cfg['project']['data_dir']) / 'manifest.jsonl')
    return cfg, directory, rows


def lane_stem(shard, shards):
    return 'qwen2_audio' if shards==1 else f'qwen2_audio.shard-{shard:05d}-of-{shards:05d}'


def prepare(config, shards, parent_shard=0, parent_shards=1):
    cfg, directory, all_rows = paths(config)
    rows=all_rows[parent_shard::parent_shards]
    parent=lane_stem(parent_shard,parent_shards)
    status_path = directory / f'{parent}.status.json'
    old_status = json.loads(status_path.read_text())
    old_pid = old_status.get('pid')
    if old_pid and Path(f'/proc/{old_pid}').exists():
        raise RuntimeError('Stop the old worker before repartitioning outputs')
    snapshot = directory / ('serial_prefix' if parent_shards==1 else parent+'_serial_prefix')
    if snapshot.exists():
        raise FileExistsError('Serial prefix already preserved; resume shards instead')
    completed = read_jsonl(directory / f'{parent}.jsonl')
    ids = {r['id'] for r in completed}
    subset = [r for r in rows if r['id'] in ids]
    validated_predictions(directory / f'{parent}.jsonl', subset, 'direct', cfg['models']['direct_st'])
    times = read_jsonl(directory / f'{parent}.timing.jsonl')
    if {r['id'] for r in times} != ids:
        raise ValueError('Interrupted row/timing pair; inspect the original outputs')
    write_jsonl(snapshot / 'predictions.jsonl', completed)
    write_jsonl(snapshot / 'timings.jsonl', times)
    write_json(snapshot / 'status.json', old_status)
    for shard in range(shards):
        child=parent_shard+shard*parent_shards
        selected = {r['id'] for r in all_rows[child::shards*parent_shards]}
        stem = lane_stem(child,shards*parent_shards)
        if (directory / f'{stem}.jsonl').exists():
            raise FileExistsError(stem)
        write_jsonl(directory / f'{stem}.jsonl', [r for r in completed if r['id'] in selected])
        write_jsonl(directory / f'{stem}.timing.jsonl', [r for r in times if r['id'] in selected])
    write_json(status_path, dict(status='resharded', completed=len(completed), expected=len(rows),
                                shards=shards, preserved_prefix=str(snapshot), pid=None))


def merge(config, shards, parent_shard=0, parent_shards=1):
    cfg, directory, all_rows = paths(config)
    rows=all_rows[parent_shard::parent_shards]
    parent=lane_stem(parent_shard,parent_shards)
    records, times = {}, {}
    for shard in range(shards):
        child=parent_shard+shard*parent_shards
        stem = lane_stem(child,shards*parent_shards)
        wait_status(directory / f'{stem}.status.json')
        index = validated_predictions(directory / f'{stem}.jsonl', all_rows[child::shards*parent_shards],
                                      'direct', cfg['models']['direct_st'])
        if records.keys() & index.keys():
            raise ValueError('Duplicate IDs across shards')
        records.update(index)
        timing = read_jsonl(directory / f'{stem}.timing.jsonl')
        if len(timing) != len(index) or {r['id'] for r in timing} != set(index):
            raise ValueError('Missing or duplicate timing records')
        times.update({r['id']: r for r in timing})
    write_jsonl(directory / f'{parent}.jsonl', [records[r['id']] for r in rows])
    write_jsonl(directory / f'{parent}.timing.jsonl', [times[r['id']] for r in rows])
    write_json(directory / f'{parent}.status.json', dict(status='complete', expected=len(rows),
               completed=len(records), independent_shards=shards, pid=os.getpid(),
               resumption='validated serial prefix plus disjoint independent GPU shards'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'merge'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--parent-shard',type=int,default=0)
    parser.add_argument('--parent-shards',type=int,default=1)
    args = parser.parse_args()
    if args.shards < 2:
        parser.error('Use at least two independent shards')
    if args.parent_shards<1 or not 0<=args.parent_shard<args.parent_shards:
        parser.error('Invalid parent lane')
    (prepare if args.stage == 'prepare' else merge)(args.config,args.shards,args.parent_shard,args.parent_shards)
