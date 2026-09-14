"""Explicit French reproduction stages; default invocation only prints a plan.

Inference is one local shard per invocation: set CUDA_VISIBLE_DEVICES externally.
Finish waits for all declared shards and shared evidence; it includes clean Qwen
NLL. Attention is an additional clean Qwen baseline. No stage trains a new head.
"""
import argparse
import shlex
import subprocess
import sys

from factrisk.core.io import read_yaml
from factrisk.pipeline.workflow import ROOT, OUT

DEFAULT_SHARDS = {'qwen': 3, 'seamless': 2}


def shard_count(backend, config, override=None):
    """CLI > optional execution.direct_shards > historical backend default."""
    count = override if override is not None else config.get('execution', {}).get(
        'direct_shards', DEFAULT_SHARDS[backend])
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('direct_shards must be a positive integer')
    return count


def commands(args):
    root = OUT/'fr_en'
    if args.group == 'stress':
        root = root/'stress'
    config = root/f'{args.backend}_config.yaml'
    count = shard_count(args.backend, read_yaml(config), args.shards)
    module = lambda name, *rest: [sys.executable, '-u', '-m', name, *map(str, rest)]
    stage = args.stage
    if stage in ('inference', 'direct'):
        if not 0 <= args.shard < count:
            raise ValueError(f'shard must be in [0, {count})')
        return [module('factrisk.backends.runner', '--config', config, '--role', 'direct',
                       '--shard', args.shard, '--shards', count)]
    if stage == 'evidence':
        if args.backend != 'qwen' or args.shard != 0 or args.shards not in (None, 1):
            raise ValueError('Evidence is shared: use qwen, shard 0, one shard')
        return [module('factrisk.backends.runner', '--config', config, '--role', 'evidence',
                       '--shard', 0, '--shards', 1)]
    if stage == 'finish':
        return [module('factrisk.french.french_finish', args.backend, '--group', args.group, '--shards', count)]
    if stage in ('attention', 'nll'):
        if args.backend != 'qwen' or args.group != 'clean':
            raise ValueError('Auxiliary baselines are defined only for clean Qwen')
        name = 'attention' if stage == 'attention' else 'likelihood'
        base = ['--config', config, '--destination', root/'qwen'/name]
        return [module('factrisk.french.french_'+name, 'extract', *base),
                module('factrisk.french.french_'+name, 'evaluate', *base,
                       '--features', root/'qwen/numeric_features.jsonl',
                       '--full-predictions', root/'qwen/numeric_clean/predictions.jsonl')]
    if stage == 'statistics':
        return [module('factrisk.eval.significance_summary', '--compute')]
    if stage == 'publication':
        return [['bash', str(ROOT/'scripts/build_paper.sh'), '--mode', args.mode]]
    return [module('factrisk.eval.readiness', '--mode', args.mode)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        'Order: inference (all shards) + shared evidence; finish (each backend); '
        'attention (clean Qwen); statistics; publication; check. Existing frozen '
        'configs/assets must exist. Default is dry-run; --execute runs the selected '
        'stage only. Counts overridden for inference must also be passed to finish.'))
    p.add_argument('stage', choices=['inference','direct','evidence','finish','attention','nll',
                                    'statistics','publication','check'])
    p.add_argument('--backend', choices=list(DEFAULT_SHARDS), default='qwen')
    p.add_argument('--group', choices=['clean','stress'], default='clean')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int)
    p.add_argument('--mode', choices=['technical','submission'], default='technical')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    try:
        plan = commands(args)
    except ValueError as exc:
        p.error(str(exc))
    for command in plan:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
