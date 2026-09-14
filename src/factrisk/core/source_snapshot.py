"""Pin the historical cohort; upstream Fact-ST manifests are mutable assets."""
from pathlib import Path
from factrisk.core.io import read_jsonl, read_yaml, sha256_file, write_json, write_jsonl, write_yaml


def pin(config, rows, historical_report):
    data=Path(config['project']['data_dir'])
    destination=data/'frozen_inputs/historical_manifest.jsonl'
    if destination.exists():
        if read_jsonl(destination)!=rows:
            raise ValueError('Frozen historical input manifest changed')
    else:
        write_jsonl(destination,rows)
    # Generic upstream prepare commands must fail closed rather than selecting
    # a different contemporary 500-pair pool. Revision prepare uses this snapshot.
    config['sources'] = dict(historical_manifest=str(destination),
        historical_manifest_sha256=sha256_file(destination),
        historical_preparation_report=str(historical_report),
        historical_preparation_report_sha256=sha256_file(historical_report),
        inference_source='saved formal outputs for unchanged waveforms; validated new outputs for repairs',
        preparation_entrypoint='python -m factrisk.pipeline.workflow prepare')
    return config


def main():
    from factrisk.pipeline.workflow import ROOT, OUT, DATA, migrated
    rows=migrated(read_jsonl(DATA/'baseline/manifest.jsonl'))
    cfg=read_yaml(OUT/'configs/resolved_config.yaml')
    prior=OUT/'frozen_inputs/prior_resolved_config.yaml'
    if not prior.exists():write_yaml(prior,cfg)
    new=pin(cfg,rows,DATA/'baseline/preparation_report.json')
    write_yaml(OUT/'configs/resolved_config.yaml',new)
    repaired=read_yaml(OUT/'configs/repaired_config.yaml')
    repaired['sources']=dict(new['sources'])
    write_yaml(OUT/'configs/repaired_config.yaml',repaired)
    upstream=Path(read_yaml(prior)['sources']['pair_manifest'])
    current=read_jsonl(upstream) if upstream.exists() else []
    old_ids={r['pair_id'] for r in rows};new_ids={r['id'] for r in current}
    write_json(OUT/'audit/upstream_manifest_drift.json',dict(
        status='historical_inputs_pinned_no_predictions_changed', historical_pairs=len(old_ids),
        upstream_manifest=str(upstream),upstream_sha256=sha256_file(upstream) if upstream.exists() else None,
        observed_current_upstream_pool=len(current) if upstream.exists() else None,
        historical_ids_absent_from_current_upstream=len(old_ids-new_ids) if upstream.exists() else None,
        current_upstream_used_for_inference=False,
        snapshot=str(DATA/'frozen_inputs/historical_manifest.jsonl'),
        reason='The shared upstream pair/alignment pools have changed since the historical experiment.'))


if __name__=='__main__':main()
