"""Join historical unchanged outputs and all repaired outputs for attention."""
from factrisk.core.io import read_jsonl,read_yaml,write_jsonl
from factrisk.pipeline.workflow import OUT,DATA,FORMAL
from factrisk.core.contracts import validated_predictions


def prepare():
    rows=read_jsonl(DATA/'manifest.jsonl')
    repaired=[r for r in rows if r.get('requires_fresh_prediction')]
    cfg=read_yaml(OUT/'configs/repaired_config.yaml')
    new=validated_predictions(OUT/'repaired/predictions/qwen2_audio.jsonl',repaired,'direct',cfg['models']['direct_st'])
    predictions={r['id']:r for r in read_jsonl(FORMAL/'predictions/qwen2_audio.jsonl')}
    predictions.update(new)
    write_jsonl(OUT/'corrected/predictions/qwen2_audio.jsonl',[predictions[r['id']] for r in rows])
    print(f'Joined {len(rows)} hypothesis records',flush=True)


if __name__=='__main__': prepare()
