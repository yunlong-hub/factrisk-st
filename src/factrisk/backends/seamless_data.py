"""Separate full second-translator manifests, preserving shared risk protocol."""
import copy
from factrisk.core.io import read_jsonl,read_yaml,write_jsonl,write_yaml
from factrisk.pipeline.workflow import ROOT,OUT,DATA


def prepare():
    for name,source in (('controlled',DATA/'manifest.jsonl'),('real',DATA/'real_full/manifest.jsonl')):
        rows=[r for r in read_jsonl(source) if r['fact_type']=='number']
        cfg=read_yaml(OUT/('configs/resolved_config.yaml' if name=='controlled' else 'real_full_config.yaml'))
        directory=DATA/f'seamless_{name}'
        write_jsonl(directory/'manifest.jsonl',rows)
        extension=copy.deepcopy(cfg['models']['extensions']['seamless_m4t'])
        extension.update(backend='seamless_m4t',num_beams=2,num_samples=3,sample_temperature=.8,batch_size=1)
        cfg['models']['direct_st']=extension
        if name=='real':
            cfg['sources']['revision_evidence_file']=str(OUT/'real_clean/evidence/whisper_nllb.jsonl')
        cfg['project'].update(data_dir=str(directory),output_dir=str(OUT/f'seamless_{name}'),
            protocol='second_translator_full_signals_'+name)
        write_yaml(OUT/f'configs/seamless_{name}_config.yaml',cfg)
    print('Seamless full-signal number manifests prepared',flush=True)


if __name__=='__main__':prepare()
