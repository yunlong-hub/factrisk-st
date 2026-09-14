"""Full SeamlessM4T-v2 ST, decoder confidence, probes and sampled outputs."""
from factrisk.backends.qwen import Qwen2AudioScorer,_transition_nll,_request_entropy,_beam_margin


class SeamlessScorer(Qwen2AudioScorer):
    def load(self):
        if self.model is not None:return
        import torch
        from transformers import AutoProcessor,SeamlessM4Tv2ForSpeechToText
        self.processor=AutoProcessor.from_pretrained(self.config['model_path'])
        self.model=SeamlessM4Tv2ForSpeechToText.from_pretrained(self.config['model_path'],
            dtype=getattr(torch,self.config.get('dtype','bfloat16')),device_map='auto').eval()

    def _generate(self,requests,*,num_beams,num_return_sequences,do_sample,temperature=None):
        import librosa
        import torch
        self.load();results=[]
        # Identical single-input policy across base, probes and sample groups.
        for request in requests:
            if request['target_language']!='en':raise ValueError('This experiment fixes English targets')
            sr=int(self.processor.feature_extractor.sampling_rate)
            audio,_=librosa.load(request['audio'],sr=sr,mono=True)
            inputs=self.processor(audio=[audio],sampling_rate=sr,return_tensors='pt',padding=True)
            device=next(self.model.parameters()).device;dtype=next(self.model.parameters()).dtype
            inputs={k:v.to(device,dtype=dtype) if k=='input_features' else v.to(device) for k,v in inputs.items()}
            kw=dict(tgt_lang='eng',max_new_tokens=self.config.get('max_new_tokens',192),
                num_beams=num_beams,num_return_sequences=num_return_sequences,do_sample=do_sample,
                return_dict_in_generate=True,output_scores=True,use_cache=False)
            if do_sample:kw['temperature']=temperature
            with torch.inference_mode():generated=self.model.generate(**inputs,**kw)
            text=[x.strip() for x in self.processor.batch_decode(generated.sequences,skip_special_tokens=True)]
            nll=_transition_nll(self.model,generated);confidence={}
            if not nll:raise RuntimeError('Seamless transition scores unavailable')
            confidence['sequence_nll']=float(nll[0])
            if not do_sample:
                confidence['token_entropy']=_request_entropy(generated.scores,0,num_beams)
                confidence['beam_margin']=_beam_margin(generated,0,num_return_sequences,nll)
            results.append(dict(translation=text[0],beam_translations=text,confidence=confidence))
        return results
