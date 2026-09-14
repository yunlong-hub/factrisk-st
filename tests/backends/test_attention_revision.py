import numpy as np
import torch
from factrisk.backends.attention import summarize
from factrisk.backends.qwen import _request_entropy


def test_attention_uses_prediction_shift_and_causal_prefix():
    weights=torch.zeros(1,5,5)
    # prompt = [text, audio, text], hypothesis = [y1, y2].
    weights[0,2,:3]=torch.tensor([.2,.6,.2])
    weights[0,3,:4]=torch.tensor([.1,.4,.1,.4])
    values=summarize(weights,torch.tensor([1]),3,2)
    assert np.isclose(values[0,0],.75)  # ratio mean of 1 and .5
    assert values[1,0]==0  # one audio frame => undefined Pearson => documented zero
    assert values[2,0]==0
    assert np.isclose(values[3,0],np.log(2))


def test_decoder_entropy_handles_masked_logits():
    entropy=_request_entropy((torch.tensor([[0.,float('-inf'),0.]]),),0,1)
    assert np.isclose(entropy,np.log(2))
