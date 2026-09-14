import torch

from factrisk.backends.whisper import _batch_entropy


def test_batch_entropy_returns_finite_positive_values() -> None:
    scores = [torch.tensor([[2.0, 0.0, float("-inf")], [0.0, 2.0, float("-inf")]])]
    values = _batch_entropy(scores, batch_size=2)
    assert len(values) == 2
    assert all(0.0 < value < 1.0 for value in values)
