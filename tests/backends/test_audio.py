from pathlib import Path

import numpy as np
import soundfile as sf

from factrisk.backends.audio import add_noise, signal_features, truncate_at_span


def test_noise_is_deterministic_and_hits_snr() -> None:
    waveform = np.sin(np.linspace(0, 20, 16000)).astype(np.float32)
    first = add_noise(waveform, 10.0, np.random.default_rng(7))
    second = add_noise(waveform, 10.0, np.random.default_rng(7))
    assert np.allclose(first, second)
    noise = first - waveform
    snr = 20.0 * np.log10(np.sqrt(np.mean(waveform**2)) / np.sqrt(np.mean(noise**2)))
    assert abs(float(snr) - 10.0) < 0.05


def test_truncate_and_signal_features(tmp_path: Path) -> None:
    waveform = np.ones(16000, dtype=np.float32) * 0.1
    truncated = truncate_at_span(waveform, 16000, [0.5, 0.8], 0.5)
    assert len(truncated) == int(0.65 * 16000)
    path = tmp_path / "audio.wav"
    sf.write(path, waveform, 16000)
    features = signal_features(path)
    assert features["audio_duration"] == 1.0
    assert features["audio_zero_fraction"] == 0.0

