from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def write_audio(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    sf.write(str(destination), clipped, sample_rate, subtype="PCM_16")


def add_noise(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    if signal_rms <= 1e-8:
        return waveform.copy()
    noise = rng.normal(0.0, 1.0, size=len(waveform)).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    return waveform + noise * (target_noise_rms / max(noise_rms, 1e-8))


def apply_gain(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    return waveform * (10.0 ** (gain_db / 20.0))


def truncate_at_span(
    waveform: np.ndarray,
    sample_rate: int,
    span: list[float] | tuple[float, float],
    retain_fraction: float,
) -> np.ndarray:
    start, end = float(span[0]), float(span[1])
    cut_seconds = start + max(0.0, min(1.0, retain_fraction)) * max(end - start, 0.0)
    cut_sample = max(1, min(len(waveform), round(cut_seconds * sample_rate)))
    return waveform[:cut_sample].copy()


def local_overlap(
    waveform: np.ndarray,
    sample_rate: int,
    donor: np.ndarray,
    donor_sample_rate: int,
    span: list[float] | tuple[float, float],
    snr_db: float,
) -> np.ndarray:
    if donor_sample_rate != sample_rate:
        divisor = math.gcd(donor_sample_rate, sample_rate)
        donor = resample_poly(donor, sample_rate // divisor, donor_sample_rate // divisor)
    start = max(0, min(len(waveform), round(float(span[0]) * sample_rate)))
    end = max(start + 1, min(len(waveform), round(float(span[1]) * sample_rate)))
    length = end - start
    if len(donor) < length:
        repeats = math.ceil(length / max(len(donor), 1))
        donor = np.tile(donor, repeats)
    donor_segment = donor[:length].astype(np.float32)
    target_segment = waveform[start:end]
    target_rms = float(np.sqrt(np.mean(np.square(target_segment), dtype=np.float64)))
    donor_rms = float(np.sqrt(np.mean(np.square(donor_segment), dtype=np.float64)))
    desired_rms = target_rms / (10.0 ** (snr_db / 20.0))
    scaled = donor_segment * (desired_rms / max(donor_rms, 1e-8))
    output = waveform.copy()
    output[start:end] += scaled
    return output


def signal_features(path: str | Path) -> dict[str, float]:
    waveform, sample_rate = read_audio(path)
    if not len(waveform):
        return {
            "audio_duration": 0.0,
            "audio_rms_db": -120.0,
            "audio_zero_fraction": 1.0,
            "audio_spectral_entropy": 0.0,
            "audio_clipping_fraction": 0.0,
        }
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    spectrum = np.abs(np.fft.rfft(waveform, n=min(max(512, 1 << (len(waveform) - 1).bit_length()), 32768)))
    power = np.square(spectrum, dtype=np.float64)
    probabilities = power / max(float(power.sum()), 1e-12)
    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    entropy /= math.log(max(len(probabilities), 2))
    return {
        "audio_duration": len(waveform) / sample_rate,
        "audio_rms_db": 20.0 * math.log10(max(rms, 1e-6)),
        "audio_zero_fraction": float(np.mean(np.abs(waveform) < 1e-5)),
        "audio_spectral_entropy": entropy,
        "audio_clipping_fraction": float(np.mean(np.abs(waveform) >= 0.999)),
    }


def materialize_transform(
    source: str | Path,
    destination: str | Path,
    spec: dict[str, Any],
    rng: np.random.Generator,
    *,
    span: list[float] | None = None,
    donor_path: str | Path | None = None,
) -> Path:
    output = Path(destination)
    if output.is_file() and output.stat().st_size > 44:
        return output.resolve()
    waveform, sample_rate = read_audio(source)
    kind = str(spec["kind"])
    if kind == "identity":
        transformed = waveform
    elif kind == "gaussian_noise":
        transformed = add_noise(waveform, float(spec["snr_db"]), rng)
    elif kind == "gain":
        transformed = apply_gain(waveform, float(spec["gain_db"]))
    elif kind == "truncate_at_slot":
        if not span:
            raise ValueError("truncate_at_slot requires an aligned span")
        transformed = truncate_at_span(
            waveform, sample_rate, span, float(spec.get("retain_slot_fraction", 0.35))
        )
    elif kind == "local_overlap":
        if not span or not donor_path:
            raise ValueError("local_overlap requires an aligned span and donor")
        donor, donor_rate = read_audio(donor_path)
        transformed = local_overlap(
            waveform,
            sample_rate,
            donor,
            donor_rate,
            span,
            float(spec.get("snr_db", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported audio transform kind: {kind}")
    write_audio(output, transformed, sample_rate)
    return output.resolve()

