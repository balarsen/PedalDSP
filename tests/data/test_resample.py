"""Tests for pedal_model.data.resample."""

import numpy as np
import pytest

from pedal_model.data.resample import downsample_96k_to_48k, resample

SR_96K = 96_000
SR_48K = 48_000


def _sine(freq: float, sr: int, duration: float = 0.1) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ── Identity ──────────────────────────────────────────────────────────────────


def test_resample_same_rate_returns_copy():
    audio = _sine(440.0, SR_48K)
    out = resample(audio, SR_48K, SR_48K)
    np.testing.assert_array_equal(out, audio)
    assert out is not audio  # must be a copy


# ── Output length ─────────────────────────────────────────────────────────────


def test_resample_96k_to_48k_output_length():
    audio = _sine(440.0, SR_96K, duration=1.0)
    out = resample(audio, SR_96K, SR_48K)
    # polyphase filter adds edge samples; allow ±1 sample tolerance
    assert abs(len(out) - SR_48K) <= 1


def test_downsample_96k_to_48k_output_length():
    audio = _sine(440.0, SR_96K, duration=1.0)
    out = downsample_96k_to_48k(audio)
    assert abs(len(out) - SR_48K) <= 1


def test_resample_48k_to_96k_output_length():
    audio = _sine(440.0, SR_48K, duration=1.0)
    out = resample(audio, SR_48K, SR_96K)
    assert abs(len(out) - SR_96K) <= 1


# ── Dtype ─────────────────────────────────────────────────────────────────────


def test_resample_output_dtype_float32():
    audio = _sine(440.0, SR_96K)
    out = resample(audio, SR_96K, SR_48K)
    assert out.dtype == np.float32


def test_downsample_output_dtype_float32():
    audio = _sine(440.0, SR_96K)
    out = downsample_96k_to_48k(audio)
    assert out.dtype == np.float32


# ── Clipping ─────────────────────────────────────────────────────────────────


def test_resample_clips_to_unity():
    # Even if the filter overshoots, output must stay in [-1, 1]
    loud = (_sine(440.0, SR_96K) * 10.0).astype(np.float32)
    out = resample(loud, SR_96K, SR_48K)
    assert np.max(np.abs(out)) <= 1.0


# ── Anti-aliasing ─────────────────────────────────────────────────────────────


def test_high_freq_attenuated_after_downsample():
    """A 30 kHz tone must be heavily attenuated after 96k→48k.

    Output Nyquist at 48k = 24 kHz.  30 kHz is 6 kHz above the cutoff and
    well into the stopband of the polyphase anti-aliasing filter.  22 kHz
    would be BELOW Nyquist and must NOT be filtered; 26 kHz is in the
    transition band; 30 kHz is safely in the stopband (>20 dB attenuation).
    """
    tone = _sine(30_000.0, SR_96K, duration=0.5)
    out = downsample_96k_to_48k(tone)
    in_rms = float(np.sqrt(np.mean(tone ** 2)))
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    assert out_rms < in_rms * 0.1, (
        f"30 kHz tone not attenuated: in_rms={in_rms:.4f} out_rms={out_rms:.4f}"
    )


def test_low_freq_preserved_after_downsample():
    """A 440 Hz tone must pass through 96k→48k essentially unattenuated."""
    tone = _sine(440.0, SR_96K, duration=0.5)
    out = downsample_96k_to_48k(tone)
    in_rms = float(np.sqrt(np.mean(tone ** 2)))
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    # Allow ±1 dB amplitude variation from the polyphase filter
    assert out_rms > in_rms * 0.89, (
        f"440 Hz tone too attenuated: in_rms={in_rms:.4f} out_rms={out_rms:.4f}"
    )


# ── Alignment (dry/wet must receive identical treatment) ──────────────────────


def test_resample_dry_wet_same_length():
    """Resampling dry and wet independently must produce equal-length arrays."""
    np.random.seed(0)
    dry = np.random.randn(SR_96K).astype(np.float32) * 0.5
    wet = np.tanh(dry * 2.0).astype(np.float32)

    dry_48 = resample(dry, SR_96K, SR_48K)
    wet_48 = resample(wet, SR_96K, SR_48K)
    assert len(dry_48) == len(wet_48)
