"""Tests for pedal_model.data.resample."""

import numpy as np
import pytest

from pedal_model.data.resample import (
    downsample_96k_to_48k,
    float32_to_int16,
    int16_to_float32,
    prepare_for_daisy,
    resample,
)

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


# ── float32_to_int16 ──────────────────────────────────────────────────────────


def test_float32_to_int16_dtype():
    audio = _sine(440.0, SR_48K)
    out = float32_to_int16(audio)
    assert out.dtype == np.int16


def test_float32_to_int16_positive_full_scale():
    audio = np.array([1.0], dtype=np.float32)
    out = float32_to_int16(audio)
    assert out[0] == 32767


def test_float32_to_int16_negative_full_scale():
    audio = np.array([-1.0], dtype=np.float32)
    out = float32_to_int16(audio)
    assert out[0] == -32767


def test_float32_to_int16_zero():
    audio = np.array([0.0], dtype=np.float32)
    out = float32_to_int16(audio)
    assert out[0] == 0


def test_float32_to_int16_clips_overflow():
    audio = np.array([1.5, -2.0], dtype=np.float32)
    out = float32_to_int16(audio)
    assert out[0] == 32767
    assert out[1] == -32767


def test_float32_to_int16_shape_preserved():
    audio = _sine(440.0, SR_48K, duration=0.5)
    out = float32_to_int16(audio)
    assert out.shape == audio.shape


# ── int16_to_float32 ──────────────────────────────────────────────────────────


def test_int16_to_float32_dtype():
    audio = np.array([0, 16384, -16384], dtype=np.int16)
    out = int16_to_float32(audio)
    assert out.dtype == np.float32


def test_int16_to_float32_range():
    audio = np.array([32767, -32767, 0], dtype=np.int16)
    out = int16_to_float32(audio)
    assert out[0] == pytest.approx(1.0, abs=1e-4)
    assert out[1] == pytest.approx(-1.0, abs=1e-4)
    assert out[2] == pytest.approx(0.0, abs=1e-6)


def test_roundtrip_float32_int16_float32():
    """float32 → int16 → float32 must be close to the original."""
    audio = _sine(440.0, SR_48K, duration=0.1)
    roundtripped = int16_to_float32(float32_to_int16(audio))
    # 16-bit quantization error is ≤ 1/32767 ≈ 3e-5
    np.testing.assert_allclose(roundtripped, audio, atol=4e-5)


# ── prepare_for_daisy ─────────────────────────────────────────────────────────


def test_prepare_for_daisy_dtype():
    audio = _sine(440.0, SR_96K, duration=0.5)
    out = prepare_for_daisy(audio)
    assert out.dtype == np.int16


def test_prepare_for_daisy_output_length():
    audio = _sine(440.0, SR_96K, duration=1.0)
    out = prepare_for_daisy(audio)
    assert abs(len(out) - SR_48K) <= 1


def test_prepare_for_daisy_range():
    audio = _sine(440.0, SR_96K, duration=0.5)
    out = prepare_for_daisy(audio)
    assert np.max(np.abs(out)) <= 32767


def test_prepare_for_daisy_440hz_not_silent():
    """A 440 Hz tone (well below 24 kHz Nyquist) must survive conversion."""
    audio = _sine(440.0, SR_96K, duration=0.5)
    out = prepare_for_daisy(audio)
    assert np.max(np.abs(out)) > 1000  # well above zero


# ── Upsampling ────────────────────────────────────────────────────────────────


def test_resample_upsample_output_length():
    """48k → 96k should produce ~2× the input length."""
    audio = _sine(440.0, SR_48K, duration=1.0)
    out = resample(audio, SR_48K, SR_96K)
    assert abs(len(out) - SR_96K) <= 2


def test_resample_upsample_dtype_float32():
    audio = _sine(440.0, SR_48K)
    out = resample(audio, SR_48K, SR_96K)
    assert out.dtype == np.float32


def test_resample_upsample_preserves_low_freq():
    """440 Hz must pass through 48k→96k essentially unattenuated."""
    tone = _sine(440.0, SR_48K, duration=0.5)
    out = resample(tone, SR_48K, SR_96K)
    in_rms = float(np.sqrt(np.mean(tone ** 2)))
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    assert out_rms > in_rms * 0.89


def test_resample_upsample_no_imaging_above_input_nyquist():
    """Upsampling must not introduce spectral images above the input Nyquist (24 kHz).

    A 10 kHz tone upsampled 48k→96k should have negligible energy at 38 kHz
    (its first image: 48k − 10k = 38k). A >20 dB rejection is expected.
    """
    f0 = 10_000.0
    tone = _sine(f0, SR_48K, duration=0.5)
    out = resample(tone, SR_48K, SR_96K)

    N = len(out)
    spectrum = np.abs(np.fft.rfft(out * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, 1.0 / SR_96K)

    fund_bin = int(round(f0 * N / SR_96K))
    fund_energy = float(spectrum[fund_bin] ** 2)

    # First image of 10 kHz at 48k→96k is at 48k − 10k = 38 kHz
    image_f = 38_000.0
    image_bin = int(round(image_f * N / SR_96K))
    image_energy = float(spectrum[image_bin] ** 2)

    ratio_db = 10.0 * np.log10(image_energy / (fund_energy + 1e-30) + 1e-30)
    assert ratio_db < -20.0, f"Image at {image_f} Hz is only {ratio_db:.1f} dB below fundamental"


def test_resample_non_power_of_2_ratio():
    """44.1k → 48k (non-trivial rational ratio) must produce the correct length."""
    SR_441 = 44_100
    audio = _sine(440.0, SR_441, duration=1.0)
    out = resample(audio, SR_441, SR_48K)
    expected = int(round(len(audio) * SR_48K / SR_441))
    assert abs(len(out) - expected) <= 2


# ── Additional float32_to_int16 coverage ─────────────────────────────────────


def test_float32_to_int16_small_signal_not_zeroed():
    """A -60 dBFS tone must survive 16-bit quantization without being zeroed."""
    amp = 10.0 ** (-60.0 / 20.0)
    audio = (_sine(440.0, SR_48K, duration=0.1) * amp).astype(np.float32)
    out = float32_to_int16(audio)
    assert np.max(np.abs(out)) > 0


def test_float32_to_int16_quantization_error():
    """Peak quantization error must be ≤ 1 LSB (1/32767 ≈ 3e-5)."""
    audio = _sine(440.0, SR_48K, duration=0.1)
    out = float32_to_int16(audio)
    reconstructed = out.astype(np.float32) / 32767.0
    max_err = float(np.max(np.abs(reconstructed - audio)))
    assert max_err <= 1.0 / 32767.0 + 1e-7


# ── Additional prepare_for_daisy coverage ────────────────────────────────────


def test_prepare_for_daisy_custom_sr_in():
    """prepare_for_daisy with sr_in=44100 must produce length ≈48000 for 1s input."""
    SR_441 = 44_100
    audio = _sine(440.0, SR_441, duration=1.0)
    out = prepare_for_daisy(audio, sr_in=SR_441)
    assert out.dtype == np.int16
    assert abs(len(out) - SR_48K) <= 2


def test_prepare_for_daisy_silence_stays_silent():
    audio = np.zeros(SR_96K, dtype=np.float32)
    out = prepare_for_daisy(audio)
    assert np.all(out == 0)
