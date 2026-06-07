"""Tests for pedal_model.utils.analysis."""
import numpy as np
import pytest

from pedal_model.utils.analysis import (
    compute_esr_skip,
    harmonic_frequencies,
    safe_trim,
)


def test_safe_trim_equal_lengths():
    a = np.ones(10, dtype=np.float32)
    b = np.zeros(10, dtype=np.float32)
    ta, tb = safe_trim(a, b)
    assert len(ta) == 10
    assert len(tb) == 10


def test_safe_trim_unequal_lengths():
    a = np.ones(50, dtype=np.float32)
    b = np.ones(30, dtype=np.float32)
    c = np.ones(40, dtype=np.float32)
    ta, tb, tc = safe_trim(a, b, c)
    assert len(ta) == 30
    assert len(tb) == 30
    assert len(tc) == 30


def test_safe_trim_preserves_content():
    a = np.arange(10, dtype=np.float32)
    b = np.arange(5, dtype=np.float32)
    ta, tb = safe_trim(a, b)
    np.testing.assert_array_equal(ta, np.arange(5, dtype=np.float32))
    np.testing.assert_array_equal(tb, b)


def test_esr_skip_zero_on_perfect():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(1000).astype(np.float32)
    assert compute_esr_skip(sig, sig) == pytest.approx(0.0, abs=1e-6)


def test_esr_skip_zero_on_perfect_with_skip():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(1000).astype(np.float32)
    assert compute_esr_skip(sig, sig, skip=100) == pytest.approx(0.0, abs=1e-6)


def test_esr_skip_nonzero_with_error():
    sig = np.ones(200, dtype=np.float32)
    pred = np.zeros(200, dtype=np.float32)
    esr = compute_esr_skip(sig, pred)
    assert esr > 0.9


def test_esr_skip_excludes_leading_samples():
    sig = np.ones(100, dtype=np.float32)
    pred = np.zeros(100, dtype=np.float32)
    # skip=99 leaves only 1 sample; with all-zeros predicted and all-ones target it should still be ~1
    esr_full = compute_esr_skip(sig, pred, skip=0)
    esr_skip = compute_esr_skip(sig, pred, skip=50)
    assert esr_full == pytest.approx(esr_skip, rel=1e-3)


def test_harmonic_frequencies_count():
    freqs = harmonic_frequencies(110.0, n_harmonics=6)
    assert len(freqs) == 6


def test_harmonic_frequencies_values():
    freqs = harmonic_frequencies(100.0, n_harmonics=4)
    np.testing.assert_allclose(freqs, [100.0, 200.0, 300.0, 400.0])


def test_harmonic_frequencies_max_freq():
    freqs = harmonic_frequencies(100.0, n_harmonics=10, max_freq=500.0)
    assert all(f <= 500.0 for f in freqs)
    assert len(freqs) == 5  # 100, 200, 300, 400, 500 (inclusive)


def test_harmonic_frequencies_max_freq_mid_step():
    # 350 Hz → keeps 100, 200, 300 (400 > 350)
    freqs = harmonic_frequencies(100.0, n_harmonics=10, max_freq=350.0)
    assert all(f <= 350.0 for f in freqs)
    assert len(freqs) == 3


# ── apply_effect, peak_db, rms_db ────────────────────────────────────────────

from pedal_model.utils.analysis import apply_effect, peak_db, rms_db


def test_apply_effect_returns_float32():
    sig = np.ones(100, dtype=np.float64)
    out = apply_effect(sig, lambda x: x * 2.0)
    assert out.dtype == np.float32


def test_apply_effect_passes_args_and_kwargs():
    sig = np.ones(100, dtype=np.float32)
    out = apply_effect(sig, lambda x, scale, shift=0.0: x * scale + shift,
                       3.0, shift=1.0)
    np.testing.assert_allclose(out, np.full(100, 4.0, dtype=np.float32), rtol=1e-5)


def test_apply_effect_shape_preserved():
    sig = np.arange(256, dtype=np.float32)
    out = apply_effect(sig, lambda x: x)
    assert out.shape == sig.shape


def test_peak_db_returns_float():
    sig = np.full(100, 0.5, dtype=np.float32)
    result = peak_db(sig)
    assert isinstance(result, float)


def test_peak_db_unity_signal_is_zero_dbfs():
    sig = np.ones(100, dtype=np.float32)
    result = peak_db(sig)
    assert result == pytest.approx(0.0, abs=1e-4)


def test_peak_db_half_amplitude_is_minus6():
    sig = np.full(100, 0.5, dtype=np.float32)
    result = peak_db(sig)
    assert result == pytest.approx(-6.0206, abs=0.01)


def test_peak_db_silence_returns_floor():
    sig = np.zeros(100, dtype=np.float32)
    result = peak_db(sig)
    assert result < -100.0


def test_rms_db_returns_float():
    sig = np.full(100, 0.5, dtype=np.float32)
    result = rms_db(sig)
    assert isinstance(result, float)


def test_rms_db_sine_near_minus3():
    """RMS of a unit sine is 1/sqrt(2) ≈ −3.01 dBFS."""
    t = np.arange(48000, dtype=np.float32) / 48000
    sig = np.sin(2 * np.pi * 1000 * t)
    result = rms_db(sig)
    assert result == pytest.approx(-3.01, abs=0.1)


def test_rms_db_silence_returns_floor():
    sig = np.zeros(100, dtype=np.float32)
    result = rms_db(sig)
    assert result < -100.0


def test_peak_db_greater_than_rms_db():
    """Peak must be ≥ RMS for any non-constant signal."""
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(4096).astype(np.float32) * 0.3
    assert peak_db(sig) >= rms_db(sig)
