"""Tests for pedal_model.metrics.harmonic."""
import numpy as np
import pytest

from pedal_model.metrics.harmonic import (
    compute_eo_ratio,
    compute_harmonic_profile,
    compute_hp_similarity,
    compute_thd,
    compute_thd_pattern_distance,
)

SR = 48000
F0 = 440.0
DURATION = 2.0  # seconds — long enough for good frequency resolution


def _pure_sine(f0: float, sr: int = SR, duration: float = DURATION, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)


def test_thd_pure_sine_is_low():
    signal = _pure_sine(F0)
    thd = compute_thd(signal, F0, SR)
    # A pure sine should have THD well below 5%
    assert thd < 5.0


def test_thd_clipped_signal_is_higher():
    sine = _pure_sine(F0, amp=0.9)
    clipped = np.clip(sine, -0.5, 0.5)
    thd_sine = compute_thd(sine, F0, SR)
    thd_clipped = compute_thd(clipped, F0, SR)
    assert thd_clipped > thd_sine


def test_thd_non_negative():
    signal = _pure_sine(F0)
    assert compute_thd(signal, F0, SR) >= 0.0


def test_harmonic_profile_fundamental_is_one():
    signal = _pure_sine(F0)
    profile = compute_harmonic_profile(signal, F0, SR)
    assert profile[0] == pytest.approx(1.0, rel=0.01)


def test_harmonic_profile_length():
    signal = _pure_sine(F0)
    n = 6
    profile = compute_harmonic_profile(signal, F0, SR, n_harmonics=n)
    assert len(profile) == n


def test_hp_similarity_identical():
    signal = _pure_sine(F0)
    sim = compute_hp_similarity(signal, signal, F0, SR)
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_hp_similarity_in_range():
    sine = _pure_sine(F0)
    clipped = np.tanh(sine * 5.0).astype(np.float32)
    sim = compute_hp_similarity(sine, clipped, F0, SR)
    assert -1.0 <= sim <= 1.0


def test_eo_ratio_non_negative():
    signal = _pure_sine(F0)
    assert compute_eo_ratio(signal, F0, SR) >= 0.0


# ── compute_thd_pattern_distance ──────────────────────────────────────────────


def test_thd_pattern_distance_identical():
    signal = _pure_sine(F0)
    assert compute_thd_pattern_distance(signal, signal, F0, SR) == pytest.approx(0.0, abs=1e-6)


def test_thd_pattern_distance_non_negative():
    sine = _pure_sine(F0)
    clipped = np.tanh(sine * 5.0).astype(np.float32)
    assert compute_thd_pattern_distance(sine, clipped, F0, SR) >= 0.0


def test_thd_pattern_distance_increases_with_distortion():
    """More distortion = larger profile distance from original sine."""
    sine = _pure_sine(F0)
    mild = np.tanh(sine * 2.0).astype(np.float32)
    heavy = np.tanh(sine * 20.0).astype(np.float32)
    d_mild = compute_thd_pattern_distance(sine, mild, F0, SR)
    d_heavy = compute_thd_pattern_distance(sine, heavy, F0, SR)
    assert d_heavy > d_mild


def test_thd_pattern_distance_symmetric():
    """Distance(a, b) == Distance(b, a)."""
    sine = _pure_sine(F0)
    clipped = np.tanh(sine * 5.0).astype(np.float32)
    d_ab = compute_thd_pattern_distance(sine, clipped, F0, SR)
    d_ba = compute_thd_pattern_distance(clipped, sine, F0, SR)
    assert d_ab == pytest.approx(d_ba, rel=1e-5)
