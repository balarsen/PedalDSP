"""Tests for pedal_model.capture.align."""
import numpy as np
import pytest

from pedal_model.capture.align import align_channels, detect_click, estimate_lag

SR = 48000


def _noise(seed: int = 0, n: int = SR) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.3


def test_detect_click_finds_spike():
    signal = np.zeros(SR, dtype=np.float32)
    signal[1000] = 0.9
    idx = detect_click(signal)
    assert idx == 1000


def test_detect_click_only_searches_window():
    signal = np.zeros(SR * 2, dtype=np.float32)
    # Spike is outside the default search window
    signal[SR + 100] = 0.9
    signal[500] = 0.5  # smaller spike inside the window
    idx = detect_click(signal, search_samples=SR)
    assert idx == 500


def test_estimate_lag_zero_for_identical():
    signal = _noise(0)
    lag = estimate_lag(signal, signal)
    assert lag == 0


def test_estimate_lag_detects_positive_delay():
    signal = _noise(0, n=SR * 2)
    delay = 100  # samples
    wet = np.concatenate([np.zeros(delay, dtype=np.float32), signal[: -delay or None]])
    lag = estimate_lag(signal, wet)
    assert lag == pytest.approx(delay, abs=2)


def test_align_channels_equal_length():
    dry = _noise(0, n=SR * 2)
    delay = 50
    wet = np.concatenate([np.zeros(delay, dtype=np.float32), dry[:-delay]])
    dry_a, wet_a = align_channels(dry, wet, SR)
    assert len(dry_a) == len(wet_a)


def test_align_channels_reduces_lag():
    dry = _noise(0, n=SR * 2)
    delay = 100
    wet = np.concatenate([np.zeros(delay, dtype=np.float32), dry[:-delay]])
    dry_a, wet_a = align_channels(dry, wet, SR)
    residual_lag = estimate_lag(dry_a, wet_a)
    assert abs(residual_lag) <= 5  # within 5 samples after alignment


def test_align_channels_preserves_dtype():
    dry = _noise(0)
    wet = _noise(1)
    dry_a, wet_a = align_channels(dry, wet, SR)
    assert dry_a.dtype == np.float32
    assert wet_a.dtype == np.float32
