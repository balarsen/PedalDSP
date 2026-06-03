"""Tests for pedal_model.metrics.perceptual."""
import numpy as np
import pytest

from pedal_model.metrics.perceptual import compute_lsd, compute_mcd

SR = 48000


def _noise(seed: int = 0, n: int = SR * 2) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.3


def test_mcd_identical_signals():
    signal = _noise(0)
    mcd = compute_mcd(signal, signal, SR)
    assert mcd == pytest.approx(0.0, abs=1e-4)


def test_mcd_non_negative():
    a, b = _noise(0), _noise(1)
    assert compute_mcd(a, b, SR) >= 0.0


def test_mcd_increases_with_distortion():
    signal = _noise(0)
    slight = (signal + 0.01 * _noise(1)).astype(np.float32)
    heavy = (signal + 1.0 * _noise(2)).astype(np.float32)
    assert compute_mcd(signal, slight, SR) < compute_mcd(signal, heavy, SR)


def test_lsd_identical():
    inp = _noise(0)
    out = inp * 0.8
    lsd = compute_lsd(out, out, inp, SR)
    assert lsd == pytest.approx(0.0, abs=1e-4)


def test_lsd_non_negative():
    inp = _noise(0)
    t = (inp * 0.8).astype(np.float32)
    p = (inp * 0.5).astype(np.float32)
    assert compute_lsd(t, p, inp, SR) >= 0.0
