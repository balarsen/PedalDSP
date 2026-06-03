"""Tests for pedal_model.metrics.frequency_domain."""
import numpy as np
import pytest

from pedal_model.metrics.frequency_domain import (
    compute_fr_error,
    compute_multiscale_stft_loss,
)


def test_stft_loss_identical_signals():
    rng = np.random.default_rng(10)
    signal = rng.standard_normal(4096).astype(np.float32) * 0.3
    loss = compute_multiscale_stft_loss(signal, signal)
    assert loss == pytest.approx(0.0, abs=1e-5)


def test_stft_loss_non_negative():
    rng = np.random.default_rng(11)
    a = rng.standard_normal(4096).astype(np.float32)
    b = rng.standard_normal(4096).astype(np.float32)
    loss = compute_multiscale_stft_loss(a, b)
    assert loss >= 0.0


def test_stft_loss_increases_with_error():
    rng = np.random.default_rng(12)
    signal = rng.standard_normal(8192).astype(np.float32)
    small_error = signal + 0.01 * rng.standard_normal(8192).astype(np.float32)
    large_error = signal + 1.0 * rng.standard_normal(8192).astype(np.float32)
    assert compute_multiscale_stft_loss(signal, small_error) < compute_multiscale_stft_loss(signal, large_error)


def test_fr_error_identical():
    rng = np.random.default_rng(13)
    inp = rng.standard_normal(4096).astype(np.float32) * 0.5
    out = inp * 0.8  # pure gain — same frequency response shape
    assert compute_fr_error(out, out, inp, 48000) == pytest.approx(0.0, abs=1e-4)


def test_fr_error_non_negative():
    rng = np.random.default_rng(14)
    inp = rng.standard_normal(4096).astype(np.float32) * 0.3
    target = rng.standard_normal(4096).astype(np.float32) * 0.3
    pred = rng.standard_normal(4096).astype(np.float32) * 0.3
    assert compute_fr_error(target, pred, inp, 48000) >= 0.0
