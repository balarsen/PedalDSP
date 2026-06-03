"""Tests for pedal_model.metrics.time_domain."""
import numpy as np
import pytest

from pedal_model.metrics.time_domain import (
    compute_dc_error,
    compute_esr,
    compute_mse,
    compute_rms_error,
)


def test_esr_perfect_prediction():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(1000).astype(np.float32)
    assert compute_esr(signal, signal) == pytest.approx(0.0, abs=1e-6)


def test_esr_silence_prediction():
    signal = np.ones(100, dtype=np.float32)
    # predicted=0 → ESR = Σ1² / Σ1² = 1
    assert compute_esr(signal, np.zeros(100, dtype=np.float32)) == pytest.approx(1.0, rel=1e-5)


def test_esr_is_non_negative():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(500).astype(np.float32)
    b = rng.standard_normal(500).astype(np.float32)
    assert compute_esr(a, b) >= 0.0


def test_mse_perfect():
    signal = np.ones(50, dtype=np.float32)
    assert compute_mse(signal, signal) == pytest.approx(0.0, abs=1e-7)


def test_mse_known_value():
    target = np.zeros(4, dtype=np.float32)
    pred = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert compute_mse(target, pred) == pytest.approx(1.0)


def test_dc_error_zero_when_equal():
    signal = np.linspace(-1, 1, 100, dtype=np.float32)
    assert compute_dc_error(signal, signal) == pytest.approx(0.0, abs=1e-7)


def test_dc_error_detects_offset():
    signal = np.zeros(100, dtype=np.float32)
    shifted = np.ones(100, dtype=np.float32) * 0.5
    assert compute_dc_error(signal, shifted) == pytest.approx(0.5, rel=1e-5)


def test_rms_error_perfect():
    signal = np.ones(100, dtype=np.float32)
    assert compute_rms_error(signal, signal) == pytest.approx(0.0, abs=1e-7)


def test_rms_equals_sqrt_mse():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(200).astype(np.float32)
    b = rng.standard_normal(200).astype(np.float32)
    assert compute_rms_error(a, b) == pytest.approx(np.sqrt(compute_mse(a, b)), rel=1e-5)
