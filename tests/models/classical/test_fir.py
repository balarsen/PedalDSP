"""Tests for pedal_model.models.classical.fir."""
import numpy as np
import pytest

from pedal_model.models.classical.fir import FIRModel

SR = 48000


def _noise(seed: int = 0, n: int = SR * 4) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.3


def test_name():
    model = FIRModel(n_taps=512)
    assert "FIR" in model.name
    assert "512" in model.name


def test_receptive_field():
    model = FIRModel(n_taps=256)
    assert model.receptive_field == 256


def test_predict_before_fit_raises():
    model = FIRModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros(100, dtype=np.float32))


def test_fit_and_predict_output_shape():
    dry = _noise(0)
    wet = (dry * 0.8).astype(np.float32)
    model = FIRModel(n_taps=256)
    model.fit(dry, wet, SR)
    out = model.predict(dry)
    assert out.shape == dry.shape


def test_predict_dtype():
    dry = _noise(0)
    wet = dry * 0.8
    model = FIRModel(n_taps=128)
    model.fit(dry, wet.astype(np.float32), SR)
    out = model.predict(dry)
    assert out.dtype == np.float32


def test_fir_recovers_gain_change():
    """FIR should fit a pure-gain pedal (wet = dry * k) to low ESR."""
    dry = _noise(0, n=SR * 8)
    wet = (dry * 0.6).astype(np.float32)
    model = FIRModel(n_taps=512)
    model.fit(dry, wet, SR)
    pred = model.predict(dry)
    n = min(len(wet), len(pred))
    esr = float(np.sum((wet[:n] - pred[:n]) ** 2) / (np.sum(wet[:n] ** 2) + 1e-8))
    assert esr < 0.05  # generous threshold for a broadband noise signal
