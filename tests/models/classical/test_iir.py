"""Tests for pedal_model.models.classical.iir."""
import numpy as np
import pytest

from pedal_model.models.classical.iir import IIRModel

SR = 48000


def _noise(seed: int = 0, n: int = SR * 4) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.3


def test_name():
    model = IIRModel(order=6)
    assert "IIR" in model.name
    assert "6" in model.name


def test_predict_before_fit_raises():
    model = IIRModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros(100, dtype=np.float32))


def test_output_shape():
    dry = _noise(0)
    wet = (dry * 0.7).astype(np.float32)
    model = IIRModel(order=4, n_freq_points=128)
    model.fit(dry, wet, SR)
    out = model.predict(dry)
    assert out.shape == dry.shape


def test_output_dtype():
    dry = _noise(0)
    wet = (dry * 0.7).astype(np.float32)
    model = IIRModel(order=4, n_freq_points=128)
    model.fit(dry, wet, SR)
    assert model.predict(dry).dtype == np.float32
