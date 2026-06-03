"""Tests for pedal_model.models.classical.wiener_hammerstein."""
import numpy as np
import pytest

from pedal_model.models.classical.wiener_hammerstein import WienerHammersteinModel

SR = 48000


def _noise(seed: int = 0, n: int = SR * 4) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.3


def test_name():
    model = WienerHammersteinModel()
    assert "WienerHammerstein" in model.name


def test_predict_before_fit_raises():
    model = WienerHammersteinModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros(100, dtype=np.float32))


def test_output_shape():
    dry = _noise(0)
    wet = np.tanh(dry * 3.0).astype(np.float32)
    model = WienerHammersteinModel(poly_order=3, n_taps=64)
    model.fit(dry, wet, SR)
    out = model.predict(dry)
    assert out.shape == dry.shape


def test_output_dtype():
    dry = _noise(0)
    wet = np.tanh(dry * 2.0).astype(np.float32)
    model = WienerHammersteinModel(n_taps=64)
    model.fit(dry, wet, SR)
    assert model.predict(dry).dtype == np.float32
