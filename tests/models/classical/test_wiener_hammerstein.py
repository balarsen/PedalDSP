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


def test_receptive_field():
    assert WienerHammersteinModel(n_taps=64).receptive_field == 128   # n_taps * 2
    assert WienerHammersteinModel(n_taps=128).receptive_field == 256


def test_output_finite():
    dry = _noise(0)
    wet = np.tanh(dry * 3.0).astype(np.float32)
    model = WienerHammersteinModel(poly_order=3, n_taps=64)
    model.fit(dry, wet, SR)
    assert np.isfinite(model.predict(dry)).all()


def test_apply_nl_shape_and_dtype():
    """_apply_nl must return the same shape and float32 dtype as input."""
    dry = _noise(0)
    wet = np.tanh(dry * 3.0).astype(np.float32)
    model = WienerHammersteinModel(poly_order=3, n_taps=64)
    model.fit(dry, wet, SR)
    out = model._apply_nl(dry)
    assert out.shape == dry.shape
    assert out.dtype == np.float32


def test_fit_fir_returns_kernel_of_correct_length():
    """_fit_fir must return a 1-D kernel of length n_taps."""
    dry = _noise(0, n=SR * 4)
    wet = (dry * 0.8).astype(np.float32)
    model = WienerHammersteinModel(n_taps=64)
    h = model._fit_fir(dry, wet)
    assert h.shape == (64,)


def test_wh_recovers_tanh():
    """WienerHammerstein should fit tanh clipping with ESR < 0.5."""
    dry = _noise(0, n=SR * 8)
    wet = np.tanh(dry * 3.0).astype(np.float32)
    model = WienerHammersteinModel(poly_order=5, n_taps=64)
    model.fit(dry, wet, SR)
    pred = model.predict(dry)
    n = min(len(wet), len(pred))
    esr = float(np.sum((wet[:n] - pred[:n]) ** 2) / (np.sum(wet[:n] ** 2) + 1e-8))
    assert esr < 0.5
