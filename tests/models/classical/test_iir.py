"""Tests for pedal_model.models.classical.iir."""
import numpy as np
import pytest

from pedal_model.models.classical.iir import IIRModel, _fit_iir_ls

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


def test_receptive_field():
    assert IIRModel(order=4).receptive_field == 40
    assert IIRModel(order=6).receptive_field == 60


def test_output_finite():
    dry = _noise(0)
    wet = (dry * 0.7).astype(np.float32)
    model = IIRModel(order=4, n_freq_points=128)
    model.fit(dry, wet, SR)
    assert np.isfinite(model.predict(dry)).all()


def test_fit_iir_ls_recovers_one_pole_filter():
    """_fit_iir_ls must recover a = [1, -0.9] for a 1-pole LP H(z) = 1/(1-0.9z^{-1})."""
    N_FREQ = 256
    w = np.linspace(0.0, np.pi, N_FREQ, endpoint=False)
    z = np.exp(1j * w)
    # pole at z=0.9 — denominator A(z)=1-0.9z^{-1}, so a[1] should be -0.9
    H = 1.0 / (1.0 - 0.9 * z ** (-1))
    b, a = _fit_iir_ls(H, w, n_b=0, n_a=1)
    assert len(b) == 1 and len(a) == 2
    assert abs(a[1] - (-0.9)) < 0.05


def test_iir_recovers_gain():
    """IIR should fit a flat-gain system (wet = 0.6*dry) with low ESR."""
    dry = _noise(0, n=SR * 8)
    wet = (dry * 0.6).astype(np.float32)
    model = IIRModel(order=2, n_freq_points=256)
    model.fit(dry, wet, SR)
    pred = model.predict(dry)
    n = min(len(wet), len(pred))
    esr = float(np.sum((wet[:n] - pred[:n]) ** 2) / (np.sum(wet[:n] ** 2) + 1e-8))
    assert esr < 0.05
