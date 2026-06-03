"""Tests for pedal_model.models.classical.volterra."""
import numpy as np
import pytest

from pedal_model.models.classical.volterra import VolterraModel

SR = 48000


def _noise(seed: int = 0, n: int = SR * 2) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.3


def test_name_contains_memory():
    model = VolterraModel(memory=15)
    assert "15" in model.name


def test_receptive_field():
    model = VolterraModel(memory=20)
    assert model.receptive_field == 20


def test_predict_before_fit_raises():
    model = VolterraModel()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros(100, dtype=np.float32))


def test_output_length_matches_input():
    dry = _noise(0)
    wet = (dry * 0.5).astype(np.float32)
    model = VolterraModel(memory=10)
    model.fit(dry, wet, SR)
    out = model.predict(dry)
    assert len(out) == len(dry)


def test_output_dtype():
    dry = _noise(0)
    wet = dry * 0.5
    model = VolterraModel(memory=5)
    model.fit(dry, wet.astype(np.float32), SR)
    assert model.predict(dry).dtype == np.float32


def test_linear_signal_low_esr():
    """Volterra should fit a linear system (wet = 0.5 * dry) with low ESR."""
    dry = _noise(0, n=SR * 4)
    wet = (dry * 0.5).astype(np.float32)
    model = VolterraModel(memory=5)
    model.fit(dry, wet, SR)
    pred = model.predict(dry)
    n = min(len(wet), len(pred))
    esr = float(np.sum((wet[:n] - pred[:n]) ** 2) / (np.sum(wet[:n] ** 2) + 1e-8))
    assert esr < 0.01


def test_too_short_signal_raises():
    model = VolterraModel(memory=50)
    with pytest.raises(ValueError):
        model._build_features(np.zeros(10, dtype=np.float64))
