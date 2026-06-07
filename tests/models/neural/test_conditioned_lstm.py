"""Tests for pedal_model.models.neural.conditioned_lstm."""
import numpy as np
import pytest
import torch

from pedal_model.models.neural.conditioned_lstm import ConditionedLSTMModel


def test_name():
    model = ConditionedLSTMModel(hidden_size=16)
    assert "ConditionedLSTM" in model.name
    assert "16" in model.name


def test_forward_output_shape():
    model = ConditionedLSTMModel(n_knobs=3, knob_embedding_dim=4, hidden_size=16)
    x = torch.randn(2, 64, 1)
    knobs = torch.rand(2, 3)
    out, (h, c) = model(x, knobs)
    assert out.shape == (2, 64, 1)
    assert h.shape == (1, 2, 16)
    assert c.shape == (1, 2, 16)


def test_different_knobs_give_different_output():
    """Conditioning on different knob values must produce different output."""
    torch.manual_seed(0)
    model = ConditionedLSTMModel(n_knobs=3, knob_embedding_dim=8, hidden_size=16)
    x = torch.randn(1, 32, 1)
    knobs_a = torch.zeros(1, 3)
    knobs_b = torch.ones(1, 3)
    out_a, _ = model(x, knobs_a)
    out_b, _ = model(x, knobs_b)
    assert not torch.allclose(out_a, out_b)


def test_hidden_state_carries_forward():
    model = ConditionedLSTMModel(n_knobs=2, hidden_size=8)
    x1 = torch.randn(1, 32, 1)
    knobs = torch.rand(1, 2)
    out1, hidden = model(x1, knobs)
    x2 = torch.randn(1, 32, 1)
    out2, _ = model(x2, knobs, hidden)
    assert out2.shape == (1, 32, 1)


def test_forward_finite():
    model = ConditionedLSTMModel(n_knobs=3, hidden_size=8)
    x = torch.randn(2, 50, 1)
    knobs = torch.rand(2, 3)
    out, _ = model(x, knobs)
    assert torch.isfinite(out).all()


def test_predict_shape():
    np.random.seed(0)
    model = ConditionedLSTMModel(n_knobs=3, hidden_size=8)
    x = np.random.randn(300).astype(np.float32) * 0.3
    knobs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    out = model.predict(x, knobs, chunk_size=64)
    assert out.shape == (300,)
    assert out.dtype == np.float32


def test_fit_raises():
    model = ConditionedLSTMModel()
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), 48000)
