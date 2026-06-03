"""Tests for pedal_model.models.neural.lstm."""
import numpy as np
import pytest
import torch

from pedal_model.models.neural.lstm import LSTMModel


def test_name():
    model = LSTMModel(hidden_size=32)
    assert "LSTM" in model.name
    assert "32" in model.name


def test_forward_output_shape():
    model = LSTMModel(hidden_size=16)
    x = torch.randn(4, 128, 1)  # (batch, time, 1)
    out, (h, c) = model(x)
    assert out.shape == (4, 128, 1)
    assert h.shape == (1, 4, 16)
    assert c.shape == (1, 4, 16)


def test_hidden_state_carries_forward():
    model = LSTMModel(hidden_size=8)
    x1 = torch.randn(1, 64, 1)
    out1, hidden = model(x1)
    x2 = torch.randn(1, 64, 1)
    out2, _ = model(x2, hidden)
    assert out2.shape == (1, 64, 1)


def test_forward_finite():
    model = LSTMModel(hidden_size=8)
    x = torch.randn(2, 50, 1)
    out, _ = model(x)
    assert torch.isfinite(out).all()


def test_predict_output_shape():
    model = LSTMModel(hidden_size=8)
    x = np.random.randn(500).astype(np.float32) * 0.3
    out = model.predict(x, chunk_size=128)
    assert out.shape == x.shape
    assert out.dtype == np.float32


def test_fit_raises():
    model = LSTMModel()
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), 48000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_on_gpu():
    model = LSTMModel(hidden_size=8).cuda()
    x = torch.randn(2, 64, 1, device="cuda")
    out, _ = model(x)
    assert out.device.type == "cuda"
