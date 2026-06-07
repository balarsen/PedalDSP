"""Tests for pedal_model.models.neural.gru."""
import numpy as np
import pytest
import torch

from pedal_model.models.neural.gru import GRUModel


def test_name():
    model = GRUModel(hidden_size=32)
    assert "GRU" in model.name
    assert "32" in model.name


def test_forward_output_shape():
    model = GRUModel(hidden_size=16)
    x = torch.randn(4, 128, 1)  # (batch, time, 1)
    out, h = model(x)
    assert out.shape == (4, 128, 1)
    assert h.shape == (1, 4, 16)


def test_hidden_state_carries_forward():
    model = GRUModel(hidden_size=8)
    x1 = torch.randn(1, 64, 1)
    out1, hidden = model(x1)
    x2 = torch.randn(1, 64, 1)
    out2, _ = model(x2, hidden)
    assert out2.shape == (1, 64, 1)


def test_hidden_state_changes_output():
    """Passing a non-None hidden state must produce different output than None."""
    torch.manual_seed(0)
    model = GRUModel(hidden_size=8)
    x = torch.randn(1, 32, 1)
    out_fresh, h = model(x)
    # Run again from the previous state — output should differ
    out_continued, _ = model(x, h)
    assert not torch.allclose(out_fresh, out_continued)


def test_forward_finite():
    model = GRUModel(hidden_size=8)
    x = torch.randn(2, 50, 1)
    out, _ = model(x)
    assert torch.isfinite(out).all()


def test_predict_output_shape():
    model = GRUModel(hidden_size=8)
    np.random.seed(0)
    x = np.random.randn(500).astype(np.float32) * 0.3
    out = model.predict(x, chunk_size=128)
    assert out.shape == x.shape
    assert out.dtype == np.float32


def test_predict_chunk_size_does_not_change_output():
    """predict() result should be the same regardless of chunk_size."""
    torch.manual_seed(0)
    model = GRUModel(hidden_size=8)
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float32) * 0.3
    out_small = model.predict(x, chunk_size=64)
    out_large = model.predict(x, chunk_size=1000)
    np.testing.assert_allclose(out_small, out_large, atol=1e-5)


def test_fit_raises():
    model = GRUModel()
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), 48000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_on_gpu():
    model = GRUModel(hidden_size=8).cuda()
    x = torch.randn(2, 64, 1, device="cuda")
    out, h = model(x)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
