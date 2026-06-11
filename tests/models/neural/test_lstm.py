"""Tests for pedal_model.models.neural.lstm."""
import numpy as np
import pytest
import torch

from pedal_model.models.neural.lstm import LSTMModel


def test_name():
    model = LSTMModel(hidden_size=32)
    assert "LSTM" in model.name
    assert "32" in model.name


def test_receptive_field():
    assert LSTMModel().receptive_field == 4096


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


# ── Regression tests ────────────────────────────────────────────────────────


def test_state_dict_roundtrip(tmp_path):
    """Save weights, load into fresh model, outputs must be identical."""
    torch.manual_seed(0)
    model = LSTMModel(hidden_size=16)
    x = torch.randn(1, 128, 1)
    with torch.no_grad():
        out_orig, _ = model(x)

    ckpt = tmp_path / "lstm.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = LSTMModel(hidden_size=16)
    model2.load_state_dict(torch.load(ckpt, weights_only=True))
    with torch.no_grad():
        out_loaded, _ = model2(x)

    torch.testing.assert_close(out_orig, out_loaded, atol=0.0, rtol=0.0)


def test_loss_backward():
    """Loss must be finite and gradients must flow to all parameters."""
    torch.manual_seed(1)
    model = LSTMModel(hidden_size=8)
    x = torch.randn(2, 64, 1)
    target = torch.randn(2, 64, 1)
    out, _ = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient contains inf/nan"


def test_predict_chunk_consistency():
    """predict() output must be identical regardless of chunk_size."""
    torch.manual_seed(2)
    np.random.seed(2)
    model = LSTMModel(hidden_size=8)
    x = np.random.randn(2000).astype(np.float32) * 0.3
    out_small = model.predict(x, chunk_size=256)
    out_large = model.predict(x, chunk_size=2000)
    np.testing.assert_allclose(out_small, out_large, atol=1e-5,
                                err_msg="LSTM predict() differs by chunk_size")


def test_forward_determinism():
    """Same seed must produce identical outputs on repeated calls."""
    torch.manual_seed(3)
    model = LSTMModel(hidden_size=8)
    torch.manual_seed(99)
    x = torch.randn(1, 64, 1)
    with torch.no_grad():
        out1, _ = model(x)
        out2, _ = model(x)  # same model, same input, fresh hidden state
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)
