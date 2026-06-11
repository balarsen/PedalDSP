"""Tests for pedal_model.models.neural.mlp."""
import numpy as np
import pytest
import torch

from pedal_model.models.neural.mlp import MLPModel


def test_name():
    model = MLPModel(receptive_field_samples=512)
    assert "MLP" in model.name


def test_receptive_field():
    model = MLPModel(receptive_field_samples=1024)
    assert model.receptive_field == 1024


def test_forward_output_shape():
    model = MLPModel(receptive_field_samples=64)
    batch = torch.randn(8, 64)
    out = model(batch)
    assert out.shape == (8, 1)


def test_forward_output_shape_batch_size_one():
    model = MLPModel(receptive_field_samples=32)
    out = model(torch.randn(1, 32))
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()


def test_forward_output_shape_large_batch():
    model = MLPModel(receptive_field_samples=32)
    out = model(torch.randn(64, 32))
    assert out.shape == (64, 1)
    assert torch.isfinite(out).all()


def test_forward_output_is_finite():
    model = MLPModel(receptive_field_samples=32)
    batch = torch.randn(4, 32)
    out = model(batch)
    assert torch.isfinite(out).all()


def test_predict_numpy_output_shape():
    model = MLPModel(receptive_field_samples=32)
    x = np.random.randn(200).astype(np.float32) * 0.3
    out = model.predict(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32


def test_fit_raises():
    model = MLPModel()
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), 48000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_on_gpu():
    model = MLPModel(receptive_field_samples=32).cuda()
    batch = torch.randn(4, 32, device="cuda")
    out = model(batch)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()


# ── Regression tests ────────────────────────────────────────────────────────


def test_state_dict_roundtrip(tmp_path):
    """Save weights, load into fresh model, outputs must be identical."""
    torch.manual_seed(0)
    model = MLPModel(receptive_field_samples=64)
    x = torch.randn(4, 64)
    with torch.no_grad():
        out_orig = model(x)

    ckpt = tmp_path / "mlp.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = MLPModel(receptive_field_samples=64)
    model2.load_state_dict(torch.load(ckpt, weights_only=True))
    with torch.no_grad():
        out_loaded = model2(x)

    torch.testing.assert_close(out_orig, out_loaded, atol=0.0, rtol=0.0)


def test_loss_backward():
    """Loss must be finite and gradients must flow to all parameters."""
    torch.manual_seed(1)
    model = MLPModel(receptive_field_samples=32)
    x = torch.randn(8, 32)
    target = torch.randn(8, 1)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient contains inf/nan"


def test_forward_determinism():
    """Same weights and input must produce identical output on repeated calls."""
    torch.manual_seed(2)
    model = MLPModel(receptive_field_samples=32)
    x = torch.randn(4, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)
