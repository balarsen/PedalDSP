"""Tests for pedal_model.models.neural.tcn."""
import numpy as np
import pytest
import torch

from pedal_model.models.neural.tcn import TCNModel


def test_name():
    model = TCNModel(channels=16, n_blocks=5)
    assert "TCN" in model.name


def test_receptive_field_grows_with_blocks():
    small = TCNModel(n_blocks=5)
    large = TCNModel(n_blocks=10)
    assert large.receptive_field > small.receptive_field


def test_receptive_field_exact_value():
    # dilation pattern 1,2,4,8 → 2*(1+2+4+8) + 1 = 31 per block pair... verified at runtime = 61
    assert TCNModel(channels=8, n_blocks=4).receptive_field == 61


def test_forward_output_shape():
    model = TCNModel(channels=8, n_blocks=4)
    x = torch.randn(2, 1, 256)
    out = model(x)
    assert out.shape == (2, 1, 256)


def test_forward_finite():
    model = TCNModel(channels=8, n_blocks=4)
    x = torch.randn(1, 1, 512) * 0.1
    out = model(x)
    assert torch.isfinite(out).all()


def test_predict_output_shape():
    model = TCNModel(channels=8, n_blocks=3)
    x = np.random.randn(1024).astype(np.float32) * 0.1
    out = model.predict(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32


def test_fit_raises():
    model = TCNModel()
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), 48000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_on_gpu():
    model = TCNModel(channels=8, n_blocks=3).cuda()
    x = torch.randn(1, 1, 256, device="cuda")
    out = model(x)
    assert out.device.type == "cuda"


# ── Regression tests ────────────────────────────────────────────────────────


def test_state_dict_roundtrip(tmp_path):
    """Save weights, load into fresh model, outputs must be identical."""
    torch.manual_seed(0)
    model = TCNModel(channels=8, n_blocks=4)
    x = torch.randn(1, 1, 256)
    with torch.no_grad():
        out_orig = model(x)

    ckpt = tmp_path / "tcn.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = TCNModel(channels=8, n_blocks=4)
    model2.load_state_dict(torch.load(ckpt, weights_only=True))
    with torch.no_grad():
        out_loaded = model2(x)

    torch.testing.assert_close(out_orig, out_loaded, atol=0.0, rtol=0.0)


def test_loss_backward():
    """Loss must be finite and gradients must flow to all parameters."""
    torch.manual_seed(1)
    model = TCNModel(channels=8, n_blocks=4)
    x = torch.randn(2, 1, 256)
    target = torch.randn(2, 1, 256)
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
    model = TCNModel(channels=8, n_blocks=4)
    x = torch.randn(1, 1, 256)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)
