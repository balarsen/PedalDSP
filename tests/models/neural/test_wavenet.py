"""Tests for pedal_model.models.neural.wavenet."""
import numpy as np
import pytest
import torch

from pedal_model.models.neural.wavenet import WaveNetModel


def test_name():
    model = WaveNetModel(channels=16)
    assert "WaveNet" in model.name
    assert "16" in model.name


def test_receptive_field_grows_with_stacks():
    m1 = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    m2 = WaveNetModel(channels=8, n_stacks=2, n_layers_per_stack=4)
    assert m2.receptive_field > m1.receptive_field


def test_receptive_field_exact_value():
    # WaveNet(channels=8, n_stacks=1, n_layers_per_stack=4) verified at runtime = 30
    assert WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4).receptive_field == 30


def test_forward_output_shape():
    model = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    x = torch.randn(2, 1, 128)
    y = model(x)
    assert y.shape == (2, 1, 128)


def test_forward_finite():
    torch.manual_seed(0)
    model = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    x = torch.randn(2, 1, 128)
    y = model(x)
    assert torch.isfinite(y).all()


def test_predict_shape():
    model = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    x = np.random.randn(500).astype(np.float32) * 0.3
    out = model.predict(x)
    assert out.shape == (500,)
    assert out.dtype == np.float32


def test_predict_dtype():
    np.random.seed(0)
    model = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    x = np.random.randn(200).astype(np.float32)
    assert model.predict(x).dtype == np.float32


def test_fit_raises():
    model = WaveNetModel(channels=8, n_stacks=1)
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), 48000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_on_gpu():
    model = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4).cuda()
    x = torch.randn(2, 1, 64, device="cuda")
    y = model(x)
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()


# ── Regression tests ────────────────────────────────────────────────────────


def test_state_dict_roundtrip(tmp_path):
    """Save weights, load into fresh model, outputs must be identical."""
    torch.manual_seed(0)
    model = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    x = torch.randn(1, 1, 128)
    with torch.no_grad():
        out_orig = model(x)

    ckpt = tmp_path / "wavenet.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    model2.load_state_dict(torch.load(ckpt, weights_only=True))
    with torch.no_grad():
        out_loaded = model2(x)

    torch.testing.assert_close(out_orig, out_loaded, atol=0.0, rtol=0.0)


def test_loss_backward():
    """Loss must be finite; every connected parameter must have a finite gradient.

    WaveNet only accumulates skip outputs, so the last block's res_conv is
    intentionally not in the output path — its gradient is expected to be None.
    """
    torch.manual_seed(1)
    model = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    x = torch.randn(2, 1, 128)
    target = torch.randn(2, 1, 128)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    assert torch.isfinite(loss)
    loss.backward()
    params_with_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"{name} gradient contains inf/nan"
            params_with_grads.append(name)
    assert len(params_with_grads) > 0, "No parameter received a gradient"


def test_forward_determinism():
    """Same weights and input must produce identical output on repeated calls."""
    torch.manual_seed(2)
    model = WaveNetModel(channels=8, n_stacks=1, n_layers_per_stack=4)
    x = torch.randn(1, 1, 128)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)
