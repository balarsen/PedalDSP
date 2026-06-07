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
