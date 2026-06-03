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
