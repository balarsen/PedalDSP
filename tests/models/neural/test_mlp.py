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
