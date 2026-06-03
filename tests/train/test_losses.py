"""Tests for pedal_model.train.losses."""
import pytest
import torch

from pedal_model.train.losses import CombinedLoss, MultiScaleSTFTLoss


def _pair(batch: int = 2, time: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    target = torch.randn(batch, time) * 0.3
    pred = target + 0.01 * torch.randn_like(target)
    return pred, target


class TestMultiScaleSTFTLoss:
    def test_identical_signals_near_zero(self):
        torch.manual_seed(0)
        signal = torch.randn(2, 4096) * 0.3
        loss_fn = MultiScaleSTFTLoss()
        loss = loss_fn(signal, signal)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_loss_is_scalar(self):
        pred, target = _pair()
        loss = MultiScaleSTFTLoss()(pred, target)
        assert loss.shape == ()

    def test_loss_is_non_negative(self):
        pred, target = _pair()
        loss = MultiScaleSTFTLoss()(pred, target)
        assert loss.item() >= 0.0

    def test_accepts_3d_input(self):
        torch.manual_seed(1)
        pred = torch.randn(2, 1, 4096)
        target = torch.randn(2, 1, 4096)
        loss = MultiScaleSTFTLoss()(pred, target)
        assert torch.isfinite(loss)

    def test_larger_error_gives_larger_loss(self):
        torch.manual_seed(2)
        signal = torch.randn(1, 4096) * 0.3
        small = signal + 0.001 * torch.randn_like(signal)
        large = signal + 1.0 * torch.randn_like(signal)
        fn = MultiScaleSTFTLoss()
        assert fn(small, signal).item() < fn(large, signal).item()


class TestCombinedLoss:
    def test_loss_is_scalar(self):
        pred, target = _pair()
        loss = CombinedLoss()(pred, target)
        assert loss.shape == ()

    def test_loss_is_non_negative(self):
        pred, target = _pair()
        loss = CombinedLoss()(pred, target)
        assert loss.item() >= 0.0

    def test_identical_signals_near_zero(self):
        torch.manual_seed(3)
        signal = torch.randn(2, 4096) * 0.3
        loss = CombinedLoss()(signal, signal)
        assert loss.item() == pytest.approx(0.0, abs=1e-3)

    def test_gradients_flow(self):
        torch.manual_seed(4)
        pred = torch.randn(2, 4096, requires_grad=True)
        target = torch.randn(2, 4096)
        loss = CombinedLoss()(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
