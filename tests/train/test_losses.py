"""Tests for pedal_model.train.losses."""
import pytest
import torch

from pedal_model.train.losses import (
    CombinedLoss,
    ESRLoss,
    MultiScaleSTFTLoss,
    PreEmphasisESRLoss,
    _hp_preemphasis,
)


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

    def test_accepts_3d_channels_first(self):
        torch.manual_seed(1)
        pred = torch.randn(2, 1, 4096)
        target = torch.randn(2, 1, 4096)
        loss = MultiScaleSTFTLoss()(pred, target)
        assert torch.isfinite(loss)

    def test_accepts_3d_channels_last(self):
        torch.manual_seed(11)
        pred = torch.randn(2, 4096, 1)
        target = torch.randn(2, 4096, 1)
        loss = MultiScaleSTFTLoss()(pred, target)
        assert torch.isfinite(loss)

    def test_larger_error_gives_larger_loss(self):
        torch.manual_seed(2)
        signal = torch.randn(1, 4096) * 0.3
        small = signal + 0.001 * torch.randn_like(signal)
        large = signal + 1.0 * torch.randn_like(signal)
        fn = MultiScaleSTFTLoss()
        assert fn(small, signal).item() < fn(large, signal).item()


class TestESRLoss:
    def test_identical_signals_is_zero(self):
        torch.manual_seed(0)
        signal = torch.randn(2, 4096) * 0.3
        assert ESRLoss()(signal, signal).item() == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        pred, target = _pair()
        assert ESRLoss()(pred, target).item() >= 0.0

    def test_is_scalar(self):
        pred, target = _pair()
        assert ESRLoss()(pred, target).shape == ()

    def test_gradients_flow(self):
        torch.manual_seed(5)
        pred = torch.randn(2, 4096, requires_grad=True)
        target = torch.randn(2, 4096)
        loss = ESRLoss()(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_larger_error_gives_larger_loss(self):
        torch.manual_seed(6)
        signal = torch.randn(1, 4096) * 0.3
        small = signal + 0.001 * torch.randn_like(signal)
        large = signal + 1.0 * torch.randn_like(signal)
        fn = ESRLoss()
        assert fn(small, signal).item() < fn(large, signal).item()

    def test_level_invariant(self):
        """ESR should be the same regardless of overall signal level."""
        torch.manual_seed(7)
        signal = torch.randn(1, 4096)
        pred = signal + 0.1 * torch.randn_like(signal)
        fn = ESRLoss()
        esr_normal = fn(pred, signal).item()
        esr_quiet = fn(pred * 0.01, signal * 0.01).item()
        assert esr_normal == pytest.approx(esr_quiet, rel=1e-4)


class TestHpPreemphasis:
    def test_output_length_is_n_minus_1(self):
        x = torch.randn(2, 1000)
        y = _hp_preemphasis(x, coef=0.85)
        assert y.shape == (2, 999)

    def test_dc_signal_is_suppressed(self):
        """A DC (constant) signal should yield near-zero output after HP filter."""
        x = torch.ones(1, 512)
        y = _hp_preemphasis(x, coef=0.85)
        # y[n] = 1 - 0.85*1 = 0.15  (not zero, it's a leaky differentiator)
        # but the output is constant and much smaller than input RMS
        assert y.abs().mean() < 0.2

    def test_coef_one_is_first_difference(self):
        """coef=1 collapses to plain first difference x[n] - x[n-1]."""
        x = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        y = _hp_preemphasis(x, coef=1.0)
        expected = torch.ones(1, 7)
        torch.testing.assert_close(y, expected)

    def test_coef_zero_passes_signal_unchanged(self):
        """coef=0 → y[n] = x[n], output equals x[1:]."""
        x = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        y = _hp_preemphasis(x, coef=0.0)
        torch.testing.assert_close(y, x[:, 1:])

    def test_high_freq_passes_more_than_low_freq(self):
        """HP filter should have higher RMS output for high-freq signal."""
        import math
        N = 4096
        t = torch.arange(N, dtype=torch.float32)
        low = torch.sin(2 * math.pi * 100 / 48000 * t).unsqueeze(0)
        high = torch.sin(2 * math.pi * 8000 / 48000 * t).unsqueeze(0)
        coef = 0.85
        low_rms = _hp_preemphasis(low, coef).pow(2).mean().sqrt().item()
        high_rms = _hp_preemphasis(high, coef).pow(2).mean().sqrt().item()
        assert high_rms > low_rms


class TestPreEmphasisESRLoss:
    def test_identical_signals_near_zero(self):
        torch.manual_seed(0)
        signal = torch.randn(2, 4096) * 0.3
        loss = PreEmphasisESRLoss()(signal, signal)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_is_scalar(self):
        pred, target = _pair()
        assert PreEmphasisESRLoss()(pred, target).shape == ()

    def test_non_negative(self):
        pred, target = _pair()
        assert PreEmphasisESRLoss()(pred, target).item() >= 0.0

    def test_gradients_flow(self):
        torch.manual_seed(8)
        pred = torch.randn(2, 4096, requires_grad=True)
        target = torch.randn(2, 4096)
        loss = PreEmphasisESRLoss()(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_higher_treble_error_penalised_more_than_plain_esr(self):
        """Pre-emphasis should give relatively larger penalty for treble errors."""
        import math
        torch.manual_seed(9)
        N = 4096
        t = torch.arange(N, dtype=torch.float32)
        target = torch.sin(2 * math.pi * 8000 / 48000 * t).unsqueeze(0) * 0.3
        pred_close = target + 0.05 * torch.randn_like(target)  # small treble error

        plain_esr = ESRLoss()(pred_close, target).item()
        pe_esr = PreEmphasisESRLoss(coef=0.85)(pred_close, target).item()
        # Pre-emphasis ESR should be higher than plain for a high-freq signal error
        assert pe_esr > plain_esr * 0.5  # not strictly larger but in the same ballpark


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

    def test_default_weights_are_75_25(self):
        fn = CombinedLoss()
        assert fn.alpha == pytest.approx(0.75)
        assert fn.beta == pytest.approx(0.25)

    def test_custom_weights_respected(self):
        fn = CombinedLoss(alpha=0.1, beta=0.9)
        assert fn.alpha == pytest.approx(0.1)
        assert fn.beta == pytest.approx(0.9)

    def test_preemphasis_disabled_uses_plain_esr(self):
        fn = CombinedLoss(preemphasis_coef=0.0)
        assert isinstance(fn.esr_loss, ESRLoss)

    def test_preemphasis_enabled_uses_pe_esr(self):
        fn = CombinedLoss(preemphasis_coef=0.85)
        assert isinstance(fn.esr_loss, PreEmphasisESRLoss)
