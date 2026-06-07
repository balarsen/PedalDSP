"""Tests for pedal_model.train.scheduler."""
import pytest
import torch
import torch.nn as nn

from pedal_model.train.scheduler import build_scheduler


def _simple_optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    param = nn.Parameter(torch.zeros(1))
    return torch.optim.Adam([param], lr=lr)


class TestBuildScheduler:
    def test_returns_sequential_lr(self):
        opt = _simple_optimizer()
        sched = build_scheduler(opt, n_epochs=100)
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    def test_lr_starts_low_during_warmup(self):
        """Initial LR should be well below the base LR during linear warmup."""
        base_lr = 1e-3
        opt = _simple_optimizer(lr=base_lr)
        sched = build_scheduler(opt, n_epochs=100, warmup_epochs=10)
        initial_lr = sched.get_last_lr()[0]
        assert initial_lr < base_lr * 0.5

    def test_lr_rises_through_warmup(self):
        """LR should increase during the warmup phase."""
        opt = _simple_optimizer(lr=1e-3)
        sched = build_scheduler(opt, n_epochs=50, warmup_epochs=5)
        lrs = []
        for _ in range(5):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        # LR should generally increase during warmup
        assert lrs[-1] > lrs[0]

    def test_lr_decays_after_warmup(self):
        """LR should decrease during the cosine decay phase."""
        opt = _simple_optimizer(lr=1e-3)
        warmup = 5
        n_epochs = 50
        sched = build_scheduler(opt, n_epochs=n_epochs, warmup_epochs=warmup)
        for _ in range(warmup):
            sched.step()
        lr_after_warmup = sched.get_last_lr()[0]
        for _ in range(20):
            sched.step()
        lr_late = sched.get_last_lr()[0]
        assert lr_late < lr_after_warmup

    def test_lr_never_negative(self):
        """LR must remain non-negative throughout training."""
        opt = _simple_optimizer(lr=1e-3)
        sched = build_scheduler(opt, n_epochs=100, warmup_epochs=5)
        for _ in range(100):
            assert sched.get_last_lr()[0] >= 0.0
            sched.step()

    def test_minimum_lr_approaches_eta_min(self):
        """At the end of cosine decay the LR should be close to eta_min=1e-6."""
        opt = _simple_optimizer(lr=1e-3)
        n_epochs = 200
        sched = build_scheduler(opt, n_epochs=n_epochs, warmup_epochs=5)
        for _ in range(n_epochs):
            sched.step()
        final_lr = sched.get_last_lr()[0]
        assert final_lr < 5e-5

    def test_single_epoch_no_crash(self):
        """n_epochs=1 should not raise (degenerate but valid input)."""
        opt = _simple_optimizer()
        sched = build_scheduler(opt, n_epochs=1, warmup_epochs=0)
        sched.step()

    def test_step_compatible_with_optimizer(self):
        """Calling scheduler.step() after optimizer.step() should not raise."""
        param = nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([param], lr=1e-3)
        sched = build_scheduler(opt, n_epochs=20)
        loss = param.sum()
        loss.backward()
        opt.step()
        sched.step()
