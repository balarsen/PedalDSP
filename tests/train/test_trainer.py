"""Tests for pedal_model.train.trainer.Trainer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from pedal_model.train.trainer import Trainer


_CHUNK = 4096  # must be ≥ largest STFT window (2048) for CombinedLoss


class _TinyModel(nn.Module):
    """Minimal model: accepts (batch, chunk, 1) and returns same shape."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _tiny_dataset(n: int = 20) -> TensorDataset:
    """Yield (input, target) pairs of shape (chunk, 1) — one chunk per item."""
    torch.manual_seed(0)
    x = torch.randn(n, _CHUNK, 1) * 0.3
    y = x * 0.5 + 0.05 * torch.randn_like(x)
    return TensorDataset(x, y)


class TestTrainerInit:
    def test_checkpoint_dir_created(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset()
        trainer = Trainer(model, ds, n_epochs=2, checkpoint_dir=tmp_path)
        assert trainer.checkpoint_dir.exists()

    def test_checkpoint_dir_contains_model_class_name(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset()
        trainer = Trainer(model, ds, n_epochs=2, checkpoint_dir=tmp_path)
        assert "_TinyModel" in str(trainer.checkpoint_dir)

    def test_device_is_cpu_when_forced(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset()
        trainer = Trainer(model, ds, n_epochs=2, device="cpu", checkpoint_dir=tmp_path)
        assert trainer.device == torch.device("cpu")

    def test_model_moved_to_device(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset()
        trainer = Trainer(model, ds, n_epochs=2, device="cpu", checkpoint_dir=tmp_path)
        for p in trainer.model.parameters():
            assert p.device == torch.device("cpu")

    def test_train_val_loaders_created(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset(n=12)
        trainer = Trainer(model, ds, n_epochs=2, val_split=0.2, checkpoint_dir=tmp_path)
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None

    def test_cuda_skipped_if_unavailable(self, tmp_path):
        """Auto-detect: should fall back to CPU if no CUDA."""
        if torch.cuda.is_available():
            pytest.skip("CUDA available — auto-detect would pick GPU")
        model = _TinyModel()
        ds = _tiny_dataset()
        trainer = Trainer(model, ds, n_epochs=2, checkpoint_dir=tmp_path)
        assert trainer.device == torch.device("cpu")


class TestTrainerRunEpoch:
    def test_returns_finite_loss(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset()
        trainer = Trainer(model, ds, n_epochs=2, device="cpu",
                          batch_size=4, checkpoint_dir=tmp_path)
        loss = trainer._run_epoch(trainer.train_loader, train=True)
        assert np.isfinite(loss)

    def test_val_epoch_does_not_update_params(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset()
        trainer = Trainer(model, ds, n_epochs=2, device="cpu",
                          batch_size=4, checkpoint_dir=tmp_path)
        params_before = [p.clone() for p in model.parameters()]
        trainer._run_epoch(trainer.val_loader, train=False)
        for before, after in zip(params_before, model.parameters()):
            torch.testing.assert_close(before, after)

    def test_train_epoch_updates_params(self, tmp_path):
        torch.manual_seed(0)
        model = _TinyModel()
        ds = _tiny_dataset(n=10)
        trainer = Trainer(model, ds, n_epochs=2, device="cpu",
                          batch_size=4, checkpoint_dir=tmp_path)
        params_before = [p.clone() for p in model.parameters()]
        trainer._run_epoch(trainer.train_loader, train=True)
        updated = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, model.parameters())
        )
        assert updated


class TestTrainerTrain:
    def test_train_runs_without_error(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset(n=12)
        trainer = Trainer(model, ds, n_epochs=2, device="cpu",
                          batch_size=4, checkpoint_dir=tmp_path)
        trainer.train()

    def test_checkpoint_saved(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset(n=12)
        trainer = Trainer(model, ds, n_epochs=2, device="cpu",
                          batch_size=4, checkpoint_dir=tmp_path)
        trainer.train()
        ckpt = trainer.checkpoint_dir / "best.pt"
        assert ckpt.exists()

    def test_checkpoint_contains_expected_keys(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset(n=12)
        trainer = Trainer(model, ds, n_epochs=2, device="cpu",
                          batch_size=4, checkpoint_dir=tmp_path)
        trainer.train()
        ckpt = torch.load(trainer.checkpoint_dir / "best.pt", map_location="cpu",
                          weights_only=False)
        assert "epoch" in ckpt
        assert "model_state" in ckpt
        assert "val_loss" in ckpt

    def test_val_loss_is_finite(self, tmp_path):
        model = _TinyModel()
        ds = _tiny_dataset(n=12)
        trainer = Trainer(model, ds, n_epochs=2, device="cpu",
                          batch_size=4, checkpoint_dir=tmp_path)
        trainer.train()
        ckpt = torch.load(trainer.checkpoint_dir / "best.pt", map_location="cpu",
                          weights_only=False)
        assert np.isfinite(ckpt["val_loss"])
