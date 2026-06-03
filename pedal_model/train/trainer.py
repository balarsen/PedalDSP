"""Training loop with checkpointing, CUDA support, and validation ESR logging."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ..metrics.time_domain import compute_esr
from .losses import CombinedLoss
from .scheduler import build_scheduler


class Trainer:
    """Trains any nn.Module pedal model on a ChunkDataset or PedalDataset.

    Usage::

        from pedal_model.models.neural.lstm import LSTMModel
        from pedal_model.data.dataset import ChunkDataset
        from pedal_model.train.trainer import Trainer

        model = LSTMModel(hidden_size=32)
        dataset = ChunkDataset(dry, wet, chunk_size=2048)
        trainer = Trainer(model, dataset, n_epochs=200)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        n_epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_split: float = 0.1,
        checkpoint_dir: Path | str = "checkpoints",
        device: str | None = None,
    ) -> None:
        """
        Args:
            model: The nn.Module to train.
            dataset: Dataset yielding (input, target) pairs.
            n_epochs: Total training epochs.
            batch_size: Mini-batch size.
            lr: Initial learning rate for AdamW.
            weight_decay: L2 regularisation for AdamW.
            val_split: Fraction of dataset held out for validation.
            checkpoint_dir: Directory to save checkpoints.
            device: 'cuda', 'cpu', or None to auto-detect.
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.loss_fn = CombinedLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = build_scheduler(self.optimizer, n_epochs)

        # Train/val split
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(checkpoint_dir) / model.__class__.__name__ / ts
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)

    def train(self) -> None:
        """Run the full training loop."""
        best_val = float("inf")
        print(f"Training on {self.device}  |  {self.n_epochs} epochs")
        for epoch in range(1, self.n_epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{self.n_epochs}  "
                      f"train={train_loss:.5f}  val={val_loss:.5f}  "
                      f"lr={self.scheduler.get_last_lr()[0]:.2e}")

            if val_loss < best_val:
                best_val = val_loss
                ckpt = self.checkpoint_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "val_loss": val_loss,
                }, ckpt)

        print(f"Training complete. Best val loss: {best_val:.5f}")
        print(f"Checkpoint: {self.checkpoint_dir / 'best.pt'}")
