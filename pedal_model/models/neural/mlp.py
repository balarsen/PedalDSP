"""MLP baseline: a context window of dry samples → one wet sample."""
import numpy as np
import torch
import torch.nn as nn

from ..base import PedalModel


class MLPModel(nn.Module, PedalModel):
    """Fully-connected model operating on a fixed-size context window.

    Establishes a lower bound: ignores temporal structure beyond the window,
    so it will struggle with envelope-dependent behaviour.

    Architecture: Linear(R, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 1)
    """

    def __init__(self, receptive_field_samples: int = 1024, hidden: int = 128) -> None:
        """
        Args:
            receptive_field_samples: Context window length R in samples.
            hidden: Width of the first hidden layer.
        """
        nn.Module.__init__(self)
        self._rf = receptive_field_samples
        self.net = nn.Sequential(
            nn.Linear(receptive_field_samples, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    @property
    def name(self) -> str:
        return f"MLP-R{self._rf}"

    @property
    def receptive_field(self) -> int:
        return self._rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one wet sample per context window.

        Args:
            x: Context windows, shape (batch, R).

        Returns:
            Predictions, shape (batch, 1).
        """
        return self.net(x)

    # ------------------------------------------------------------------ #
    # PedalModel interface (NumPy wrappers for eval/compare)              #
    # ------------------------------------------------------------------ #

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        """Not implemented: train via train/trainer.py instead."""
        raise NotImplementedError("Use train/trainer.py to train neural models.")

    def predict(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Batched inference using stride-trick sliding windows.

        Args:
            x: Dry input, shape (N,), float32.
            batch_size: Windows per forward pass. Tune for GPU memory.

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        self.eval()
        R = self._rf
        N = len(x)
        padded = np.pad(x, (R - 1, 0))
        # Zero-copy sliding-window view: shape (N, R)
        stride = padded.strides[0]
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(N, R), strides=(stride, stride)
        )
        out = np.empty(N, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = torch.tensor(
                    windows[start:end].copy(), dtype=torch.float32
                )
                out[start:end] = self(batch).squeeze(-1).cpu().numpy()
        return out
