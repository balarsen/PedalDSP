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

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference over a full signal sample-by-sample.

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        self.eval()
        R = self._rf
        out = np.zeros(len(x), dtype=np.float32)
        padded = np.pad(x, (R - 1, 0))
        with torch.no_grad():
            for n in range(len(x)):
                window = torch.tensor(padded[n : n + R], dtype=torch.float32).unsqueeze(0)
                out[n] = self(window).item()
        return out
