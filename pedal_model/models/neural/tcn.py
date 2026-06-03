"""Temporal Convolutional Network: dilated causal convolutions, fully parallel training."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import PedalModel


class _TCNBlock(nn.Module):
    """Single dilated residual block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self._causal_trim = pad  # remove non-causal future samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)

        Returns:
            Residual output, same shape.
        """
        y = self.conv(x)[..., : x.shape[-1]]  # causal trim
        # LayerNorm over channel dim: transpose to (batch, time, channels)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return self.act(y) + x


class TCNModel(nn.Module, PedalModel):
    """TCN with exponentially growing dilations.

    Receptive field with n_blocks=10, kernel=3, doubling dilations:
        RF = 1 + 2 · Σᵢ (kernel-1) · 2ⁱ  ≈ 4095 samples (~85ms at 48kHz)

    5–10× faster to train than LSTM because all timesteps are parallel.
    """

    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 3,
        n_blocks: int = 10,
    ) -> None:
        """
        Args:
            channels: Number of convolutional channels.
            kernel_size: Kernel size (3 is standard).
            n_blocks: Number of dilated blocks. Receptive field doubles each block.
        """
        nn.Module.__init__(self)
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks

        self.input_conv = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList([
            _TCNBlock(channels, kernel_size, dilation=2 ** i)
            for i in range(n_blocks)
        ])
        self.output_conv = nn.Conv1d(channels, 1, 1)

    @property
    def name(self) -> str:
        return f"TCN-C{self.channels}x{self.n_blocks}"

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum((self.kernel_size - 1) * (2 ** i) for i in range(self.n_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, time)

        Returns:
            Output, shape (batch, 1, time).
        """
        y = self.input_conv(x)
        for block in self.blocks:
            y = block(y)
        return self.output_conv(y)

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        raise NotImplementedError("Use train/trainer.py to train neural models.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Full-signal inference (causal — no look-ahead).

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            out = self(t)
        return out.squeeze().cpu().numpy().astype(np.float32)
