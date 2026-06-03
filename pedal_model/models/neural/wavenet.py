"""Reduced WaveNet: gated residual blocks with dilated causal convolutions."""
import numpy as np
import torch
import torch.nn as nn

from ..base import PedalModel


class _WaveNetBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        pad = dilation  # kernel_size=2 causal padding
        self.dilated_conv = nn.Conv1d(channels, channels * 2, kernel_size=2, dilation=dilation, padding=pad)
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        self._pad = dilation  # trim to restore causal alignment

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.dilated_conv(x)[..., : x.shape[-1]]  # causal trim
        tanh_part, sigmoid_part = y.chunk(2, dim=1)
        gated = torch.tanh(tanh_part) * torch.sigmoid(sigmoid_part)
        skip = self.skip_conv(gated)
        residual = self.res_conv(gated) + x
        return residual, skip


class WaveNetModel(nn.Module, PedalModel):
    """Reduced WaveNet (16–32 channels, 3 stacks of 10 blocks).

    Expected: highest accuracy of all models, slowest inference.
    Use as the quality ceiling reference when comparing model classes.

    Receptive field ≈ 3 stacks × (2^10 − 1) × 2 ≈ 6138 samples (~128ms at 48kHz).
    """

    def __init__(
        self,
        channels: int = 16,
        n_stacks: int = 3,
        n_layers_per_stack: int = 10,
    ) -> None:
        """
        Args:
            channels: Residual + skip channel width.
            n_stacks: Number of dilation stacks (dilations reset to 1 each stack).
            n_layers_per_stack: Layers per stack; dilations = 1, 2, 4, …, 2^(n-1).
        """
        nn.Module.__init__(self)
        self.channels = channels

        self.input_conv = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList()
        for _ in range(n_stacks):
            for i in range(n_layers_per_stack):
                self.blocks.append(_WaveNetBlock(channels, dilation=2 ** i))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, 1),
        )

    @property
    def name(self) -> str:
        return f"WaveNet-C{self.channels}"

    @property
    def receptive_field(self) -> int:
        return sum(2 ** (i % 10) for i in range(len(self.blocks))) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, time)

        Returns:
            Output, shape (batch, 1, time).
        """
        y = self.input_conv(x)
        skip_sum = torch.zeros_like(y)
        for block in self.blocks:
            y, skip = block(y)
            skip_sum = skip_sum + skip
        return self.output(skip_sum)

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        raise NotImplementedError("Use train/trainer.py to train neural models.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            out = self(t)
        return out.squeeze().cpu().numpy().astype(np.float32)
