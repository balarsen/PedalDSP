"""Conditioned TCN: FiLM-modulated dilated causal convolutions with knob conditioning.

FiLM (Feature-wise Linear Modulation) applies a per-block gamma/beta affine
transform to the convolutional feature maps, where gamma and beta are predicted
from the knob vector by a shared MLP. This is architecturally aligned with EQ:
each dilated block captures a frequency band, and the knobs scale which bands
are emphasised — mirroring how EQ bands additively combine in log-frequency.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning
Layer", AAAI 2018. Audio application: Steinmetz & Reiss, "Automatic Multitrack
Mixing with a Differentiable Mixing Console", ICASSP 2021.
"""
import numpy as np
import torch
import torch.nn as nn

from ..base import PedalModel
from .tcn import _TCNBlock


class ConditionedTCNModel(nn.Module, PedalModel):
    """TCN with FiLM knob conditioning.

    One model covers the full slider parameter space. The 8-dim GE-7 knob
    vector is mapped by a shared MLP to per-block gamma/beta pairs that
    scale and shift each block's feature maps after the dilated convolution.

    Interface matches ConditionedLSTMModel: forward(x, knobs, hidden=None)
    returns (output, None) so train_ge7_conditioned.py works without changes.

    Receptive field (n_blocks=10, kernel=3):
        RF = 1 + 2 · Σᵢ (kernel-1) · 2ⁱ = 4095 samples ≈ 85 ms at 48 kHz
    """

    def __init__(
        self,
        n_knobs: int = 8,
        channels: int = 32,
        kernel_size: int = 3,
        n_blocks: int = 10,
        film_hidden: int = 64,
    ) -> None:
        """
        Args:
            n_knobs: Knob vector dimension (8 for GE-7: 7 bands + level).
            channels: Conv channels in each dilated block.
            kernel_size: Dilated conv kernel size. 3 is standard.
            n_blocks: Number of dilated blocks. Doubles dilation each block.
            film_hidden: Hidden units in the FiLM MLP.
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

        # Shared FiLM MLP: n_knobs → film_hidden → n_blocks × 2 × channels
        # Last layer initialised near-zero so gamma ≈ 1 and beta ≈ 0 at startup,
        # giving a near-passthrough model that learns EQ deviations from there.
        # Without this, random gamma values compound across 10 blocks (up to 2^10×
        # amplification), causing ESR explosions in the first few epochs.
        _film_out = nn.Linear(film_hidden, n_blocks * 2 * channels)
        nn.init.normal_(_film_out.weight, std=0.01)
        nn.init.zeros_(_film_out.bias)
        self.film_mlp = nn.Sequential(
            nn.Linear(n_knobs, film_hidden),
            nn.GELU(),
            _film_out,
        )

    @property
    def name(self) -> str:
        return f"ConditionedTCN-C{self.channels}x{self.n_blocks}"

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(
            (self.kernel_size - 1) * (2 ** i) for i in range(self.n_blocks)
        )

    def forward(
        self,
        x: torch.Tensor,
        knobs: torch.Tensor,
        hidden: None = None,
    ) -> tuple[torch.Tensor, None]:
        """Process an audio chunk with knob conditioning.

        Args:
            x: Audio chunk, shape (batch, time, 1), float32.
            knobs: Knob positions, shape (batch, n_knobs), values in [-1, 1].
            hidden: Unused — present for interface compatibility with
                ConditionedLSTMModel. Always returns None.

        Returns:
            (output, None) — output shape (batch, time, 1).
        """
        # Compute all FiLM parameters in one MLP forward pass
        film = self.film_mlp(knobs)                          # (B, n_blocks*2*C)
        film = film.view(-1, self.n_blocks, 2, self.channels)
        gammas = 1.0 + film[:, :, 0, :]   # (B, n_blocks, C) — near 1.0 at init
        betas  = film[:, :, 1, :]          # (B, n_blocks, C)

        y = x.permute(0, 2, 1)            # (B, 1, T)
        y = self.input_conv(y)             # (B, C, T)

        skip_sum = torch.zeros_like(y)
        for i, block in enumerate(self.blocks):
            y, skip = block(y)
            g = gammas[:, i, :].unsqueeze(-1)   # (B, C, 1)
            b = betas[:, i, :].unsqueeze(-1)
            y    = g * y + b
            skip = g * skip + b
            skip_sum = skip_sum + skip

        out = self.output_conv(skip_sum)   # (B, 1, T)
        return out.permute(0, 2, 1), None  # (B, T, 1), None

    def predict(
        self,
        x: np.ndarray,
        knobs: np.ndarray,
        chunk_size: int = 131072,
    ) -> np.ndarray:
        """Chunked inference with a fixed knob position.

        Each chunk is prefixed with receptive_field samples of context so the
        causal convolutions see the correct signal history at every position.

        Args:
            x: Dry input, shape (N,), float32.
            knobs: Normalised knob vector, shape (n_knobs,), float32, in [-1, 1].
            chunk_size: Output samples per chunk. Default 131072 (≈2.7 s at 48 kHz).

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        self.eval()
        device = next(self.parameters()).device
        rf = self.receptive_field
        out = np.zeros(len(x), dtype=np.float32)
        k = torch.tensor(knobs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            for start in range(0, len(x), chunk_size):
                ctx_start = max(0, start - rf)
                chunk = x[ctx_start : start + chunk_size]
                t = (torch.tensor(chunk, dtype=torch.float32)
                     .unsqueeze(0).unsqueeze(-1).to(device))   # (1, T, 1)
                pred, _ = self(t, k)
                pred_np = pred.squeeze().cpu().numpy().astype(np.float32)
                offset = start - ctx_start
                out[start : start + chunk_size] = pred_np[offset : offset + chunk_size]
        return out

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        raise NotImplementedError("Use train_ge7_conditioned.py to train.")
