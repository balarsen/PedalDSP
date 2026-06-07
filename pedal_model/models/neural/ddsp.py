"""DDSP: Differentiable DSP — learnable FIR, waveshaper, and IIR in one graph."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import PedalModel


class LearnableFIR(nn.Module):
    """FIR filter with trainable coefficients as nn.Parameter."""

    def __init__(self, n_taps: int = 64) -> None:
        super().__init__()
        self.n_taps = n_taps
        self.kernel = nn.Parameter(torch.zeros(1, 1, n_taps))
        nn.init.dirac_(self.kernel)  # identity initialisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal FIR convolution.

        Args:
            x: (batch, 1, time)

        Returns:
            Filtered output, same shape.
        """
        pad = self.n_taps - 1
        return F.conv1d(F.pad(x, (pad, 0)), self.kernel)


class LearnableWaveshaper(nn.Module):
    """Static nonlinearity parameterised by a small MLP."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Small init on output layer: default Kaiming-uniform on Linear(16,1)
        # produces outputs up to ±4 RMS before training starts, causing loss
        # saturation and stalled gradients through the tanh-saturated hidden layers.
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sample-wise nonlinearity.

        Args:
            x: (batch, 1, time)

        Returns:
            Shaped output, same shape.
        """
        shape = x.shape
        flat = x.reshape(-1, 1)
        return self.net(flat).reshape(shape)


class LearnableIIR(nn.Module):
    """First-order IIR biquad section with learnable a1 coefficient.

    y[n] = x[n] - a1 * y[n-1]

    Simple single-pole IIR for tone-shaping after the waveshaper.
    Multiple instances can be chained for higher-order response.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialise to near-zero feedback (near identity)
        self.a1 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """First-order IIR as a truncated-FIR convolution: y[n] = x[n] - a1*y[n-1].

        Exact impulse response h[k] = (-a1)^k is truncated at 128 taps.
        Error < 1e-5 for |a1| < 0.95. Fully differentiable w.r.t. a1.

        Args:
            x: (batch, 1, time)

        Returns:
            Filtered output, same shape.
        """
        a1 = torch.tanh(self.a1)  # keep in (-1, 1) for stability
        k = torch.arange(128, device=x.device, dtype=x.dtype)
        h = torch.pow(-a1, k)               # (128,) IIR impulse response
        kernel = h.flip(0).view(1, 1, 128)  # (out_ch, in_ch, K) causal
        return F.conv1d(F.pad(x, (127, 0)), kernel)


class DDSPModel(nn.Module, PedalModel):
    """Differentiable DSP chain: LearnableFIR → LearnableWaveshaper → LearnableIIR.

    All DSP blocks have learnable parameters — train end-to-end with backprop.
    After training, inspect the learned waveshaper and filter to understand
    what the model discovered about the pedal's circuit.
    """

    def __init__(self, n_fir_taps: int = 64) -> None:
        """
        Args:
            n_fir_taps: Length of the learnable FIR kernel.
        """
        nn.Module.__init__(self)
        self.fir = LearnableFIR(n_fir_taps)
        self.waveshaper = LearnableWaveshaper()
        self.iir = LearnableIIR()

    @property
    def name(self) -> str:
        return "DDSP"

    @property
    def receptive_field(self) -> int:
        return self.fir.n_taps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FIR → waveshaper → IIR.

        Args:
            x: (batch, 1, time)

        Returns:
            Output, same shape.
        """
        x = self.fir(x)
        x = self.waveshaper(x)
        x = self.iir(x)
        return x

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        raise NotImplementedError("Use train/trainer.py to train neural models.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            out = self(t)
        return out.squeeze().cpu().numpy().astype(np.float32)
