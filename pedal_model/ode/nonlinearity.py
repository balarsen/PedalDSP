"""Nonlinearity models for physics-informed Wiener-Hammerstein fitting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class OddSymmetricMLP(nn.Module):
    """Structurally odd-symmetric MLP nonlinearity: NL(x) = x · g(x²).

    Odd symmetry is enforced by construction — not via a regularization penalty.
    This guarantees NL(0)=0, NL(−x)=−NL(x), and NL′(0)>0.

    Architecture: g is a 3-layer MLP with hidden_size neurons. The final
    activation is softplus (ensures g(x²)>0, so NL has positive slope at origin).

    Args:
        hidden_size: Width of the two hidden layers.
    """

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),  # g(x²) > 0 always
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the odd-symmetric nonlinearity pointwise.

        Args:
            x: Input signal, shape (N,) or (B, N).

        Returns:
            NL(x) = x · g(x²), same shape as x.
        """
        flat = x.reshape(-1, 1)
        g_val = self.g(flat * flat).reshape(x.shape)
        return x * g_val

    def plot_shape(
        self,
        x_range: tuple[float, float] = (-2.0, 2.0),
        n_points: int = 400,
        ax: object | None = None,
    ) -> object:
        """Plot NL(x) vs tanh(x) on a given x range.

        Args:
            x_range: (x_min, x_max) for the plot domain.
            n_points: Number of evaluation points.
            ax: Matplotlib Axes to plot into (creates new figure if None).

        Returns:
            The Axes object.
        """
        import matplotlib.pyplot as plt

        x_np = np.linspace(x_range[0], x_range[1], n_points, dtype=np.float32)
        x_t = torch.from_numpy(x_np)
        with torch.no_grad():
            y_nl = self.forward(x_t).numpy()

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x_np, y_nl, label="NL(x)")
        ax.plot(x_np, np.tanh(x_np), "--", label="tanh(x)", alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("NL(x)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def r2_vs_tanh(
        self,
        x_range: tuple[float, float] = (-1.0, 1.0),
        n_points: int = 1000,
    ) -> float:
        """R² of the fitted NL against tanh on the given x range.

        Args:
            x_range: Evaluation domain.
            n_points: Number of evaluation points.

        Returns:
            R² coefficient of determination (1.0 = perfect match).
        """
        device = next(self.parameters()).device
        x_t = torch.linspace(x_range[0], x_range[1], n_points, device=device)
        with torch.no_grad():
            y_pred = self.forward(x_t).cpu().numpy()
        y_true = np.tanh(x_t.cpu().numpy())
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / (ss_tot + 1e-30)


class UnconstrainedMLP(nn.Module):
    """Unconstrained MLP nonlinearity for asymmetric clipping (e.g. real Notaklon).

    No symmetry assumption — the shape emerges entirely from data. Use this
    when the clipping is known to be asymmetric (germanium diodes, etc.).

    Args:
        hidden_size: Width of the two hidden layers.
    """

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the unconstrained nonlinearity pointwise.

        Args:
            x: Input signal, shape (N,) or (B, N).

        Returns:
            NL(x), same shape as x.
        """
        flat = x.reshape(-1, 1)
        return self.net(flat).reshape(x.shape)

    def r2_vs_tanh(
        self,
        x_range: tuple[float, float] = (-1.0, 1.0),
        n_points: int = 1000,
    ) -> float:
        """R² against tanh on the given x range.

        Args:
            x_range: Evaluation domain.
            n_points: Number of evaluation points.

        Returns:
            R² coefficient of determination.
        """
        device = next(self.parameters()).device
        x_t = torch.linspace(x_range[0], x_range[1], n_points, device=device)
        with torch.no_grad():
            y_pred = self.forward(x_t).cpu().numpy()
        y_true = np.tanh(x_t.cpu().numpy())
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / (ss_tot + 1e-30)
