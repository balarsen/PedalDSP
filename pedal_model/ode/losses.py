"""Loss functions for PedalODE KlonSim parameter recovery."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from pedal_model.ode.nonlinearity import OddSymmetricMLP, UnconstrainedMLP


# ── Stage 1: frequency-domain loss ───────────────────────────────────────────


def log_magnitude_loss(
    log_mag_model: Tensor,
    log_mag_measured: Tensor,
    freq_weights: Tensor,
) -> Tensor:
    """Weighted mean-squared error between log-magnitude transfer functions.

    Args:
        log_mag_model: Predicted log|H(f)|, shape (M,). Must have grad.
        log_mag_measured: Target log|H_measured(f)|, shape (M,). No grad needed.
        freq_weights: Non-negative frequency weights w(f), shape (M,).

    Returns:
        Scalar loss L₁ = mean_f[ w(f) · (log|H_model| − log|H_meas|)² ].
    """
    diff_sq = (log_mag_model - log_mag_measured) ** 2
    return (freq_weights * diff_sq).mean()


def make_frequency_weights(freqs_hz: Tensor, sr: int) -> Tensor:
    """Build frequency weights for Stage 1: zero below 20 Hz, unit 20–10 kHz, taper to 0 at 15 kHz.

    Args:
        freqs_hz: Frequency axis in Hz, shape (M,).
        sr: Sample rate in Hz.

    Returns:
        Non-negative weights, shape (M,), same device as freqs_hz.
    """
    w = torch.ones_like(freqs_hz)
    # Zero below 50 Hz — bins below 50 Hz are biased by NL harmonic contamination
    # even at -24 dBFS (drive=4 puts ~0.25 at tanh input → 23% nonlinearity error).
    # Step 0 Bode curve_fit using 50 Hz+ correctly recovers fc1=81 Hz vs 64 Hz here.
    w = torch.where(freqs_hz < 50.0, torch.zeros_like(w), w)
    # Taper from 10 kHz to 15 kHz (linear ramp to zero)
    ramp_start, ramp_end = 10_000.0, 15_000.0
    in_taper = (freqs_hz >= ramp_start) & (freqs_hz < ramp_end)
    taper = 1.0 - (freqs_hz - ramp_start) / (ramp_end - ramp_start)
    w = torch.where(in_taper, taper.clamp(0.0, 1.0), w)
    # Zero above 15 kHz
    w = torch.where(freqs_hz >= ramp_end, torch.zeros_like(w), w)
    # Zero above Nyquist
    nyq = sr / 2.0
    w = torch.where(freqs_hz > nyq, torch.zeros_like(w), w)
    return w


# ── Stage 2 / Stage 3: time-domain loss ──────────────────────────────────────


def time_domain_mse(y_hat: Tensor, y_wet: Tensor) -> Tensor:
    """Standard time-domain mean squared error.

    Args:
        y_hat: Model prediction, shape (N,).
        y_wet: Target signal, shape (N,).

    Returns:
        Scalar MSE.
    """
    return F.mse_loss(y_hat, y_wet)


# ── Stage 2: NL smoothness regularizer ───────────────────────────────────────


def nl_smoothness_loss(
    nl: OddSymmetricMLP | UnconstrainedMLP,
    x_range: tuple[float, float] = (-2.0, 2.0),
    n_points: int = 200,
) -> Tensor:
    """Second-derivative smoothness penalty on the nonlinearity.

    Approximates ∫ [NL''(x)]² dx via finite differences on a uniform grid.

    Args:
        nl: Nonlinearity module (OddSymmetricMLP or UnconstrainedMLP).
        x_range: Evaluation domain for the penalty.
        n_points: Number of equally-spaced evaluation points.

    Returns:
        Scalar smoothness penalty.
    """
    device = next(nl.parameters()).device
    dtype = next(nl.parameters()).dtype
    x = torch.linspace(x_range[0], x_range[1], n_points, device=device, dtype=dtype)
    y = nl(x)  # (n_points,)
    # Central-difference second derivative
    d2y = y[2:] - 2.0 * y[1:-1] + y[:-2]
    return (d2y ** 2).mean()


# ── Stage 3: physics constraint penalties ─────────────────────────────────────


def _soft_penalty(val: Tensor, lower: float | None = None, upper: float | None = None) -> Tensor:
    """ReLU² penalty for val out of [lower, upper].

    Args:
        val: Scalar tensor to constrain.
        lower: Lower bound (None = unconstrained below).
        upper: Upper bound (None = unconstrained above).

    Returns:
        Scalar non-negative penalty.
    """
    penalty = torch.zeros(1, device=val.device, dtype=val.dtype)
    if lower is not None:
        penalty = penalty + F.relu(lower - val) ** 2
    if upper is not None:
        penalty = penalty + F.relu(val - upper) ** 2
    return penalty.squeeze()


def physics_penalty(
    fc1_hz: float | Tensor,
    fc2_hz: float | Tensor,
    drive: float | Tensor,
    output_level: float | Tensor,
) -> Tensor:
    """Soft ReLU² physics constraint penalties for Stage 3.

    Args:
        fc1_hz: Pre-filter cutoff frequency in Hz.
        fc2_hz: Post-filter cutoff frequency in Hz.
        drive: Pre-gain scalar.
        output_level: Output gain scalar.

    Returns:
        Scalar penalty (zero when all constraints satisfied).
    """
    def _t(v: float | Tensor) -> Tensor:
        if isinstance(v, float):
            return torch.tensor(v)
        return v

    p = (
        _soft_penalty(_t(fc1_hz), lower=20.0, upper=500.0)
        + _soft_penalty(_t(fc2_hz), lower=100.0, upper=2000.0)
        + _soft_penalty(_t(drive), lower=1.0)
        + _soft_penalty(_t(output_level), upper=1.0)
    )
    return p


# ── Stage 3: composite loss ───────────────────────────────────────────────────


def composite_loss_stage3(
    y_hat: Tensor,
    y_wet: Tensor,
    nl: OddSymmetricMLP | UnconstrainedMLP,
    fc1_hz: Tensor,
    fc2_hz: Tensor,
    drive: Tensor,
    output_level: Tensor,
    sr: int,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 1e-5,
    delta: float = 0.01,
) -> tuple[Tensor, dict[str, float]]:
    """Composite Stage 3 loss: time MSE + spectral + NL smoothness + physics.

    Args:
        y_hat: Model prediction, shape (N,).
        y_wet: Target signal, shape (N,).
        nl: Nonlinearity module.
        fc1_hz: Pre-filter cutoff frequency (scalar Tensor with grad).
        fc2_hz: Post-filter cutoff frequency (scalar Tensor with grad).
        drive: Drive scalar (scalar Tensor with grad).
        output_level: Output level scalar (scalar Tensor with grad).
        sr: Sample rate in Hz.
        alpha: Weight for time-domain MSE (primary loss).
        beta: Weight for log-magnitude spectral loss.
        gamma: Weight for NL smoothness penalty.
        delta: Weight for physics constraint penalty.

    Returns:
        (total_loss, loss_components) where loss_components is a dict with
        keys 'time', 'spectral', 'smooth', 'physics'.
    """
    # Time domain
    l_time = time_domain_mse(y_hat, y_wet)

    # Spectral: log-magnitude MSE via FFT
    N = y_hat.shape[0]
    Y_hat = torch.fft.rfft(y_hat, n=N)
    Y_wet = torch.fft.rfft(y_wet.detach(), n=N)
    eps = 1e-6
    log_mag_hat = torch.log(Y_hat.abs() + eps)
    log_mag_wet = torch.log(Y_wet.abs() + eps)
    l_spectral = F.mse_loss(log_mag_hat, log_mag_wet)

    # NL smoothness
    l_smooth = nl_smoothness_loss(nl)

    # Physics constraints
    l_phys = physics_penalty(fc1_hz, fc2_hz, drive, output_level)

    total = alpha * l_time + beta * l_spectral + gamma * l_smooth + delta * l_phys

    components = {
        "time": float(l_time.detach()),
        "spectral": float(l_spectral.detach()),
        "smooth": float(l_smooth.detach()),
        "physics": float(l_phys.detach()),
    }
    return total, components
