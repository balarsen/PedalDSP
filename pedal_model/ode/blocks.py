"""Continuous-time transfer-function blocks (torch), evaluated on s = jω, for block-diagram models.

Every block takes physical parameters (corner frequencies in Hz, Q, strengths) and returns a complex
response on a frequency grid, so a pedal model is just these blocks multiplied and summed the way
the pedal's block diagram connects them. Nothing here filters audio: rendering to the time domain
(bilinear transform) is a separate step.

Shapes: ``s`` is (F,) complex; per-setting quantities broadcast as (N, ..., F).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

TWO_PI = 2.0 * math.pi


def s_axis(freqs_hz: Tensor) -> Tensor:
    """s = jω for a frequency grid in Hz (complex128)."""
    return 1j * TWO_PI * freqs_hz.to(torch.float64)


def highpass1(s: Tensor, fc_hz: Tensor) -> Tensor:
    """First-order high-pass s / (s + ωc): AC coupling."""
    w = TWO_PI * fc_hz
    return s / (s + w)


def lowpass1(s: Tensor, fc_hz: Tensor) -> Tensor:
    """First-order low-pass ωc / (s + ωc)."""
    w = TWO_PI * fc_hz
    return w / (s + w)


def zero1(s: Tensor, fz_hz: Tensor) -> Tensor:
    """First-order zero 1 + s/ωz: rises 6 dB/octave above fz (the GE-7 input pre-emphasis)."""
    return 1.0 + s / (TWO_PI * fz_hz)


def bandpass2(s: Tensor, fc_hz: Tensor, q: Tensor) -> Tensor:
    """Second-order band-pass (ω/Q)s / (s² + (ω/Q)s + ω²), peak exactly 1 at fc.

    A series R-L-C branch's admittance is exactly a strength times this block, which is why a band
    of a graphic EQ can be described by (fc, Q, strength) without naming parts.
    """
    w = TWO_PI * fc_hz
    bw = w / q
    return bw * s / (s * s + bw * s + w * w)


def pot_split_stage(
    x: Tensor, branches: Tensor, rho_in: Tensor, rho_fb: Tensor, end: Tensor
) -> Tensor:
    """A summing stage with branches hung on pot wipers between its input node and feedback node.

    The GE-7's EQ bank (and, per the teardown, its LEVEL stage): each branch k sits on the wiper of
    a pot whose ends go to the input node (P) and the feedback node (N). With the pot halves as
    conductances g1 = 1/(end + x) toward P and g2 = 1/(end + 1 - x) toward N (in units of the pot's
    resistance), and branch admittance y_k, the ideal-op-amp node equations give

        H = [1 + rho_fb Σ g2 y/(g1 + g2 + y)] / [1 + rho_in Σ g1 y/(g1 + g2 + y)]

    so x = 1 (wiper at N) boosts by 1 + rho_fb y, x = 0 cuts by 1/(1 + rho_in y), x = 0.5 is flat
    when rho_in = rho_fb. The pot halves sit in series with each branch, so the effective Q depends
    on x, and all branches share the two nodes, so they interact.

    Args:
        x: Wiper positions, shape (N, K), in [0, 1] (0 = toward the input node = cut).
        branches: Branch admittances y_k(s), shape (K, F) complex, in units of 1 / pot resistance.
        rho_in: Input resistor / pot resistance (scalar tensor).
        rho_fb: Feedback resistor / pot resistance (scalar tensor).
        end: End resistance of each pot track / pot resistance (scalar tensor), keeps g finite.

    Returns:
        H, shape (N, F) complex.
    """
    g1 = (1.0 / (end + x)).unsqueeze(-1)  # (N, K, 1)
    g2 = (1.0 / (end + 1.0 - x)).unsqueeze(-1)
    y = branches.unsqueeze(0)  # (1, K, F)
    t = y / (g1 + g2 + y)
    num = 1.0 + rho_fb * (g2 * t).sum(dim=1)
    den = 1.0 + rho_in * (g1 * t).sum(dim=1)
    return num / den
