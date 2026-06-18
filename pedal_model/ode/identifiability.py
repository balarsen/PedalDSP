"""Identifiability utilities for Wiener-Hammerstein parameter recovery."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


# ── Amplitude-stratified section selection ────────────────────────────────────


def select_amplitude_sections(
    dry: np.ndarray,
    wet: np.ndarray,
    sections: list[dict],
    amplitude_threshold_dbfs: float,
    above: bool = False,
    section_types: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract audio windows from manifest sections at the given amplitude level.

    Args:
        dry: Full dry audio, shape (N,), float32.
        wet: Full wet audio, shape (N,), float32.
        sections: List of section dicts from the JSON manifest, each with keys
            'start_sample', 'end_sample', 'amplitude_dbfs', and 'type'.
        amplitude_threshold_dbfs: dBFS threshold.
        above: If False (default), select sections at or below the threshold.
            If True, select sections at or above the threshold.
        section_types: If given, only include sections whose 'type' is in this
            list. If None, all types are accepted.

    Returns:
        (dry_sel, wet_sel): Concatenated dry and wet audio from selected sections.
    """
    dry_parts: list[np.ndarray] = []
    wet_parts: list[np.ndarray] = []

    for sec in sections:
        amp = float(sec.get("amplitude_dbfs", 0.0))
        qualified = amp >= amplitude_threshold_dbfs if above else amp <= amplitude_threshold_dbfs
        if not qualified:
            continue
        if section_types is not None and sec.get("type") not in section_types:
            continue
        start = int(sec["start_sample"])
        end = int(sec["end_sample"])
        dry_parts.append(dry[start:end])
        wet_parts.append(wet[start:end])

    if not dry_parts:
        raise ValueError(
            f"No sections found matching amplitude {'>' if above else '<='} "
            f"{amplitude_threshold_dbfs} dBFS"
            + (f" and types {section_types}" if section_types else "")
        )

    return np.concatenate(dry_parts), np.concatenate(wet_parts)


# ── Transfer function estimation ──────────────────────────────────────────────


def compute_transfer_function(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    n_fft: int = 8192,
    hop: int = 4096,
    min_dry_rms: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the system transfer function via Welch-style windowed FFT averaging.

    H(f) is estimated as the average of FFT(wet[i]) / FFT(dry[i]) over
    non-overlapping Hann-windowed frames where the dry RMS exceeds min_dry_rms.

    Args:
        dry: Dry signal, shape (N,), float32.
        wet: Wet signal, shape (N,).
        sr: Sample rate in Hz.
        n_fft: FFT size.
        hop: Hop size between frames.
        min_dry_rms: Minimum dry frame RMS for inclusion (excludes silence).

    Returns:
        (freqs_hz, H_avg) where:
            freqs_hz: Frequency axis in Hz, shape (n_fft//2 + 1,).
            H_avg: Complex transfer function estimate, shape (n_fft//2 + 1,).
    """
    window = np.hanning(n_fft).astype(np.float64)
    # H1 cross-spectrum estimator: H = Σ(W·conj(D)) / Σ|D|²
    # Frame-by-frame W/D averaging blows up for sweep signals because most bins
    # have near-zero dry energy in any given frame; the cross-spectrum form is
    # stable because numerator and denominator are both small at those bins.
    cross_sum = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    power_sum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    count = 0

    N = len(dry)
    for start in range(0, N - n_fft + 1, hop):
        d_frame = dry[start : start + n_fft].astype(np.float64) * window
        w_frame = wet[start : start + n_fft].astype(np.float64) * window
        rms_dry = float(np.sqrt(np.mean(d_frame ** 2)))
        if rms_dry < min_dry_rms:
            continue
        D = np.fft.rfft(d_frame)
        W = np.fft.rfft(w_frame)
        cross_sum += W * np.conj(D)
        power_sum += np.abs(D) ** 2
        count += 1

    if count == 0:
        raise ValueError("No usable frames found: all dry frames are below min_dry_rms.")

    H_avg = cross_sum / np.maximum(power_sum, 1e-20)
    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return freqs_hz.astype(np.float32), H_avg.astype(np.complex64)


# ── Energy-balance anchor for drive/output_level degeneracy ──────────────────


def compute_energy_balance_anchor(
    nl_output: np.ndarray,
    inv_post_wet: np.ndarray,
) -> float:
    """Compute the unique output_level satisfying the energy-balance constraint.

    Resolves the drive × output_level scaling degeneracy. Given NL(pre_filter(dry))
    and the inverse-post-filtered wet signal, output_level is the unique scalar
    that makes their RMS match:

        output_level = RMS(inv_post_wet) / RMS(nl_output)

    Args:
        nl_output: NL(z[n]) where z = pre_filter(dry) × drive, shape (N,).
        inv_post_wet: Inverse post-filter applied to wet signal, shape (N,).

    Returns:
        Estimated output_level scalar in (0, ∞).
    """
    rms_nl = float(np.sqrt(np.mean(nl_output ** 2)))
    rms_inv = float(np.sqrt(np.mean(inv_post_wet ** 2)))
    if rms_nl < 1e-10:
        raise ValueError("NL output RMS is effectively zero — check drive / NL params.")
    return rms_inv / rms_nl


# ── Wiener inverse filter ─────────────────────────────────────────────────────


def wiener_inverse_filter(
    h: np.ndarray,
    n: int,
    beta: float = 1e-4,
) -> np.ndarray:
    """Minimum-phase Wiener deconvolution filter for post-filter inversion.

    Computes h_inv such that h_inv ★ h ≈ δ, regularised to avoid noise
    amplification at frequencies where |H(f)| is small.

    Args:
        h: Impulse response of the filter to invert, shape (L,).
        n: Output length of the inverse filter.
        beta: Regularisation parameter (larger = smoother inversion).

    Returns:
        Inverse filter impulse response, shape (n,).
    """
    n_fft = max(2 * n, 2 * len(h))
    H = np.fft.rfft(h, n=n_fft)
    H_inv = np.conj(H) / (np.abs(H) ** 2 + beta)
    h_inv = np.fft.irfft(H_inv, n=n_fft)
    return h_inv[:n].astype(np.float32)


# ── Linear-regime nonlinearity error estimate ─────────────────────────────────


def tanh_linearity_error(drive: float, amplitude_dbfs: float) -> float:
    """Estimate the fractional error of linearizing tanh(drive·x) at the given amplitude.

    For small x, tanh(x) ≈ x − x³/3. The fractional third-order error is
    approximately |drive·x|²/3.

    Args:
        drive: Pre-gain scalar.
        amplitude_dbfs: Peak input level in dBFS.

    Returns:
        Fractional nonlinearity error (0 = perfectly linear, 1 = fully saturated).
    """
    amplitude_linear = 10.0 ** (amplitude_dbfs / 20.0)
    x_at_tanh = drive * amplitude_linear
    # Relative error from cubic term: (x³/3) / x = x²/3
    return min(float(x_at_tanh ** 2 / 3.0), 1.0)
