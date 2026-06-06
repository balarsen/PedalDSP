"""Harmonic distortion metrics."""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq


def _harmonic_amplitudes(signal: np.ndarray, f0: float, sr: int, n_harmonics: int = 8) -> np.ndarray:
    """Extract amplitudes at f0, 2*f0, ..., n_harmonics*f0 via DFT peak picking.

    Args:
        signal: Audio signal, shape (N,), float32.
        f0: Fundamental frequency in Hz.
        sr: Sample rate in Hz.
        n_harmonics: Number of harmonics to extract (including fundamental).

    Returns:
        Array of amplitudes, shape (n_harmonics,).
    """
    n = len(signal)
    freqs = rfftfreq(n, d=1.0 / sr)
    spectrum = np.abs(rfft(signal * np.hanning(n)))
    bin_width = sr / n

    # half-window of ±2 bins around each harmonic
    half = max(2, int(2 * bin_width))
    amps = np.zeros(n_harmonics)
    for k in range(1, n_harmonics + 1):
        center = k * f0
        mask = (freqs >= center - half) & (freqs <= center + half)
        if mask.any():
            amps[k - 1] = spectrum[mask].max()
    return amps


def compute_thd(signal: np.ndarray, f0: float, sr: int, n_harmonics: int = 8) -> float:
    """Total Harmonic Distortion as a percentage.

    Args:
        signal: Audio signal, shape (N,), float32.
        f0: Fundamental frequency in Hz (e.g. 440.0).
        sr: Sample rate in Hz.
        n_harmonics: Total harmonics to consider (including fundamental).

    Returns:
        THD in percent. 0 = no distortion.
    """
    amps = _harmonic_amplitudes(signal, f0, sr, n_harmonics)
    a1 = amps[0] + 1e-12
    harmonics_power = np.sum(amps[1:] ** 2)
    return float(np.sqrt(harmonics_power) / a1 * 100.0)


def compute_harmonic_profile(signal: np.ndarray, f0: float, sr: int, n_harmonics: int = 8) -> np.ndarray:
    """Harmonic amplitudes normalised to the fundamental.

    Args:
        signal: Audio signal, shape (N,).
        f0: Fundamental frequency in Hz.
        sr: Sample rate in Hz.
        n_harmonics: Number of harmonics (including fundamental).

    Returns:
        Normalised profile, shape (n_harmonics,). First element is always 1.0.
    """
    amps = _harmonic_amplitudes(signal, f0, sr, n_harmonics)
    return amps / (amps[0] + 1e-12)


def compute_hp_similarity(
    target: np.ndarray,
    predicted: np.ndarray,
    f0: float,
    sr: int,
    n_harmonics: int = 8,
) -> float:
    """Cosine similarity between harmonic profiles of target and predicted.

    Args:
        target: Reference audio, shape (N,).
        predicted: Model output, same shape.
        f0: Fundamental frequency in Hz.
        sr: Sample rate in Hz.
        n_harmonics: Number of harmonics to compare.

    Returns:
        Similarity in [-1, 1]. 1.0 = identical harmonic profile.
    """
    p_target = compute_harmonic_profile(target, f0, sr, n_harmonics)
    p_pred = compute_harmonic_profile(predicted, f0, sr, n_harmonics)
    denom = np.linalg.norm(p_target) * np.linalg.norm(p_pred) + 1e-12
    return float(np.dot(p_target, p_pred) / denom)


def compute_thd_pattern_distance(
    target: np.ndarray,
    predicted: np.ndarray,
    f0: float,
    sr: int,
    n_harmonics: int = 8,
) -> float:
    """L2 distance between the normalised harmonic profiles of target and predicted.

    Complementary to HP similarity: captures absolute magnitude differences
    between harmonic profiles (not just their angular separation).
    0 = identical profiles; typical well-matched models: < 0.1.

    Args:
        target: Reference audio, shape (N,).
        predicted: Model output, same shape.
        f0: Fundamental frequency in Hz.
        sr: Sample rate in Hz.
        n_harmonics: Number of harmonics to compare (including fundamental).

    Returns:
        L2 profile distance ∈ [0, ∞).
    """
    p_t = compute_harmonic_profile(target, f0, sr, n_harmonics)
    p_p = compute_harmonic_profile(predicted, f0, sr, n_harmonics)
    return float(np.linalg.norm(p_t - p_p))


def compute_eo_ratio(signal: np.ndarray, f0: float, sr: int) -> float:
    """Even-to-odd harmonic ratio.

    Args:
        signal: Audio signal, shape (N,).
        f0: Fundamental frequency in Hz.
        sr: Sample rate in Hz.

    Returns:
        even_sum / odd_sum. High = tube character, high odd = fuzz character.
    """
    amps = _harmonic_amplitudes(signal, f0, sr, n_harmonics=8)
    # amps[0]=A1, amps[1]=A2, ...
    even = amps[1] + amps[3] + amps[5] + amps[7]  # A2 A4 A6 A8
    odd = amps[2] + amps[4] + amps[6]              # A3 A5 A7
    return float(even / (odd + 1e-12))
