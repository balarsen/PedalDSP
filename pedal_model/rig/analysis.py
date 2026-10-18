"""Measurements on recorded takes: levels, lag, alignment, transfer function, smoothing."""

from __future__ import annotations

import numpy as np
from scipy.signal import csd, welch

from pedal_model.capture.hardware_delay import estimate_lag

EPS = 1e-12


def peak_dbfs(x: np.ndarray) -> float:
    """Sample peak in dBFS (no oversampling: a true-peak meter would read slightly higher)."""
    return float(20 * np.log10(np.max(np.abs(x)) + EPS)) if len(x) else -200.0


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS (a full-scale sine reads −3.01 dBFS)."""
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + EPS)) if len(x) else -200.0


def find_lag(
    ref: np.ndarray, sig: np.ndarray, sr: int, start: int, stop: int, max_lag_s: float = 0.05
) -> int:
    """Lag of *sig* behind *ref* in samples, from cross-correlation over ``[start, stop)``.

    Wraps ``pedal_model.capture.hardware_delay.estimate_lag``. Pick a quiet, broadband window (the
    −24 dBFS sweep) so a nonlinear pedal stays near-linear and the correlation peak is sharp.
    """
    return estimate_lag(ref, sig, sr, start_s=start / sr, end_s=stop / sr, max_lag_s=max_lag_s)


def aligned(rec: np.ndarray, pre_roll: int, lag: int, length: int) -> np.ndarray:
    """Cut the recorded channel to the dry signal's span, removing pre-roll and round-trip lag."""
    start = pre_roll + lag
    out = np.zeros(length, dtype=np.float32)
    seg = rec[max(start, 0) : start + length]
    out[: len(seg)] = seg
    return out


def transfer_function(
    dry: np.ndarray, wet: np.ndarray, sr: int, nperseg: int = 8192
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H1 estimate of the frequency response, |S_dw| / S_dd, plus coherence.

    The H1 estimator averages over Welch segments, so uncorrelated noise on the return averages out
    of the magnitude instead of biasing it upward (as a plain |FFT(wet)| / |FFT(dry)| would).

    Returns:
        (freqs_hz, magnitude_db, coherence in [0, 1]).
    """
    f, s_dd = welch(dry, sr, nperseg=nperseg)
    _, s_ww = welch(wet, sr, nperseg=nperseg)
    _, s_dw = csd(dry, wet, sr, nperseg=nperseg)
    h = np.abs(s_dw) / (s_dd + EPS)
    coh = np.abs(s_dw) ** 2 / (s_dd * s_ww + EPS)
    return f, 20 * np.log10(h + EPS), np.clip(coh, 0.0, 1.0)


def log_grid(fmin: float, fmax: float, points_per_octave: int = 24) -> np.ndarray:
    """Log-spaced frequencies from fmin to fmax."""
    n = int(np.ceil(np.log2(fmax / fmin) * points_per_octave)) + 1
    return np.geomspace(fmin, fmax, n)


def smooth_octave(
    f: np.ndarray, db: np.ndarray, grid: np.ndarray, fraction: float = 1 / 6
) -> np.ndarray:
    """Average *db* in a band of ±fraction/2 octave around each grid frequency (in power, then back
    to dB).
    """
    out = np.empty(len(grid))
    power = 10 ** (db / 10)
    half = 2 ** (fraction / 2)
    for i, fc in enumerate(grid):
        sel = (f >= fc / half) & (f <= fc * half)
        out[i] = 10 * np.log10(np.mean(power[sel]) + EPS) if np.any(sel) else np.interp(fc, f, db)
    return out


def spectrum_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed magnitude spectrum of one block, in dBFS (full-scale sine ≈ 0 dB)."""
    n = len(x)
    if n == 0:
        return np.zeros(1), np.full(1, -200.0)
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * w)) * 2 / np.sum(w)
    return np.fft.rfftfreq(n, 1 / sr), 20 * np.log10(mag + EPS)
