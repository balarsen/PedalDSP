"""Measurements on recorded takes: levels, lag, alignment, transfer function, smoothing."""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate

EPS = 1e-12
# Lag search radius. The 1824c's ASIO round trip measured 3722 samples (77.5 ms) at 48 kHz with
# 1024-sample blocks (2026-10-18), far more than the Studio One-era 47 samples (Studio One
# compensates its own latency). Bigger ASIO buffers add more, hence the margin.
MAX_LAG_S = 0.5
REL_FLOOR = 1e-12  # spectral floor, relative to the peak (see frequency_response)


def peak_dbfs(x: np.ndarray) -> float:
    """Sample peak in dBFS (no oversampling: a true-peak meter would read slightly higher)."""
    return float(20 * np.log10(np.max(np.abs(x)) + EPS)) if len(x) else -200.0


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS (a full-scale sine reads −3.01 dBFS)."""
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + EPS)) if len(x) else -200.0


def find_lag(
    ref: np.ndarray, sig: np.ndarray, sr: int, start: int, stop: int, max_lag_s: float = MAX_LAG_S
) -> int:
    """Delay of *sig* behind *ref* in samples (0 … max_lag), by cross-correlation.

    ``ref[start:stop]`` is slid along ``sig[start : stop + max_lag]``, so the window overlaps fully
    at every candidate lag, and only causal lags are searched: a recording cannot precede what was
    played. Pick a quiet, broadband window (a sweep) so a nonlinear pedal stays near-linear and the
    correlation peak is sharp. A result equal to the max lag means the true delay is longer than the
    search: treat it as a failure.
    """
    max_lag = int(max_lag_s * sr)
    r = ref[start:stop].astype(np.float64)
    s = np.zeros(len(r) + max_lag)
    seg = sig[start : stop + max_lag].astype(np.float64)
    s[: len(seg)] = seg
    c = correlate(s, r, mode="valid", method="fft")  # c[k] = Σ s[n + k] r[n], k = 0 … max_lag
    return int(np.argmax(np.abs(c)))


def aligned(rec: np.ndarray, pre_roll: int, lag: int, length: int) -> np.ndarray:
    """Cut the recorded channel to the dry signal's span, removing pre-roll and round-trip lag."""
    start = pre_roll + lag
    out = np.zeros(length, dtype=np.float32)
    seg = rec[max(start, 0) : start + length]
    out[: len(seg)] = seg
    return out


def frequency_response(
    dry: np.ndarray, wet: np.ndarray, sr: int, nperseg: int = 16384
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured frequency response H(f) = S_dw / S_dd (the H1 estimate), and the coherence.

    Welch averaging: Hann-windowed segments of ``nperseg`` samples, 50 % overlap, each segment's
    mean removed. Per segment D = FFT(dry), W = FFT(wet); then

        S_dd = Σ |D|²,   S_ww = Σ |W|²,   S_dw = Σ conj(D)·W
        H = S_dw / S_dd,   coherence = |S_dw|² / (S_dd·S_ww)

    Averaging the cross-spectrum lets noise on the return average out instead of biasing |H| up.
    H is complex: its phase holds whatever delay is left between dry and wet. Coherence near 1 means
    the wet signal is a linear function of the dry one at that frequency.

    Returns:
        (frequencies in Hz, complex H, coherence in [0, 1]), each nperseg // 2 + 1 long.
    """
    nperseg = min(nperseg, len(dry))
    n = np.arange(nperseg)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * n / nperseg)  # periodic Hann, as scipy uses
    s_dd = np.zeros(nperseg // 2 + 1)
    s_ww = np.zeros(nperseg // 2 + 1)
    s_dw = np.zeros(nperseg // 2 + 1, dtype=complex)
    starts = list(range(0, len(dry) - nperseg + 1, nperseg // 2))
    # One more segment flush with the end, so the last samples are analysed too: a log sweep spends
    # its last ~0.1 s on the top octave, which whole-segment Welch steps would skip (2026-10-20).
    if starts and starts[-1] != len(dry) - nperseg:
        starts.append(len(dry) - nperseg)
    for a in starts:
        d = dry[a : a + nperseg].astype(np.float64)
        w = wet[a : a + nperseg].astype(np.float64)
        big_d = np.fft.rfft((d - d.mean()) * window)
        big_w = np.fft.rfft((w - w.mean()) * window)
        s_dd += np.abs(big_d) ** 2
        s_ww += np.abs(big_w) ** 2
        s_dw += np.conj(big_d) * big_w
    # Floors relative to the peak, not an absolute EPS: a -20 dBFS sweep has S_dd·S_ww ~ 1e-13,
    # which an absolute floor would swamp. Bins the signal never reached read as deep notches.
    floor_dd = REL_FLOOR * s_dd.max() + 1e-300
    floor_prod = REL_FLOOR * (s_dd * s_ww).max() + 1e-300
    h = s_dw / np.maximum(s_dd, floor_dd)
    coherence = np.abs(s_dw) ** 2 / np.maximum(s_dd * s_ww, floor_prod)
    return np.fft.rfftfreq(nperseg, 1 / sr), h, np.clip(coherence, 0.0, 1.0)


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


def log_bin_db(
    f: np.ndarray, db: np.ndarray, points_per_octave: int = 24, fmin: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Average a dB spectrum into log-spaced bins (in power), for display.

    An 8192-point FFT has 4096 bins, almost all above 1 kHz on a log axis; drawing them costs far
    more than the eye can use. Bins with no FFT line (low frequencies, at 1/24 octave) are dropped.

    Returns:
        (bin centre frequencies in Hz, power-averaged dB), about ``points_per_octave`` per octave.
    """
    fmax = float(f[-1])
    n = int(np.ceil(np.log2(fmax / fmin) * points_per_octave)) + 1
    edges = np.geomspace(fmin, fmax, n)
    idx = np.minimum(np.digitize(f, edges) - 1, n - 2)  # the top edge belongs to the last bin
    ok = (idx >= 0) & (idx < n - 1)
    power = np.bincount(idx[ok], weights=10 ** (db[ok] / 10), minlength=n - 1)
    count = np.bincount(idx[ok], minlength=n - 1)
    keep = count > 0
    centres = np.sqrt(edges[:-1] * edges[1:])
    return centres[keep], 10 * np.log10(power[keep] / count[keep] + EPS)


def spectrum_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed magnitude spectrum of one block, in dBFS (full-scale sine ≈ 0 dB)."""
    n = len(x)
    if n == 0:
        return np.zeros(1), np.full(1, -200.0)
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * w)) * 2 / np.sum(w)
    return np.fft.rfftfreq(n, 1 / sr), 20 * np.log10(mag + EPS)
