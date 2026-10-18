"""Fit one EQ band to a measured response: centre frequency, Q / bandwidth, gain (slider map).

The models are the RBJ Audio EQ Cookbook peaking filter and high shelf, the same family Ge7Sim uses,
so a fitted band can be compared with the model directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import freqz

# Fit range: above the DC-blocking region, below the converter roll-off.
FIT_BAND_HZ = (30.0, 16_000.0)
# A band's centre is searched within this many octaves of its panel frequency. Without the limit a
# peak can wander far outside its band and imitate a shelf (it did, for the 6.4k band, at 14 kHz).
CENTRE_SEARCH_OCT = 1.5
SHELF_SLOPE_RANGE = (
    0.2,
    1.0,
)  # RBJ shelf slope S; 1 = steepest without overshoot (Ge7Sim uses 0.7)
# For a band expected to be a shelf, keep the shelf unless a peak fits this much better (RMS ratio).
SHELF_PREFERENCE = 1.25


@dataclass
class BandFit:
    """Result of fitting one band.

    Attributes:
        model: "peak", "shelf" or "gain" (a flat level change, for the LEVEL slider).
        fc_hz: Centre (peak) or corner (shelf) frequency in Hz; None for "gain".
        q: Peak Q; None for "shelf" and "gain".
        bw_oct: Bandwidth in octaves from Q (RBJ definition: between the gain/2 points in
            dB); None unless "peak".
        gain_db: Gain at the centre (peak), above the corner (shelf), or overall ("gain").
        rms_err_db: RMS difference between the fitted curve and the measurement over FIT_BAND_HZ.
        slope: Shelf slope S; None unless "shelf".
    """

    model: str
    fc_hz: float | None
    q: float | None
    bw_oct: float | None
    gain_db: float
    rms_err_db: float
    slope: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.model == "gain":
            return f"flat gain {self.gain_db:+.2f} dB"
        if self.model == "shelf":
            return (
                f"high shelf at {self.fc_hz:.0f} Hz, slope {self.slope:.2f}, {self.gain_db:+.2f} dB"
            )
        return (
            f"peak at {self.fc_hz:.0f} Hz, Q {self.q:.2f} ({self.bw_oct:.2f} oct), "
            f"{self.gain_db:+.2f} dB"
        )


def q_to_bw_oct(q: float) -> float:
    """RBJ cookbook relation between Q and bandwidth in octaves."""
    return float(2 / np.log(2) * np.arcsinh(1 / (2 * q)))


def _peak_coeffs(fc: float, q: float, gain_db: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return b, den


def _shelf_coeffs(
    fc: float, gain_db: float, slope: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / 2 * np.sqrt((a + 1 / a) * (1 / slope - 1) + 2)
    c, sq = np.cos(w0), 2 * np.sqrt(a) * alpha
    b = np.array(
        [
            a * ((a + 1) + (a - 1) * c + sq),
            -2 * a * ((a - 1) + (a + 1) * c),
            a * ((a + 1) + (a - 1) * c - sq),
        ]
    )
    den = np.array(
        [(a + 1) - (a - 1) * c + sq, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - sq]
    )
    return b, den


def _response_db(b: np.ndarray, a: np.ndarray, f: np.ndarray, sr: int) -> np.ndarray:
    _, h = freqz(b, a, worN=f, fs=sr)
    return 20 * np.log10(np.abs(h) + 1e-12)


def peak_db(f: np.ndarray, fc: float, q: float, gain_db: float, sr: int) -> np.ndarray:
    """Magnitude (dB) of an RBJ peaking filter at frequencies *f* (Hz)."""
    return _response_db(*_peak_coeffs(fc, q, gain_db, sr), f, sr)


def shelf_db(f: np.ndarray, fc: float, gain_db: float, sr: int, slope: float = 1.0) -> np.ndarray:
    """Magnitude (dB) of an RBJ high shelf at frequencies *f* (Hz)."""
    return _response_db(*_shelf_coeffs(fc, gain_db, slope, sr), f, sr)


def fit_band(
    freqs_hz: np.ndarray,
    measured_db: np.ndarray,
    sr: int,
    nominal_hz: float | None,
    nominal_db: float,
    try_shelf: bool = False,
) -> BandFit:
    """Fit one band to a measured, chain-normalised response.

    Args:
        freqs_hz: Frequency grid (log-spaced is best), Hz.
        measured_db: Pedal-only response on that grid, dB (0 dB = unity).
        sr: Sample rate in Hz (the filters are digital, as in Ge7Sim).
        nominal_hz: Panel frequency of the band (the starting point), or None for a level control.
        nominal_db: Slider setting in dB (the starting gain).
        try_shelf: The band is expected to be a high shelf (the GE-7's 6.4k band): fit both and
            keep the shelf unless a peak fits clearly better (SHELF_PREFERENCE).

    Returns:
        The best fit.
    """
    top = min(FIT_BAND_HZ[1], 0.45 * sr)
    sel = (freqs_hz >= FIT_BAND_HZ[0]) & (freqs_hz <= top) & np.isfinite(measured_db)
    f, m = freqs_hz[sel], measured_db[sel]
    if nominal_hz is None:
        g = float(np.median(m))
        return BandFit("gain", None, None, None, g, float(np.sqrt(np.mean((m - g) ** 2))))

    lo = np.log10(max(FIT_BAND_HZ[0], nominal_hz / 2**CENTRE_SEARCH_OCT))
    hi = np.log10(min(top, nominal_hz * 2**CENTRE_SEARCH_OCT))
    start_db = nominal_db or 1.0

    def peak_resid(p):
        return peak_db(f, 10 ** p[0], np.exp(p[1]), p[2], sr) - m

    r = least_squares(
        peak_resid,
        [np.clip(np.log10(nominal_hz), lo, hi), np.log(1.4), start_db],
        bounds=([lo, np.log(0.1), -30], [hi, np.log(20), 30]),
    )
    fc, q, g = 10 ** r.x[0], float(np.exp(r.x[1])), float(r.x[2])
    best = BandFit("peak", float(fc), q, q_to_bw_oct(q), g, float(np.sqrt(np.mean(r.fun**2))))
    if try_shelf:

        def shelf_resid(p):
            return shelf_db(f, 10 ** p[0], p[1], sr, p[2]) - m

        s = least_squares(
            shelf_resid,
            [np.clip(np.log10(nominal_hz), lo, hi), start_db, 0.7],
            bounds=([lo, -30, SHELF_SLOPE_RANGE[0]], [hi, 30, SHELF_SLOPE_RANGE[1]]),
        )
        err = float(np.sqrt(np.mean(s.fun**2)))
        if err <= SHELF_PREFERENCE * best.rms_err_db:
            best = BandFit(
                "shelf", float(10 ** s.x[0]), None, None, float(s.x[1]), err, float(s.x[2])
            )
    return best
