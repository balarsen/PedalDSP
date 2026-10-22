"""Fit one EQ band to a measured response: centre frequency, Q / bandwidth, gain (slider map).

The models are the analog prototypes of the RBJ Audio EQ Cookbook peaking filter and high shelf,
written as |H(jω)|: continuous time, like the GE-7 block model, so no sample rate is involved.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

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


def band_fit(
    model: str,
    fc_hz: float | None,
    q: float | None,
    gain_db: float,
    rms_err_db: float,
    slope: float | None = None,
) -> dict:
    """One band's fit as a dict, with a one-line summary.

    Keys: model ("peak", "shelf" or "gain": a flat level change, for the LEVEL slider); fc_hz
    (centre for a peak, corner for a shelf, None for gain); q and bw_oct (peak only; bw_oct is the
    RBJ bandwidth between the gain/2 points in dB); gain_db (at the centre, above the corner, or
    overall); rms_err_db (fit vs measurement over FIT_BAND_HZ); slope (shelf slope S, shelf
    only); summary.
    """
    if model == "gain":
        summary = f"flat gain {gain_db:+.2f} dB"
    elif model == "shelf":
        summary = f"high shelf at {fc_hz:.0f} Hz, slope {slope:.2f}, {gain_db:+.2f} dB"
    else:
        summary = f"peak at {fc_hz:.0f} Hz, Q {q:.2f} ({q_to_bw_oct(q):.2f} oct), {gain_db:+.2f} dB"
    return {
        "model": model,
        "fc_hz": fc_hz,
        "q": q,
        "bw_oct": q_to_bw_oct(q) if model == "peak" else None,
        "gain_db": gain_db,
        "rms_err_db": rms_err_db,
        "slope": slope,
        "summary": summary,
    }


def q_to_bw_oct(q: float) -> float:
    """RBJ cookbook relation between Q and bandwidth in octaves."""
    return float(2 / np.log(2) * np.arcsinh(1 / (2 * q)))


def peak_db(f: np.ndarray, fc: float, q: float, gain_db: float) -> np.ndarray:
    """|H(jω)| in dB of the analog peaking EQ (RBJ cookbook prototype), at frequencies f (Hz):

        H(s) = (s² + (A/Q)·s + 1) / (s² + s/(A·Q) + 1),   s = j·f/fc,   A = 10^(gain_dB/40)

    Gain is ``gain_db`` at fc and 0 dB far from it.
    """
    a = 10 ** (gain_db / 40)
    s = 1j * np.asarray(f) / fc
    h = (s * s + s * a / q + 1) / (s * s + s / (a * q) + 1)
    return 20 * np.log10(np.abs(h))


def shelf_db(f: np.ndarray, fc: float, gain_db: float, slope: float = 1.0) -> np.ndarray:
    """|H(jω)| in dB of the analog high shelf (RBJ cookbook prototype), at frequencies f (Hz):

        H(s) = A·(A·s² + (√A/Q)·s + 1) / (s² + (√A/Q)·s + A),   s = j·f/fc,
        1/Q = √((A + 1/A)·(1/S − 1) + 2),   A = 10^(gain_dB/40)

    0 dB at low frequency, ``gain_db`` at high frequency; slope S = 1 is the steepest without
    overshoot.
    """
    a = 10 ** (gain_db / 40)
    inv_q = np.sqrt((a + 1 / a) * (1 / slope - 1) + 2)
    s = 1j * np.asarray(f) / fc
    h = a * (a * s * s + np.sqrt(a) * inv_q * s + 1) / (s * s + np.sqrt(a) * inv_q * s + a)
    return 20 * np.log10(np.abs(h))


def fit_band(
    freqs_hz: np.ndarray,
    measured_db: np.ndarray,
    sr: int,
    nominal_hz: float | None,
    nominal_db: float,
    try_shelf: bool = False,
) -> dict:
    """Fit one band to a measured, chain-normalised response.

    Args:
        freqs_hz: Frequency grid (log-spaced is best), Hz.
        measured_db: Pedal-only response on that grid, dB (0 dB = unity).
        sr: Sample rate in Hz (only to keep the fit below the converter roll-off).
        nominal_hz: Panel frequency of the band (the starting point), or None for a level control.
        nominal_db: Slider setting in dB (the starting gain).
        try_shelf: The band is expected to be a high shelf (the GE-7's 6.4k band): fit both and
            keep the shelf unless a peak fits clearly better (SHELF_PREFERENCE).

    Returns:
        The best fit, as ``band_fit`` describes it.
    """
    top = min(FIT_BAND_HZ[1], 0.45 * sr)
    sel = (freqs_hz >= FIT_BAND_HZ[0]) & (freqs_hz <= top) & np.isfinite(measured_db)
    f, m = freqs_hz[sel], measured_db[sel]
    if nominal_hz is None:
        g = float(np.median(m))
        return band_fit("gain", None, None, g, float(np.sqrt(np.mean((m - g) ** 2))))

    lo = np.log10(max(FIT_BAND_HZ[0], nominal_hz / 2**CENTRE_SEARCH_OCT))
    hi = np.log10(min(top, nominal_hz * 2**CENTRE_SEARCH_OCT))
    start_db = nominal_db or 1.0

    def peak_resid(p):
        return peak_db(f, 10 ** p[0], np.exp(p[1]), p[2]) - m

    r = least_squares(
        peak_resid,
        [np.clip(np.log10(nominal_hz), lo, hi), np.log(1.4), start_db],
        bounds=([lo, np.log(0.1), -30], [hi, np.log(20), 30]),
    )
    fc, q, g = 10 ** r.x[0], float(np.exp(r.x[1])), float(r.x[2])
    best = band_fit("peak", float(fc), q, g, float(np.sqrt(np.mean(r.fun**2))))
    if try_shelf:

        def shelf_resid(p):
            return shelf_db(f, 10 ** p[0], p[1], p[2]) - m

        s = least_squares(
            shelf_resid,
            [np.clip(np.log10(nominal_hz), lo, hi), start_db, 0.7],
            bounds=([lo, -30, SHELF_SLOPE_RANGE[0]], [hi, 30, SHELF_SLOPE_RANGE[1]]),
        )
        err = float(np.sqrt(np.mean(s.fun**2)))
        if err <= SHELF_PREFERENCE * best["rms_err_db"]:
            best = band_fit("shelf", float(10 ** s.x[0]), None, float(s.x[1]), err, float(s.x[2]))
    return best
