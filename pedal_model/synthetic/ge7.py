"""Synthetic Boss GE-7 graphic equalizer — calibrated to hardware captures.

Calibration data from recordings/data/ge7/captures/single_slider/ measurements:
  - Centre frequencies: measured by peak-finding in single-slider frequency response
  - Gain scaling: measured peak gain at +10 dB slider position vs nominal
  - Top band (nominal 6400 Hz): behaves as a high shelf, not a peaking filter
  - LEVEL slider: log-taper pot with non-linear dB response
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import sosfilt

# ── Calibrated band parameters ────────────────────────────────────────────────

# Measured centre frequencies (Hz) from single-slider captures.
# Nominal panel labels: 100, 200, 400, 800, 1600, 3200, 6400 Hz.
GE7_BAND_HZ: tuple[int, ...] = (100, 200, 400, 800, 1600, 3200, 6400)
_BAND_HZ_CALIBRATED: tuple[float, ...] = (87.0, 190.0, 383.0, 792.0, 1599.0, 3380.0)

# Q factor for peaking bands. Octave-spaced graphic EQ → Q ≈ 1.4.
_BAND_Q: float = 1.4

# Per-band gain correction factors: at slider = +10 dB the measured peak gain
# (relative to flat, in dB) was less than 10 dB due to pot taper non-linearity.
# Factor = measured_peak_dB / slider_dB, measured at each band's actual peak.
# Bands 1–6 (100–3200 Hz nominal); band 7 (6400 Hz) uses a separate shelf model.
_BAND_GAIN_FACTOR: tuple[float, ...] = (0.969, 0.861, 0.718, 0.752, 0.741, 0.842)

# High-shelf parameters for band 7 (nominal 6400 Hz).
# The hardware band is a high shelf: gain rises from ~6 kHz, peaks at ~13.2 kHz.
# Shelf frequency = half-gain point; slope S controls transition steepness.
_SHELF_FREQ_HZ: float = 6400.0
_SHELF_SLOPE: float = 0.7    # S parameter: 1 = standard shelving Q, <1 = gentler slope
_SHELF_GAIN_FACTOR: float = 0.717  # measured at +10 dB slider

# LEVEL slider calibration — log-taper pot (quadratic correction):
#   actual_dB = _LEVEL_A × slider_dB + _LEVEL_B × slider_dB²
# Fit from: +10 dB → +12.1 dB actual, −10 dB → −14.4 dB actual
_LEVEL_A: float = 1.325
_LEVEL_B: float = -0.0115

_SLIDER_MIN_DB: float = -15.0
_SLIDER_MAX_DB: float = +15.0


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Ge7Sliders:
    """Physical slider positions for the Boss GE-7 (8 sliders).

    Each value is in dB, range [−15, +15].  0.0 = centre detent = flat.

    Attributes:
        hz_100: 100 Hz band gain in dB.
        hz_200: 200 Hz band gain in dB.
        hz_400: 400 Hz band gain in dB.
        hz_800: 800 Hz band gain in dB.
        hz_1600: 1.6 kHz band gain in dB.
        hz_3200: 3.2 kHz band gain in dB.
        hz_6400: 6.4 kHz band gain in dB.
        level: Output level in dB (±15 dB).
    """

    hz_100: float = 0.0
    hz_200: float = 0.0
    hz_400: float = 0.0
    hz_800: float = 0.0
    hz_1600: float = 0.0
    hz_3200: float = 0.0
    hz_6400: float = 0.0
    level: float = 0.0

    def gains_db(self) -> list[float]:
        """Return the 7 band gains as a list in frequency order."""
        return [
            self.hz_100, self.hz_200, self.hz_400, self.hz_800,
            self.hz_1600, self.hz_3200, self.hz_6400,
        ]

    def as_array(self) -> np.ndarray:
        """Return all 8 slider values as a float32 array [band×7 + level]."""
        return np.array(self.gains_db() + [self.level], dtype=np.float32)

    def clamp(self) -> "Ge7Sliders":
        """Return a copy with all values clamped to [−15, +15] dB."""
        def _c(v: float) -> float:
            return float(np.clip(v, _SLIDER_MIN_DB, _SLIDER_MAX_DB))

        return Ge7Sliders(
            hz_100=_c(self.hz_100), hz_200=_c(self.hz_200),
            hz_400=_c(self.hz_400), hz_800=_c(self.hz_800),
            hz_1600=_c(self.hz_1600), hz_3200=_c(self.hz_3200),
            hz_6400=_c(self.hz_6400), level=_c(self.level),
        )

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "Ge7Sliders":
        """Construct from a plain dict (e.g. parsed from preset JSON).

        Args:
            d: Keys matching Ge7Sliders field names, values in dB.

        Returns:
            Ge7Sliders instance.
        """
        return cls(**{k: float(v) for k, v in d.items()})


@dataclass
class Ge7Params:
    """Internal parameters for Ge7Sim (physical slider values + Q).

    Attributes:
        sliders: Per-band and level gains in dB.
        band_q: Q factor for all peaking biquads (default 1.4 = octave spacing).
        calibrated: If True (default), use measured centre frequencies and gain
            scaling from hardware captures. If False, use nominal panel frequencies
            and no gain correction — useful for testing against the original model.
    """

    sliders: Ge7Sliders = field(default_factory=Ge7Sliders)
    band_q: float = _BAND_Q
    calibrated: bool = True


# ── Main class ────────────────────────────────────────────────────────────────

class Ge7Sim:
    """Calibrated Boss GE-7 7-band graphic equalizer.

    Signal chain:
      Six peaking biquads (bands 1–6, corrected centre frequencies) cascaded
      with one high-shelf biquad (band 7, the 6400 Hz panel label) and a LEVEL
      gain stage with calibrated pot taper correction.

    Calibration vs. nominal model:
      - Bands 1–6 use measured centre frequencies (87–3380 Hz) rather than the
        panel labels (100–3200 Hz).
      - Band 7 uses a high-shelf (Audio EQ Cookbook) instead of a peaking filter.
        The hardware response rises continuously from 6 kHz and never returns to
        unity within the audible range, consistent with a shelf topology.
      - Per-band gain factors correct for the log-taper pot non-linearity.
      - LEVEL slider applies quadratic dB correction fitted to hardware measurements.

    With all sliders at 0 dB the model is unity gain (flat response).
    """

    VERSION = "2.0"

    def __init__(self, params: Ge7Params | None = None) -> None:
        self.params = params or Ge7Params()

    @classmethod
    def from_sliders(
        cls,
        sliders: Ge7Sliders,
        band_q: float = _BAND_Q,
        calibrated: bool = True,
    ) -> "Ge7Sim":
        """Construct directly from a Ge7Sliders instance.

        Args:
            sliders: Per-band gains and output level.
            band_q: Peaking filter Q. Default 1.4 (octave-spaced bands).
            calibrated: Use measured hardware frequencies and gain scaling.

        Returns:
            Ge7Sim instance.
        """
        return cls(Ge7Params(sliders=sliders, band_q=band_q, calibrated=calibrated))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process dry audio through the equalizer.

        Args:
            audio: Dry audio, shape (N,), float32, nominally in [−1, 1].
            sr: Sample rate in Hz.

        Returns:
            Wet audio, same shape, float32.
        """
        p = self.params
        x = audio.astype(np.float64)
        gains = p.sliders.gains_db()

        if p.calibrated:
            # Bands 1–6: peaking EQ at measured centre frequencies with gain scaling
            for i, (freq_hz, gain_db, scale) in enumerate(
                zip(_BAND_HZ_CALIBRATED, gains[:6], _BAND_GAIN_FACTOR)
            ):
                effective_db = gain_db * scale
                if abs(effective_db) < 1e-4:
                    continue
                sos = _peaking_eq_sos(freq_hz, effective_db, p.band_q, sr)
                x = sosfilt(sos, x)

            # Band 7: high shelf at 6400 Hz with gain scaling
            shelf_db = gains[6] * _SHELF_GAIN_FACTOR
            if abs(shelf_db) > 1e-4:
                sos = _high_shelf_sos(_SHELF_FREQ_HZ, shelf_db, _SHELF_SLOPE, sr)
                x = sosfilt(sos, x)

            # LEVEL: quadratic pot taper correction
            x = x * _level_linear(p.sliders.level)
        else:
            # Uncalibrated: nominal frequencies, no gain scaling, linear level
            for freq_hz, gain_db in zip(GE7_BAND_HZ, gains):
                if abs(gain_db) < 1e-4:
                    continue
                sos = _peaking_eq_sos(freq_hz, gain_db, p.band_q, sr)
                x = sosfilt(sos, x)
            x = x * (10.0 ** (p.sliders.level / 20.0))

        return x.astype(np.float32)

    def frequency_response(
        self, sr: int, n_fft: int = 4096
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the magnitude frequency response analytically.

        Args:
            sr: Sample rate in Hz.
            n_fft: Number of frequency points (rfft size).

        Returns:
            ``(freqs_hz, magnitude_db)`` arrays of length ``n_fft // 2 + 1``.
        """
        from scipy.signal import freqz

        p = self.params
        w_points = np.linspace(0, np.pi, n_fft // 2 + 1)
        h_total = np.ones(n_fft // 2 + 1, dtype=complex)

        gains = p.sliders.gains_db()

        if p.calibrated:
            for freq_hz, gain_db, scale in zip(
                _BAND_HZ_CALIBRATED, gains[:6], _BAND_GAIN_FACTOR
            ):
                effective_db = gain_db * scale
                if abs(effective_db) < 1e-4:
                    continue
                for row in _peaking_eq_sos(freq_hz, effective_db, p.band_q, sr):
                    _, h = freqz(row[:3], row[3:], worN=w_points)
                    h_total *= h

            shelf_db = gains[6] * _SHELF_GAIN_FACTOR
            if abs(shelf_db) > 1e-4:
                for row in _high_shelf_sos(_SHELF_FREQ_HZ, shelf_db, _SHELF_SLOPE, sr):
                    _, h = freqz(row[:3], row[3:], worN=w_points)
                    h_total *= h

            h_total *= _level_linear(p.sliders.level)
        else:
            for freq_hz, gain_db in zip(GE7_BAND_HZ, gains):
                if abs(gain_db) < 1e-4:
                    continue
                for row in _peaking_eq_sos(freq_hz, gain_db, p.band_q, sr):
                    _, h = freqz(row[:3], row[3:], worN=w_points)
                    h_total *= h
            h_total *= 10.0 ** (p.sliders.level / 20.0)

        freqs_hz = w_points / np.pi * (sr / 2.0)
        mag_db = 20.0 * np.log10(np.abs(h_total) + 1e-12)
        return freqs_hz, mag_db

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON."""
        return {
            "pedal": "boss_ge7",
            "pedal_class": "linear_eq",
            "pedal_version": self.VERSION,
            "fit_hints": {
                "recommended_models": ["FIR", "IIR"],
                "skip_models": ["WienerHammerstein", "Volterra", "Hammerstein"],
                "fir_length_ms": 100,
                "iir_order": 14,
            },
            "sliders": asdict(self.params.sliders),
            "band_q": self.params.band_q,
            "calibrated": self.params.calibrated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ge7Sim":
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Args:
            d: Dict with ``"sliders"`` and optionally ``"band_q"`` / ``"calibrated"`` keys.

        Returns:
            Ge7Sim instance.
        """
        sliders = Ge7Sliders.from_dict(d["sliders"])
        band_q = float(d.get("band_q", _BAND_Q))
        calibrated = bool(d.get("calibrated", True))
        return cls(Ge7Params(sliders=sliders, band_q=band_q, calibrated=calibrated))


# ── Filter helpers ────────────────────────────────────────────────────────────


def _peaking_eq_sos(
    freq_hz: float,
    gain_db: float,
    Q: float,
    sr: int,
) -> np.ndarray:
    """Peaking EQ biquad via Audio EQ Cookbook, returned as SOS (1×6).

    Args:
        freq_hz: Centre frequency in Hz.
        gain_db: Peak gain in dB (positive = boost, negative = cut).
        Q: Quality factor (bandwidth control).
        sr: Sample rate in Hz.

    Returns:
        SOS array of shape (1, 6) for use with ``scipy.signal.sosfilt``.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * Q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _high_shelf_sos(
    freq_hz: float,
    gain_db: float,
    slope: float,
    sr: int,
) -> np.ndarray:
    """High-shelf biquad via Audio EQ Cookbook, returned as SOS (1×6).

    The shelf frequency is the −3 dB transition point. At frequencies well
    above ``freq_hz`` the gain asymptotes to ``gain_db``.

    Args:
        freq_hz: Shelf (−3 dB) frequency in Hz.
        gain_db: Shelf gain in dB (positive = boost high freqs).
        slope: Shelf slope parameter S (0 < S ≤ 1). S=1 gives the steepest
            slope without overshoot; lower values give a gentler roll-off.
        sr: Sample rate in Hz.

    Returns:
        SOS array of shape (1, 6) for use with ``scipy.signal.sosfilt``.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)

    b0 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 =  A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha)
    a0 =       (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * np.sqrt(A) * alpha
    a1 =  2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 =       (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * np.sqrt(A) * alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _level_linear(slider_db: float) -> float:
    """Convert LEVEL slider dB position to linear gain with pot-taper correction.

    The LEVEL pot is log-taper: at +10 dB slider position the measured gain is
    +12.1 dB, and at −10 dB it is −14.4 dB. This quadratic correction was fit
    to both calibration points.

    Args:
        slider_db: Slider position in dB, range [−15, +15].

    Returns:
        Linear gain multiplier.
    """
    actual_db = _LEVEL_A * slider_db + _LEVEL_B * slider_db ** 2
    return float(10.0 ** (actual_db / 20.0))
