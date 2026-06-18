"""Synthetic Boss GE-7 graphic equalizer for pipeline development and testing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import sosfilt

# GE-7 band centre frequencies in Hz (octave-spaced, 100 Hz – 6.4 kHz)
GE7_BAND_HZ: tuple[int, ...] = (100, 200, 400, 800, 1600, 3200, 6400)

# Slider gain range in dB
_SLIDER_MIN_DB: float = -15.0
_SLIDER_MAX_DB: float = +15.0

# Q factor for each peaking band.  GE-7 bands are octave-spaced so Q ≈ 1.4
# gives the canonical graphic-EQ overlap between adjacent bands.
_BAND_Q: float = 1.4


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
    """

    sliders: Ge7Sliders = field(default_factory=Ge7Sliders)
    band_q: float = _BAND_Q


class Ge7Sim:
    """Synthetic Boss GE-7 7-band graphic equalizer.

    Signal chain:
      Cascade of 7 peak-EQ biquads (Audio EQ Cookbook peaking filter),
      one per band at 100 / 200 / 400 / 800 / 1600 / 3200 / 6400 Hz,
      followed by a linear output-level scalar derived from the LEVEL slider.

    With all sliders at 0 dB the model is unity gain (flat response) plus any
    rounding error in the biquad arithmetic — this is intentional so that
    ``preset_flat`` can serve as a reference / null test.
    """

    VERSION = "1.0"

    def __init__(self, params: Ge7Params | None = None) -> None:
        self.params = params or Ge7Params()

    @classmethod
    def from_sliders(cls, sliders: Ge7Sliders, band_q: float = _BAND_Q) -> "Ge7Sim":
        """Construct directly from a Ge7Sliders instance.

        Args:
            sliders: Per-band gains and output level.
            band_q: Peaking filter Q. Default 1.4 (octave-spaced bands).

        Returns:
            Ge7Sim instance.
        """
        return cls(Ge7Params(sliders=sliders, band_q=band_q))

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
        for freq_hz, gain_db in zip(GE7_BAND_HZ, gains):
            if abs(gain_db) < 1e-4:
                continue  # skip unity bands for speed
            sos = _peaking_eq_sos(freq_hz, gain_db, p.band_q, sr)
            x = sosfilt(sos, x)

        # Output level scalar
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
        for freq_hz, gain_db in zip(GE7_BAND_HZ, gains):
            if abs(gain_db) < 1e-4:
                continue
            sos = _peaking_eq_sos(freq_hz, gain_db, p.band_q, sr)
            b = np.convolve(sos[0, :3], sos[1, :3]) if len(sos) == 2 else sos[0, :3]
            a = np.convolve(sos[0, 3:], sos[1, 3:]) if len(sos) == 2 else sos[0, 3:]
            # Use freqz on each biquad and multiply
            for row in sos:
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
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ge7Sim":
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Args:
            d: Dict with ``"sliders"`` and optionally ``"band_q"`` keys.

        Returns:
            Ge7Sim instance.
        """
        sliders = Ge7Sliders.from_dict(d["sliders"])
        band_q = float(d.get("band_q", _BAND_Q))
        return cls(Ge7Params(sliders=sliders, band_q=band_q))


# ── Internal helpers ──────────────────────────────────────────────────────────


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
    A = 10.0 ** (gain_db / 40.0)  # sqrt of linear gain
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * Q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
    return sos
