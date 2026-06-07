"""Synthetic Klon-style soft-overdrive pedal for pipeline development and testing."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

# ── Knob positions ────────────────────────────────────────────────────────── #

# Klon knob travel: 6 o'clock (min) → 5 o'clock next day (max)
# Normalised to [0, 1].  Key reference positions:
#   6 o'clock = 0.000   (fully CCW)
#   9 o'clock = 0.273   (≈ 3/11 of travel)
#  12 o'clock = 0.545   (centre)
#   3 o'clock = 0.818   (3/4 of travel)
#   5 o'clock = 1.000   (fully CW)
_NINE_OCLOCK: float = 3.0 / 11.0

# Gain knob → drive: log scale 1.5× to 12× (0.5 to 4 in log10 = 21.6 dB range)
_GAIN_MIN: float = 1.5
_GAIN_MAX: float = 12.0

# Treble knob → presence_db: linear, −3 dB to +12 dB
_TREBLE_MIN_DB: float = -3.0
_TREBLE_MAX_DB: float = 12.0

# Volume knob → output_level: linear, 0.2 to 1.0
_VOLUME_MIN: float = 0.2
_VOLUME_MAX: float = 1.0


@dataclass
class KlonKnobs:
    """Physical knob positions for the three Klon controls.

    Each value is in [0, 1] where 0 = 6 o'clock (min) and 1 = 5 o'clock (max).
    The convenience constant ``NINE_OCLOCK`` (≈ 0.273) maps each control to
    the default training position.

    Attributes:
        gain: Gain knob position ∈ [0, 1]. Controls pre-gain into the clip stage.
        treble: Treble knob position ∈ [0, 1]. Controls high-shelf presence.
        volume: Volume knob position ∈ [0, 1]. Controls output level.
    """

    gain: float = _NINE_OCLOCK
    treble: float = _NINE_OCLOCK
    volume: float = _NINE_OCLOCK

    NINE_OCLOCK: float = _NINE_OCLOCK  # class-level alias for convenience

    def to_params_delta(self) -> dict[str, float]:
        """Map knob positions to KlonSimParams physical values.

        Returns:
            Dict with keys 'drive', 'presence_db', 'output_level' suitable
            for overriding corresponding KlonSimParams fields.
        """
        # Gain → drive: logarithmic mapping 1.5× to 12×
        log_min = np.log10(_GAIN_MIN)
        log_max = np.log10(_GAIN_MAX)
        drive = float(10.0 ** (log_min + self.gain * (log_max - log_min)))

        # Treble → presence_db: linear
        presence_db = float(_TREBLE_MIN_DB + self.treble * (_TREBLE_MAX_DB - _TREBLE_MIN_DB))

        # Volume → output_level: linear
        output_level = float(_VOLUME_MIN + self.volume * (_VOLUME_MAX - _VOLUME_MIN))

        return {"drive": drive, "presence_db": presence_db, "output_level": output_level}

    def as_tensor(self) -> "np.ndarray":
        """Return knob positions as a float32 array [gain, treble, volume]."""
        return np.array([self.gain, self.treble, self.volume], dtype=np.float32)


@dataclass
class KlonSimParams:
    """All parameters that define one KlonSim instance.

    Attributes:
        drive: Pre-gain multiplier into the clipping stage. Default 4.0 ≈ 12 dB.
        input_hpf_hz: Input coupling-cap HPF cutoff in Hz. Default 80 Hz.
        clip_hpf_hz: Post-clip HPF cutoff in Hz — gives mid-forward Klon voice.
            Default 350 Hz.
        presence_hz: High-shelf centre frequency in Hz. Default 3000 Hz.
        presence_db: High-shelf gain in dB (positive = boost). Default +2 dB.
        output_level: Final output gain scalar in (0, 1]. Default 0.7.
        noise_floor_dbfs: Gaussian noise RMS level in dBFS. Default −70 dBFS.
        noise_seed: RNG seed for fully reproducible noise. Default 42.
    """

    drive: float = 4.0
    input_hpf_hz: float = 80.0
    clip_hpf_hz: float = 350.0
    presence_hz: float = 3_000.0
    presence_db: float = 2.0
    output_level: float = 0.7
    noise_floor_dbfs: float = -70.0
    noise_seed: int = 42


class KlonSim:
    """Synthetic Klon-style soft-overdrive pedal.

    Signal chain:
      1. Input HPF  (2nd-order Butterworth) — coupling-cap roll-off below ~80 Hz
      2. Drive gain — pre-amp boost into saturation
      3. tanh clip  — symmetric op-amp + diode soft saturation
      4. Post HPF   (2nd-order Butterworth) — mid-forward character, ~350 Hz
      5. High shelf (biquad, Audio EQ Cookbook) — +2 dB presence at 3 kHz
      6. Level      — output-stage gain
      7. Noise      — seeded Gaussian floor at −70 dBFS

    The filter–NL–filter topology makes this a Wiener–Hammerstein structure,
    so among classical models only Wiener–Hammerstein should fit it well.
    Neural models (TCN, LSTM) should outperform all classical baselines.
    """

    VERSION = "1.0"

    def __init__(self, params: KlonSimParams | None = None) -> None:
        self.params = params or KlonSimParams()

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process dry audio through the synthetic pedal.

        Args:
            audio: Dry audio, shape (N,), float32, nominally in [−1, 1].
            sr: Sample rate in Hz.

        Returns:
            Wet audio, same shape, float32, clipped to [−1, 1].
        """
        p = self.params
        x = audio.astype(np.float64)

        # 1. Input HPF — 2nd-order Butterworth, simulates coupling capacitor
        sos_in = butter(2, p.input_hpf_hz / (sr / 2), btype="high", output="sos")
        x = sosfilt(sos_in, x)

        # 2. Drive — pre-amp boost into the clipping stage
        x = x * p.drive

        # 3. Soft clip — tanh models symmetric op-amp + diode saturation
        x = np.tanh(x)

        # 4. Post-clip HPF — Klon's mid-forward character
        sos_post = butter(2, p.clip_hpf_hz / (sr / 2), btype="high", output="sos")
        x = sosfilt(sos_post, x)

        # 5. High shelf — presence boost (Audio EQ Cookbook biquad)
        b, a = _high_shelf_biquad(p.presence_hz, p.presence_db, sr)
        x = lfilter(b, a, x)

        # 6. Output level
        x = x * p.output_level

        # 7. Seeded Gaussian noise floor
        rng = np.random.default_rng(p.noise_seed)
        noise_amp = 10.0 ** (p.noise_floor_dbfs / 20.0)
        x = x + noise_amp * rng.standard_normal(len(x))

        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON."""
        return {
            "pedal_type": "klon_sim",
            "pedal_version": self.VERSION,
            "params": asdict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KlonSim":
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Args:
            d: Dict with a ``"params"`` key containing KlonSimParams fields.

        Returns:
            KlonSim instance with the stored parameters.
        """
        return cls(KlonSimParams(**d["params"]))


# ── Internal helpers ──────────────────────────────────────────────────────────


def _high_shelf_biquad(
    freq_hz: float,
    gain_db: float,
    sr: int,
    Q: float = 0.707,
) -> tuple[np.ndarray, np.ndarray]:
    """Biquad high-shelf filter coefficients via the Audio EQ Cookbook.

    Args:
        freq_hz: Shelf transition frequency in Hz.
        gain_db: Shelf gain in dB (positive = boost).
        sr: Sample rate in Hz.
        Q: Shelf slope quality factor. 0.707 = maximally flat shelf.

    Returns:
        ``(b, a)`` arrays of length 3 for use with ``scipy.signal.lfilter``.
    """
    A = 10.0 ** (gain_db / 40.0)  # sqrt(10^(gain_db/20))
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * Q)
    sqrt_A = np.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a
