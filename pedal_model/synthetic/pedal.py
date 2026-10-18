"""Synthetic JHS Notaklon (Klon-style soft-overdrive) for pipeline dev and testing.

Signal chain (parallel clean-blend topology — matches real Klon circuit):

    dry ─┬─ HPF1(fc1) → ×drive → asym_soft_clip → HPF2(fc2) ─────────────────────────┐
         │                                                                              + → ×level → out
         └─ HiShelf(treble_db) → ×clean_blend ──────────────────────────────────────→─┘

Shamrock Mod (switch, adds after the blend):
    ... → blend → peak_EQ(800Hz, +4dB) → hard_clip → ×shamrock_boost → ×level → out

Controls (from JHS spec):
  Output  — Overall volume, left=less, right=more.  Max = +16 dB.
  Treble  — Treble boost/cut.  Noon is neutral (0 dB shelf). Left cuts, right boosts.
  Gain    — Overdrive amount.  Fully left = clean boost, fully right = max overdrive.
            As gain increases clockwise, mid-frequency range increases (blend shifts
            from clean-path-dominated → clip-path-dominated; clip path is more mid-forward
            because HPF2 rolls off bass from the clip path).
  Shamrock Mod Switch — +4 dB overall gain, mid-frequency increase (~800 Hz), plus
            an additional stage of hard-edge clipping (Si-diode ceiling after soft clip).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

# ── Knob travel ────────────────────────────────────────────────────────────── #

# Dial travel: 6 o'clock (full CCW, min) → 4 o'clock (full CW, max) = 10 hours.
# 6 o'clock = 0.0,  noon (12 o'clock) = 0.6,  4 o'clock = 1.0
_DIAL_HOURS: float = 10.0
_NOON: float = 6.0 / _DIAL_HOURS          # 0.6
_NINE_OCLOCK: float = 3.0 / _DIAL_HOURS   # 0.3
_ELEVEN_OCLOCK: float = 5.0 / _DIAL_HOURS # 0.5


def _oclock_to_norm(hours: float) -> float:
    """Convert o'clock position (6 = min, 4 = max, 10-hour travel) to [0, 1]."""
    return float(np.clip((hours - 6.0) / _DIAL_HOURS, 0.0, 1.0))


# ── Gain knob → drive + blend ─────────────────────────────────────────────── #

# Log taper: drive goes 1.0× (silence-adjacent, clean) to 12× (max overdrive)
_DRIVE_MIN: float = 1.0
_DRIVE_MAX: float = 12.0

# At min gain: almost entirely clean path.  At max gain: almost entirely clip path.
# clean_blend: fraction of the signal that comes from the clean (un-clipped) path.
_BLEND_AT_MIN: float = 0.95   # nearly pure clean at min gain → enables the +16dB boost
_BLEND_AT_MAX: float = 0.05   # nearly pure clip at max gain

# ── Treble knob → high-shelf ──────────────────────────────────────────────── #

# Noon = 0 dB (flat).  Full CCW = −TREBLE_MAX_DB (cut).  Full CW = +TREBLE_MAX_DB (boost).
_TREBLE_MAX_DB: float = 8.0
_TREBLE_SHELF_HZ: float = 3_500.0   # shelf transition frequency (Hz)

# ── Volume knob → output_level ────────────────────────────────────────────── #

# Log taper: min = near-silent, max = +16 dB (6.31× linear).
# "Fully left is clean [boost] … up to +16 dB … in conjunction with the output."
# At noon: ≈ 0.74× (−2.6 dB), appropriate for typical instrument levels.
_VOLUME_MIN: float = 0.05
_VOLUME_MAX: float = 6.31   # 10^(16/20)

# ── Shamrock Mod ───────────────────────────────────────────────────────────── #

# Peaking EQ centre for the Shamrock mid-boost (Hz)
_SHAMROCK_MID_HZ: float = 800.0
_SHAMROCK_MID_DB: float = 4.0
_SHAMROCK_MID_Q: float = 0.7
# Hard-clip threshold (fraction of ±1). After soft clip output is in [−1, 1];
# the hard clipper catches peaks from the additional Shamrock gain stage.
_SHAMROCK_HARD_THRESHOLD: float = 0.65
# Gain after hard clip normalises to ±1 and adds the +4 dB specification
_SHAMROCK_BOOST: float = 10.0 ** (4.0 / 20.0)   # ≈ 1.585×


# ── Dataclasses ───────────────────────────────────────────────────────────── #


@dataclass
class KlonKnobs:
    """Physical knob positions for the JHS Notaklon (or Klon-style pedal).

    Each position is normalised to [0, 1] where 0 = 6 o'clock (min) and
    1 = 5 o'clock / fully CW (max).

    Attributes:
        gain: Gain knob ∈ [0, 1]. Controls drive AND clean blend ratio simultaneously.
        treble: Treble knob ∈ [0, 1]. 0.5 ≈ noon = neutral. Left cuts, right boosts.
        volume: Volume (output level) knob ∈ [0, 1]. Max maps to +16 dB.
        shamrock: Shamrock Mod switch. False = off (standard), True = on.
    """

    gain: float = _NINE_OCLOCK
    treble: float = _NOON
    volume: float = _NINE_OCLOCK
    shamrock: bool = False

    # Convenience class-level aliases
    NINE_OCLOCK: float = _NINE_OCLOCK
    NOON: float = _NOON
    ELEVEN_OCLOCK: float = _ELEVEN_OCLOCK

    def to_params(self) -> "KlonSimParams":
        """Map knob positions to KlonSimParams physical values.

        Returns:
            KlonSimParams instance derived from the knob positions.
        """
        # Drive: log-taper from DRIVE_MIN to DRIVE_MAX
        log_min = np.log10(_DRIVE_MIN)
        log_max = np.log10(_DRIVE_MAX)
        drive = float(10.0 ** (log_min + self.gain * (log_max - log_min)))

        # Clean blend: linear, BLEND_AT_MIN (clean) → BLEND_AT_MAX (dirty)
        clean_blend = float(_BLEND_AT_MIN + self.gain * (_BLEND_AT_MAX - _BLEND_AT_MIN))

        # Treble: noon (0.6) = 0 dB, asymmetric slopes so CCW and CW each reach ±TREBLE_MAX_DB
        t = self.treble
        if t >= _NOON:
            treble_db = float((t - _NOON) / (1.0 - _NOON) * _TREBLE_MAX_DB)
        else:
            treble_db = float((t - _NOON) / _NOON * _TREBLE_MAX_DB)

        # Volume: log taper — gives −2.6 dB at noon, +16 dB at max
        output_level = float(_VOLUME_MIN * (_VOLUME_MAX / _VOLUME_MIN) ** self.volume)

        return KlonSimParams(
            drive=drive,
            clean_blend=clean_blend,
            treble_db=treble_db,
            output_level=output_level,
            shamrock=self.shamrock,
        )

    def to_params_delta(self) -> dict:
        """Alias for to_params() returned as a plain dict."""
        return asdict(self.to_params())

    def as_tensor(self) -> np.ndarray:
        """Return [gain, treble, volume] as a float32 array (shamrock excluded)."""
        return np.array([self.gain, self.treble, self.volume], dtype=np.float32)

    @classmethod
    def at_oclock(
        cls,
        gain: float,
        treble: float = 12.0,
        volume: float = 12.0,
        shamrock: bool = False,
    ) -> "KlonKnobs":
        """Construct from o'clock positions (6 = min, 12 = noon, 17 = max).

        Args:
            gain: Gain in hours (e.g. 9 = low, 11 = sweet spot, 15 = high overdrive).
            treble: Treble in hours. 12 = noon = neutral (default).
            volume: Volume in hours. 12 = noon ≈ −2.6 dB.
            shamrock: Shamrock Mod switch state.

        Returns:
            KlonKnobs instance.
        """
        return cls(
            gain=_oclock_to_norm(gain),
            treble=_oclock_to_norm(treble),
            volume=_oclock_to_norm(volume),
            shamrock=shamrock,
        )


@dataclass
class KlonSimParams:
    """All internal parameters for one KlonSim instance.

    Attributes:
        drive: Pre-gain multiplier into the clipping stage (1.0 to 12.0).
        input_hpf_hz: Input coupling-cap HPF cutoff in Hz. Default 80 Hz.
        clip_hpf_hz: Post-clip HPF cutoff — gives mid-forward character. Default 350 Hz.
        presence_hz: Output high-shelf frequency in Hz.
        presence_db: Output high-shelf gain in dB.
        output_level: Final linear output gain (0.05 to ~6.31 for +16 dB).
        clean_blend: Fraction of the clean path in the output mix.
            0 = pure clip, 1 = pure clean. Typically 0.05–0.95.
        treble_db: High-shelf gain on the clean blend path in dB.
            0 = neutral (noon), negative = treble cut (CCW), positive = boost (CW).
        clip_asymmetry: Half-wave asymmetry (0 = symmetric tanh, 0.15 = Klon-like).
        shamrock: Shamrock Mod switch state. When True, adds mid boost + hard clip.
        noise_floor_dbfs: Gaussian noise floor RMS in dBFS. Default −70 dBFS.
        noise_seed: RNG seed for reproducible noise. Default 42.
    """

    drive: float = 4.0
    input_hpf_hz: float = 80.0
    clip_hpf_hz: float = 350.0
    presence_hz: float = 3_000.0
    presence_db: float = 2.0
    output_level: float = 0.7
    clean_blend: float = 0.35
    treble_db: float = 0.0        # replaces old treble_hz; 0 = neutral
    clip_asymmetry: float = 0.15
    shamrock: bool = False
    noise_floor_dbfs: float = -70.0
    noise_seed: int = 42


# ── Main class ────────────────────────────────────────────────────────────── #


class KlonSim:
    """JHS Notaklon synthetic model with parallel clean-blend topology.

    Standard signal chain::

        dry ─┬─ HPF1 → ×drive → asym_clip → HPF2 ─────────────────────────────────┐
             │                                                                       + → ×level → noise → out
             └─ HiShelf(treble_db) → ×clean_blend ──────────────────────────────→──┘

    With Shamrock Mod (shamrock=True), an additional stage is inserted after the blend::

        blend_out → PeakEQ(800Hz,+4dB) → hard_clip(±0.65) → ×(1/0.65) → ×shamrock_boost → ×level → out

    Knob physics:
      - Gain increases drive AND shifts blend from clean-dominated to clip-dominated.
        This naturally creates the mid-frequency increase stated in the JHS spec,
        because the clip path (with its post-HPF bass roll-off) is more mid-forward.
      - Treble is a high shelf on the clean path: noon = flat, CCW = cut, CW = boost.
      - Volume is log-taper: noon ≈ −2.6 dB, max = +16 dB.

    Usage::

        # Song-specific setting (Gain 11 o'clock, treble noon, volume noon)
        sim = KlonSim.from_knobs(KlonKnobs.at_oclock(gain=11))
        wet = sim.process(dry, sr=48000)

        # With Shamrock Mod
        sim_sh = KlonSim.from_knobs(KlonKnobs.at_oclock(gain=11, shamrock=True))
    """

    VERSION = "2.1"

    def __init__(self, params: KlonSimParams | None = None) -> None:
        self.params = params or KlonSimParams()

    @classmethod
    def from_knobs(cls, knobs: KlonKnobs) -> "KlonSim":
        """Construct from physical knob positions.

        Args:
            knobs: KlonKnobs instance (use KlonKnobs.at_oclock() for convenience).

        Returns:
            KlonSim instance.
        """
        return cls(knobs.to_params())

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process dry audio through the pedal.

        Args:
            audio: Dry input, shape (N,), float32, nominally in [−1, 1].
            sr: Sample rate in Hz.

        Returns:
            Wet output, same shape, float32, soft-clipped to [−1, 1].
        """
        p = self.params
        x = audio.astype(np.float64)

        # ── Input HPF (coupling cap roll-off) ────────────────────────────── #
        sos_in = butter(2, p.input_hpf_hz / (sr / 2.0), btype="high", output="sos")
        x_hpf = sosfilt(sos_in, x)

        # ── Clip path ────────────────────────────────────────────────────── #
        clipped = _asymmetric_soft_clip(x_hpf * p.drive, p.clip_asymmetry)
        sos_post = butter(2, p.clip_hpf_hz / (sr / 2.0), btype="high", output="sos")
        clipped = sosfilt(sos_post, clipped)

        # ── Clean path (with treble shelf) ───────────────────────────────── #
        if abs(p.treble_db) > 0.05:
            b_t, a_t = _high_shelf_biquad(
                _TREBLE_SHELF_HZ, p.treble_db, sr, Q=0.707
            )
            clean = lfilter(b_t, a_t, x)
        else:
            clean = x.copy()

        # ── Parallel blend ───────────────────────────────────────────────── #
        y = (1.0 - p.clean_blend) * clipped + p.clean_blend * clean

        # ── Shamrock Mod ─────────────────────────────────────────────────── #
        if p.shamrock:
            # Mid-frequency boost (peaking EQ at 800 Hz)
            sos_mid = _peaking_eq_sos(
                _SHAMROCK_MID_HZ, _SHAMROCK_MID_DB, _SHAMROCK_MID_Q, sr
            )
            y = sosfilt(sos_mid, y)
            # Hard-edge clipping stage: Si-diode ceiling, then renormalise
            y = np.clip(y, -_SHAMROCK_HARD_THRESHOLD, _SHAMROCK_HARD_THRESHOLD)
            y = y / _SHAMROCK_HARD_THRESHOLD   # renormalise to ±1
            # +4 dB overall boost (the combined effect of the above equals +4 dB)
            y = y * _SHAMROCK_BOOST

        # ── Output presence shelf ─────────────────────────────────────────── #
        b_p, a_p = _high_shelf_biquad(p.presence_hz, p.presence_db, sr)
        y = lfilter(b_p, a_p, y)

        # ── Output level ─────────────────────────────────────────────────── #
        y = y * p.output_level

        # ── Noise floor ──────────────────────────────────────────────────── #
        rng = np.random.default_rng(p.noise_seed)
        noise_amp = 10.0 ** (p.noise_floor_dbfs / 20.0)
        y = y + noise_amp * rng.standard_normal(len(y))

        # Soft ceiling at ±1 (interface ADC limit)
        return np.tanh(y).astype(np.float32)

    def frequency_response(
        self, sr: int, n_fft: int = 4096, level_dbfs: float = -24.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Small-signal frequency response estimate via log-chirp deconvolution.

        Args:
            sr: Sample rate in Hz.
            n_fft: FFT size.
            level_dbfs: Input level for the chirp (−24 dBFS = well below clipping).

        Returns:
            (freqs_hz, magnitude_db) of length n_fft // 2 + 1.
        """
        from scipy.signal import chirp as _chirp
        dur = 2.0 * n_fft / sr
        t = np.linspace(0.0, dur, int(dur * sr), endpoint=False, dtype=np.float32)
        amp = 10.0 ** (level_dbfs / 20.0)
        dry = (amp * _chirp(t, f0=20.0, f1=sr / 2.0 - 100.0, t1=dur, method="log")).astype(np.float32)
        wet = self.process(dry, sr)
        D = np.fft.rfft(dry, n=n_fft)
        W = np.fft.rfft(wet[:n_fft], n=n_fft)
        H = W / (D + 1e-10)
        freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        return freqs_hz, 20.0 * np.log10(np.abs(H) + 1e-12)

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON."""
        return {
            "pedal_type": "notaklon_sim",
            "pedal_version": self.VERSION,
            "params": asdict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KlonSim":
        """Reconstruct from a dict produced by to_dict.

        Gracefully ignores unknown keys so old-format JSON still loads.

        Args:
            d: Dict with a ``"params"`` key.

        Returns:
            KlonSim instance.
        """
        known = set(KlonSimParams.__dataclass_fields__)  # type: ignore[attr-defined]
        raw = d.get("params", d)
        # Migrate old treble_hz → treble_db
        if "treble_hz" in raw and "treble_db" not in raw:
            hz = float(raw.pop("treble_hz"))
            # rough inverse: map hz → db (noon ~ 9kHz → 0 dB)
            raw["treble_db"] = float(np.clip((hz - 9000.0) / 9000.0 * _TREBLE_MAX_DB, -_TREBLE_MAX_DB, _TREBLE_MAX_DB))
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(KlonSimParams(**filtered))


# ── Filter / nonlinearity helpers ─────────────────────────────────────────── #


def _asymmetric_soft_clip(x: np.ndarray, asymmetry: float) -> np.ndarray:
    """Soft clipper with Ge/Si diode asymmetry (2nd-harmonic generation).

    Positive half (Ge — lower Vf): shallower slope, softer clip onset.
    Negative half (Si — higher Vf): steeper slope, sharper clip onset.

    Args:
        x: Input signal, float64.
        asymmetry: 0 = symmetric tanh, 0.10–0.20 = Klon-range. Must be < 1.0.

    Returns:
        Clipped signal, same shape.
    """
    # a_pos = 1 − asymmetry  (softer for positive half-wave, Ge-like)
    # a_neg = 1 + asymmetry  (harder for negative half-wave, Si-like)
    a = 2.0 - (1.0 + asymmetry * np.sign(x))
    return np.tanh(a * x)


def _peaking_eq_sos(
    freq_hz: float, gain_db: float, Q: float, sr: int
) -> np.ndarray:
    """Peaking EQ biquad (Audio EQ Cookbook), returned as SOS (1×6).

    Args:
        freq_hz: Centre frequency in Hz.
        gain_db: Peak gain in dB.
        Q: Quality factor.
        sr: Sample rate in Hz.

    Returns:
        SOS array of shape (1, 6).
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


def _high_shelf_biquad(
    freq_hz: float, gain_db: float, sr: int, Q: float = 0.707
) -> tuple[np.ndarray, np.ndarray]:
    """High-shelf biquad (Audio EQ Cookbook).

    Args:
        freq_hz: Shelf transition frequency in Hz.
        gain_db: Shelf gain in dB (positive = boost, negative = cut).
        sr: Sample rate in Hz.
        Q: Slope quality factor. 0.707 = maximally flat.

    Returns:
        (b, a) coefficient arrays of length 3.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * Q)
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
