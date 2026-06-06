"""Parameterized audio signal generator for pedal characterization.

Builds on the log-sweep, amplitude-sweep, IM-tone, noise and transient
primitives already prototyped in generate_capture_signal.py, extends them
with pink noise, stepped tone grids, impulse trains, AM chirps, decaying
plucks, multitone stacks and layered sections, and wraps everything in a
structured JSON manifest so any section can be queried by label.

Usage (CLI):
    python -m pedal_model.signals.generate --signal train --output data/signals/
    python -m pedal_model.signals.generate --signal val   --output data/signals/
    python -m pedal_model.signals.generate --signal both  --output data/signals/

Usage (library):
    from pedal_model.signals import generate, TrainParams
    wav, js = generate("train", "data/signals/", train_params=TrainParams(seed=99))
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

GENERATOR_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Parameter dataclasses — every field is a CLI / config knob
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainParams:
    """Full parameter set for train_signal_v1 (~13 min at 96 kHz).

    Amplitude conventions:
        deterministic signals (sweeps, tones): amplitude_dbfs = peak level
        stochastic signals (pink/white noise):  amplitude_dbfs = RMS level
    """

    sample_rate: int = 96_000
    seed: int = 1234
    silence_between_s: float = 0.5  # gap inserted between logical groups

    # Log sine sweeps (pure, 3 levels)
    sweep_f_start_hz: float = 20.0
    sweep_f_end_hz: float = 20_000.0
    sweep_duration_s: float = 20.0
    sweep_amplitudes_dbfs: list[float] = field(
        default_factory=lambda: [-24.0, -12.0, -3.0]
    )

    # Stepped sine tones (freqs × levels grid)
    stepped_freqs_hz: list[float] = field(
        default_factory=lambda: [
            82.4, 110.0, 165.0, 220.0, 330.0, 440.0,
            660.0, 880.0, 1320.0, 2640.0, 5280.0, 10_560.0,
        ]
    )
    stepped_levels_dbfs: list[float] = field(
        default_factory=lambda: [-24.0, -18.0, -12.0, -6.0, -3.0]
    )
    stepped_tone_duration_s: float = 1.0

    # Pink noise (3 RMS levels, 20 s each)
    pink_amplitudes_dbfs: list[float] = field(
        default_factory=lambda: [-24.0, -12.0, -3.0]
    )
    pink_duration_s: float = 20.0

    # White noise (2 RMS levels, 15 s each)
    white_amplitudes_dbfs: list[float] = field(
        default_factory=lambda: [-18.0, -6.0]
    )
    white_duration_s: float = 15.0

    # Amplitude ramps (continuous level traversal per freq)
    ramp_freqs_hz: list[float] = field(
        default_factory=lambda: [110.0, 220.0, 440.0, 880.0]
    )
    ramp_duration_s: float = 10.0
    ramp_start_dbfs: float = -40.0
    ramp_end_dbfs: float = -2.0

    # Impulse train (varying spacing)
    impulse_duration_s: float = 30.0
    impulse_amplitude_dbfs: float = -6.0
    impulse_spacings_s: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.05, 0.15, 0.30, 0.05, 0.08]
    )

    # AM chirp (swept modulation freq × carrier)
    am_chirp_duration_s: float = 60.0
    am_chirp_amplitude_dbfs: float = -9.0
    am_chirp_f_carrier_hz: float = 440.0
    am_chirp_f_mod_start_hz: float = 0.5
    am_chirp_f_mod_end_hz: float = 20.0
    am_chirp_mod_depth: float = 0.8

    # Decaying plucks (guitar-like attack/decay)
    pluck_freqs_hz: list[float] = field(
        default_factory=lambda: [82.4, 110.0, 165.0, 220.0, 330.0, 440.0, 880.0]
    )
    pluck_decay_tau_s: float = 0.4
    pluck_amplitude_dbfs: float = -6.0
    pluck_total_duration_s: float = 40.0

    # Two-tone sweeps (fixed + swept partner)
    two_tone_fixed_freqs_hz: list[float] = field(
        default_factory=lambda: [110.0, 220.0, 440.0]
    )
    two_tone_sweep_start_hz: float = 20.0
    two_tone_sweep_end_hz: float = 8_000.0
    two_tone_subsection_duration_s: float = 10.0
    two_tone_amplitude_dbfs: float = -9.0

    # Multitone stacks (3 levels)
    multitone_freqs_hz: list[float] = field(
        default_factory=lambda: [82.4, 165.0, 247.0, 330.0, 412.0, 495.0, 660.0, 880.0]
    )
    multitone_amplitudes_dbfs: list[float] = field(
        default_factory=lambda: [-18.0, -9.0, -3.0]
    )
    multitone_duration_s: float = 30.0

    # Layered sections
    layered_duration_s: float = 45.0


@dataclass
class ValParams:
    """Full parameter set for val_signal_v1 (~1 min at 96 kHz).

    Fixed seed makes this a reproducible, cross-project benchmark.
    """

    sample_rate: int = 96_000
    seed: int = 42
    silence_gap_s: float = 1.0

    # Amplitude-stepped sweep (each level = one sweep)
    stepped_sweep_amplitudes_dbfs: list[float] = field(
        default_factory=lambda: [-24.0, -12.0, -3.0]
    )
    stepped_sweep_duration_s: float = 5.0
    stepped_sweep_f_start_hz: float = 20.0
    stepped_sweep_f_end_hz: float = 20_000.0

    # Stepped tones
    val_stepped_freqs_hz: list[float] = field(
        default_factory=lambda: [110.0, 220.0, 440.0, 880.0, 1_760.0, 3_520.0, 7_040.0]
    )
    val_stepped_levels_dbfs: list[float] = field(
        default_factory=lambda: [-18.0, -9.0, -3.0]
    )
    val_stepped_tone_duration_s: float = 0.5

    # Pink noise (2 RMS levels)
    val_pink_amplitudes_dbfs: list[float] = field(
        default_factory=lambda: [-18.0, -6.0]
    )
    val_pink_duration_s: float = 5.0

    # Two-tone IM segments
    val_two_tone_pairs: list[list[float]] = field(
        default_factory=lambda: [[220.0, 330.0], [440.0, 660.0], [220.0, 880.0]]
    )
    val_two_tone_duration_s: float = 3.33

    # Transient impulses
    val_impulse_duration_s: float = 3.0
    val_impulse_amplitude_dbfs: float = -6.0
    val_impulse_spacings_s: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.05, 0.15]
    )

    # Decaying plucks
    val_pluck_freqs_hz: list[float] = field(
        default_factory=lambda: [110.0, 220.0, 440.0]
    )
    val_pluck_decay_tau_s: float = 0.4
    val_pluck_amplitude_dbfs: float = -6.0


# ─────────────────────────────────────────────────────────────────────────────
# Low-level synthesis primitives
# ─────────────────────────────────────────────────────────────────────────────


def _dbfs(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def _fade(audio: np.ndarray, fade_n: int) -> np.ndarray:
    f = min(fade_n, len(audio) // 4)
    if f < 1:
        return audio
    audio = audio.copy()
    audio[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
    audio[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return audio


def _log_sweep(n: int, sr: int, f_start: float, f_end: float) -> np.ndarray:
    """Farina exponential sine sweep, unit peak. Reuses the proven formula from
    generate_capture_signal.py: φ(t) = 2π·f1·L·(exp(t/L)−1), L = T/ln(f2/f1)."""
    t = np.linspace(0.0, n / sr, n, endpoint=False, dtype=np.float64)
    L = (n / sr) / np.log(f_end / f_start)
    phase = 2.0 * np.pi * f_start * L * (np.exp(t / L) - 1.0)
    return np.sin(phase).astype(np.float32)


def _sine(n: int, sr: int, freq: float) -> np.ndarray:
    t = np.linspace(0.0, n / sr, n, endpoint=False, dtype=np.float64)
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS pink (1/f) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid DC singularity
    scale = 1.0 / np.sqrt(freqs)
    scale[0] = 0.0  # zero DC component
    pink = np.fft.irfft(np.fft.rfft(white) * scale, n=n)
    return (pink / (float(np.std(pink)) + 1e-10)).astype(np.float32)


def _white_noise(n: int, rng: np.random.Generator, bandlimit: bool = False,
                 low_hz: float = 80.0, high_hz: float = 8_000.0, sr: int = 96_000) -> np.ndarray:
    """Unit-RMS white noise, optionally bandlimited (4th-order Butterworth)."""
    sig = rng.standard_normal(n).astype(np.float64)
    if bandlimit and low_hz < high_hz:
        sos = butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
        sig = sosfilt(sos, sig)
    return (sig / (float(np.std(sig)) + 1e-10)).astype(np.float32)


def _amp_ramp(n: int, sr: int, freq: float, start_dbfs: float, end_dbfs: float) -> np.ndarray:
    """Sine tone with exponential amplitude ramp (log-linear in dB).
    Reuses the 'log' shape from gen_amplitude_sweep in generate_capture_signal.py."""
    t = np.linspace(0.0, n / sr, n, endpoint=False, dtype=np.float64)
    u = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    envelope = _dbfs(start_dbfs) * (_dbfs(end_dbfs) / (_dbfs(start_dbfs) + 1e-30)) ** u
    return (np.sin(2.0 * np.pi * freq * t) * envelope).astype(np.float32)


def _impulse_train(n: int, sr: int, spacings_s: list[float]) -> np.ndarray:
    """Unit-peak impulse train with cycling inter-impulse spacings."""
    audio = np.zeros(n, dtype=np.float32)
    pos, idx = 0, 0
    while pos < n:
        audio[pos] = 1.0
        pos += max(1, int(spacings_s[idx % len(spacings_s)] * sr))
        idx += 1
    return audio


def _am_chirp(n: int, sr: int, f_carrier: float,
              f_mod_start: float, f_mod_end: float, mod_depth: float) -> np.ndarray:
    """AM signal with linearly sweeping modulation frequency, unit peak."""
    t = np.linspace(0.0, n / sr, n, endpoint=False, dtype=np.float64)
    f_mod = f_mod_start + (f_mod_end - f_mod_start) * t / (n / sr)
    mod_phase = 2.0 * np.pi * np.cumsum(f_mod) / sr
    sig = np.sin(2.0 * np.pi * f_carrier * t) * (1.0 + mod_depth * np.sin(mod_phase))
    peak = float(np.max(np.abs(sig))) + 1e-10
    return (sig / peak).astype(np.float32)


def _decaying_pluck(sr: int, freq: float, decay_tau: float) -> np.ndarray:
    """Single exponentially-decaying sinusoid (5 time constants). Unit peak."""
    n = max(1, int(5.0 * decay_tau * sr))
    t = np.linspace(0.0, n / sr, n, endpoint=False, dtype=np.float64)
    sig = np.sin(2.0 * np.pi * freq * t) * np.exp(-t / decay_tau)
    return sig.astype(np.float32)


def _two_tone(n: int, sr: int, f1: float, f2: float) -> np.ndarray:
    """Equal-amplitude two-tone mix (reuses gen_im_tones math). Unit peak."""
    t = np.linspace(0.0, n / sr, n, endpoint=False, dtype=np.float64)
    sig = np.sin(2.0 * np.pi * f1 * t) + np.sin(2.0 * np.pi * f2 * t)
    return (sig / (float(np.max(np.abs(sig))) + 1e-10)).astype(np.float32)


def _two_tone_sweep(n: int, sr: int, f_fixed: float,
                    f_sweep_start: float, f_sweep_end: float) -> np.ndarray:
    """Fixed tone + exponentially-swept partner. Unit peak."""
    t = np.linspace(0.0, n / sr, n, endpoint=False, dtype=np.float64)
    duration = n / sr
    L = duration / np.log(f_sweep_end / f_sweep_start)
    phase_swept = 2.0 * np.pi * f_sweep_start * L * (np.exp(t / L) - 1.0)
    sig = np.sin(2.0 * np.pi * f_fixed * t) + np.sin(phase_swept)
    return (sig / (float(np.max(np.abs(sig))) + 1e-10)).astype(np.float32)


def _multitone(n: int, sr: int, freqs: list[float]) -> np.ndarray:
    """Equal-amplitude multitone with Schroeder phases (low crest factor). Unit peak."""
    t = np.linspace(0.0, n / sr, n, endpoint=False, dtype=np.float64)
    k_total = len(freqs)
    sig = np.zeros(n, dtype=np.float64)
    for k, f in enumerate(freqs):
        phase = np.pi * k * (k + 1) / k_total  # Schroeder phase
        sig += np.sin(2.0 * np.pi * f * t + phase)
    return (sig / (float(np.max(np.abs(sig))) + 1e-10)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Section builder
# ─────────────────────────────────────────────────────────────────────────────


class _Builder:
    """Accumulates audio chunks and section metadata."""

    def __init__(self, sr: int, fade_n: int) -> None:
        self.sr = sr
        self.fade_n = fade_n
        self._chunks: list[np.ndarray] = []
        self._sections: list[dict[str, Any]] = []
        self._pos = 0
        self._idx = 0

    def silence(self, duration_s: float) -> None:
        n = max(0, int(duration_s * self.sr))
        if n:
            self._chunks.append(np.zeros(n, dtype=np.float32))
            self._pos += n

    def add(
        self,
        audio: np.ndarray,
        label: str,
        section_type: str,
        amplitude_dbfs: float,
        params: dict[str, Any],
        *,
        apply_fade: bool = True,
        gap_before: float = 0.0,
    ) -> None:
        if gap_before > 0.0:
            self.silence(gap_before)
        if apply_fade:
            audio = _fade(audio, self.fade_n)
        n = len(audio)
        self._sections.append({
            "index": self._idx,
            "label": label,
            "type": section_type,
            "start_sample": self._pos,
            "end_sample": self._pos + n,
            "start_s": round(self._pos / self.sr, 6),
            "end_s": round((self._pos + n) / self.sr, 6),
            "amplitude_dbfs": amplitude_dbfs,
            "params": params,
        })
        self._chunks.append(audio)
        self._pos += n
        self._idx += 1

    def build(self) -> np.ndarray:
        return np.concatenate(self._chunks).astype(np.float32) if self._chunks else np.array([], dtype=np.float32)

    @property
    def sections(self) -> list[dict[str, Any]]:
        return list(self._sections)


# ─────────────────────────────────────────────────────────────────────────────
# Train signal assembly
# ─────────────────────────────────────────────────────────────────────────────


def _build_train(p: TrainParams, rng: np.random.Generator) -> tuple[np.ndarray, list[dict[str, Any]]]:
    sr = p.sample_rate
    fade_n = max(1, int(0.010 * sr))  # 10 ms fade
    b = _Builder(sr, fade_n)
    gap = p.silence_between_s

    # ── 1. Log sine sweeps: low / mid / high ──────────────────────────────────
    for amp, lbl in zip(p.sweep_amplitudes_dbfs, ["low", "mid", "high"]):
        n = int(p.sweep_duration_s * sr)
        audio = _dbfs(amp) * _log_sweep(n, sr, p.sweep_f_start_hz, p.sweep_f_end_hz)
        b.add(audio, f"sweep_{lbl}", "log_sine_sweep", amp, {
            "f_start_hz": p.sweep_f_start_hz,
            "f_end_hz": p.sweep_f_end_hz,
            "duration_s": p.sweep_duration_s,
            "amplitude_convention": "peak_dbfs",
        }, gap_before=gap)

    # ── 2. Stepped sine tones: freqs × levels grid ────────────────────────────
    # No gaps between individual tones; one gap before the entire block.
    first = True
    for freq in p.stepped_freqs_hz:
        for amp in p.stepped_levels_dbfs:
            n = int(p.stepped_tone_duration_s * sr)
            audio = _dbfs(amp) * _sine(n, sr, freq)
            label = f"stepped_tone_f{int(round(freq))}_l{int(round(amp))}"
            b.add(audio, label, "stepped_sine_tone", amp, {
                "freq_hz": freq,
                "duration_s": p.stepped_tone_duration_s,
                "amplitude_convention": "peak_dbfs",
            }, gap_before=gap if first else 0.0)
            first = False

    # ── 3. Pink noise: 3 RMS levels ───────────────────────────────────────────
    first = True
    for amp in p.pink_amplitudes_dbfs:
        n = int(p.pink_duration_s * sr)
        audio = _dbfs(amp) * _pink_noise(n, rng)
        b.add(audio, f"pink_noise_l{int(round(amp))}", "pink_noise", amp, {
            "duration_s": p.pink_duration_s,
            "amplitude_convention": "rms_dbfs",
        }, apply_fade=False, gap_before=gap if first else 0.0)
        first = False

    # ── 4. White noise: 2 RMS levels ──────────────────────────────────────────
    first = True
    for amp in p.white_amplitudes_dbfs:
        n = int(p.white_duration_s * sr)
        audio = _dbfs(amp) * _white_noise(n, rng, sr=sr)
        b.add(audio, f"white_noise_l{int(round(amp))}", "white_noise", amp, {
            "duration_s": p.white_duration_s,
            "amplitude_convention": "rms_dbfs",
        }, apply_fade=False, gap_before=gap if first else 0.0)
        first = False

    # ── 5. Amplitude ramps: log-linear level traversal per freq ───────────────
    first = True
    for freq in p.ramp_freqs_hz:
        n = int(p.ramp_duration_s * sr)
        audio = _amp_ramp(n, sr, freq, p.ramp_start_dbfs, p.ramp_end_dbfs)
        b.add(audio, f"amp_ramp_f{int(round(freq))}", "amplitude_ramp", p.ramp_end_dbfs, {
            "freq_hz": freq,
            "duration_s": p.ramp_duration_s,
            "amp_start_dbfs": p.ramp_start_dbfs,
            "amp_end_dbfs": p.ramp_end_dbfs,
        }, apply_fade=False, gap_before=gap if first else 0.0)
        first = False

    # ── 6. Impulse train ──────────────────────────────────────────────────────
    n = int(p.impulse_duration_s * sr)
    audio = _dbfs(p.impulse_amplitude_dbfs) * _impulse_train(n, sr, p.impulse_spacings_s)
    b.add(audio, "impulse_train", "impulse_train", p.impulse_amplitude_dbfs, {
        "duration_s": p.impulse_duration_s,
        "spacings_s": p.impulse_spacings_s,
        "amplitude_convention": "peak_dbfs",
    }, apply_fade=False, gap_before=gap)

    # ── 7. AM chirp ───────────────────────────────────────────────────────────
    n = int(p.am_chirp_duration_s * sr)
    raw = _am_chirp(n, sr, p.am_chirp_f_carrier_hz,
                    p.am_chirp_f_mod_start_hz, p.am_chirp_f_mod_end_hz, p.am_chirp_mod_depth)
    b.add(_dbfs(p.am_chirp_amplitude_dbfs) * raw, "am_chirp", "am_chirp", p.am_chirp_amplitude_dbfs, {
        "duration_s": p.am_chirp_duration_s,
        "f_carrier_hz": p.am_chirp_f_carrier_hz,
        "f_mod_start_hz": p.am_chirp_f_mod_start_hz,
        "f_mod_end_hz": p.am_chirp_f_mod_end_hz,
        "mod_depth": p.am_chirp_mod_depth,
        "amplitude_convention": "peak_dbfs",
    }, gap_before=gap)

    # ── 8. Decaying plucks: cycle through freqs to fill total duration ────────
    pluck_chunks: list[np.ndarray] = []
    target_n = int(p.pluck_total_duration_s * sr)
    total = 0
    fi = 0
    while total < target_n:
        pluck = _dbfs(p.pluck_amplitude_dbfs) * _decaying_pluck(sr, p.pluck_freqs_hz[fi % len(p.pluck_freqs_hz)], p.pluck_decay_tau_s)
        pluck_chunks.append(pluck)
        total += len(pluck)
        fi += 1
    pluck_audio = np.concatenate(pluck_chunks)[:target_n].astype(np.float32)
    b.add(pluck_audio, "decaying_plucks", "decaying_plucks", p.pluck_amplitude_dbfs, {
        "freqs_hz": p.pluck_freqs_hz,
        "decay_tau_s": p.pluck_decay_tau_s,
        "total_duration_s": p.pluck_total_duration_s,
        "amplitude_convention": "peak_dbfs",
    }, apply_fade=False, gap_before=gap)

    # ── 9. Two-tone sweeps: one fixed freq × swept partner ────────────────────
    first = True
    for f_fixed in p.two_tone_fixed_freqs_hz:
        n = int(p.two_tone_subsection_duration_s * sr)
        raw = _two_tone_sweep(n, sr, f_fixed, p.two_tone_sweep_start_hz, p.two_tone_sweep_end_hz)
        label = f"two_tone_sweep_f{int(round(f_fixed))}"
        b.add(_dbfs(p.two_tone_amplitude_dbfs) * raw, label, "two_tone_sweep",
              p.two_tone_amplitude_dbfs, {
            "f_fixed_hz": f_fixed,
            "f_sweep_start_hz": p.two_tone_sweep_start_hz,
            "f_sweep_end_hz": p.two_tone_sweep_end_hz,
            "duration_s": p.two_tone_subsection_duration_s,
            "amplitude_convention": "peak_dbfs",
        }, gap_before=gap if first else 0.0)
        first = False

    # ── 10. Multitone stacks: 3 amplitude levels ──────────────────────────────
    first = True
    for amp in p.multitone_amplitudes_dbfs:
        n = int(p.multitone_duration_s * sr)
        raw = _multitone(n, sr, p.multitone_freqs_hz)
        b.add(_dbfs(amp) * raw, f"multitone_l{int(round(amp))}", "multitone", amp, {
            "freqs_hz": p.multitone_freqs_hz,
            "duration_s": p.multitone_duration_s,
            "amplitude_convention": "peak_dbfs",
        }, gap_before=gap if first else 0.0)
        first = False

    # ── 11. Layered: sweep + pink noise ───────────────────────────────────────
    n = int(p.layered_duration_s * sr)
    sweep_part = _dbfs(-9.0) * _log_sweep(n, sr, p.sweep_f_start_hz, p.sweep_f_end_hz)
    noise_part = _dbfs(-12.0) * _pink_noise(n, rng)
    layered = sweep_part + noise_part
    layered = (_dbfs(-6.0) * layered / (float(np.max(np.abs(layered))) + 1e-10)).astype(np.float32)
    b.add(layered, "layered_sweep_noise", "layered", -6.0, {
        "components": ["log_sine_sweep@-9dBFS", "pink_noise@-12dBFS"],
        "output_peak_dbfs": -6.0,
        "duration_s": p.layered_duration_s,
    }, apply_fade=False, gap_before=gap)

    # ── 12. Layered: multitone + impulse transients ───────────────────────────
    n = int(p.layered_duration_s * sr)
    mt_part = _dbfs(-9.0) * _multitone(n, sr, p.multitone_freqs_hz)
    imp_part = _dbfs(-12.0) * _impulse_train(n, sr, p.impulse_spacings_s)
    layered = mt_part + imp_part
    layered = (_dbfs(-6.0) * layered / (float(np.max(np.abs(layered))) + 1e-10)).astype(np.float32)
    b.add(layered, "layered_multitone_transients", "layered", -6.0, {
        "components": ["multitone@-9dBFS", "impulse_train@-12dBFS"],
        "output_peak_dbfs": -6.0,
        "duration_s": p.layered_duration_s,
    }, apply_fade=False, gap_before=gap)

    # ── 13. Layered: three-way (tone + white noise + impulses) ────────────────
    n = int(p.layered_duration_s * sr)
    tone_part = _dbfs(-9.0) * _sine(n, sr, 440.0)
    noise_part = _dbfs(-15.0) * _white_noise(n, rng, sr=sr)
    imp_part = _dbfs(-18.0) * _impulse_train(n, sr, p.impulse_spacings_s)
    layered = tone_part + noise_part + imp_part
    layered = (_dbfs(-6.0) * layered / (float(np.max(np.abs(layered))) + 1e-10)).astype(np.float32)
    b.add(layered, "layered_three_way", "layered", -6.0, {
        "components": ["sine_440Hz@-9dBFS", "white_noise@-15dBFS", "impulse_train@-18dBFS"],
        "output_peak_dbfs": -6.0,
        "duration_s": p.layered_duration_s,
    }, apply_fade=False, gap_before=gap)

    return b.build(), b.sections


# ─────────────────────────────────────────────────────────────────────────────
# Validation signal assembly
# ─────────────────────────────────────────────────────────────────────────────


def _build_val(p: ValParams, rng: np.random.Generator) -> tuple[np.ndarray, list[dict[str, Any]]]:
    sr = p.sample_rate
    fade_n = max(1, int(0.010 * sr))
    b = _Builder(sr, fade_n)
    gap = p.silence_gap_s

    # ── 1. Amplitude-stepped sweep ────────────────────────────────────────────
    first = True
    for amp in p.stepped_sweep_amplitudes_dbfs:
        n = int(p.stepped_sweep_duration_s * sr)
        audio = _dbfs(amp) * _log_sweep(n, sr, p.stepped_sweep_f_start_hz, p.stepped_sweep_f_end_hz)
        b.add(audio, f"val_sweep_l{int(round(amp))}", "log_sine_sweep", amp, {
            "f_start_hz": p.stepped_sweep_f_start_hz,
            "f_end_hz": p.stepped_sweep_f_end_hz,
            "duration_s": p.stepped_sweep_duration_s,
            "amplitude_convention": "peak_dbfs",
        }, gap_before=gap if first else 0.0)
        first = False

    # ── 2. Stepped tones ──────────────────────────────────────────────────────
    first = True
    for freq in p.val_stepped_freqs_hz:
        for amp in p.val_stepped_levels_dbfs:
            n = int(p.val_stepped_tone_duration_s * sr)
            audio = _dbfs(amp) * _sine(n, sr, freq)
            label = f"val_stepped_tone_f{int(round(freq))}_l{int(round(amp))}"
            b.add(audio, label, "stepped_sine_tone", amp, {
                "freq_hz": freq,
                "duration_s": p.val_stepped_tone_duration_s,
                "amplitude_convention": "peak_dbfs",
            }, gap_before=gap if first else 0.0)
            first = False

    # ── 3. Pink noise ─────────────────────────────────────────────────────────
    first = True
    for amp in p.val_pink_amplitudes_dbfs:
        n = int(p.val_pink_duration_s * sr)
        audio = _dbfs(amp) * _pink_noise(n, rng)
        b.add(audio, f"val_pink_l{int(round(amp))}", "pink_noise", amp, {
            "duration_s": p.val_pink_duration_s,
            "amplitude_convention": "rms_dbfs",
        }, apply_fade=False, gap_before=gap if first else 0.0)
        first = False

    # ── 4. Two-tone IM segments ───────────────────────────────────────────────
    first = True
    for f1, f2 in p.val_two_tone_pairs:
        n = int(p.val_two_tone_duration_s * sr)
        raw = _two_tone(n, sr, f1, f2)
        label = f"val_two_tone_f{int(round(f1))}_f{int(round(f2))}"
        b.add(_dbfs(-9.0) * raw, label, "two_tone", -9.0, {
            "f1_hz": f1,
            "f2_hz": f2,
            "duration_s": p.val_two_tone_duration_s,
            "amplitude_convention": "peak_dbfs",
        }, gap_before=gap if first else 0.0)
        first = False

    # ── 5. Impulse train ──────────────────────────────────────────────────────
    n_imp = int(p.val_impulse_duration_s * sr)
    audio = _dbfs(p.val_impulse_amplitude_dbfs) * _impulse_train(n_imp, sr, p.val_impulse_spacings_s)
    b.add(audio, "val_impulse_train", "impulse_train", p.val_impulse_amplitude_dbfs, {
        "duration_s": p.val_impulse_duration_s,
        "spacings_s": p.val_impulse_spacings_s,
        "amplitude_convention": "peak_dbfs",
    }, apply_fade=False, gap_before=gap)

    # ── 6. Decaying plucks ────────────────────────────────────────────────────
    for freq in p.val_pluck_freqs_hz:
        pluck = _dbfs(p.val_pluck_amplitude_dbfs) * _decaying_pluck(sr, freq, p.val_pluck_decay_tau_s)
        b.add(pluck, f"val_pluck_f{int(round(freq))}", "decaying_pluck", p.val_pluck_amplitude_dbfs, {
            "freq_hz": freq,
            "decay_tau_s": p.val_pluck_decay_tau_s,
            "amplitude_convention": "peak_dbfs",
        }, apply_fade=False, gap_before=0.0)

    return b.build(), b.sections


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def generate(
    signal: str,
    output_dir: Path | str,
    *,
    train_params: TrainParams | None = None,
    val_params: ValParams | None = None,
) -> tuple[Path, Path]:
    """Generate a WAV + JSON manifest pair and write to output_dir.

    Args:
        signal: ``"train"`` or ``"val"``.
        output_dir: Directory to write ``{name}.wav`` and ``{name}.json``.
        train_params: Override defaults for the train signal.
        val_params: Override defaults for the val signal.

    Returns:
        ``(wav_path, json_path)``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if signal == "train":
        p = train_params or TrainParams()
        rng = np.random.default_rng(p.seed)
        audio, sections = _build_train(p, rng)
        name = "train_signal_v1"
        params_dict: dict[str, Any] = asdict(p)
    elif signal == "val":
        p = val_params or ValParams()
        rng = np.random.default_rng(p.seed)
        audio, sections = _build_val(p, rng)
        name = "val_signal_v1"
        params_dict = asdict(p)
    else:
        raise ValueError(f"Unknown signal: {signal!r}. Choose 'train' or 'val'.")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "signal_name": name,
        "sample_rate": p.sample_rate,
        "bit_depth": 32,
        "format": "float",
        "seed": p.seed,
        "total_samples": len(audio),
        "total_duration_s": round(len(audio) / p.sample_rate, 6),
        "params": params_dict,
        "sections": sections,
    }

    wav_path = output_dir / f"{name}.wav"
    json_path = output_dir / f"{name}.json"

    sf.write(str(wav_path), audio, p.sample_rate, subtype="FLOAT")
    json_path.write_text(json.dumps(manifest, indent=2))

    return wav_path, json_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate PedalDSP training/validation signals (WAV + JSON manifest).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--signal", choices=["train", "val", "both"], default="both",
        help="Which signal(s) to generate.",
    )
    parser.add_argument("--output", default="data/signals", help="Output directory.")
    parser.add_argument("--sr", type=int, default=96_000, help="Sample rate in Hz.")
    parser.add_argument("--seed-train", type=int, default=1234, help="RNG seed for train signal.")
    parser.add_argument("--seed-val", type=int, default=42, help="RNG seed for val signal.")
    args = parser.parse_args(argv)

    signals = ["train", "val"] if args.signal == "both" else [args.signal]
    for sig in signals:
        if sig == "train":
            p_train = TrainParams(sample_rate=args.sr, seed=args.seed_train)
            p_val = None
        else:
            p_train = None
            p_val = ValParams(sample_rate=args.sr, seed=args.seed_val)

        print(f"Generating {sig} signal ...", end=" ", flush=True)
        wav_path, json_path = generate(sig, args.output, train_params=p_train, val_params=p_val)
        duration = json.loads(json_path.read_text())["total_duration_s"]
        n_sections = len(json.loads(json_path.read_text())["sections"])
        print(f"done  {wav_path.name}  ({duration:.1f} s, {n_sections} sections)  {json_path.name}")


if __name__ == "__main__":
    main()
