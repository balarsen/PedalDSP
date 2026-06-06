#!/usr/bin/env python3
"""
generate_capture_signal.py — CLI entry point for PedalDSP signal generation.

NEW (v2) — train/val signals with full JSON manifest:
  python generate_capture_signal.py --signal train --output data/signals/
  python generate_capture_signal.py --signal val   --output data/signals/
  python generate_capture_signal.py --signal both  --output data/signals/

  Delegates to pedal_model.signals.generate; WAV + JSON written together.
  All parameters (seed, sample rate, durations, freqs, levels) are dataclass
  fields — see pedal_model/signals/generate.py for the full spec.

LEGACY (v1) — preset-based 48 kHz capture signals:
  python generate_capture_signal.py --preset master
  python generate_capture_signal.py --preset fuzz
  python generate_capture_signal.py --preset overdrive --output-dir ./my_signals
  python generate_capture_signal.py --preset custom
  python generate_capture_signal.py --list-presets

  MASTER REFERENCE: generate once with --preset master; never regenerate.
  All capture sessions must use the same master WAV as the dry source so
  models trained across sessions remain directly comparable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — all parameters here, no magic numbers elsewhere
# ─────────────────────────────────────────────────────────────────────────────

CONFIG: dict = {

    # ── Global ────────────────────────────────────────────────────────────────

    "SAMPLE_RATE": 48000,
    # Hz. Target sample rate. 48 kHz is standard for audio interfaces.
    # Changing this scales every duration/sample-count in the script.

    "BIT_DEPTH": 24,
    # Output WAV bit depth: 16 or 24.
    # 24-bit gives 144 dB dynamic range — always use 24 for captures.

    "OUTPUT_DIR": "./data/signals/legacy",
    # Directory for all output files. Created if it does not exist.

    "MASTER_FILENAME": "master_reference.wav",
    # Filename for the permanent master reference. Never change this string
    # after your first master capture, or existing references break.

    "SESSION_FILENAME": "capture_{preset}_{timestamp}.wav",
    # Template for non-master output filenames.
    # {preset} and {timestamp} are substituted at runtime.

    "SEED": 42,
    # RNG seed for the noise section. Fixed seed → identical noise across
    # runs, essential for the master reference concept.

    "NORMALIZE_HEADROOM_DB": -3.0,
    # dBFS peak of the final normalized output. -3 dB leaves 3 dB of
    # headroom to protect against interface ADC clipping.
    # Make more negative if your interface clips; less if you need more level.

    # ── Alignment Click ───────────────────────────────────────────────────────

    "CLICK_ENABLED": True,
    # Single-spike alignment marker at the very start of the file.
    # Used to time-align the dry and wet recordings after capture.
    # Disable only if your interface provides hardware word-clock sync.

    "CLICK_DURATION_MS": 10.0,
    # Total duration of the click section in ms.
    # The spike is at sample 0; the rest is silence within this window.

    "CLICK_AMPLITUDE": 0.9,
    # Peak amplitude of the spike (0–1). High enough to be unambiguous
    # in cross-correlation alignment; below 1.0 to avoid ADC overload.

    "CLICK_SILENCE_AFTER_MS": 500.0,
    # Silence after the click before the test signals begin.
    # Gives the interface time to settle and keeps the click isolated.

    # ── Log Sweep (Farina exponential sweep) ──────────────────────────────────

    "SWEEP_ENABLED": True,
    # Full-bandwidth logarithmic sine sweep. Covers the frequency response
    # of the pedal and is the basis for impulse response extraction.

    "SWEEP_FREQ_START": 20.0,
    # Hz. Start frequency. 20 Hz captures sub-bass and low-frequency
    # input-coupling capacitor roll-offs common in fuzz circuits.

    "SWEEP_FREQ_END": 20000.0,
    # Hz. End frequency. Full audible range.

    "SWEEP_DURATION_SEC": 10.0,
    # Duration of each sweep repetition. Longer = better SNR via averaging
    # but more total file length. 10 s is the standard for guitar pedals.

    "SWEEP_REPETITIONS": 2,
    # Number of times to repeat the full sweep. Repetitions allow the
    # training pipeline to average out random noise, or check consistency.

    "SWEEP_SILENCE_BETWEEN_SEC": 1.0,
    # Silence inserted between sweep repetitions.

    "SWEEP_GUITAR_RANGE_ENABLED": True,
    # An additional sweep focused on the guitar frequency range (see below).
    # Spends proportionally more energy where guitar content actually lives.

    "SWEEP_GUITAR_FREQ_START": 80.0,
    # Hz. Low E string fundamental (82.4 Hz). Start of focused sweep.

    "SWEEP_GUITAR_FREQ_END": 8000.0,
    # Hz. Well above the highest significant guitar harmonic.

    "SWEEP_GUITAR_DURATION_SEC": 10.0,
    # Duration of the focused guitar-range sweep.

    # ── Amplitude Envelope Sweep (critical for fuzz / saturation) ─────────────

    "AMP_SWEEP_ENABLED": False,
    # Plays a sustained sine at a fixed frequency while the amplitude rises
    # from near-silence to near-clipping. Exposes the level-dependent
    # nonlinear behavior: where fuzz cleans up vs. where it saturates hard.
    # Essential for fuzz and overdrive; less important for linear pedals.

    "AMP_SWEEP_FREQ_HZ": 440.0,
    # Primary carrier frequency for the amplitude sweep. A4 = 440 Hz is
    # a good default (near the middle of the guitar range).

    "AMP_SWEEP_ADDITIONAL_FREQS": [110.0, 220.0, 880.0],
    # Additional carrier frequencies to run the amplitude sweep at.
    # Multiple frequencies reveal whether clipping threshold is
    # frequency-dependent (it often is in fuzz circuits).

    "AMP_SWEEP_DURATION_SEC": 8.0,
    # Duration of the amplitude sweep per carrier frequency.

    "AMP_SWEEP_MIN_AMPLITUDE": 0.001,
    # Starting amplitude. Near-silence so we capture the linear regime
    # and clean-up behavior before saturation kicks in.

    "AMP_SWEEP_MAX_AMPLITUDE": 0.95,
    # Peak amplitude. Below 1.0 to avoid pre-ADC clipping of the dry signal.

    "AMP_SWEEP_SHAPE": "log",
    # Envelope shape: 'linear' | 'log' | 'triangle'.
    # 'log': amplitude grows exponentially — spends more time at quiet levels
    #        where fuzz behavior is richest (recommended for fuzz).
    # 'linear': uniform time across all amplitude levels.
    # 'triangle': rises to peak then falls back — captures hysteresis effects.

    # ── Intermodulation Test Tones (fuzz / saturation) ────────────────────────

    "IM_TONES_ENABLED": False,
    # Plays two simultaneous sine waves. Nonlinear devices produce sum and
    # difference frequencies not present in the input — these intermodulation
    # products are characteristic of fuzz/saturation circuits.
    # A model that learns IM correctly will sound more authentic.

    "IM_TONE_PAIRS": [[220, 330], [440, 660], [880, 1320], [110, 440]],
    # List of [f1, f2] pairs in Hz. Ratios matter: 2:3 (perfect fifth) and
    # 3:2 are common guitar intervals. 110:440 tests wide-interval IM.

    "IM_TONE_DURATION_SEC": 3.0,
    # Duration per tone pair. Long enough to measure steady-state IM.

    "IM_TONE_AMPLITUDE": 0.4,
    # Amplitude per tone (0–1). Two tones summed peak at 2× this value,
    # so 0.4 per tone → 0.8 combined — stays below clipping in the dry path.

    # ── Transient Tests (attack / decay dynamics) ─────────────────────────────

    "TRANSIENT_ENABLED": False,
    # Generates note-like events with controlled attack and decay.
    # Fuzz circuits react differently to slow vs. fast pick attacks due to
    # capacitor charge/discharge time constants. This section captures that.

    "TRANSIENT_FREQ_HZ": 220.0,
    # Carrier frequency for transient tests. A3 = 220 Hz is a typical
    # guitar note in the middle of the fuzz's working range.

    "TRANSIENT_SLOW_ATTACK_MS": 50.0,
    # Fade-in time for slow-attack notes (violin-bow or volume-knob roll-on).
    # Fuzz often stays "clean" during a slow attack even at high final level.

    "TRANSIENT_FAST_ATTACK_MS": 1.0,
    # Fade-in time for fast-attack notes (hard pick strike). Near-instant
    # onset — exposes how the circuit responds to transient overload.

    "TRANSIENT_SUSTAIN_MS": 500.0,
    # Duration of the sustained portion at full amplitude.

    "TRANSIENT_DECAY_MS": 200.0,
    # Fade-out time simulating natural note decay.

    "TRANSIENT_REPETITIONS": 8,
    # Number of notes per attack type (slow and fast are separate sequences).

    "TRANSIENT_SILENCE_BETWEEN_MS": 300.0,
    # Silence between individual notes. For delay preset: make this long
    # (800+ ms) so delay echoes are fully captured before the next note.

    # ── White Noise Burst ─────────────────────────────────────────────────────

    "NOISE_ENABLED": True,
    # Broadband random signal. Exercises all frequencies simultaneously.
    # Useful as a final catch-all after the structured sweeps.

    "NOISE_DURATION_SEC": 5.0,
    # Duration of the noise burst.

    "NOISE_AMPLITUDE": 0.5,
    # RMS-equivalent amplitude (0–1). The random signal is normalized then
    # scaled to this value, so actual peaks will be higher (crest factor ~3).

    "NOISE_BANDPASS_ENABLED": True,
    # Apply a 4th-order Butterworth bandpass before output.
    # Limits noise to the frequency range the pedal actually operates in.

    "NOISE_BANDPASS_LOW_HZ": 80.0,
    # Lower bandpass cutoff. Below this is typically not useful for guitar.

    "NOISE_BANDPASS_HIGH_HZ": 8000.0,
    # Upper bandpass cutoff.

    # ── Silence Gaps ──────────────────────────────────────────────────────────

    "SILENCE_BETWEEN_SECTIONS_SEC": 1.0,
    # Silence inserted between every major section. Gives the pedal's
    # internal state time to settle between sections. Also makes the
    # segment map easier to read in a DAW.

    "SILENCE_AT_END_SEC": 3.0,
    # Silence at the very end of the file. Critical for delay and reverb
    # captures — must be longer than the longest expected effect tail.

    # ── Metadata ──────────────────────────────────────────────────────────────

    "WRITE_METADATA_JSON": True,
    # Write a sidecar _meta.json with all resolved CONFIG values, preset
    # name, git hash, and timestamp. Required for reproducibility.

    "WRITE_SEGMENT_MAP": True,
    # Write a sidecar _segments.json with sample-accurate start/end and
    # label for every section. Training pipeline uses this to slice sections
    # without hardcoding byte offsets.
}


# ─────────────────────────────────────────────────────────────────────────────
# PRESETS  — override only the parameters relevant to each pedal type
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {

    "master": {
        # Conservative, broad, reproducible. Three sweep repetitions for
        # SNR averaging. No nonlinear-specific sections — this is a clean
        # frequency-response reference, not a fuzz-optimized training set.
        # Parameters touched: SWEEP_REPETITIONS, SWEEP_GUITAR_RANGE_ENABLED,
        #                     SILENCE_AT_END_SEC, SEED
        "SWEEP_REPETITIONS": 3,
        "SWEEP_GUITAR_RANGE_ENABLED": True,
        "NOISE_AMPLITUDE": 0.5,
        "SILENCE_AT_END_SEC": 5.0,
        "SEED": 42,   # Fixed forever — never change for master.
    },

    "fuzz": {
        # Heavy nonlinear effects. Amplitude sweeps expose where clipping
        # kicks in. IM tones reveal intermodulation character. Transients
        # capture slow/fast pick attack response. Hotter noise exercises
        # full saturation. Extended low-end (30 Hz start) because some fuzzes
        # have unexpected coupling-cap behavior below 80 Hz.
        # Parameters touched: AMP_SWEEP_*, IM_TONES_*, TRANSIENT_*,
        #                     SWEEP_FREQ_START, NOISE_AMPLITUDE, SILENCE_AT_END_SEC
        "SWEEP_FREQ_START": 30.0,
        "AMP_SWEEP_ENABLED": True,
        "AMP_SWEEP_SHAPE": "log",
        "AMP_SWEEP_MIN_AMPLITUDE": 0.001,
        "AMP_SWEEP_MAX_AMPLITUDE": 0.95,
        "AMP_SWEEP_FREQ_HZ": 440.0,
        "AMP_SWEEP_ADDITIONAL_FREQS": [110.0, 220.0, 880.0],
        "IM_TONES_ENABLED": True,
        "IM_TONE_PAIRS": [[220, 330], [440, 660], [880, 1320], [110, 440]],
        "TRANSIENT_ENABLED": True,
        "TRANSIENT_SLOW_ATTACK_MS": 50.0,
        "TRANSIENT_FAST_ATTACK_MS": 1.0,
        "TRANSIENT_REPETITIONS": 8,
        "TRANSIENT_SILENCE_BETWEEN_MS": 300.0,
        "NOISE_AMPLITUDE": 0.7,
        "SILENCE_AT_END_SEC": 3.0,
    },

    "overdrive": {
        # Moderate nonlinearity. Emphasis on mid-frequency harmonic content.
        # Amplitude sweep uses linear shape (overdrive saturates more gradually
        # than fuzz — uniform amplitude coverage matters more). IM tones at
        # musically relevant intervals. No transient test needed (overdrive
        # attack dynamics are less extreme than fuzz).
        # Parameters touched: AMP_SWEEP_*, IM_TONES_*, SWEEP_GUITAR_*,
        #                     NOISE_AMPLITUDE, SILENCE_AT_END_SEC
        "AMP_SWEEP_ENABLED": True,
        "AMP_SWEEP_SHAPE": "linear",
        "AMP_SWEEP_FREQ_HZ": 440.0,
        "AMP_SWEEP_ADDITIONAL_FREQS": [220.0, 880.0, 1760.0],
        "AMP_SWEEP_MIN_AMPLITUDE": 0.005,
        "AMP_SWEEP_MAX_AMPLITUDE": 0.85,
        "IM_TONES_ENABLED": True,
        "IM_TONE_PAIRS": [[220, 330], [440, 880], [660, 1320]],
        "IM_TONE_AMPLITUDE": 0.4,
        "TRANSIENT_ENABLED": False,
        "SWEEP_GUITAR_RANGE_ENABLED": True,
        "NOISE_AMPLITUDE": 0.5,
        "SILENCE_AT_END_SEC": 2.0,
    },

    "delay": {
        # Time-based effect. Long silence gaps let echo tails fully decay
        # before the next section. Transient section acts like an impulse
        # train with long inter-note silences to expose each delay repeat.
        # No amplitude sweep or IM tones needed — delay is linear.
        # Parameters touched: TRANSIENT_*, SILENCE_*, NOISE_DURATION_SEC
        "AMP_SWEEP_ENABLED": False,
        "IM_TONES_ENABLED": False,
        "TRANSIENT_ENABLED": True,
        "TRANSIENT_FREQ_HZ": 220.0,
        "TRANSIENT_FAST_ATTACK_MS": 1.0,
        "TRANSIENT_SLOW_ATTACK_MS": 5.0,
        "TRANSIENT_SUSTAIN_MS": 150.0,
        "TRANSIENT_DECAY_MS": 50.0,
        "TRANSIENT_REPETITIONS": 16,
        "TRANSIENT_SILENCE_BETWEEN_MS": 800.0,
        "SILENCE_BETWEEN_SECTIONS_SEC": 2.0,
        "SILENCE_AT_END_SEC": 6.0,
        "NOISE_DURATION_SEC": 3.0,
    },

    "reverb": {
        # Reverb preset. Very long silence gaps (room tails can exceed 2 s).
        # Focused sweep range 200–4 kHz where room modes are most audible.
        # Transient section acts as impulse-response stimulus with very long
        # inter-note silences to fully capture room decay.
        # Parameters touched: SWEEP_FREQ_*, SWEEP_GUITAR_RANGE_ENABLED,
        #                     TRANSIENT_*, NOISE_BANDPASS_*, SILENCE_*
        "SWEEP_FREQ_START": 200.0,
        "SWEEP_FREQ_END": 4000.0,
        "SWEEP_GUITAR_RANGE_ENABLED": False,
        "AMP_SWEEP_ENABLED": False,
        "IM_TONES_ENABLED": False,
        "TRANSIENT_ENABLED": True,
        "TRANSIENT_FREQ_HZ": 440.0,
        "TRANSIENT_FAST_ATTACK_MS": 1.0,
        "TRANSIENT_SLOW_ATTACK_MS": 10.0,
        "TRANSIENT_SUSTAIN_MS": 300.0,
        "TRANSIENT_DECAY_MS": 100.0,
        "TRANSIENT_REPETITIONS": 10,
        "TRANSIENT_SILENCE_BETWEEN_MS": 2500.0,
        "NOISE_BANDPASS_LOW_HZ": 200.0,
        "NOISE_BANDPASS_HIGH_HZ": 4000.0,
        "SILENCE_BETWEEN_SECTIONS_SEC": 3.0,
        "SILENCE_AT_END_SEC": 8.0,
    },

    "clean": {
        # Linear/clean pedals: compressors, EQ, chorus, buffers.
        # Full amplitude range with fine frequency granularity (six extra
        # amplitude-sweep carriers cover the full guitar fretboard range).
        # No IM or transient tests — these pedals are approximately linear
        # and level-independent. More sweep repetitions for better averaging.
        # Parameters touched: AMP_SWEEP_*, SWEEP_REPETITIONS,
        #                     SWEEP_GUITAR_RANGE_ENABLED, SILENCE_AT_END_SEC
        "AMP_SWEEP_ENABLED": True,
        "AMP_SWEEP_SHAPE": "linear",
        "AMP_SWEEP_FREQ_HZ": 440.0,
        "AMP_SWEEP_ADDITIONAL_FREQS": [82.0, 165.0, 330.0, 660.0, 1320.0, 2640.0],
        "AMP_SWEEP_MIN_AMPLITUDE": 0.01,
        "AMP_SWEEP_MAX_AMPLITUDE": 0.90,
        "IM_TONES_ENABLED": False,
        "TRANSIENT_ENABLED": False,
        "SWEEP_REPETITIONS": 3,
        "SWEEP_GUITAR_RANGE_ENABLED": True,
        "NOISE_AMPLITUDE": 0.5,
        "SILENCE_AT_END_SEC": 2.0,
    },

    "custom": {},
    # Uses CONFIG values exactly as written above. No overrides.
    # Change CONFIG directly to customise.
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _silence(n_samples: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


def _apply_fades(sig: np.ndarray, sr: int, fade_ms: float = 5.0) -> np.ndarray:
    """Apply equal-power fade-in and fade-out to avoid inter-section clicks."""
    fade_n = min(int(fade_ms / 1000.0 * sr), len(sig) // 2)
    if fade_n < 1:
        return sig
    result = sig.copy()
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    result[:fade_n] *= ramp
    result[-fade_n:] *= ramp[::-1]
    return result


def _get_git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _bit_depth_to_subtype(bit_depth: int) -> str:
    mapping = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
    if bit_depth not in mapping:
        raise ValueError(f"Unsupported BIT_DEPTH {bit_depth}. Use 16, 24, or 32.")
    return mapping[bit_depth]


# ─────────────────────────────────────────────────────────────────────────────
# Signal generators  — each returns a float32 numpy array, unfaded
# ─────────────────────────────────────────────────────────────────────────────

def gen_click(cfg: dict) -> np.ndarray:
    """Single-sample spike for channel alignment via cross-correlation."""
    n = max(1, int(cfg["CLICK_DURATION_MS"] / 1000.0 * cfg["SAMPLE_RATE"]))
    sig = np.zeros(n, dtype=np.float32)
    sig[0] = float(cfg["CLICK_AMPLITUDE"])
    return sig   # no fades — spike must remain sharp


def gen_log_sweep(f1: float, f2: float, duration: float, sr: int, amplitude: float = 1.0) -> np.ndarray:
    """
    Farina exponential sine sweep from f1 to f2.

    Instantaneous frequency: f(t) = f1 * (f2/f1)^(t/T)
    Phase integral:          φ(t) = 2π · f1 · L · (exp(t/L) - 1)
    where L = T / ln(f2/f1).
    """
    n = int(duration * sr)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    L = duration / np.log(f2 / f1)
    phase = 2.0 * np.pi * f1 * L * (np.exp(t / L) - 1.0)
    return (amplitude * np.sin(phase)).astype(np.float32)


def gen_amplitude_sweep(freq: float, duration: float, sr: int,
                        min_amp: float, max_amp: float, shape: str) -> np.ndarray:
    """
    Fixed-frequency sine with a time-varying amplitude envelope.

    Exposes level-dependent nonlinearities (fuzz clean-up, saturation onset).
    """
    n = int(duration * sr)
    u = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)  # normalised time

    if shape == "linear":
        envelope = min_amp + (max_amp - min_amp) * u
    elif shape == "log":
        # Exponential growth — more resolution at quiet levels where fuzz cleans up.
        envelope = min_amp * (max_amp / min_amp) ** u
    elif shape == "triangle":
        # Rise to peak then fall back — reveals hysteresis in the nonlinearity.
        envelope = min_amp + (max_amp - min_amp) * (1.0 - np.abs(2.0 * u - 1.0))
    else:
        raise ValueError(f"Unknown AMP_SWEEP_SHAPE '{shape}'. Use 'linear', 'log', or 'triangle'.")

    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    carrier = np.sin(2.0 * np.pi * freq * t)
    return (envelope * carrier).astype(np.float32)


def gen_im_tones(f1: float, f2: float, duration: float, sr: int, amplitude: float) -> np.ndarray:
    """
    Two simultaneous sine waves at f1 and f2.

    Nonlinear devices produce sum (f1+f2) and difference (f2-f1) frequencies
    not present in the input — a fingerprint of the circuit's nonlinearity.
    """
    n = int(duration * sr)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    tone = amplitude * np.sin(2.0 * np.pi * f1 * t) + amplitude * np.sin(2.0 * np.pi * f2 * t)
    return tone.astype(np.float32)


def gen_transient_sequence(freq: float, attack_ms: float, sustain_ms: float,
                           decay_ms: float, repetitions: int,
                           silence_between_ms: float, sr: int) -> np.ndarray:
    """
    Repeated note-like events with controlled attack/decay envelopes.

    Fuzz circuits respond differently to fast vs. slow attacks because the
    capacitor charge/discharge time constant creates an input-impedance
    nonlinearity that varies with signal rate-of-change.
    """
    atk_n  = max(1, int(attack_ms        / 1000.0 * sr))
    sus_n  = max(1, int(sustain_ms       / 1000.0 * sr))
    dec_n  = max(1, int(decay_ms         / 1000.0 * sr))
    sil_n  = max(0, int(silence_between_ms / 1000.0 * sr))

    note_n = atk_n + sus_n + dec_n
    t_note = np.linspace(0.0, note_n / sr, note_n, endpoint=False, dtype=np.float64)

    # ADSR envelope: linear attack, flat sustain, linear decay
    envelope = np.empty(note_n, dtype=np.float64)
    envelope[:atk_n]             = np.linspace(0.0, 1.0, atk_n)
    envelope[atk_n:atk_n+sus_n]  = 1.0
    envelope[atk_n+sus_n:]       = np.linspace(1.0, 0.0, dec_n)

    carrier = np.sin(2.0 * np.pi * freq * t_note)
    note = (envelope * carrier).astype(np.float32)
    gap  = _silence(sil_n)

    chunks = []
    for _ in range(repetitions):
        chunks.append(note)
        chunks.append(gap)
    return np.concatenate(chunks)


def gen_noise_burst(duration: float, sr: int, amplitude: float,
                    bandpass: bool, low_hz: float, high_hz: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Bandlimited white noise burst.

    Exercises all frequencies simultaneously. 4th-order Butterworth limits
    energy to the guitar operating range so the interface ADC isn't stressed
    by ultra-low or ultra-high out-of-band content.
    """
    n = int(duration * sr)
    sig = rng.standard_normal(n).astype(np.float64)

    if bandpass and low_hz < high_hz:
        sos = butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
        sig = sosfilt(sos, sig)

    # Normalise to unit peak, then scale to requested amplitude.
    peak = np.max(np.abs(sig))
    if peak > 1e-9:
        sig = sig / peak * amplitude
    return sig.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Signal assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_signal(cfg: dict) -> tuple[np.ndarray, list[dict]]:
    """
    Assemble all enabled sections into a single flat float32 array.

    Returns (signal, segments) where segments is a list of dicts:
        {"label": str, "start_sample": int, "end_sample": int}
    start_sample is inclusive, end_sample is exclusive.
    """
    sr   = cfg["SAMPLE_RATE"]
    rng  = np.random.default_rng(cfg["SEED"])
    gap  = _silence(int(cfg["SILENCE_BETWEEN_SECTIONS_SEC"] * sr))
    fade_ms = 5.0  # applied to every section except the click

    sections: list[np.ndarray] = []
    labels:   list[tuple[str, int]] = []  # (label, n_samples_before_this_section)

    def _add(label: str, sig: np.ndarray, with_fade: bool = True) -> None:
        if with_fade:
            sig = _apply_fades(sig, sr, fade_ms)
        start = sum(len(s) for s in sections)
        sections.append(sig)
        sections.append(gap.copy())
        labels.append((label, start, start + len(sig)))

    # ── Alignment click ───────────────────────────────────────────────────────
    if cfg["CLICK_ENABLED"]:
        click_sig = gen_click(cfg)
        start = sum(len(s) for s in sections)
        sections.append(click_sig)
        labels.append(("alignment_click", start, start + len(click_sig)))
        sections.append(_silence(int(cfg["CLICK_SILENCE_AFTER_MS"] / 1000.0 * sr)))

    # ── Log sweeps ────────────────────────────────────────────────────────────
    if cfg["SWEEP_ENABLED"]:
        f1, f2 = cfg["SWEEP_FREQ_START"], cfg["SWEEP_FREQ_END"]
        dur    = cfg["SWEEP_DURATION_SEC"]
        for rep in range(cfg["SWEEP_REPETITIONS"]):
            sig = gen_log_sweep(f1, f2, dur, sr)
            _add(f"log_sweep_{rep+1}", sig)
            if rep < cfg["SWEEP_REPETITIONS"] - 1:
                # Extra silence between repetitions
                sections.append(_silence(int(cfg["SWEEP_SILENCE_BETWEEN_SEC"] * sr)))

        if cfg["SWEEP_GUITAR_RANGE_ENABLED"]:
            g1, g2 = cfg["SWEEP_GUITAR_FREQ_START"], cfg["SWEEP_GUITAR_FREQ_END"]
            gdur   = cfg["SWEEP_GUITAR_DURATION_SEC"]
            sig = gen_log_sweep(g1, g2, gdur, sr)
            _add("guitar_range_sweep", sig)

    # ── Amplitude sweeps ──────────────────────────────────────────────────────
    if cfg["AMP_SWEEP_ENABLED"]:
        all_freqs = [cfg["AMP_SWEEP_FREQ_HZ"]] + list(cfg["AMP_SWEEP_ADDITIONAL_FREQS"])
        for freq in all_freqs:
            sig = gen_amplitude_sweep(
                freq=freq,
                duration=cfg["AMP_SWEEP_DURATION_SEC"],
                sr=sr,
                min_amp=cfg["AMP_SWEEP_MIN_AMPLITUDE"],
                max_amp=cfg["AMP_SWEEP_MAX_AMPLITUDE"],
                shape=cfg["AMP_SWEEP_SHAPE"],
            )
            _add(f"amp_sweep_{int(freq)}Hz", sig)

    # ── Intermodulation tone pairs ─────────────────────────────────────────────
    if cfg["IM_TONES_ENABLED"]:
        for f1, f2 in cfg["IM_TONE_PAIRS"]:
            sig = gen_im_tones(
                f1=f1, f2=f2,
                duration=cfg["IM_TONE_DURATION_SEC"],
                sr=sr,
                amplitude=cfg["IM_TONE_AMPLITUDE"],
            )
            _add(f"im_tones_{int(f1)}_{int(f2)}Hz", sig)

    # ── Transients (slow and fast attack, separate sequences) ────────────────
    if cfg["TRANSIENT_ENABLED"]:
        for attack_ms, label_tag in [
            (cfg["TRANSIENT_SLOW_ATTACK_MS"], "slow_attack"),
            (cfg["TRANSIENT_FAST_ATTACK_MS"], "fast_attack"),
        ]:
            sig = gen_transient_sequence(
                freq=cfg["TRANSIENT_FREQ_HZ"],
                attack_ms=attack_ms,
                sustain_ms=cfg["TRANSIENT_SUSTAIN_MS"],
                decay_ms=cfg["TRANSIENT_DECAY_MS"],
                repetitions=cfg["TRANSIENT_REPETITIONS"],
                silence_between_ms=cfg["TRANSIENT_SILENCE_BETWEEN_MS"],
                sr=sr,
            )
            _add(f"transients_{label_tag}", sig)

    # ── Noise burst ───────────────────────────────────────────────────────────
    if cfg["NOISE_ENABLED"]:
        sig = gen_noise_burst(
            duration=cfg["NOISE_DURATION_SEC"],
            sr=sr,
            amplitude=cfg["NOISE_AMPLITUDE"],
            bandpass=cfg["NOISE_BANDPASS_ENABLED"],
            low_hz=cfg["NOISE_BANDPASS_LOW_HZ"],
            high_hz=cfg["NOISE_BANDPASS_HIGH_HZ"],
            rng=rng,
        )
        _add("noise_burst", sig)

    # ── Trailing silence ──────────────────────────────────────────────────────
    tail = _silence(int(cfg["SILENCE_AT_END_SEC"] * sr))
    tail_start = sum(len(s) for s in sections)
    sections.append(tail)
    labels.append(("tail_silence", tail_start, tail_start + len(tail)))

    # ── Concatenate and normalise ─────────────────────────────────────────────
    full = np.concatenate(sections)
    peak = np.max(np.abs(full))
    if peak > 1e-9:
        target = 10.0 ** (cfg["NORMALIZE_HEADROOM_DB"] / 20.0)
        full = (full * (target / peak)).astype(np.float32)

    segments = [
        {"label": lbl, "start_sample": int(s), "end_sample": int(e)}
        for lbl, s, e in labels
    ]
    return full, segments


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    signal: np.ndarray,
    segments: list[dict],
    cfg: dict,
    preset: str,
    output_dir: Path,
    base_name: str,
) -> dict[str, Path]:
    """Write WAV, metadata JSON, and segment map JSON. Returns paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subtype = _bit_depth_to_subtype(cfg["BIT_DEPTH"])
    paths: dict[str, Path] = {}

    # WAV
    wav_path = output_dir / base_name
    sf.write(str(wav_path), signal, cfg["SAMPLE_RATE"], subtype=subtype)
    paths["wav"] = wav_path

    stem = wav_path.stem

    # Metadata JSON
    if cfg["WRITE_METADATA_JSON"]:
        meta = {
            "preset": preset,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_hash": _get_git_hash(),
            "total_samples": len(signal),
            "total_duration_sec": len(signal) / cfg["SAMPLE_RATE"],
            "peak_amplitude": float(np.max(np.abs(signal))),
            "config": cfg,
        }
        meta_path = output_dir / f"{stem}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        paths["meta"] = meta_path

    # Segment map JSON
    if cfg["WRITE_SEGMENT_MAP"]:
        seg_path = output_dir / f"{stem}_segments.json"
        seg_path.write_text(json.dumps(segments, indent=2))
        paths["segments"] = seg_path

    return paths


def print_summary(signal: np.ndarray, segments: list[dict], cfg: dict, preset: str,
                  paths: dict[str, Path]) -> None:
    sr = cfg["SAMPLE_RATE"]
    dur = len(signal) / sr
    peak_db = 20.0 * np.log10(max(np.max(np.abs(signal)), 1e-9))

    print()
    print("═" * 60)
    print("  Capture signal generated")
    print("═" * 60)
    print(f"  Preset         : {preset}")
    print(f"  Sample rate    : {sr} Hz")
    print(f"  Bit depth      : {cfg['BIT_DEPTH']}-bit")
    print(f"  Total duration : {dur:.1f} s  ({dur/60:.1f} min)")
    print(f"  Total samples  : {len(signal):,}")
    print(f"  Peak level     : {peak_db:.2f} dBFS")
    print(f"  Sections       : {len(segments)}")
    print()
    print("  Sections:")
    for seg in segments:
        sec_dur = (seg["end_sample"] - seg["start_sample"]) / sr
        print(f"    [{seg['start_sample']:>8} – {seg['end_sample']:>8}]  "
              f"{sec_dur:6.2f} s  {seg['label']}")
    print()
    print("  Output files:")
    for key, path in paths.items():
        print(f"    {key:<10}: {path}")
    print("═" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def resolve_config(preset: str) -> dict:
    """Merge preset overrides on top of CONFIG defaults."""
    if preset not in PRESETS:
        print(f"ERROR: unknown preset '{preset}'. Use --list-presets to see options.",
              file=sys.stderr)
        sys.exit(1)
    cfg = dict(CONFIG)
    cfg.update(PRESETS[preset])
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PedalDSP capture/training signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── New v2 flags (delegate to pedal_model.signals.generate) ───────────────
    parser.add_argument(
        "--signal", default=None, choices=["train", "val", "both"],
        help="Generate train/val signal with JSON manifest (v2). "
             "Overrides --preset; output goes to --output.",
    )
    parser.add_argument(
        "--output", default="data/signals",
        help="Output directory for --signal (v2 mode).",
    )
    parser.add_argument("--sr", type=int, default=96_000, help="Sample rate for --signal.")
    parser.add_argument("--seed-train", type=int, default=1234)
    parser.add_argument("--seed-val", type=int, default=42)
    # ── Legacy v1 flags ───────────────────────────────────────────────────────
    parser.add_argument(
        "--preset", default="custom",
        choices=list(PRESETS.keys()),
        help="(Legacy) signal preset. 'master' generates the permanent reference file.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="(Legacy) override CONFIG['OUTPUT_DIR'].",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="(Legacy) allow overwriting master_reference.wav.",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="(Legacy) print available presets and exit.",
    )
    args = parser.parse_args()

    # ── v2 path ────────────────────────────────────────────────────────────────
    if args.signal is not None:
        from pedal_model.signals.generate import TrainParams, ValParams
        from pedal_model.signals.generate import generate as _generate
        import json as _json

        sigs = ["train", "val"] if args.signal == "both" else [args.signal]
        for sig in sigs:
            p_train = TrainParams(sample_rate=args.sr, seed=args.seed_train) if sig == "train" else None
            p_val = ValParams(sample_rate=args.sr, seed=args.seed_val) if sig == "val" else None
            print(f"Generating {sig} signal ...", end=" ", flush=True)
            wav_path, json_path = _generate(sig, args.output, train_params=p_train, val_params=p_val)
            info = _json.loads(json_path.read_text())
            print(f"done  {wav_path.name}  ({info['total_duration_s']:.1f}s, "
                  f"{len(info['sections'])} sections)  {json_path.name}")
        return

    if args.list_presets:
        descriptions = {
            "master":    "Permanent reference signal. Generate once, never regenerate.",
            "fuzz":      "Amplitude sweeps, IM tones, transients. Optimised for Big Muff-style circuits.",
            "overdrive": "Mid-frequency emphasis, linear amplitude sweep, IM tones.",
            "delay":     "Impulse trains with long silence gaps to capture echo tails.",
            "reverb":    "Like delay but longer gaps; room-frequency 200–4000 Hz focus.",
            "clean":     "Broad coverage for compressors, EQ, chorus, and buffers.",
            "custom":    "Uses CONFIG values exactly as written in the script.",
        }
        print("\nAvailable presets:\n")
        for name, desc in descriptions.items():
            print(f"  {name:<12}  {desc}")
        print()
        return

    cfg = resolve_config(args.preset)

    if args.output_dir is not None:
        cfg["OUTPUT_DIR"] = args.output_dir

    output_dir = Path(cfg["OUTPUT_DIR"])

    # Determine output filename
    if args.preset == "master":
        base_name = cfg["MASTER_FILENAME"]
        master_path = output_dir / base_name
        if master_path.exists() and not args.force:
            print(
                f"\nERROR: {master_path} already exists.\n"
                "The master reference must never be regenerated once captured sessions\n"
                "depend on it. Pass --force only if you are certain this is intentional\n"
                "and have archived the existing file.\n",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = cfg["SESSION_FILENAME"].format(preset=args.preset, timestamp=timestamp)

    print(f"\nGenerating '{args.preset}' signal → {output_dir / base_name} ...")

    signal, segments = build_signal(cfg)
    paths = write_outputs(signal, segments, cfg, args.preset, output_dir, base_name)
    print_summary(signal, segments, cfg, args.preset, paths)


if __name__ == "__main__":
    main()
