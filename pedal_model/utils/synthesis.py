"""Synthetic audio signal generation for test data and system identification."""
from __future__ import annotations

import numpy as np


def guitar_note(
    f0: float,
    sr: int = 48000,
    duration: float = 2.0,
    amp: float = 1.0,
    n_harmonics: int = 6,
    rolloff: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """Synthesise a guitar-like note: fundamental + harmonics with random phases.

    Amplitude of the k-th harmonic scales as 1/k^rolloff.
    Random phases break the sawtooth alignment that pure 1/k sine sums produce.

    Args:
        f0: Fundamental frequency in Hz.
        sr: Sample rate in Hz.
        duration: Signal length in seconds.
        amp: Peak amplitude after normalisation (< 1 to leave headroom).
        n_harmonics: Number of harmonics including the fundamental.
        rolloff: Harmonic amplitude exponent. 1.0 → sawtooth-like, 2.0 → triangle-like.
        seed: RNG seed for reproducible phases. Defaults to int(f0) for consistent notes.

    Returns:
        Synthesised note, shape (sr * duration,), float32.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    rng = np.random.default_rng(seed if seed is not None else int(f0))
    phases = rng.uniform(0, 2 * np.pi, n_harmonics)
    signal = sum(
        (1.0 / k ** rolloff) * np.sin(2 * np.pi * k * f0 * t + phases[k - 1])
        for k in range(1, n_harmonics + 1)
    )
    signal /= np.max(np.abs(signal))
    return (signal * amp).astype(np.float32)


def g_major_chord(
    sr: int = 48000,
    duration: float = 2.0,
    amp: float = 0.2,
    n_harmonics: int = 6,
) -> np.ndarray:
    """G major chord: D3 (147 Hz) + G3 (196 Hz) + B3 (247 Hz).

    Each note is synthesised with guitar_note and summed without clipping.
    amp=0.2 keeps the combined peak at roughly -6 dBFS, leaving headroom for effects.

    Args:
        sr: Sample rate in Hz.
        duration: Signal length in seconds.
        amp: Per-note amplitude (combined chord peaks at ~3 * amp * note_mix).
        n_harmonics: Harmonics per note.

    Returns:
        G major chord signal, shape (sr * duration,), float32.
    """
    d3 = guitar_note(147, sr=sr, duration=duration, amp=amp, n_harmonics=n_harmonics)
    g3 = guitar_note(196, sr=sr, duration=duration, amp=amp, n_harmonics=n_harmonics)
    b3 = guitar_note(247, sr=sr, duration=duration, amp=amp, n_harmonics=n_harmonics)
    return (d3 + g3 + b3).astype(np.float32)


def white_noise_id(
    sr: int = 48000,
    duration: float = 4.0,
    amplitude: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Broadband white noise for system identification.

    White noise has (approximately) equal energy at every frequency, making
    H(ω) = WET(ω) / DRY(ω) well-conditioned at all bins. Always use this
    (or a log sweep) rather than a tonal signal when fitting FIR/IIR models.

    Args:
        sr: Sample rate in Hz.
        duration: Length of the identification signal in seconds.
        amplitude: RMS amplitude (σ of the Gaussian draw).
        seed: RNG seed for reproducibility.

    Returns:
        White noise signal, shape (sr * duration,), float32.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sr * duration)) * amplitude).astype(np.float32)


def pure_sine(
    f0: float,
    sr: int = 48000,
    duration: float = 1.0,
    amp: float = 0.5,
) -> np.ndarray:
    """A single-frequency sine wave — useful for THD measurement.

    Args:
        f0: Frequency in Hz.
        sr: Sample rate in Hz.
        duration: Duration in seconds.
        amp: Peak amplitude.

    Returns:
        Sine wave, shape (sr * duration,), float32.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
