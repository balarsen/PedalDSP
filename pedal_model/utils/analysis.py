"""Signal analysis helpers used in notebooks and evaluation pipelines."""
from __future__ import annotations

import numpy as np


def safe_trim(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Trim all arrays to the length of the shortest one.

    Replaces the recurring ``n = min(len(a), len(b), ...); a[:n], b[:n]``
    boilerplate when comparing model predictions against targets of slightly
    different lengths (e.g. Volterra prepends zeros).

    Args:
        *arrays: One or more 1-D arrays.

    Returns:
        Tuple of views, all length min(len(a) for a in arrays).
    """
    n = min(len(a) for a in arrays)
    return tuple(a[:n] for a in arrays)


def compute_esr_skip(
    target: np.ndarray,
    predicted: np.ndarray,
    skip: int = 0,
) -> float:
    """Error-to-Signal Ratio, optionally skipping an initial warmup window.

    The warmup skip is important for causal filters and recurrent models:
    the first `skip` samples are edge effects, not meaningful prediction errors.

    Args:
        target: Reference signal, shape (N,), float32.
        predicted: Model output, same or shorter shape.
        skip: Number of leading samples to exclude from both signals.

    Returns:
        ESR in [0, ∞). 0 = perfect, 1 = output same RMS as silence.
    """
    n = min(len(target), len(predicted))
    t = target[skip:n]
    p = predicted[skip:n]
    return float(np.sum((t - p) ** 2) / (np.sum(t ** 2) + 1e-8))


def harmonic_frequencies(
    f0: float,
    n_harmonics: int = 8,
    max_freq: float | None = None,
) -> list[float]:
    """Return a list of harmonic frequencies for a given fundamental.

    Args:
        f0: Fundamental frequency in Hz.
        n_harmonics: Maximum number of harmonics to include (including f0).
        max_freq: If given, exclude harmonics above this frequency.

    Returns:
        List [f0, 2*f0, ..., n*f0], filtered by max_freq.
    """
    freqs = [k * f0 for k in range(1, n_harmonics + 1)]
    if max_freq is not None:
        freqs = [f for f in freqs if f <= max_freq]
    return freqs


def apply_effect(
    signal: np.ndarray,
    effect_fn,
    *args,
    **kwargs,
) -> np.ndarray:
    """Apply an effect function to a signal and return float32 output.

    Thin wrapper that ensures consistent dtype and avoids repeated
    ``.astype(np.float32)`` calls in notebooks.

    Args:
        signal: Input audio, shape (N,), float32.
        effect_fn: Callable with signature ``effect_fn(signal, *args, **kwargs) -> np.ndarray``.
        *args: Positional arguments forwarded to effect_fn.
        **kwargs: Keyword arguments forwarded to effect_fn.

    Returns:
        Effect output, shape (N,), float32.
    """
    return effect_fn(signal, *args, **kwargs).astype(np.float32)


def peak_db(signal: np.ndarray) -> float:
    """Peak amplitude in dBFS.

    Args:
        signal: Audio signal, shape (N,).

    Returns:
        20 * log10(max|signal|), clipped to -200 dBFS floor.
    """
    peak = float(np.max(np.abs(signal)))
    return float(20.0 * np.log10(max(peak, 1e-10)))


def rms_db(signal: np.ndarray) -> float:
    """RMS level in dBFS.

    Args:
        signal: Audio signal, shape (N,).

    Returns:
        20 * log10(rms), clipped to -200 dBFS floor.
    """
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    return float(20.0 * np.log10(max(rms, 1e-10)))
