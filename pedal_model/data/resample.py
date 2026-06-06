"""Audio resampling and bit-depth conversion for the deployment pipeline.

Daisy Seed export requires two conversions from the 96k/float32 archive:
  1. Sample rate:  96 kHz → 48 kHz  (downsample_96k_to_48k)
  2. Bit depth:    float32 → int16   (float32_to_int16)

Both are lossless within the dynamic range of 16-bit PCM (~96 dB), which
exceeds any guitar pedal's SNR (~80–90 dB).  Apply rate first, depth second.
Use prepare_for_daisy() to do both in one call.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

INT16_MAX = np.iinfo(np.int16).max  # 32767


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample *audio* from *sr_in* to *sr_out* Hz with anti-aliasing.

    Uses polyphase filtering (scipy resample_poly) which automatically applies
    a FIR low-pass filter at min(sr_in, sr_out)/2 before rate conversion, so
    there is no aliasing on downsampling and no imaging on upsampling.

    Args:
        audio: Input samples, shape ``(N,)``, float32.
        sr_in: Input sample rate in Hz.
        sr_out: Output sample rate in Hz.

    Returns:
        Resampled audio, shape ``(N * sr_out // sr_in,)`` approximately,
        float32, clipped to ``[-1, 1]``.
    """
    if sr_in == sr_out:
        return audio.copy()

    from math import gcd
    g = gcd(sr_out, sr_in)
    up, down = sr_out // g, sr_in // g

    out = resample_poly(audio.astype(np.float64), up, down)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def downsample_96k_to_48k(audio: np.ndarray) -> np.ndarray:
    """Resample from 96 kHz to 48 kHz (integer 2:1 ratio, fast path).

    Args:
        audio: Input samples at 96 kHz, shape ``(N,)``, float32.

    Returns:
        Resampled audio at 48 kHz, shape ``(N // 2,)`` approximately, float32.
    """
    # 96k→48k is an exact 1:2 ratio — resample_poly uses a short FIR kernel
    out = resample_poly(audio.astype(np.float64), 1, 2)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio in ``[-1, 1]`` to int16 PCM.

    Clips before conversion to prevent wrap-around on values that exceed
    ``±1`` after any processing.  The conversion maps ``1.0 → 32767`` and
    ``-1.0 → -32768`` (standard dithering is not applied — use a dedicated
    dithering step if needed for final mastering).

    Args:
        audio: Float32 samples, shape ``(N,)``, nominally in ``[-1, 1]``.

    Returns:
        Int16 samples, shape ``(N,)``, range ``[-32768, 32767]``.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * INT16_MAX).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in ``[-1, 1]``.

    Args:
        audio: Int16 samples, shape ``(N,)``.

    Returns:
        Float32 samples, shape ``(N,)``, range ``[-1, 1]`` approximately.
    """
    return audio.astype(np.float32) / INT16_MAX


def prepare_for_daisy(
    audio: np.ndarray,
    sr_in: int = 96_000,
) -> np.ndarray:
    """Downsample from *sr_in* to 48 kHz and convert to int16 for Daisy Seed.

    Applies both pipeline steps in the correct order:
      1. Anti-aliased resample to 48 kHz (float32).
      2. Clip and convert to int16 PCM.

    Args:
        audio: Source audio, shape ``(N,)``, float32.
        sr_in: Input sample rate in Hz. Default 96 000 (the archive rate).

    Returns:
        Int16 samples at 48 kHz, shape ``(N * 48000 // sr_in,)`` approximately.
    """
    at_48k = resample(audio, sr_in, 48_000)
    return float32_to_int16(at_48k)
