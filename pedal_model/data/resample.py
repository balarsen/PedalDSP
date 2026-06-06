"""Audio resampling utilities for the 96k→48k deployment pipeline."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


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
