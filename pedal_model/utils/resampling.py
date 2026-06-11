"""Audio resampling and bit-depth utilities for the DAW-ready inference pipeline.

Uses soxr for all sample-rate conversion. See docs/architecture.md for rationale.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

INT24_MAX: int = (1 << 23) - 1  # 8_388_607
INT24_MIN: int = -(1 << 23)     # -8_388_608

_LOG = logging.getLogger(__name__)


def resample_audio(
    audio: np.ndarray,
    src_sr: int,
    dst_sr: int,
    quality: str = "HQ",
) -> np.ndarray:
    """Resample audio using soxr. Returns float32 ndarray, same shape as input.

    Handles mono (1D) and multichannel (2D, channels-first). Quality default
    is HQ — sufficient for 44.1↔96 kHz. VHQ is ~2× slower with no audible
    difference for this application.

    Args:
        audio: Float32 input, shape (N,) or (C, N) channels-first.
        src_sr: Source sample rate in Hz.
        dst_sr: Destination sample rate in Hz.
        quality: soxr quality preset: 'HQ', 'VHQ', 'MQ', 'LQ', 'QQ'.

    Returns:
        Resampled float32 array, same shape convention as input.
    """
    if src_sr == dst_sr:
        return audio.copy().astype(np.float32)

    if audio.ndim == 2:
        # soxr expects (time, channels); we store (channels, time)
        out_tc = soxr.resample(audio.T, src_sr, dst_sr, quality=quality)
        return out_tc.T.astype(np.float32)

    return soxr.resample(audio, src_sr, dst_sr, quality=quality).astype(np.float32)


def quantize_to_int24(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int32 scaled for 24-bit depth.

    Clips to [-1, 1] before conversion. No dither — adequate for guitar pedal
    emulation at 24-bit depth.

    Args:
        audio: Float32 samples, shape (N,), nominally in [-1, 1].

    Returns:
        Int32 samples in [-8_388_608, 8_388_607]:
          +1.0 → 8_388_607, -1.0 → -8_388_608, 0.0 → 0.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    # Scale by 2^23 so -1.0 hits INT24_MIN exactly; clip positive headroom to INT24_MAX.
    raw = np.round(clipped.astype(np.float64) * (1 << 23)).astype(np.int64)
    return np.clip(raw, INT24_MIN, INT24_MAX).astype(np.int32)


def load_wav_to_float32(path: Path | str) -> tuple[np.ndarray, int]:
    """Load a WAV file, returning float32 samples in [-1, 1].

    Args:
        path: Path to a mono WAV file.

    Returns:
        (audio, sr) where audio has shape (N,), float32.

    Raises:
        ValueError: If the file has more than one channel.
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        raise ValueError(
            f"Expected mono WAV, got {audio.shape[1]} channels: {path}"
        )
    return audio, int(sr)


def write_wav_44k(
    path: Path | str,
    audio: np.ndarray,
    src_sr: int = 96_000,
) -> None:
    """Downsample to 44.1 kHz, quantize to 24-bit, and write a WAV file.

    Logs peak level before and after downsampling so clipping is visible
    in the run log.

    Args:
        path: Output file path. Parent directories are created if absent.
        audio: Float32 mono audio at src_sr.
        src_sr: Sample rate of the incoming audio in Hz. Default 96 000.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    peak_in_db = 20.0 * np.log10(float(np.max(np.abs(audio))) + 1e-9)
    _LOG.info("write_wav_44k %s: peak in  = %.1f dBFS", path.name, peak_in_db)

    at_44k = resample_audio(audio, src_sr, 44_100, quality="HQ")

    peak_out_db = 20.0 * np.log10(float(np.max(np.abs(at_44k))) + 1e-9)
    _LOG.info("write_wav_44k %s: peak out = %.1f dBFS", path.name, peak_out_db)

    sf.write(str(path), at_44k, 44_100, subtype="PCM_24")
