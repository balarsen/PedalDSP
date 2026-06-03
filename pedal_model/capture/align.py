"""Channel alignment: detect the click and correct the sample offset."""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate


def detect_click(signal: np.ndarray, search_samples: int = 48000) -> int:
    """Find the sample index of the alignment click (largest absolute peak).

    Args:
        signal: Audio signal, shape (N,), float32.
        search_samples: Only search within the first this many samples.

    Returns:
        Sample index of the click peak.
    """
    window = signal[: min(search_samples, len(signal))]
    return int(np.argmax(np.abs(window)))


def estimate_lag(dry: np.ndarray, wet: np.ndarray, max_lag_samples: int = 4800) -> int:
    """Cross-correlation lag between dry and wet channels.

    Positive lag means wet is delayed relative to dry.

    Args:
        dry: Dry channel, shape (N,), float32.
        wet: Wet channel, same shape.
        max_lag_samples: Search window in samples (±). Default ≈ 100 ms at 48 kHz.

    Returns:
        Lag in samples (can be negative).
    """
    n = min(len(dry), len(wet), max_lag_samples * 8)
    corr = correlate(dry[:n], wet[:n], mode="full")
    centre = len(corr) // 2
    search = corr[centre - max_lag_samples : centre + max_lag_samples + 1]
    # scipy.signal.correlate convention: peak at position p means wet leads by
    # (p - max_lag) samples, so positive lag = wet is delayed behind dry.
    lag = max_lag_samples - int(np.argmax(np.abs(search)))
    return lag


def align_channels(dry: np.ndarray, wet: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Correct sample-offset between dry and wet and trim to equal length.

    Uses the alignment click to find the offset, then slices both arrays so
    they start at the same physical moment and have the same number of samples.

    Args:
        dry: Dry channel, shape (N,), float32.
        wet: Wet channel, shape (M,), float32.
        sr: Sample rate in Hz.

    Returns:
        (aligned_dry, aligned_wet) — equal-length float32 arrays.
    """
    lag = estimate_lag(dry, wet, max_lag_samples=int(sr * 0.1))

    if lag > 0:
        # wet is delayed: drop first `lag` samples of wet
        wet_aligned = wet[lag:]
        dry_aligned = dry[: len(wet_aligned)]
    elif lag < 0:
        # dry is delayed: drop first `|lag|` samples of dry
        dry_aligned = dry[-lag:]
        wet_aligned = wet[: len(dry_aligned)]
    else:
        dry_aligned, wet_aligned = dry.copy(), wet.copy()

    n = min(len(dry_aligned), len(wet_aligned))
    return dry_aligned[:n].astype(np.float32), wet_aligned[:n].astype(np.float32)


def load_and_align(path: Path | str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a stereo capture WAV and return aligned (dry, wet) arrays.

    Expects a stereo file where channel 0 = dry, channel 1 = wet.

    Args:
        path: Path to the stereo WAV file.

    Returns:
        (dry, wet, sr) — aligned float32 arrays and sample rate.
    """
    path = Path(path)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    dry, wet = audio[:, 0], audio[:, 1]
    dry_aligned, wet_aligned = align_channels(dry, wet, sr)
    lag = estimate_lag(dry, wet)
    print(f"Lag: {lag} samples ({lag / sr * 1000:.2f} ms)")
    return dry_aligned, wet_aligned, sr
