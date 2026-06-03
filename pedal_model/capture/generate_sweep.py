"""Generate test signals for pedal capture sessions."""
from pathlib import Path

import numpy as np
import soundfile as sf


def generate_log_sweep(
    f_start: float = 20.0,
    f_end: float = 20000.0,
    duration: float = 10.0,
    sr: int = 48000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Logarithmic (Farina) sine sweep.

    Args:
        f_start: Start frequency in Hz.
        f_end: End frequency in Hz.
        duration: Sweep duration in seconds.
        sr: Sample rate in Hz.
        amplitude: Peak amplitude (< 1.0 to avoid clipping).

    Returns:
        Sweep signal, shape (sr * duration,), float32.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Exponential instantaneous frequency: ω(t) = ω_start · (ω_end/ω_start)^(t/T)
    k = np.log(f_end / f_start)
    phase = 2.0 * np.pi * f_start * duration / k * (np.exp(t / duration * k) - 1.0)
    return (amplitude * np.sin(phase)).astype(np.float32)


def generate_alignment_click(sr: int = 48000, duration: float = 0.25) -> np.ndarray:
    """Single-sample spike for channel alignment.

    Args:
        sr: Sample rate in Hz.
        duration: Buffer length in seconds around the click.

    Returns:
        Buffer with a 0.9 spike at sample 0, shape (sr * duration,), float32.
    """
    click = np.zeros(int(sr * duration), dtype=np.float32)
    click[0] = 0.9
    return click


def build_capture_signal(sr: int = 48000) -> np.ndarray:
    """Full capture signal: silence + click + sweeps + noise + silence.

    Produces ~45 seconds of signal suitable for system identification:
    - 1 s silence
    - Alignment click (0.25 s)
    - Full-range log sweep 20–20 kHz (10 s)
    - 1 s silence
    - Guitar-range log sweep 80–8 kHz (10 s)
    - 1 s silence
    - Repeat full-range sweep (10 s) — for averaging
    - 1 s silence
    - White noise (5 s)
    - 2 s trailing silence

    Args:
        sr: Sample rate in Hz.

    Returns:
        Mono capture signal, float32.
    """
    silence_1s = np.zeros(sr, dtype=np.float32)
    silence_2s = np.zeros(sr * 2, dtype=np.float32)

    click = generate_alignment_click(sr)
    sweep_full_1 = generate_log_sweep(20, 20_000, 10.0, sr)
    sweep_guitar = generate_log_sweep(80, 8_000, 10.0, sr)
    sweep_full_2 = generate_log_sweep(20, 20_000, 10.0, sr)
    noise = np.clip(
        np.random.default_rng(0).standard_normal(sr * 5) * 0.3, -0.9, 0.9
    ).astype(np.float32)

    return np.concatenate([
        silence_1s, click,
        sweep_full_1, silence_1s,
        sweep_guitar, silence_1s,
        sweep_full_2, silence_1s,
        noise, silence_2s,
    ])


def save_capture_signal(path: Path | str = "capture_sweep.wav", sr: int = 48000) -> Path:
    """Build and write the capture signal to a WAV file.

    Args:
        path: Output WAV path.
        sr: Sample rate in Hz.

    Returns:
        Resolved output path.
    """
    path = Path(path)
    signal = build_capture_signal(sr)
    sf.write(str(path), signal, sr, subtype="PCM_24")
    print(f"Generated {len(signal) / sr:.1f}s  →  {path}")
    return path


if __name__ == "__main__":
    save_capture_signal()
