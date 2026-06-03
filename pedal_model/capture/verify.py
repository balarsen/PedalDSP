"""Visual and statistical verification of a capture pair."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .align import load_and_align


def check_capture(dry: np.ndarray, wet: np.ndarray, sr: int) -> dict[str, float]:
    """Run statistical checks on an aligned dry/wet pair.

    Args:
        dry: Aligned dry audio, shape (N,), float32.
        wet: Aligned wet audio, same shape.
        sr: Sample rate in Hz.

    Returns:
        Dict with keys: dry_peak, wet_peak, dry_dc, wet_dc, duration_s.
        Prints warnings if any check fails.
    """
    stats = {
        "dry_peak": float(np.max(np.abs(dry))),
        "wet_peak": float(np.max(np.abs(wet))),
        "dry_dc": float(np.mean(dry)),
        "wet_dc": float(np.mean(wet)),
        "duration_s": len(dry) / sr,
    }
    if stats["dry_peak"] >= 0.99:
        print("WARNING: DRY channel may be clipping")
    if stats["wet_peak"] >= 0.99:
        print("WARNING: WET channel may be clipping")
    if abs(stats["dry_dc"]) > 1e-3:
        print(f"WARNING: DRY DC offset = {stats['dry_dc']:.4f}")
    if abs(stats["wet_dc"]) > 1e-3:
        print(f"WARNING: WET DC offset = {stats['wet_dc']:.4f}")
    return stats


def plot_alignment(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    seconds: float = 2.0,
    save_path: Path | str | None = None,
) -> None:
    """Plot first `seconds` of dry vs wet to visually confirm alignment.

    Args:
        dry: Dry audio, shape (N,).
        wet: Wet audio, shape (N,).
        sr: Sample rate in Hz.
        seconds: How many seconds to plot.
        save_path: If given, save the figure here instead of displaying.
    """
    n = int(sr * seconds)
    t = np.arange(n) / sr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, dry[:n], label="DRY", alpha=0.7)
    ax.plot(t, wet[:n], label="WET", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"DRY vs WET — first {seconds:.1f}s")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
    else:
        plt.show()
    plt.close(fig)


def verify_file(path: Path | str, plot: bool = True) -> dict[str, float]:
    """Load, align, check, and optionally plot a capture WAV.

    Args:
        path: Path to stereo capture WAV (ch0=dry, ch1=wet).
        plot: If True, display the alignment plot.

    Returns:
        Stats dict from check_capture.
    """
    dry, wet, sr = load_and_align(path)
    stats = check_capture(dry, wet, sr)
    print(f"Duration: {stats['duration_s']:.1f}s  |  "
          f"DRY peak: {stats['dry_peak']:.3f}  |  WET peak: {stats['wet_peak']:.3f}")
    if plot:
        save = Path(path).with_suffix(".png")
        plot_alignment(dry, wet, sr, save_path=save)
        print(f"Saved plot → {save}")
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pedal_model.capture.verify <capture.wav>")
        sys.exit(1)
    verify_file(sys.argv[1])
