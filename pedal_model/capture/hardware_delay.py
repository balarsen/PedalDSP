"""Measure and visualise hardware round-trip delay between dry and wet captures.

Provides lag estimation via cross-correlation on a low-amplitude (linear-regime)
section of the signal, plus a suite of diagnostic plots saved to a directory.

Typical usage::

    from pedal_model.capture.hardware_delay import analyze_delay
    results = analyze_delay(
        dry="path/to/dry.wav",
        wet={"G9": "path/to/wet_g9.wav", "G11": "path/to/wet_g11.wav"},
        sr=48000,
        out_dir="outputs/delay_analysis",
    )
    # results == {"G9": 0, "G11": 2}  — lag in samples
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import correlate


# ──────────────────────────────────────────────────────────────────────────────
# Lag estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_lag(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    *,
    start_s: float = 0.5,
    end_s: float = 20.0,
    max_lag_s: float = 0.5,
) -> int:
    """Cross-correlation lag on a time-limited window of the signals.

    Use a low-amplitude section (e.g. quiet sweep, −24 dBFS) so the device
    under test stays in its linear regime and the correlation peak is sharp.
    Running xcorr on a loud / distorted section gives a broad peak that can
    be misidentified by tens or hundreds of samples.

    Args:
        dry: Full dry signal, shape (N,), float32.
        wet: Full wet signal, shape (M,), float32.
        sr: Sample rate in Hz.
        start_s: Window start in seconds. Default 0.5 (skip any silence/click).
        end_s: Window end in seconds. Default 20.0 (use first ~20 s).
        max_lag_s: Search radius in seconds (±). Default 0.5.

    Returns:
        Lag in samples. Positive = wet is delayed behind dry.
    """
    s0 = int(start_s * sr)
    s1 = int(min(end_s * sr, len(dry), len(wet)))
    max_lag = int(max_lag_s * sr)

    d = dry[s0:s1].astype(np.float64)
    w = wet[s0:s1].astype(np.float64)

    corr = correlate(d, w, mode="full")
    centre = len(corr) // 2
    search = corr[centre - max_lag: centre + max_lag + 1]
    lag = max_lag - int(np.argmax(np.abs(search)))
    return lag


def estimate_lag_from_onset(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    *,
    search_start_s: float = 0.0,
    search_end_s: float = 5.0,
    threshold: float = 0.005,
) -> int | None:
    """Onset-based lag: first sample above threshold in dry minus first in wet.

    Reliable when there is a clean transient onset (click, ramp, impulse) with
    silence before it. Returns None if no onset is found in either channel.

    Args:
        dry: Full dry signal, shape (N,), float32.
        wet: Full wet signal, shape (M,), float32.
        sr: Sample rate in Hz.
        search_start_s: Start of search window in seconds.
        search_end_s: End of search window in seconds.
        threshold: Absolute amplitude threshold for onset.

    Returns:
        Lag in samples (positive = wet delayed), or None.
    """
    s0 = int(search_start_s * sr)
    s1 = int(min(search_end_s * sr, len(dry), len(wet)))
    d_hits = np.where(np.abs(dry[s0:s1]) > threshold)[0]
    w_hits = np.where(np.abs(wet[s0:s1]) > threshold)[0]
    if len(d_hits) == 0 or len(w_hits) == 0:
        return None
    return int((s0 + d_hits[0]) - (s0 + w_hits[0]))


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load(src: np.ndarray | Path | str, sr: int) -> np.ndarray:
    """Load or pass-through an audio array, mono float32."""
    if isinstance(src, np.ndarray):
        return src.astype(np.float32)
    path = Path(src)
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if file_sr != sr:
        raise ValueError(f"{path.name}: expected {sr} Hz, got {file_sr} Hz")
    return audio[:, 0]


def _normalise_wets(
    wet: np.ndarray | Path | str | dict[str, np.ndarray | Path | str],
) -> dict[str, np.ndarray | Path | str]:
    """Accept single or dict wet input and return a label→source dict."""
    if isinstance(wet, dict):
        return wet
    return {"wet": wet}


# ──────────────────────────────────────────────────────────────────────────────
# Individual plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_xcorr(
    dry: np.ndarray,
    wets: dict[str, np.ndarray],
    sr: int,
    lags: dict[str, int],
    *,
    start_s: float = 0.5,
    end_s: float = 20.0,
    max_lag_s: float = 0.05,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Normalised cross-correlation curves for each wet capture.

    Args:
        dry: Dry reference, shape (N,), float32.
        wets: Label → wet array dict.
        sr: Sample rate in Hz.
        lags: Label → estimated lag in samples (for marking peaks).
        start_s: Analysis window start in seconds.
        end_s: Analysis window end in seconds.
        max_lag_s: Display range (±) in seconds.
        ax: Axes to draw on; creates a new figure if None.

    Returns:
        Figure containing the plot.
    """
    fig, ax = (plt.subplots(figsize=(8, 4)) if ax is None else (ax.figure, ax))
    s0 = int(start_s * sr)
    s1 = int(min(end_s * sr, len(dry)))
    max_lag = int(max_lag_s * sr)
    lags_ms = np.arange(-max_lag, max_lag + 1) / sr * 1000

    d = dry[s0:s1].astype(np.float64)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, w) in enumerate(wets.items()):
        ww = w[s0:min(s1, len(w))].astype(np.float64)
        n = min(len(d), len(ww))
        corr = correlate(d[:n], ww[:n], mode="full")
        centre = len(corr) // 2
        snippet = corr[centre - max_lag: centre + max_lag + 1]
        peak = np.max(np.abs(snippet))
        if peak > 0:
            snippet = snippet / peak
        col = colours[idx % len(colours)]
        ax.plot(lags_ms, snippet, color=col, alpha=0.8, label=label)
        lag_smp = lags[label]
        ax.axvline(lag_smp / sr * 1000, color=col, linestyle="--", linewidth=0.8)

    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalised correlation")
    ax.set_title(f"Cross-correlation (analysis window {start_s:.1f}–{end_s:.1f} s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def plot_waveform_zoom(
    dry: np.ndarray,
    wets: dict[str, np.ndarray],
    sr: int,
    lags: dict[str, int],
    *,
    t_center_s: float,
    half_ms: float = 30.0,
    title: str = "Waveform zoom",
) -> plt.Figure:
    """Overlay dry and wet waveforms around a reference time, lag-corrected.

    Each wet array is shifted by its estimated lag so that a perfectly aligned
    signal would overlay the dry exactly.

    Args:
        dry: Dry reference, shape (N,), float32.
        wets: Label → wet array dict.
        sr: Sample rate in Hz.
        lags: Label → lag in samples (positive = wet delayed).
        t_center_s: Centre of the zoom window in seconds.
        half_ms: Half-width of the zoom window in milliseconds.
        title: Axes title.

    Returns:
        Figure containing the plot.
    """
    n_plots = 1 + len(wets)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharey=True)
    fig.suptitle(title, y=1.01)

    half = int(half_ms / 1000 * sr)
    center = int(t_center_s * sr)
    t_ms = np.arange(-half, half + 1) / sr * 1000

    # Dry reference panel
    ax = axes[0] if n_plots > 1 else axes
    chunk = dry[center - half: center + half + 1]
    ax.plot(t_ms[:len(chunk)], chunk, color="gray", linewidth=0.8, label="dry")
    ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Amp")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, w) in enumerate(wets.items()):
        ax = axes[idx + 1] if n_plots > 1 else axes
        lag = lags[label]
        # Shift wet window by lag to compensate delay
        w0 = center - half + lag
        w1 = center + half + 1 + lag
        w0c, w1c = max(0, w0), min(len(w), w1)
        chunk = w[w0c:w1c]
        offset_ms = (w0c - w0) / sr * 1000
        t_shifted = t_ms[w0c - w0: w0c - w0 + len(chunk)]
        col = colours[idx % len(colours)]
        ax.plot(t_shifted, chunk, color=col, linewidth=0.8,
                label=f"{label} (lag={lag:+d} smp)")
        ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
        ax.set_ylabel("Amp")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time offset (ms)")
    fig.tight_layout()
    return fig


def plot_level_summary(
    wets: dict[str, np.ndarray],
    sr: int,
) -> plt.Figure:
    """Bar chart of peak and RMS levels for each wet capture.

    Args:
        wets: Label → wet array dict.
        sr: Sample rate in Hz.

    Returns:
        Figure with two sub-axes (peak dBFS, RMS dBFS).
    """
    labels = list(wets.keys())
    peaks = [20 * np.log10(np.max(np.abs(w)) + 1e-12) for w in wets.values()]
    rms = [20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-12) for w in wets.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(6, len(labels) * 1.5), 4))
    x = np.arange(len(labels))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax1.bar(x, peaks, color=colours[:len(labels)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Peak (dBFS)")
    ax1.set_title("Peak level")
    ax1.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x, peaks):
        ax1.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(x, rms, color=colours[:len(labels)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("RMS (dBFS)")
    ax2.set_title("RMS level")
    ax2.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x, rms):
        ax2.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def analyze_delay(
    dry: np.ndarray | Path | str,
    wet: np.ndarray | Path | str | dict[str, np.ndarray | Path | str],
    sr: int,
    out_dir: Path | str,
    *,
    xcorr_start_s: float = 0.5,
    xcorr_end_s: float = 20.0,
    max_lag_s: float = 0.5,
    t_zoom_s: float | None = None,
    zoom_half_ms: float = 30.0,
    zoom_title: str = "Waveform zoom",
) -> dict[str, int]:
    """Full delay analysis: estimate lags, save plots and JSON results.

    Args:
        dry: Dry reference — numpy array, file path, or Path. Mono float32.
        wet: Wet capture(s). May be:
            - A single array / path → labelled "wet".
            - A dict ``{label: array_or_path}`` for multi-capture comparison.
        sr: Sample rate in Hz. Used to verify file sample rates.
        out_dir: Directory where plots and ``results.json`` are written.
        xcorr_start_s: Start of the xcorr analysis window in seconds.
            Use a low-amplitude / linear-regime section (e.g. quiet sweep).
        xcorr_end_s: End of the xcorr analysis window in seconds.
        max_lag_s: Xcorr search radius (±) in seconds.
        t_zoom_s: Centre time for the waveform zoom plot (in seconds).
            If None, the zoom plot is skipped.
        zoom_half_ms: Half-width of the zoom window in ms. Default 30 ms.
        zoom_title: Title for the waveform zoom plot.

    Returns:
        Dict mapping each label to its estimated lag in samples.
        Positive lag = wet is delayed behind dry.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dry_arr = _load(dry, sr)
    wet_sources = _normalise_wets(wet)
    wet_arrs: dict[str, np.ndarray] = {
        label: _load(src, sr) for label, src in wet_sources.items()
    }

    # ── Lag estimation ────────────────────────────────────────────────────────
    lags: dict[str, int] = {}
    for label, w in wet_arrs.items():
        lag = estimate_lag(
            dry_arr, w, sr,
            start_s=xcorr_start_s,
            end_s=xcorr_end_s,
            max_lag_s=max_lag_s,
        )
        lags[label] = lag
        print(f"  {label:20s}  lag = {lag:+5d} smp  ({lag / sr * 1000:+.2f} ms)")

    # ── Plot 1: cross-correlation ─────────────────────────────────────────────
    fig = plot_xcorr(
        dry_arr, wet_arrs, sr, lags,
        start_s=xcorr_start_s, end_s=xcorr_end_s,
        max_lag_s=min(max_lag_s, 0.05),
    )
    p = out_dir / "01_xcorr.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

    # ── Plot 2: waveform zoom (optional) ─────────────────────────────────────
    if t_zoom_s is not None:
        fig = plot_waveform_zoom(
            dry_arr, wet_arrs, sr, lags,
            t_center_s=t_zoom_s,
            half_ms=zoom_half_ms,
            title=zoom_title,
        )
        p = out_dir / "02_waveform_zoom.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {p}")

    # ── Plot 3: level summary ─────────────────────────────────────────────────
    fig = plot_level_summary(wet_arrs, sr)
    p = out_dir / "03_level_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

    # ── JSON results ──────────────────────────────────────────────────────────
    results = {
        label: {
            "lag_samples": lag,
            "lag_ms": round(lag / sr * 1000, 3),
        }
        for label, lag in lags.items()
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"  saved {results_path}")

    return lags
