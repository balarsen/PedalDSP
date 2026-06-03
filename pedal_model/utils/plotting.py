"""Reusable matplotlib helpers for signal visualisation in notebooks."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import rfft, rfftfreq


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def db_spectrum(signal: np.ndarray, sr: int, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_Hz, magnitudes_dBFS) for a real signal.

    Args:
        signal: Audio signal, shape (N,), float32.
        sr: Sample rate in Hz.
        n_fft: FFT size. Defaults to len(signal).

    Returns:
        (freqs, mag_db) — both 1-D arrays of length n_fft // 2 + 1.
    """
    n = n_fft or len(signal)
    freqs = rfftfreq(n, d=1.0 / sr)
    mag = np.abs(rfft(signal, n=n))
    mag_db = 20.0 * np.log10(mag / n + 1e-12)
    return freqs, mag_db


def freq_response_db(kernel: np.ndarray, sr: int, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Frequency response (dB) of an FIR kernel via zero-padded FFT.

    Args:
        kernel: FIR impulse response, shape (M,).
        sr: Sample rate in Hz.
        n_fft: Zero-padded FFT size for smooth curve.

    Returns:
        (freqs, H_db) — both 1-D arrays.
    """
    freqs = rfftfreq(n_fft, d=1.0 / sr)
    H = rfft(kernel, n=n_fft)
    H_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    return freqs, H_db


# ──────────────────────────────────────────────────────────────────────────────
# Single-axis plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_waveform(
    signal: np.ndarray,
    sr: int,
    label: str = "",
    color: str = "steelblue",
    ax: plt.Axes | None = None,
    ms: float = 30.0,
    lw: float = 0.9,
) -> plt.Axes:
    """Plot the first `ms` milliseconds of a signal.

    Args:
        signal: Audio signal, shape (N,).
        sr: Sample rate in Hz.
        label: Legend label.
        color: Line colour.
        ax: Target axes (creates figure if None).
        ms: Time window to display in milliseconds.
        lw: Line width.

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    n = int(sr * ms / 1000)
    t_ms = np.arange(min(n, len(signal))) / sr * 1000
    ax.plot(t_ms, signal[:n], color=color, lw=lw, label=label)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    if label:
        ax.legend(fontsize=9)
    return ax


def plot_spectrum(
    signal: np.ndarray,
    sr: int,
    label: str = "",
    color: str = "steelblue",
    ax: plt.Axes | None = None,
    max_freq: float = 5000.0,
    floor_db: float = -80.0,
    lw: float = 0.9,
) -> plt.Axes:
    """Plot the dB magnitude spectrum of a signal.

    Args:
        signal: Audio signal, shape (N,).
        sr: Sample rate in Hz.
        label: Legend label.
        color: Line colour.
        ax: Target axes (creates figure if None).
        max_freq: Upper x-axis frequency limit in Hz.
        floor_db: Lower y-axis limit in dB.
        lw: Line width.

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    freqs, mag_db = db_spectrum(signal, sr)
    ax.plot(freqs, mag_db, color=color, lw=lw, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_xlim(0, max_freq)
    ax.set_ylim(floor_db, 5)
    if label:
        ax.legend(fontsize=9)
    return ax


def plot_impulse_response(
    kernel: np.ndarray,
    label: str = "h[k]",
    color: str = "steelblue",
    ax: plt.Axes | None = None,
    n_show: int = 80,
) -> plt.Axes:
    """Stem plot of FIR impulse response.

    Args:
        kernel: FIR kernel, shape (M,).
        label: Axes title.
        color: Stem colour.
        ax: Target axes.
        n_show: Number of taps to display.

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    n = min(n_show, len(kernel))
    markerline, stemlines, baseline = ax.stem(
        np.arange(n), kernel[:n],
        markerfmt=f"o", linefmt=f"-", basefmt="k-",
    )
    markerline.set(color=color, markersize=3)
    stemlines.set(color=color, linewidth=0.8)
    ax.set_xlabel("Tap index k")
    ax.set_ylabel("Amplitude")
    ax.set_title(label)
    return ax


def plot_freq_response(
    kernel: np.ndarray,
    sr: int,
    label: str = "FIR freq response",
    color: str = "steelblue",
    ax: plt.Axes | None = None,
    max_freq: float = 5000.0,
    floor_db: float = -60.0,
    lw: float = 1.2,
) -> plt.Axes:
    """Plot the frequency response of an FIR kernel in dB.

    Args:
        kernel: FIR kernel, shape (M,).
        sr: Sample rate in Hz.
        label: Legend label.
        color: Line colour.
        ax: Target axes.
        max_freq: Upper x-axis limit in Hz.
        floor_db: Lower y-axis limit in dB.
        lw: Line width.

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    freqs, H_db = freq_response_db(kernel, sr)
    ax.plot(freqs, H_db, color=color, lw=lw, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H(ω)  (dB)")
    ax.set_xlim(0, max_freq)
    ax.set_ylim(floor_db, 10)
    if label:
        ax.legend(fontsize=9)
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# Multi-signal comparison panels
# ──────────────────────────────────────────────────────────────────────────────

def compare_waveforms(
    signals: dict[str, tuple[np.ndarray, str]],
    sr: int,
    title: str = "",
    ms: float = 30.0,
    figsize: tuple[float, float] = (10, 3),
) -> plt.Figure:
    """Overlay multiple signals in one time-domain axes.

    Args:
        signals: {label: (signal_array, color)} dict, plotted in order.
        sr: Sample rate in Hz.
        title: Figure title.
        ms: Duration to display in milliseconds.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, (sig, color) in signals.items():
        plot_waveform(sig, sr, label=label, color=color, ax=ax, ms=ms)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def compare_spectra(
    signals: dict[str, tuple[np.ndarray, str]],
    sr: int,
    title: str = "",
    max_freq: float = 5000.0,
    floor_db: float = -80.0,
    vlines: dict[str, float] | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Overlay multiple signal spectra in one frequency-domain axes.

    Args:
        signals: {label: (signal_array, color)} dict.
        sr: Sample rate in Hz.
        title: Figure title.
        max_freq: Upper x-axis limit in Hz.
        floor_db: Lower y-axis limit in dB.
        vlines: {label: frequency_Hz} — vertical reference lines to mark.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, (sig, color) in signals.items():
        plot_spectrum(sig, sr, label=label, color=color, ax=ax,
                      max_freq=max_freq, floor_db=floor_db)
    if vlines:
        for vlabel, vf in vlines.items():
            ax.axvline(vf, color="salmon", lw=0.8, ls="--")
            ax.text(vf + max_freq * 0.005, floor_db + 5, vlabel,
                    color="salmon", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def signal_dashboard(
    signals: dict[str, tuple[np.ndarray, str]],
    sr: int,
    title: str = "",
    ms: float = 30.0,
    max_freq: float = 5000.0,
    floor_db: float = -80.0,
    vlines: dict[str, float] | None = None,
    figsize: tuple[float, float] = (13, 4),
) -> plt.Figure:
    """Side-by-side time-domain + frequency-domain panel for multiple signals.

    Args:
        signals: {label: (signal_array, color)} dict.
        sr: Sample rate in Hz.
        title: Figure suptitle.
        ms: Time window in milliseconds.
        max_freq: Upper frequency limit in Hz.
        floor_db: Lower spectrum limit in dB.
        vlines: Vertical reference lines for spectrum axes.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=figsize)
    for label, (sig, color) in signals.items():
        plot_waveform(sig, sr, label=label, color=color, ax=ax_t, ms=ms)
        plot_spectrum(sig, sr, label=label, color=color, ax=ax_f,
                      max_freq=max_freq, floor_db=floor_db)
    if vlines:
        for vlabel, vf in (vlines or {}).items():
            ax_f.axvline(vf, color="salmon", lw=0.8, ls="--")
            ax_f.text(vf + max_freq * 0.005, floor_db + 4, vlabel,
                      color="salmon", fontsize=8)
    ax_t.set_title("Time Domain")
    ax_f.set_title("Frequency Spectrum")
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def model_fit_panel(
    target: np.ndarray,
    prediction: np.ndarray,
    kernel: np.ndarray | None,
    sr: int,
    model_name: str = "Model",
    target_color: str = "darkorange",
    pred_color: str = "steelblue",
    ms: float = 30.0,
    max_freq: float = 5000.0,
    n_ir_taps: int = 80,
) -> plt.Figure:
    """3-panel figure: impulse response (or blank), time overlay, spectrum overlay.

    Args:
        target: Ground-truth wet signal.
        prediction: Model's predicted wet signal.
        kernel: FIR kernel (optional — pass None to skip IR panel).
        sr: Sample rate in Hz.
        model_name: Used in titles.
        target_color: Colour for target curve.
        pred_color: Colour for prediction curve.
        ms: Time window in milliseconds.
        max_freq: Upper frequency limit in Hz.
        n_ir_taps: Number of IR taps to display.

    Returns:
        matplotlib Figure.
    """
    n_panels = 3 if kernel is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    if kernel is not None:
        plot_impulse_response(kernel, label=f"{model_name} kernel h[k]",
                              color=pred_color, ax=axes[0], n_show=n_ir_taps)
        ax_t, ax_f = axes[1], axes[2]
    else:
        ax_t, ax_f = axes[0], axes[1]

    n = min(len(target), len(prediction))
    plot_waveform(target[:n],     sr, label="Target wet", color=target_color, ax=ax_t, ms=ms, lw=1.2)
    plot_waveform(prediction[:n], sr, label=f"{model_name} prediction", color=pred_color, ax=ax_t, ms=ms, lw=0.9)
    ax_t.set_title("Time: Target vs Prediction")

    plot_spectrum(target[:n],     sr, label="Target wet", color=target_color, ax=ax_f, max_freq=max_freq, lw=1.2)
    plot_spectrum(prediction[:n], sr, label=f"{model_name} prediction", color=pred_color, ax=ax_f, max_freq=max_freq, lw=0.9)
    ax_f.set_title("Spectrum: Target vs Prediction")

    fig.suptitle(f"{model_name} — Fit Quality", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
