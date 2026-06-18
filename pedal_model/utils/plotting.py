"""Reusable matplotlib helpers for signal visualisation in notebooks.

Spectrograms and waveforms use ``librosa.display.specshow`` /
``librosa.display.waveshow`` for axis labelling (time, mel, Hz) and
proper aspect handling. Primitive helpers (spectrum, IR, freq response)
use plain matplotlib so they have no librosa dependency.

Multi-signal comparison functions (``compare_*``) take a ``signals`` dict
of ``{label: array}`` and a ``ref_key`` string identifying the reference
signal. They return a ``plt.Figure`` ready to save. All comparison functions
use only numpy/scipy so they have no librosa dependency.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq


# ──────────────────────────────────────────────────────────────────────────────
# Spectral analysis primitives
# ──────────────────────────────────────────────────────────────────────────────

def db_spectrum(
    signal: np.ndarray,
    sr: int,
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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


def freq_response_db(
    kernel: np.ndarray,
    sr: int,
    n_fft: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency response (dB) of an FIR kernel via zero-padded FFT.

    Args:
        kernel: FIR impulse response, shape (M,).
        sr: Sample rate in Hz.
        n_fft: Zero-padded FFT size for a smooth, interpolated curve.

    Returns:
        (freqs, H_db) — both 1-D arrays.
    """
    freqs = rfftfreq(n_fft, d=1.0 / sr)
    H = rfft(kernel, n=n_fft)
    H_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    return freqs, H_db


def filter_impulse_response(sos: np.ndarray, n: int = 4096) -> np.ndarray:
    """Impulse response of an SOS filter (Kronecker delta input).

    Args:
        sos: Second-order sections array, shape (n_sections, 6).
        n: Length of the impulse response to compute.

    Returns:
        Impulse response, shape (n,), float32.
    """
    from scipy.signal import sosfilt
    impulse = np.zeros(n, dtype=np.float32)
    impulse[0] = 1.0
    return sosfilt(sos, impulse).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Axis annotation helpers
# ──────────────────────────────────────────────────────────────────────────────

def mark_harmonics(
    ax: plt.Axes,
    f0: float,
    n_harmonics: int = 6,
    max_freq: float = 5000.0,
    color: str = "gray",
    text_y_db: float = -10.0,
    alpha: float = 0.6,
) -> None:
    """Draw vertical dashed lines at harmonic frequencies on a spectrum axes.

    Args:
        ax: Target axes (must already have a spectrum plotted).
        f0: Fundamental frequency in Hz.
        n_harmonics: Number of harmonics to mark (including fundamental).
        max_freq: Do not mark harmonics above this frequency.
        color: Line and text colour.
        text_y_db: Vertical position (dBFS) for the harmonic labels.
        alpha: Line transparency.
    """
    for k in range(1, n_harmonics + 1):
        fk = k * f0
        if fk > max_freq:
            break
        ax.axvline(fk, color=color, lw=0.6, ls="--", alpha=alpha)
        ax.text(fk + max_freq * 0.005, text_y_db, f"{k}f₀",
                fontsize=7, color=color, alpha=alpha)


def mark_vlines(
    ax: plt.Axes,
    vlines: dict[str, float],
    max_freq: float = 5000.0,
    floor_db: float = -80.0,
    color: str = "salmon",
    lw: float = 0.8,
    ls: str = "--",
) -> None:
    """Draw labelled vertical reference lines on a spectrum axes.

    Args:
        ax: Target axes.
        vlines: {label: frequency_Hz} mapping.
        max_freq: Used to compute text x-offset.
        floor_db: Used to compute text y position.
        color: Line and text colour.
        lw: Line width.
        ls: Line style.
    """
    for label, freq in vlines.items():
        ax.axvline(freq, color=color, lw=lw, ls=ls)
        ax.text(freq + max_freq * 0.005, floor_db + 4, label,
                color=color, fontsize=8)


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


def plot_waveshow(
    signal: np.ndarray,
    sr: int,
    label: str = "",
    color: str = "steelblue",
    ax: plt.Axes | None = None,
    max_points: int = 200_000,
) -> plt.Axes:
    """Full-signal waveform via librosa.display.waveshow with proper time axis.

    Uses envelope shading when the signal is longer than max_points, otherwise
    draws the raw samples. Prefer this over plot_waveform for full-length
    signals where the time axis should show seconds, not milliseconds.

    Args:
        signal: Audio signal, shape (N,).
        sr: Sample rate in Hz.
        label: Legend label (applied as ax title if provided).
        color: Waveform colour.
        ax: Target axes (creates figure if None).
        max_points: Samples above this length trigger envelope mode.

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(signal, sr=sr, color=color, ax=ax,
                              max_points=max_points)
    if label:
        ax.set_title(label)
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


def plot_mel_spectrogram(
    signal: np.ndarray,
    sr: int,
    ax: plt.Axes | None = None,
    n_mels: int = 128,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 8_000.0,
    cmap: str = "magma",
    title: str = "",
    colorbar: bool = True,
) -> plt.Axes:
    """Mel spectrogram using librosa.display.specshow.

    Uses librosa's y_axis='mel' which draws frequency ticks in Hz and labels
    the y axis correctly for mel-scale spectrograms.

    Args:
        signal: Audio signal, shape (N,), float32.
        sr: Sample rate in Hz.
        ax: Target axes (creates figure if None).
        n_mels: Number of mel filterbank channels.
        hop_length: STFT hop length in samples.
        fmin: Lowest mel frequency in Hz.
        fmax: Highest mel frequency in Hz.
        cmap: Colormap name.
        title: Axes title.
        colorbar: Whether to draw a colorbar.

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(
        y=signal.astype(np.float32), sr=sr,
        n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_db, y_axis="mel", x_axis="time", ax=ax,
        sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, cmap=cmap,
    )
    if colorbar:
        ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")
    if title:
        ax.set_title(title)
    return ax


def plot_cqt_spectrogram(
    signal: np.ndarray,
    sr: int,
    ax: plt.Axes | None = None,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float | None = None,
    cmap: str = "magma",
    title: str = "",
    colorbar: bool = True,
) -> plt.Axes:
    """CQT spectrogram using librosa.display.specshow with y_axis='cqt_hz'.

    The constant-Q transform gives equal pitch resolution across all octaves —
    ideal for visualising guitar content where harmonics fall on a log scale.

    Args:
        signal: Audio signal, shape (N,), float32.
        sr: Sample rate in Hz.
        ax: Target axes (creates figure if None).
        hop_length: Hop length in samples.
        n_bins: Total number of CQT bins.
        bins_per_octave: Frequency resolution per octave.
        fmin: Lowest CQT frequency. Defaults to librosa's C1.
        cmap: Colormap name.
        title: Axes title.
        colorbar: Whether to draw a colorbar.

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    if fmin is None:
        fmin = librosa.note_to_hz("C1")
    C = np.abs(librosa.cqt(
        signal.astype(np.float32), sr=sr, hop_length=hop_length,
        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave,
    ))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    img = librosa.display.specshow(
        C_db, y_axis="cqt_hz", x_axis="time", ax=ax,
        sr=sr, hop_length=hop_length,
        fmin=fmin, bins_per_octave=bins_per_octave, cmap=cmap,
    )
    if colorbar:
        ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")
    if title:
        ax.set_title(title)
    return ax


def plot_impulse_response(
    kernel: np.ndarray,
    label: str = "h[k]",
    color: str = "steelblue",
    ax: plt.Axes | None = None,
    n_show: int = 80,
) -> plt.Axes:
    """Stem plot of an FIR impulse response.

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
        np.arange(n), kernel[:n], markerfmt="o", linefmt="-", basefmt="k-",
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
    """Frequency response of an FIR kernel in dB.

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
# Compound figures
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
    harmonics: dict[float, str] | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Overlay multiple signal spectra in one frequency-domain axes.

    Args:
        signals: {label: (signal_array, color)} dict.
        sr: Sample rate in Hz.
        title: Figure title.
        max_freq: Upper x-axis limit in Hz.
        floor_db: Lower y-axis limit in dB.
        vlines: {label: frequency_Hz} — vertical reference lines.
        harmonics: {f0: note_name} — mark harmonic series for each fundamental.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, (sig, color) in signals.items():
        plot_spectrum(sig, sr, label=label, color=color, ax=ax,
                      max_freq=max_freq, floor_db=floor_db)
    if vlines:
        mark_vlines(ax, vlines, max_freq=max_freq, floor_db=floor_db)
    if harmonics:
        for f0 in harmonics:
            mark_harmonics(ax, f0, max_freq=max_freq)
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
    harmonics: dict[float, str] | None = None,
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
        vlines: Vertical reference lines on the spectrum axes.
        harmonics: {f0: note_name} — mark harmonic series on spectrum axes.
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
        mark_vlines(ax_f, vlines, max_freq=max_freq, floor_db=floor_db)
    if harmonics:
        for f0 in harmonics:
            mark_harmonics(ax_f, f0, max_freq=max_freq)
    ax_t.set_title("Time Domain")
    ax_f.set_title("Frequency Spectrum")
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def waveform_spectrogram_panel(
    signal: np.ndarray,
    sr: int,
    title: str = "",
    n_mels: int = 128,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 8_000.0,
    waveform_color: str = "steelblue",
    cmap: str = "magma",
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Mel spectrogram + waveform panel sharing a time axis.

    Two rows, shared X (time):
      - Top (3/4 height): mel spectrogram via librosa.display.specshow,
        time on X, mel frequency on Y.
      - Bottom (1/4 height): waveform via librosa.display.waveshow.

    Args:
        signal: Audio signal, shape (N,), float32.
        sr: Sample rate in Hz.
        title: Figure title (applied to spectrogram axes).
        n_mels: Mel filterbank channels.
        hop_length: STFT hop in samples.
        fmin: Lower mel frequency bound in Hz.
        fmax: Upper mel frequency bound in Hz.
        waveform_color: Waveform fill colour.
        cmap: Spectrogram colourmap.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    S = librosa.feature.melspectrogram(
        y=signal.astype(np.float32), sr=sr,
        n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, (ax_S, ax_w) = plt.subplots(
        2, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    img = librosa.display.specshow(
        S_db, ax=ax_S, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", fmin=fmin, fmax=fmax, cmap=cmap,
    )
    fig.colorbar(img, ax=ax_S, format="%+2.0f dB", fraction=0.03, pad=0.02)
    ax_S.set_title(title or "Mel spectrogram")
    ax_S.set_xlabel("")  # time label lives on the waveform row instead

    librosa.display.waveshow(signal, sr=sr, color=waveform_color,
                              ax=ax_w, max_points=200_000)
    ax_w.set_ylabel("Amplitude")

    # Align the time axis of the waveform to the spectrogram
    ax_w.set_xlim(ax_S.get_xlim())

    return fig


def error_analysis_panel(
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    skip: int = 0,
    ms: float = 30.0,
    max_freq: float = 5000.0,
    target_color: str = "darkorange",
    pred_color: str = "navy",
    title: str = "Prediction Error Analysis",
) -> plt.Figure:
    """2×2 panel: time overlay, residual waveform, spectrum overlay, error spectrum.

    Args:
        target: Reference wet signal, shape (N,).
        predicted: Model prediction, shape (N,) or shorter.
        sr: Sample rate in Hz.
        skip: Warmup samples to exclude from error computation and spectra.
        ms: Time window in milliseconds for waveform panels.
        max_freq: Upper frequency limit in Hz.
        target_color: Colour for the target curve.
        pred_color: Colour for the prediction curve.
        title: Figure suptitle.

    Returns:
        matplotlib Figure.
    """
    n = min(len(target), len(predicted))
    error = target[:n] - predicted[:n]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    n_show = int(sr * ms / 1000)
    t_ms = np.arange(n_show) / sr * 1000

    axes[0, 0].plot(t_ms, target[:n_show], color=target_color, lw=1.2, label="Target")
    axes[0, 0].plot(t_ms, predicted[:n_show], color=pred_color, lw=0.9, ls="--",
                    label="Prediction")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title("Time: Target vs Prediction")
    axes[0, 0].legend(fontsize=9)

    rms_err = float(np.sqrt(np.mean(error[skip:] ** 2)))
    axes[0, 1].plot(t_ms, error[:n_show], color="tomato", lw=0.7)
    axes[0, 1].axhline(0, color="black", lw=0.5)
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Error amplitude")
    axes[0, 1].set_title(f"Residual (target − prediction)  RMS = {rms_err:.2e}")

    for sig, label, color, lw in [
        (target[:n], "Target",     target_color, 1.2),
        (predicted[:n], "Prediction", pred_color, 0.9),
    ]:
        freqs_s, m = db_spectrum(sig, sr)
        axes[1, 0].plot(freqs_s, m, color=color, lw=lw, label=label)
    axes[1, 0].set_xlim(0, max_freq)
    axes[1, 0].set_ylim(-80, 5)
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("dBFS")
    axes[1, 0].set_title("Spectrum: Target vs Prediction")
    axes[1, 0].legend(fontsize=9)

    freqs_err, m_err = db_spectrum(error[skip:], sr)
    axes[1, 1].plot(freqs_err, m_err, color="tomato", lw=0.9)
    axes[1, 1].set_xlim(0, max_freq)
    axes[1, 1].set_ylim(-120, 0)
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("dBFS")
    axes[1, 1].set_title("Error Spectrum  (lower = better)")

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def ir_overlay(
    kernels: dict[str, tuple[np.ndarray, str]],
    ax: plt.Axes | None = None,
    n_show: int = 100,
    title: str = "Impulse Response Comparison",
) -> plt.Axes:
    """Overlay multiple impulse responses on one stem-plot axes.

    Args:
        kernels: {label: (kernel_array, color)} dict.
        ax: Target axes (creates figure if None).
        n_show: Number of taps to display.
        title: Axes title.

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    markers = ["o", "s", "^", "D"]
    linefmts = ["-", "--", "-.", ":"]
    for i, (label, (kernel, color)) in enumerate(kernels.items()):
        n = min(n_show, len(kernel))
        ml, sl, bl = ax.stem(
            np.arange(n), kernel[:n],
            markerfmt=markers[i % len(markers)],
            linefmt=linefmts[i % len(linefmts)],
            basefmt="k-",
            label=label,
        )
        ml.set(color=color, markersize=4)
        sl.set(color=color, linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Tap index k")
    ax.set_ylabel("h[k]")
    ax.set_title(title)
    ax.legend(fontsize=9)
    return ax


def freq_response_overlay(
    kernels: dict[str, tuple[np.ndarray, str]],
    sr: int,
    ax: plt.Axes | None = None,
    max_freq: float = 5000.0,
    floor_db: float = -60.0,
    title: str = "Frequency Response Comparison",
    linewidths: list[float] | None = None,
) -> plt.Axes:
    """Overlay frequency responses of multiple FIR kernels on one axes.

    Args:
        kernels: {label: (kernel_array, color)} dict.
        sr: Sample rate in Hz.
        ax: Target axes (creates figure if None).
        max_freq: Upper x-axis limit in Hz.
        floor_db: Lower y-axis limit in dB.
        title: Axes title.
        linewidths: Per-kernel line widths. Defaults to [2.0, 1.2, 1.0, ...].

    Returns:
        The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    lws = linewidths or [2.0] + [1.2] * 10
    for i, (label, (kernel, color)) in enumerate(kernels.items()):
        plot_freq_response(kernel, sr, label=label, color=color, ax=ax,
                           max_freq=max_freq, floor_db=floor_db, lw=lws[i])
    ax.set_title(title)
    return ax


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
    """3-panel figure: impulse response (optional), time overlay, spectrum overlay.

    Args:
        target: Ground-truth wet signal.
        prediction: Model's predicted wet signal.
        kernel: FIR kernel — pass None to skip the IR panel.
        sr: Sample rate in Hz.
        model_name: Used in panel titles.
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
    plot_waveform(target[:n],     sr, label="Target",       color=target_color,
                  ax=ax_t, ms=ms, lw=1.2)
    plot_waveform(prediction[:n], sr, label=model_name,     color=pred_color,
                  ax=ax_t, ms=ms, lw=0.9)
    ax_t.set_title("Time: Target vs Prediction")

    plot_spectrum(target[:n],     sr, label="Target",       color=target_color,
                  ax=ax_f, max_freq=max_freq, lw=1.2)
    plot_spectrum(prediction[:n], sr, label=model_name,     color=pred_color,
                  ax=ax_f, max_freq=max_freq, lw=0.9)
    ax_f.set_title("Spectrum: Target vs Prediction")

    fig.suptitle(f"{model_name} — Fit Quality", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Results-analysis plots (used in zoo-results and diagnostics notebooks)
# ──────────────────────────────────────────────────────────────────────────────

# Metrics where a lower value is better (used to invert heatmap colour)
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"ESR", "STFT", "FR_err_dB", "THD_err", "THD_pattern_dist"}
)


def db_spectrum_smooth(
    signal: np.ndarray,
    sr: int,
    n_bins: int = 512,
    f_min: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-smoothed magnitude spectrum: resample raw FFT to n_bins log-spaced bins.

    Raw FFT bins are linearly spaced — at high frequencies thousands of bins per
    octave create a filled blob on a log-freq plot. Resampling to ~512 log-spaced
    bins gives one clean line per trace and guarantees identical smoothing when
    n_bins is the same across all signals in a figure.

    Args:
        signal: Audio signal, shape (N,), float32.
        sr: Sample rate in Hz.
        n_bins: Number of log-spaced output bins. Default 512.
        f_min: Lowest output frequency in Hz. Default 20 Hz.

    Returns:
        (log_freqs, log_db) — both shape (n_bins,).
    """
    n = len(signal)
    w = np.hanning(n)
    X = rfft(signal * w)
    mag = np.abs(X) / (n / 2)
    freqs = rfftfreq(n, d=1.0 / sr)
    db_raw = 20.0 * np.log10(np.maximum(mag, 1e-9))
    # Skip DC (index 0); work with positive frequencies
    f_pos, db_pos = freqs[1:], db_raw[1:]
    f_start = max(f_pos[0], f_min)
    log_f = np.logspace(np.log10(f_start), np.log10(f_pos[-1]), n_bins)
    log_db = np.interp(log_f, f_pos, db_pos)
    return log_f, log_db


def compare_spectra_smooth(
    signals: dict[str, tuple[np.ndarray, str]],
    sr: int,
    n_bins: int = 512,
    f_min: float = 20.0,
    floor_db: float = -120.0,
    figsize: tuple[float, float] = (10, 4),
    title: str = "",
    lw_default: float = 0.9,
) -> plt.Figure:
    """Overlay log-smoothed spectra for multiple signals on a log-frequency axis.

    All traces use the same ``n_bins`` log-spaced bins so smoothing is identical
    across every model. The last signal in the dict is drawn on top (highest zorder).

    Args:
        signals: Ordered dict of {label: (signal_array, color)}.
            Put the reference / target signal last so it renders on top.
        sr: Sample rate in Hz.
        n_bins: Log-spaced bins for smoothing. Same value applied to every trace.
        f_min: Lowest x-axis frequency in Hz.
        floor_db: Lower y-axis limit (clips quantisation noise). Default -120 dB.
        figsize: Figure size.
        title: Axes title.
        lw_default: Line width for all traces.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_sig = len(signals)
    for z, (label, (sig, color)) in enumerate(signals.items()):
        f, db = db_spectrum_smooth(sig, sr, n_bins=n_bins, f_min=f_min)
        db = np.maximum(db, floor_db)
        ax.semilogx(f, db, color=color, lw=lw_default, label=label, zorder=z)
    ax.set_xlim(f_min, sr // 2)
    ax.set_ylim(bottom=floor_db)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_metrics_heatmap(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    metric_cols: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Summary metrics — mean across all sections",
) -> plt.Figure:
    """Normalised heatmap of model zoo metrics. Green = good regardless of metric direction.

    Lower-is-better metrics (ESR, STFT, FR_err_dB, THD_err, THD_pattern_dist) are
    inverted before colouring so the colour scale is always 0 = worst → 1 = best.

    Args:
        df: DataFrame indexed by model name with metric columns. Typically the
            output of ``ZooResults.to_dataframe()``.
        metric_cols: Columns to display. Default: recognised metrics present in df.
        figsize: Figure size. Default auto-sized by number of models and metrics.
        title: Axes title.

    Returns:
        matplotlib Figure.
    """
    import seaborn as sns
    import pandas as pd  # noqa: F401 — checked via isinstance

    _preferred = [
        "ESR", "null_depth_dB", "STFT", "FR_err_dB",
        "THD_err", "THD_pattern_dist", "HP_sim", "MCD",
    ]
    if metric_cols is None:
        metric_cols = [m for m in _preferred if m in df.columns]
    if not metric_cols:
        metric_cols = [c for c in df.columns if c not in ("type", "fit_s", "predict_s")]

    df_heat = df[metric_cols].copy()

    # Per-column normalise: 0 = worst, 1 = best
    df_norm = df_heat.copy()
    for col in df_norm.columns:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - mn) / max(mx - mn, 1e-9)
    df_norm = df_norm.fillna(0.5)

    # Invert lower-is-better metrics so green = good everywhere
    for col in _LOWER_IS_BETTER:
        if col in df_norm.columns:
            df_norm[col] = 1.0 - df_norm[col]

    w = max(8, len(metric_cols) * 1.2)
    h = max(4, len(df_heat) * 0.55)
    fig, ax = plt.subplots(figsize=figsize or (w, h))
    annot = df_heat.map(lambda v: f"{v:.3g}" if v == v else "—")  # NaN → "—"
    sns.heatmap(
        df_norm,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        linewidths=0.4,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "0 = worst  →  1 = best (normalised per metric)"},
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_esr_scatter(
    esr_df: "pd.DataFrame",  # type: ignore[name-defined]
    figsize: tuple[float, float] = (14, 4),
    esr_clip: float = 2.0,
    target_esr: float = 0.01,
    color_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Scatter plot of ESR per section for each model.

    Args:
        esr_df: Long-form DataFrame with columns: model, section, ESR.
        figsize: Figure size.
        esr_clip: Clip displayed ESR at this maximum.
        target_esr: Draw a horizontal reference line at this ESR value.
        color_map: {model_name: color}. Default: tab10 auto-assignment.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    models = list(esr_df["model"].unique())
    cmap = plt.cm.tab10
    for i, key in enumerate(models):
        color = (color_map or {}).get(key, cmap(i / max(1, len(models))))
        sub = esr_df[esr_df["model"] == key]
        ax.scatter(
            sub["section"],
            sub["ESR"].clip(upper=esr_clip),
            label=key,
            s=40,
            alpha=0.8,
            color=color,
        )
    ax.axhline(target_esr, color="green", ls="--", lw=1, label=f"ESR={target_esr} target")
    ax.set_xlabel("section")
    ax.set_ylabel(f"ESR (clipped at {esr_clip})")
    ax.set_title("ESR per section (lower is better)")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    return fig


def plot_esr_by_section_type(
    esr_df: "pd.DataFrame",  # type: ignore[name-defined]
    model_names: list[str] | None = None,
    figsize: tuple[float, float] = (16, 5),
    color_map: dict[str, str] | None = None,
    target_esr: float = 0.01,
) -> plt.Figure:
    """Grouped bar chart of mean ESR per section type for each model.

    Requires a ``type`` column in esr_df (section type label, e.g. 'val_sweep',
    'val_stepped_tone'). ESR axis is log-scaled so both excellent and poor models
    are readable simultaneously.

    Args:
        esr_df: Long-form DataFrame with columns: model, section, type, ESR.
        model_names: Ordered model keys to show. Default: all models in esr_df.
        figsize: Figure size.
        color_map: {model_name: color}. Default: tab10 auto-assignment.
        target_esr: Draw reference lines at ESR=1 (silence) and target_esr.

    Returns:
        matplotlib Figure.
    """
    if model_names is None:
        model_names = list(esr_df["model"].unique())

    df_grp = esr_df.groupby(["model", "type"])["ESR"].mean().reset_index()
    types = sorted(df_grp["type"].unique())
    x = np.arange(len(types))
    w = 0.8 / max(len(model_names), 1)
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(model_names):
        color = (color_map or {}).get(name, cmap(i / max(1, len(model_names))))
        sub = df_grp[df_grp["model"] == name].set_index("type")
        vals = [float(sub.loc[t, "ESR"]) if t in sub.index else float("nan") for t in types]
        ax.bar(x + i * w - 0.4, vals, w, label=name, color=color, alpha=0.85)

    ax.set_yscale("log")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="ESR=1 (worse than silence)")
    ax.axhline(target_esr, color="green", ls="--", lw=1, label=f"ESR={target_esr} (good target)")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylabel("ESR (mean within type, log scale)")
    ax.set_title("ESR per section type — lower is better")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    return fig


def plot_rms_envelope(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    sections: list[tuple[int, int]] | None = None,
    win_ms: float = 10.0,
    figsize: tuple[float, float] = (18, 5),
    dry_color: str = "steelblue",
    wet_color: str = "tomato",
) -> plt.Figure:
    """Two-panel RMS envelope overview: dry (top) and wet/target (bottom).

    Args:
        dry: Dry signal, shape (N,), float32.
        wet: Wet (target) signal, shape (N,), float32.
        sr: Sample rate in Hz.
        sections: List of (start_sample, end_sample) pairs to shade on the wet
            panel. Pass ``[(s.start_sample, s.end_sample) for s in manifest.sections]``
            to show section boundaries.
        win_ms: RMS window length in milliseconds.
        figsize: Figure size.
        dry_color: Line colour for the dry panel.
        wet_color: Line colour for the wet panel.

    Returns:
        matplotlib Figure.
    """
    win = max(1, int(sr * win_ms / 1000.0))
    n_frames = min(len(dry), len(wet)) // win
    t_env = np.arange(n_frames) * win / sr

    def _rms_env(x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(x[: n_frames * win].reshape(n_frames, win) ** 2, axis=1))

    dry_env = _rms_env(dry)
    wet_env = _rms_env(wet)

    fig, (ax_d, ax_w) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax_d.plot(t_env, dry_env, lw=0.7, color=dry_color, label="dry")
    ax_d.set_ylabel("RMS amplitude")
    ax_d.set_title(f"Dry signal RMS envelope ({win_ms:.0f} ms)")
    ax_d.legend()

    ax_w.plot(t_env, wet_env, lw=0.7, color=wet_color, label="wet / target")
    ax_w.set_ylabel("RMS amplitude")
    ax_w.set_xlabel("time (s)")
    ax_w.set_title(f"Wet signal RMS envelope ({win_ms:.0f} ms)")
    ax_w.legend()

    if sections:
        for start_s, end_s in sections:
            ax_w.axvspan(start_s / sr, end_s / sr, alpha=0.04, color="gray")

    fig.tight_layout()
    return fig


def plot_topology_weights(
    topology: dict[str, float],
    title: str = "CircuitDictionary — learned topology (α weights)",
    figsize: tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Horizontal bar chart of CircuitDictionary architecture weights.

    Bars are sorted descending by weight. A dashed vertical line marks the
    uniform-prior weight (1/n_primitives) for reference.

    Args:
        topology: {primitive_name: softmax_weight} as stored in
            ``ModelResult.topology`` after a CircuitDictionary eval run.
        title: Axes title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    pairs = sorted(topology.items(), key=lambda kv: kv[1], reverse=True)
    names = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    uniform_prior = 1.0 / max(len(names), 1)
    max_w = max(weights) if weights else 1.0

    colors = plt.cm.RdYlGn([0.2 + 0.6 * w / max(max_w, 1e-9) for w in weights])
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(names)), weights, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("softmax weight α")
    ax.set_title(title)
    ax.axvline(uniform_prior, color="gray", ls="--", lw=0.8,
                label=f"uniform prior ({uniform_prior:.2f})")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Multi-signal comparison panels
# ──────────────────────────────────────────────────────────────────────────────
# All ``compare_*`` functions accept:
#   signals : dict[str, np.ndarray]   — ordered, label → audio array
#   ref_key : str                     — key of the reference signal in signals
# They return a plt.Figure ready to savefig().
# ──────────────────────────────────────────────────────────────────────────────

_CMP_COLORS = ["steelblue", "darkorange", "forestgreen", "crimson"]


def _avg_magnitude(sig: np.ndarray, n_fft: int = 4096) -> np.ndarray:
    """Average magnitude spectrum across non-overlapping Hann-windowed frames."""
    win = np.hanning(n_fft)
    n_frames = len(sig) // n_fft
    if n_frames == 0:
        return np.abs(np.fft.rfft(sig * np.hanning(len(sig)), n=n_fft))
    frames = sig[: n_frames * n_fft].reshape(n_frames, n_fft) * win
    return np.mean(np.abs(np.fft.rfft(frames, axis=1)), axis=0)


def _cross_H(dry: np.ndarray, sig: np.ndarray, n_fft: int = 4096) -> np.ndarray:
    """Averaged transfer function H(f) = sig/dry via Welch cross-spectrum estimate."""
    win = np.hanning(n_fft)
    N = min(len(dry), len(sig))
    n_frames = N // n_fft
    H = np.zeros(n_fft // 2 + 1, dtype=complex)
    for i in range(n_frames):
        s, e = i * n_fft, (i + 1) * n_fft
        X = np.fft.rfft(dry[s:e] * win)
        Y = np.fft.rfft(sig[s:e] * win)
        H += Y * np.conj(X) / (np.abs(X) ** 2 + 1e-10)
    return H / max(n_frames, 1)


def _mel_fb(sr: int, n_fft: int, n_mels: int = 80) -> np.ndarray:
    """Mel filterbank matrix, shape (n_mels, n_fft//2+1)."""
    def hz2mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz2mel(20.0), hz2mel(sr / 2.0), n_mels + 2)
    bins = np.clip(
        np.floor((n_fft + 1) * np.array([mel2hz(m) for m in mel_pts]) / sr).astype(int),
        0, n_fft // 2,
    )
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        lo, ctr, hi = bins[m], bins[m + 1], bins[m + 2]
        if ctr > lo:
            fb[m, lo:ctr] = np.linspace(0.0, 1.0, ctr - lo)
        if hi > ctr:
            fb[m, ctr:hi] = np.linspace(1.0, 0.0, hi - ctr)
    return fb


def _smooth1d(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def _energetic_start(sig: np.ndarray, sr: int, window_s: float) -> int:
    """Return the start sample of the most energetic non-overlapping window."""
    n = int(window_s * sr)
    rms = np.array([
        float(np.sqrt(np.mean(sig[i: i + n] ** 2)))
        for i in range(0, min(len(sig) - n, sr * 10), n)
    ])
    return int(np.argmax(rms)) * n


def compare_waveform_error_panel(
    dry: np.ndarray,
    signals: dict[str, np.ndarray],
    ref_key: str,
    sr: int,
    window_s: float = 0.1,
) -> plt.Figure:
    """Multi-panel waveform comparison with per-signal error traces.

    Layout (rows):
      0  : dry signal
      1  : all signals overlaid on the same axes
      2+ : for each predicted signal — left: signal vs ref overlay;
           right: error (ref − pred)

    Args:
        dry: Dry input signal.
        signals: Ordered dict of label → audio array.
        ref_key: Key of the reference signal in ``signals``.
        sr: Sample rate in Hz.
        window_s: Time window to display in seconds.

    Returns:
        matplotlib Figure.
    """
    ref = signals[ref_key]
    preds = {k: v for k, v in signals.items() if k != ref_key}
    n = int(window_s * sr)
    start = _energetic_start(dry, sr, window_s)
    t_ms = np.arange(n) * 1000.0 / sr

    n_pred = len(preds)
    n_rows = 2 + n_pred
    fig = plt.figure(figsize=(16, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.5)

    # Row 0: dry (spans both columns)
    ax_dry = fig.add_subplot(gs[0, :])
    ax_dry.plot(t_ms, dry[start: start + n], lw=0.8, color="gray")
    ax_dry.set_title("Dry input")
    ax_dry.set_ylabel("Amplitude")
    ax_dry.grid(True, alpha=0.3)

    # Row 1: all signals overlaid (spans both columns)
    ax_all = fig.add_subplot(gs[1, :])
    for (label, sig), color in zip(signals.items(), _CMP_COLORS):
        ax_all.plot(t_ms, sig[start: start + n], lw=0.9, color=color,
                    label=label, alpha=0.85)
    ax_all.set_title("All signals overlaid")
    ax_all.set_ylabel("Amplitude")
    ax_all.legend(fontsize=8, loc="upper right")
    ax_all.grid(True, alpha=0.3)

    # Rows 2+: per-prediction — left: overlay with ref; right: error
    for row, ((label, pred), color) in enumerate(zip(preds.items(), _CMP_COLORS[1:]), start=2):
        N = min(len(ref), len(pred))
        ax_l = fig.add_subplot(gs[row, 0])
        ax_l.plot(t_ms, ref[start: start + n], lw=1.0, color=_CMP_COLORS[0],
                  label=ref_key, alpha=0.9)
        ax_l.plot(t_ms, pred[start: start + n], lw=0.8, color=color,
                  label=label, alpha=0.85, linestyle="--")
        ax_l.set_title(f"{ref_key} vs {label}")
        ax_l.set_ylabel("Amplitude")
        ax_l.legend(fontsize=8)
        ax_l.grid(True, alpha=0.3)

        ax_r = fig.add_subplot(gs[row, 1])
        err = ref[start: start + n].astype(np.float64) - pred[start: start + n].astype(np.float64)
        rms_err = float(np.sqrt(np.mean(
            (ref[:N].astype(np.float64) - pred[:N].astype(np.float64)) ** 2
        )))
        ax_r.plot(t_ms, err, lw=0.8, color=color)
        ax_r.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax_r.set_title(f"{ref_key} − {label}  (full-length RMS = {rms_err:.5f})")
        ax_r.set_ylabel("Error")
        ax_r.grid(True, alpha=0.3)

    for ax in fig.axes:
        ax.set_xlabel("Time (ms)")

    fig.suptitle(f"Waveform & Error Comparison ({window_s*1000:.0f} ms excerpt)", fontsize=13)
    fig.tight_layout()
    return fig


def compare_spectrum_analysis_panel(
    dry: np.ndarray,
    signals: dict[str, np.ndarray],
    ref_key: str,
    sr: int,
    n_fft: int = 4096,
) -> plt.Figure:
    """3-column panel: averaged spectrum | transfer function | coherence.

    Args:
        dry: Dry input signal.
        signals: label → audio array.
        ref_key: Reference signal key.
        sr: Sample rate in Hz.
        n_fft: FFT size.

    Returns:
        matplotlib Figure.
    """
    ref = signals[ref_key]
    preds = {k: v for k, v in signals.items() if k != ref_key}
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Column 0: averaged magnitude spectra
    ax = axes[0]
    for (label, sig), color in zip(signals.items(), _CMP_COLORS):
        mag = _avg_magnitude(sig, n_fft)
        ax.semilogx(freqs[1:], 20 * np.log10(mag[1:] + 1e-12),
                    label=label, color=color, lw=1.2, alpha=0.85)
    ax.set_xlim(20, sr / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Averaged Magnitude Spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Column 1: transfer function |H(f)| = wet/dry
    ax = axes[1]
    for (label, sig), color in zip(signals.items(), _CMP_COLORS):
        H = _cross_H(dry, sig, n_fft)
        ax.semilogx(freqs[1:], 20 * np.log10(np.abs(H[1:]) + 1e-12),
                    label=label, color=color, lw=1.2, alpha=0.85)
    ax.set_xlim(20, sr / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Transfer Function |H(f)| = wet/dry")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Column 2: coherence of each pred vs ref
    ax = axes[2]
    for (label, pred), color in zip(preds.items(), _CMP_COLORS[1:]):
        N = min(len(ref), len(pred))
        f_coh, cxy = scipy_signal.coherence(ref[:N], pred[:N], fs=sr, nperseg=n_fft)
        ax.semilogx(f_coh[1:], cxy[1:], label=label, color=color, lw=1.2)
    ax.set_xlim(20, sr / 2)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title(f"Coherence vs {ref_key} (1.0 = perfect)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Spectral Analysis", fontsize=13)
    fig.tight_layout()
    return fig


def compare_phase_analysis_panel(
    dry: np.ndarray,
    signals: dict[str, np.ndarray],
    ref_key: str,
    sr: int,
    n_fft: int = 8192,
) -> plt.Figure:
    """2-column panel: transfer function phase | group delay.

    Args:
        dry: Dry input signal.
        signals: label → audio array.
        ref_key: Reference signal key.
        sr: Sample rate in Hz.
        n_fft: FFT size (larger = smoother group delay).

    Returns:
        matplotlib Figure.
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    omega = 2 * np.pi * freqs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for (label, sig), color in zip(signals.items(), _CMP_COLORS):
        H = _cross_H(dry, sig, n_fft)
        phase = np.unwrap(np.angle(H))

        axes[0].semilogx(freqs[2:], np.degrees(phase[2:]),
                         label=label, color=color, lw=1.0, alpha=0.85)

        gd_ms = -np.gradient(phase, omega) * 1000.0
        gd_ms = _smooth1d(gd_ms, 40)
        axes[1].semilogx(freqs[2:], gd_ms[2:],
                         label=label, color=color, lw=1.2, alpha=0.85)

    for ax, (title, ylabel) in zip(axes, [
        ("Transfer Function Phase (wet/dry)", "Phase (degrees)"),
        ("Group Delay (smoothed)", "Group delay (ms)"),
    ]):
        ax.set_xlim(20, sr / 2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[1].set_ylim(-20, 150)
    fig.suptitle("Phase Analysis", fontsize=13)
    fig.tight_layout()
    return fig


def compare_rms_envelopes(
    signals: dict[str, np.ndarray],
    sr: int,
    window_ms: float = 50.0,
) -> plt.Figure:
    """Rolling RMS envelope for all signals on a single axes.

    Args:
        signals: label → audio array.
        sr: Sample rate in Hz.
        window_ms: RMS window length in milliseconds.

    Returns:
        matplotlib Figure.
    """
    window = int(window_ms * sr / 1000)
    hop = window // 2
    fig, ax = plt.subplots(figsize=(16, 5))
    for (label, sig), color in zip(signals.items(), _CMP_COLORS):
        n_frames = (len(sig) - window) // hop
        if n_frames < 1:
            continue
        t = np.arange(n_frames) * hop / sr
        rms = np.array([
            float(np.sqrt(np.mean(sig[i * hop: i * hop + window] ** 2)))
            for i in range(n_frames)
        ])
        ax.plot(t, 20 * np.log10(rms + 1e-10), label=label, color=color,
                lw=0.8, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS level (dBFS)")
    ax.set_title(f"RMS Envelope ({window_ms:.0f} ms window)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def compare_mel_spectrograms(
    signals: dict[str, np.ndarray],
    sr: int,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 80,
) -> plt.Figure:
    """Side-by-side mel spectrograms on a shared colour scale.

    Args:
        signals: label → audio array.
        sr: Sample rate in Hz.
        n_fft: STFT window size.
        hop: STFT hop in samples.
        n_mels: Number of mel bands.

    Returns:
        matplotlib Figure.
    """
    fb = _mel_fb(sr, n_fft, n_mels)
    win = np.hanning(n_fft)
    mel_specs: dict[str, np.ndarray] = {}
    for label, sig in signals.items():
        n_frames = (len(sig) - n_fft) // hop
        if n_frames < 1:
            continue
        mag = np.stack([
            np.abs(np.fft.rfft(sig[i * hop: i * hop + n_fft] * win))
            for i in range(n_frames)
        ], axis=1)
        mel_specs[label] = 20 * np.log10(fb @ mag + 1e-10)

    if not mel_specs:
        return plt.figure()

    vmax = max(m.max() for m in mel_specs.values())
    vmin = vmax - 80.0
    n_sigs = len(mel_specs)
    fig, axes = plt.subplots(1, n_sigs, figsize=(8 * n_sigs, 5),
                              sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (label, mel_db) in zip(axes, mel_specs.items()):
        duration = len(signals[label]) / sr
        im = ax.imshow(mel_db, aspect="auto", origin="lower",
                       extent=[0, duration, 0, n_mels],
                       vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Mel band index")
    plt.colorbar(im, ax=axes[-1], label="dB")
    fig.suptitle("Mel Spectrogram Comparison", fontsize=13)
    fig.tight_layout()
    return fig


def compare_phase_portraits(
    signals: dict[str, np.ndarray],
    ref_key: str,
) -> plt.Figure:
    """Lissajous phase portrait for each predicted signal vs reference.

    A perfect match lies on the 45° diagonal. Deviations reveal amplitude
    or nonlinear error. Points are density-coloured (log count).

    Args:
        signals: label → audio array.
        ref_key: Key of the reference signal.

    Returns:
        matplotlib Figure.
    """
    ref = signals[ref_key]
    preds = {k: v for k, v in signals.items() if k != ref_key}
    if not preds:
        return plt.figure()

    n_pred = len(preds)
    fig, axes = plt.subplots(1, n_pred, figsize=(6 * n_pred, 6), squeeze=False)
    axes = axes[0]
    for ax, (label, pred) in zip(axes, preds.items()):
        N = min(len(ref), len(pred))
        r = ref[:N].astype(np.float64)
        p = pred[:N].astype(np.float64)
        counts, xedges, yedges = np.histogram2d(r, p, bins=300)
        ax.imshow(
            np.log1p(counts).T, origin="lower", aspect="equal",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="hot", interpolation="bilinear",
        )
        lim = max(float(np.abs(r).max()), float(np.abs(p).max())) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "w--", lw=1.2, alpha=0.7, label="ideal (45°)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{ref_key} amplitude")
        ax.set_ylabel(f"{label} amplitude")
        ax.set_title(f"Phase portrait: {ref_key} vs {label}")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Lissajous Phase Portrait (density coloured, ideal = diagonal)", fontsize=13)
    fig.tight_layout()
    return fig


def compare_error_distributions(
    signals: dict[str, np.ndarray],
    ref_key: str,
) -> plt.Figure:
    """2-row panel: error histogram (top) + scatter density (bottom) per predicted signal.

    Args:
        signals: label → audio array.
        ref_key: Key of the reference signal.

    Returns:
        matplotlib Figure.
    """
    ref = signals[ref_key]
    preds = {k: v for k, v in signals.items() if k != ref_key}
    if not preds:
        return plt.figure()

    n_pred = len(preds)
    fig, axes = plt.subplots(2, n_pred, figsize=(6 * n_pred, 10), squeeze=False)

    for col, (label, pred) in enumerate(preds.items()):
        color = _CMP_COLORS[(col + 1) % len(_CMP_COLORS)]
        N = min(len(ref), len(pred))
        r = ref[:N].astype(np.float64)
        p = pred[:N].astype(np.float64)
        err = r - p

        # Top: histogram + Gaussian fit
        ax = axes[0, col]
        ax.hist(err, bins=400, density=True, alpha=0.55, color=color, label=label)
        std = float(np.std(err))
        x = np.linspace(-5 * std, 5 * std, 600)
        gauss = np.exp(-x ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, gauss, color=color, lw=2.0, ls="--",
                label=f"Gaussian (σ={std:.5f})")
        ax.set_xlabel(f"ref − {label}")
        ax.set_ylabel("Density")
        ax.set_title(f"Error distribution: {ref_key} − {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom: 2D density scatter
        ax = axes[1, col]
        counts, xedges, yedges = np.histogram2d(r, p, bins=256)
        im = ax.imshow(
            np.log1p(counts).T, origin="lower", aspect="equal",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis", interpolation="bilinear",
        )
        plt.colorbar(im, ax=ax, label="log(count+1)")
        lim = max(float(np.abs(r).max()), float(np.abs(p).max())) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.5, alpha=0.8, label="ideal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{ref_key}")
        ax.set_ylabel(label)
        ax.set_title(f"Scatter density: {ref_key} vs {label}")
        ax.legend(fontsize=8)

    fig.suptitle("Error Distribution Analysis", fontsize=13)
    fig.tight_layout()
    return fig


def compare_harmonic_fingerprint(
    dry: np.ndarray,
    signals: dict[str, np.ndarray],
    sr: int,
) -> plt.Figure:
    """Relative harmonic energy H1–H5 for each signal.

    Detects the dominant fundamental frequency from the dry signal
    (guitar range 80–500 Hz), then measures spectral energy at H1–H5
    in each wet signal. All values are normalised to H1 = 0 dB.

    Args:
        dry: Dry input signal.
        signals: label → audio array.
        sr: Sample rate in Hz.

    Returns:
        matplotlib Figure.
    """
    n_fft = max(4096, min(65536, 2 ** int(np.log2(max(len(dry) // 2, 4096)))))
    win = np.hanning(n_fft)
    n_frames = len(dry) // n_fft

    mag_dry = np.zeros(n_fft // 2 + 1)
    for i in range(min(n_frames, 16)):
        frame = dry[i * n_fft: (i + 1) * n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        mag_dry += np.abs(np.fft.rfft(frame * win))
    mag_dry /= max(min(n_frames, 16), 1)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    guitar_mask = (freqs >= 80) & (freqs <= 500)
    f0 = float(freqs[guitar_mask][np.argmax(mag_dry[guitar_mask])])
    harmonics = [f0 * h for h in range(1, 6)]
    bw = f0 * 0.08

    x = np.arange(5)
    width = 0.8 / len(signals)
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (label, sig) in enumerate(signals.items()):
        n_frames2 = len(sig) // n_fft
        mag_sig = np.zeros(n_fft // 2 + 1)
        for j in range(min(n_frames2, 16)):
            frame = sig[j * n_fft: (j + 1) * n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            mag_sig += np.abs(np.fft.rfft(frame * win))
        mag_sig /= max(min(n_frames2, 16), 1)

        h_energy = []
        for hf in harmonics:
            band = (freqs >= hf - bw) & (freqs <= hf + bw) & (freqs < sr / 2)
            h_energy.append(float(np.mean(mag_sig[band])) if band.any() else 0.0)

        h1 = h_energy[0] if h_energy[0] > 0 else 1.0
        h_db = [20 * np.log10(e / h1 + 1e-12) for e in h_energy]
        ax.bar(x + i * width, h_db, width, label=label,
               color=_CMP_COLORS[i % len(_CMP_COLORS)], alpha=0.85)

    ax.set_xticks(x + width * (len(signals) - 1) / 2)
    ax.set_xticklabels([f"H{h} ({harmonics[h-1]:.0f} Hz)" for h in range(1, 6)], rotation=10)
    ax.set_ylabel("Level relative to H1 (dB)")
    ax.set_title(f"Harmonic Fingerprint  (f₀ ≈ {f0:.0f} Hz detected from dry)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def compare_radar_chart(
    results: dict[str, dict[str, float]],
    higher_is_better: frozenset[str] = frozenset({"HP_sim", "null_depth_dB"}),
) -> plt.Figure:
    """Radar (spider) chart of normalised key metrics — smaller polygon = better.

    Args:
        results: {pair_label: {metric_name: value}} as returned by compute_all_metrics.
        higher_is_better: Metrics where a larger value is preferred (they are
            inverted so the radar axis is always "lower = better").

    Returns:
        matplotlib Figure.
    """
    keys = ["ESR", "STFT", "FR_err_dB", "THD_err", "HP_sim", "MCD"]
    axis_labels = ["ESR", "STFT", "FR err\n(dB)", "THD err", "1−HP_sim", "MCD"]

    normalized: dict[str, dict[str, float]] = {}
    for key in keys:
        raw = {pair: results[pair].get(key, float("nan")) for pair in results}
        vals = [v for v in raw.values() if not np.isnan(v)]
        mn, mx = (min(vals), max(vals)) if vals else (0.0, 1.0)
        span = mx - mn if mx != mn else 1.0
        if key in higher_is_better:
            normalized[key] = {p: (mx - raw[p]) / span for p in results}
        else:
            normalized[key] = {p: (raw[p] - mn) / span for p in results}

    N = len(keys)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for (pair, _), color in zip(results.items(), _CMP_COLORS):
        vals = [normalized[k][pair] for k in keys] + [normalized[keys[0]][pair]]
        ax.plot(angles, vals, color=color, lw=2, label=pair)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "worst"], size=8)
    ax.set_title("Normalised Metrics Radar\n(smaller polygon = better match)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    fig.tight_layout()
    return fig


def compare_sliding_esr(
    signals: dict[str, np.ndarray],
    ref_key: str,
    sr: int,
    window_s: float = 1.0,
) -> plt.Figure:
    """ESR computed in sliding windows — reveals where the model struggles.

    Args:
        signals: label → audio array.
        ref_key: Key of the reference signal.
        sr: Sample rate in Hz.
        window_s: Window length in seconds (50% overlap).

    Returns:
        matplotlib Figure.
    """
    ref = signals[ref_key]
    preds = {k: v for k, v in signals.items() if k != ref_key}
    window = int(window_s * sr)
    hop = window // 2

    fig, ax = plt.subplots(figsize=(16, 5))
    for (label, pred), color in zip(preds.items(), _CMP_COLORS[1:]):
        N = min(len(ref), len(pred))
        n_windows = (N - window) // hop
        if n_windows < 1:
            continue
        t = np.arange(n_windows) * hop / sr
        esr = np.array([
            float(
                np.sum((ref[i * hop: i * hop + window] - pred[i * hop: i * hop + window]) ** 2)
                / (np.sum(ref[i * hop: i * hop + window] ** 2) + 1e-10)
            )
            for i in range(n_windows)
        ])
        ax.plot(t, 10 * np.log10(esr + 1e-10), label=label, color=color, lw=0.9)

    ax.axhline(-20, color="gray", lw=1.0, ls="--", alpha=0.7,
               label="ESR=0.01 (PASS threshold)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ESR (dB)")
    ax.set_title(f"Sliding ESR — {window_s:.0f} s window, 50% overlap  (lower = better)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
