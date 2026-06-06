"""Reusable matplotlib helpers for signal visualisation in notebooks.

Spectrograms and waveforms use ``librosa.display.specshow`` /
``librosa.display.waveshow`` for axis labelling (time, mel, Hz) and
proper aspect handling. Primitive helpers (spectrum, IR, freq response)
use plain matplotlib so they have no librosa dependency.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
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
