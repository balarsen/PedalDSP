"""Deep diagnostic plots for evaluating pedal model quality.

Each public function accepts (dry, target, predicted, sr) arrays plus an
optional save_path, and returns a matplotlib Figure. All figures use a dark
background with a consistent palette:

    C_TARGET = '#00FF88'  (green)  — pedal wet / ground truth
    C_PRED   = '#FF6B6B'  (coral)  — model prediction
    C_DRY    = '#4FC3F7'  (blue)   — dry input
    C_ERR    = '#FFD700'  (gold)   — error / residual signal

Call run_all_diagnostics() to generate every plot in a single pass.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, coherence as _coherence, medfilt, sosfilt

# ── Palette / style ───────────────────────────────────────────────────────────

C_TARGET = "#00FF88"
C_PRED   = "#FF6B6B"
C_DRY    = "#4FC3F7"
C_ERR    = "#FFD700"

_STYLE = "dark_background"
_LOG_TICKS = [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000]


# ── Private helpers ───────────────────────────────────────────────────────────

def _log_freq_axis(ax: plt.Axes, fmin: float = 20.0, fmax: float = 20_000.0) -> None:
    """Set log-scale x-axis with human-readable Hz / kHz tick labels."""
    ax.set_xscale("log")
    ticks = [t for t in _LOG_TICKS if fmin <= t <= fmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1_000}k" if t >= 1_000 else str(t) for t in ticks])
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")


def _wiener_tf(
    dry: np.ndarray,
    wet: np.ndarray,
    noise_floor: float = 1e-6,
) -> np.ndarray:
    """Wiener-regularised transfer function H(f) = X*(f)·Y(f) / (|X(f)|² + ε)."""
    n = max(len(dry), len(wet))
    X = rfft(dry.astype(np.float64), n=n)
    Y = rfft(wet.astype(np.float64), n=n)
    Sxx = np.abs(X) ** 2
    reg = noise_floor * (np.max(Sxx) + 1e-30)
    return (np.conj(X) * Y) / (Sxx + reg)


def _wiener_ir(dry: np.ndarray, wet: np.ndarray, noise_floor: float = 1e-6) -> np.ndarray:
    """Estimate impulse response from dry/wet pair via Wiener deconvolution."""
    return np.real(irfft(_wiener_tf(dry, wet, noise_floor))).astype(np.float32)


def _aweight_db(freqs: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting in dB, normalised to 0 dB at 1 kHz."""
    f = np.asarray(freqs, dtype=np.float64)
    f2 = f * f
    with np.errstate(divide="ignore", invalid="ignore"):
        RA = (12194.0 ** 2 * f2 ** 2) / (
            (f2 + 20.6 ** 2)
            * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
            * (f2 + 12194.0 ** 2)
        )
    RA = np.where(f > 0, RA, 0.0)
    f1k2 = 1_000.0 ** 2
    RA1k = (12194.0 ** 2 * f1k2 ** 2) / (
        (f1k2 + 20.6 ** 2)
        * np.sqrt((f1k2 + 107.7 ** 2) * (f1k2 + 737.9 ** 2))
        * (f1k2 + 12194.0 ** 2)
    )
    return 20.0 * np.log10(np.maximum(RA / (RA1k + 1e-30), 1e-12))


def _synth_sine(f0: float, sr: int, duration: float, amp: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amp * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)


def _fft_amp_at(signal: np.ndarray, sr: int, f0: float) -> float:
    """Peak FFT amplitude at f0 (caller is responsible for windowing)."""
    X = np.abs(rfft(signal.astype(np.float64)))
    freqs = rfftfreq(len(signal), 1.0 / sr)
    k = int(np.argmin(np.abs(freqs - f0)))
    return float(X[k]) * 2.0 / len(signal)


def _harmonic_amps(
    signal: np.ndarray, sr: int, f0: float, n_harmonics: int = 8,
) -> np.ndarray:
    """Amplitudes of harmonics 1..n_harmonics via Hann-windowed FFT."""
    win = signal.astype(np.float64) * np.hanning(len(signal))
    X = np.abs(rfft(win))
    freqs = rfftfreq(len(signal), 1.0 / sr)
    amps = np.zeros(n_harmonics)
    for k in range(1, n_harmonics + 1):
        idx = int(np.argmin(np.abs(freqs - k * f0)))
        amps[k - 1] = float(X[idx]) * 2.0 / len(signal)
    return amps


def _smooth_log_octave(
    freqs: np.ndarray, values: np.ndarray, frac: float = 1.0 / 3.0,
) -> np.ndarray:
    """1/3-octave smoothing via uniform resampling in log-frequency space."""
    valid = freqs > 0
    if valid.sum() < 4:
        return values.copy()
    lf = np.log2(freqs[valid])
    vals = values[valid]
    n_grid = 1_000
    lf_uni = np.linspace(lf[0], lf[-1], n_grid)
    v_uni = np.interp(lf_uni, lf, vals)
    log_range = max(lf[-1] - lf[0], 1e-6)
    win = max(3, int(frac * n_grid / log_range))
    v_sm = uniform_filter1d(v_uni, size=win, mode="nearest")
    out = values.copy()
    out[valid] = np.interp(lf, lf_uni, v_sm)
    return out


def _bandpass_sos(f_center: float, sr: int, width: float = 0.2) -> np.ndarray:
    """4th-order Butterworth bandpass SOS centred at f_center."""
    flo = max(f_center * (1 - width), 20.0)
    fhi = min(f_center * (1 + width), sr * 0.49)
    return butter(4, [flo, fhi], btype="bandpass", fs=sr, output="sos")


def _save(fig: plt.Figure, path) -> None:
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())


def _stamp_figure(
    fig: plt.Figure,
    model_name: str,
    dry_name: str,
    wet_name: str,
) -> None:
    """Add a small footer line to fig identifying the signals and model.

    Placed in figure coordinates so it never overlaps axes content.
    Shows: dry · wet · model · date
    """
    import datetime
    date_str = datetime.date.today().isoformat()
    label = f"dry: {dry_name}  ·  wet: {wet_name}  ·  model: {model_name}  ·  {date_str}"
    fig.text(
        0.5, 0.005, label,
        ha="center", va="bottom",
        fontsize=7, color="#888888",
        transform=fig.transFigure,
    )


# ── 1. STATIC NONLINEARITY ────────────────────────────────────────────────────

def plot_static_transfer_curve(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Sample-by-sample input→output scatter plot.

    Reveals the static nonlinearity: clipping shape, soft vs hard knee,
    and symmetry between positive and negative halves. A linear system
    traces a straight diagonal; hard clipping flattens the extremes.

    Diffuse scatter (not a tight curve) indicates memory — the output depends
    on history, not just the instantaneous input. Model divergence from target
    at high amplitudes reveals a wrong saturation level or knee shape.

    Args:
        dry: Dry input signal, shape (N,), float32, range [-1, 1].
        target: Pedal wet output, same shape.
        predicted: Model output, same shape.
        sr: Sample rate in Hz (unused; kept for API consistency).
        save_path: Optional path to save PNG.

    Returns:
        matplotlib Figure.
    """
    n = min(len(dry), len(target), len(predicted))
    n_sub = min(n, 50_000)
    idx = np.round(np.linspace(0, n - 1, n_sub)).astype(int)
    d, t, p = dry[idx], target[idx], predicted[idx]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, sig, color, label in [
            (axes[0], t, C_TARGET, "Target"),
            (axes[1], p, C_PRED, "Predicted"),
        ]:
            ax.scatter(d, sig, s=1.5, alpha=0.2, color=color,
                       label=label, rasterized=True)
            diag = np.array([-1.05, 1.05])
            ax.plot(diag, diag, color="white", lw=0.6, ls="--",
                    alpha=0.25, label="Linear (unity)")
            ax.set_xlabel("Dry amplitude")
            ax.set_ylabel("Wet amplitude")
            ax.set_title(f"Transfer curve — {label}")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal")
            ax.legend(markerscale=8, fontsize=9)

        fig.suptitle("Static Transfer Curve  (50 k subsampled points)", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_describing_function(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    model_fn: Callable | None = None,
    save_path=None,
) -> plt.Figure:
    """Effective gain vs input level at 440 Hz.

    Sweeps input amplitude from -40 dBFS to 0 dBFS and measures the
    fundamental-frequency output component. Effective gain = output_f0 /
    input_amplitude (dB). A linear system is flat; saturation causes gain
    compression as input level rises.

    With model_fn provided: synthesises test sines and passes them through
    the callable — gives a precise per-level measurement.
    Without model_fn: estimates from pre-computed arrays using overlapping
    50 ms Hann-windowed FFTs — noisy but directionally correct.

    Look for: gain compression onset, soft vs hard knee shape, and whether
    the model's compression curve tracks the target's saturation.

    Args:
        dry: Dry input array.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        model_fn: Optional callable (x: np.ndarray) -> np.ndarray. If given,
            the predicted curve is re-computed from synthesised sines.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    f0 = 440.0
    n_levels = 60
    level_dbs = np.linspace(-40, 0, n_levels)
    amps = 10.0 ** (level_dbs / 20.0)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        # ── Estimate from pre-computed arrays via short-time windowed FFT ──
        win_n = min(int(sr * 0.05), len(dry) // 4)
        hop_n = win_n // 4
        hann  = np.hanning(win_n)
        i_a, t_a, p_a = [], [], []
        for i in range(0, len(dry) - win_n, hop_n):
            ia = _fft_amp_at(dry[i:i + win_n] * hann, sr, f0)
            if ia < 1e-4:
                continue
            i_a.append(ia)
            t_a.append(_fft_amp_at(target[i:i + win_n] * hann, sr, f0))
            p_a.append(_fft_amp_at(predicted[i:i + win_n] * hann, sr, f0))

        if len(i_a) >= 6:
            i_a = np.array(i_a)
            sort = np.argsort(i_a)
            i_db = 20.0 * np.log10(i_a[sort] + 1e-10)
            t_g  = 20.0 * np.log10(np.array(t_a)[sort] / (i_a[sort] + 1e-10) + 1e-10)
            p_g  = 20.0 * np.log10(np.array(p_a)[sort] / (i_a[sort] + 1e-10) + 1e-10)
            ax.plot(i_db, medfilt(t_g, 7), color=C_TARGET, lw=1.8,
                    label="Target (from arrays)")
            if model_fn is None:
                ax.plot(i_db, medfilt(p_g, 7), color=C_PRED, lw=1.8, ls="--",
                        label="Predicted (from arrays)")

        # ── Synthesised sines through model_fn (more accurate) ─────────────
        if model_fn is not None:
            dur = max(0.05, 5 / f0)       # at least 5 full cycles
            m_gains = []
            for amp in amps:
                x = _synth_sine(f0, sr, dur, amp)
                y = np.atleast_1d(model_fn(x)).astype(np.float32)
                hann_y = np.hanning(len(y))
                g = _fft_amp_at(y * hann_y, sr, f0) / (amp + 1e-10)
                m_gains.append(g)
            ax.plot(level_dbs, 20.0 * np.log10(np.array(m_gains) + 1e-10),
                    color=C_PRED, lw=2.0, label="Predicted (model_fn)")

        ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.3, label="0 dB (linear)")
        ax.set_xlabel("Input level (dBFS)")
        ax.set_ylabel("Effective gain (dB)")
        ax.set_title("Describing Function  (440 Hz gain compression)")
        ax.legend(fontsize=9)
        ax.set_xlim(-42, 2)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 2. MEMORY / DYNAMICS ─────────────────────────────────────────────────────

def plot_lag_error(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    max_lag: int = 50,
    save_path=None,
) -> plt.Figure:
    """2D delay-coordinate error map: which (x[n], x[n-k]) regions fail most.

    For a selection of lags k, plots the input phase-space plane coloured by
    mean absolute error |target[n] - predicted[n]| per 50×50 bin. A model
    with no memory fails in specific delay-coordinate regions; a purely static
    model fails in all off-diagonal bins equally.

    Look for: bright (high-error) stripes that rotate with lag → indicates
    the model has the right nonlinearity but wrong memory depth.

    Args:
        dry: Dry input signal.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz (unused; kept for consistency).
        max_lag: Upper lag limit in samples. Displayed lags are a subset.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    show_lags = sorted({0, 1, 2, 5, 10, 25, max_lag} & set(range(max_lag + 1)))
    n_lags = len(show_lags)
    n_cols = min(4, n_lags)
    n_rows = (n_lags + n_cols - 1) // n_cols

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.2 * n_cols, 4.0 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        err = np.abs(
            target[:min(len(target), len(predicted))]
            - predicted[:min(len(target), len(predicted))]
        )

        for ax_i, k in enumerate(show_lags):
            x_now     = dry[k:].astype(np.float64)
            x_delayed = dry[:len(dry) - k].astype(np.float64) if k > 0 else dry.astype(np.float64)
            e = err[k:] if k > 0 else err
            n = min(len(x_now), len(x_delayed), len(e))

            bins = 50
            h,   xe, ye = np.histogram2d(x_now[:n], x_delayed[:n],
                                          bins=bins, range=[[-1, 1], [-1, 1]])
            herr, _,  _  = np.histogram2d(x_now[:n], x_delayed[:n],
                                           bins=bins, range=[[-1, 1], [-1, 1]],
                                           weights=e[:n])
            with np.errstate(invalid="ignore"):
                mean_err = np.where(h > 0, herr / h, np.nan)

            ax = axes[ax_i]
            vmax = float(np.nanpercentile(mean_err, 95)) or 0.01
            im = ax.imshow(
                mean_err.T, origin="lower", extent=[-1, 1, -1, 1],
                aspect="auto", cmap="hot", vmin=0, vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label="Mean |error|", fraction=0.046, pad=0.04)
            ax.set_xlabel("x[n]")
            ax.set_ylabel(f"x[n-{k}]")
            ax.set_title(f"Lag k = {k}")

        for j in range(n_lags, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Delay-Coordinate Error Map", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_impulse_response(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """IR via sweep deconvolution: first 50 ms for target and predicted.

    Computes IR = IFFT(Wiener-deconvolution(wet, dry)) for both target and
    predicted. Also plots the difference IR. A perfect model has a difference
    IR of zero.

    Reliable only when the dry signal has broadband content (log sweep, noise).
    On a tonal signal the deconvolution is ill-conditioned.

    Look for: IR length mismatch, pre-ringing (non-causal artefacts),
    and whether the model's tail energy matches the pedal's.

    Args:
        dry: Dry input; should be broadband (sweep or noise) for best results.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(dry), len(target), len(predicted))
    ir_t = _wiener_ir(dry[:n], target[:n])
    ir_p = _wiener_ir(dry[:n], predicted[:n])
    n_show = int(sr * 0.05)  # first 50 ms
    t_ms = np.arange(n_show) / sr * 1000.0

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

        for ax, ir, color, label in [
            (axes[0], ir_t, C_TARGET, "Target IR"),
            (axes[1], ir_p, C_PRED,   "Predicted IR"),
        ]:
            ax.plot(t_ms, ir[:n_show], color=color, lw=0.9)
            ax.axhline(0, color="white", lw=0.4, alpha=0.3)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")
            ax.set_title(label)

        axes[2].plot(t_ms, ir_t[:n_show] - ir_p[:n_show], color=C_ERR, lw=0.9)
        axes[2].axhline(0, color="white", lw=0.4, alpha=0.3)
        axes[2].set_xlabel("Time (ms)")
        axes[2].set_title("IR difference  (target − predicted)")

        fig.suptitle("Impulse Response via Wiener Deconvolution  (first 50 ms)",
                     fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_step_response(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Average step response across all detected hard transitions.

    Finds large sample-to-sample transitions in the dry signal, extracts a
    ±5 ms window around each, and averages them. Capacitor charge/discharge
    timescales appear as slow settling tails. A model without dynamic state
    will settle instantly.

    Look for: settling time difference between target and predicted, and
    asymmetry between rising and falling edges.

    Args:
        dry: Dry input signal.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(dry), len(target), len(predicted))
    dry, target, predicted = dry[:n], target[:n], predicted[:n]

    delta = np.abs(np.diff(dry.astype(np.float64)))
    threshold = np.percentile(delta[delta > 0], 97)
    raw_idx = np.where(delta > threshold)[0]

    half = int(sr * 0.005)  # 5 ms
    win = 2 * half
    # De-duplicate: keep only one step per window-length stretch
    steps: list[int] = []
    for si in raw_idx:
        if not steps or si - steps[-1] > win:
            steps.append(int(si))

    t_wins, p_wins = [], []
    for si in steps:
        s = si - half
        e = si + half
        if s >= 0 and e <= n:
            t_wins.append(target[s:e])
            p_wins.append(predicted[s:e])

    t_ms = (np.arange(win) - half) / sr * 1000.0

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        if len(t_wins) >= 2:
            t_mean = np.mean(t_wins, axis=0)
            p_mean = np.mean(p_wins, axis=0)
            axes[0].plot(t_ms, t_mean, color=C_TARGET, lw=1.8, label="Target")
            axes[0].plot(t_ms, p_mean, color=C_PRED,   lw=1.8, ls="--", label="Predicted")
            axes[1].plot(t_ms, t_mean - p_mean, color=C_ERR, lw=1.4)
            axes[1].set_title(f"Step error  (n = {len(t_wins)} averaged)")
        else:
            axes[0].text(0.5, 0.5, "Not enough transitions detected",
                         ha="center", va="center", transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, "n/a",
                         ha="center", va="center", transform=axes[1].transAxes)

        axes[0].axvline(0, color="white", lw=0.5, ls="--", alpha=0.5)
        axes[0].set_xlabel("Time relative to step (ms)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Mean step response")
        axes[0].legend(fontsize=9)

        axes[1].axhline(0, color="white", lw=0.4, alpha=0.3)
        axes[1].axvline(0, color="white", lw=0.5, ls="--", alpha=0.5)
        axes[1].set_xlabel("Time relative to step (ms)")
        axes[1].set_ylabel("Error amplitude")

        fig.suptitle("Step Response  (averaged over all detected transitions)",
                     fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 3. FREQUENCY DOMAIN ───────────────────────────────────────────────────────

def plot_transfer_function(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Transfer function magnitude (dB) and phase (degrees), log frequency axis.

    Computed via Wiener deconvolution from dry/wet pairs. Phase is unwrapped
    before display. Requires broadband dry content for reliable estimates.

    Look for: magnitude deviation > 1-2 dB or phase deviation > 10° in the
    guitar operating range (80–8 kHz) — audible even at low levels.

    Args:
        dry: Dry input; broadband signal gives best results.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(dry), len(target), len(predicted))
    H_t = _wiener_tf(dry[:n], target[:n])
    H_p = _wiener_tf(dry[:n], predicted[:n])
    freqs = rfftfreq(n, 1.0 / sr)

    mag_t = 20.0 * np.log10(np.abs(H_t) + 1e-12)
    mag_p = 20.0 * np.log10(np.abs(H_p) + 1e-12)
    ph_t  = np.degrees(np.unwrap(np.angle(H_t)))
    ph_p  = np.degrees(np.unwrap(np.angle(H_p)))

    with plt.style.context(_STYLE):
        fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        mask = (freqs >= 20) & (freqs <= 20_000)
        ax_m.plot(freqs[mask], mag_t[mask], color=C_TARGET, lw=1.2, label="Target")
        ax_m.plot(freqs[mask], mag_p[mask], color=C_PRED,   lw=1.0, ls="--", label="Predicted")
        ax_m.set_ylabel("Magnitude (dB)")
        ax_m.set_title("Transfer Function — Magnitude")
        ax_m.legend(fontsize=9)
        ax_m.set_ylim(-60, 20)
        _log_freq_axis(ax_m, 20, 20_000)

        ax_p.plot(freqs[mask], ph_t[mask], color=C_TARGET, lw=1.2, label="Target")
        ax_p.plot(freqs[mask], ph_p[mask], color=C_PRED,   lw=1.0, ls="--", label="Predicted")
        ax_p.set_ylabel("Phase (degrees)")
        ax_p.set_title("Transfer Function — Phase")
        ax_p.legend(fontsize=9)
        _log_freq_axis(ax_p, 20, 20_000)

        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_group_delay(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Group delay (ms) vs log frequency with 1/3-octave smoothing.

    Group delay = -d(phase)/dω in seconds, converted to milliseconds.
    Smoothed over a 1/3-octave window to suppress numerical noise from the
    deconvolution. A flat group delay means all frequencies are delayed equally.

    Look for: group delay peaks at low frequencies (capacitor coupling),
    and whether the model reproduces the pedal's delay profile.

    Args:
        dry: Dry input.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(dry), len(target), len(predicted))
    H_t = _wiener_tf(dry[:n], target[:n])
    H_p = _wiener_tf(dry[:n], predicted[:n])
    freqs = rfftfreq(n, 1.0 / sr)

    dw = 2.0 * np.pi * (freqs[1] - freqs[0])
    gd_t = -np.diff(np.unwrap(np.angle(H_t))) / dw * 1000.0  # ms
    gd_p = -np.diff(np.unwrap(np.angle(H_p))) / dw * 1000.0
    f_mid = 0.5 * (freqs[:-1] + freqs[1:])

    mask = (f_mid >= 30) & (f_mid <= 18_000)
    gd_t_sm = _smooth_log_octave(f_mid[mask], gd_t[mask])
    gd_p_sm = _smooth_log_octave(f_mid[mask], gd_p[mask])

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(f_mid[mask], gd_t_sm, color=C_TARGET, lw=1.5, label="Target")
        ax.plot(f_mid[mask], gd_p_sm, color=C_PRED,   lw=1.5, ls="--", label="Predicted")
        ax.set_ylabel("Group delay (ms)")
        ax.set_title("Group Delay  (1/3-octave smoothed)")
        ax.legend(fontsize=9)
        _log_freq_axis(ax, 30, 18_000)
        # Clip to ±50 ms for readability
        ylo, yhi = np.percentile(np.concatenate([gd_t_sm, gd_p_sm]), [2, 98])
        ax.set_ylim(ylo - 1, yhi + 1)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_coherence(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Magnitude-squared coherence γ²(f) for (dry→target) and (dry→predicted).

    γ²=1 means the output is fully predictable from the input at that
    frequency; γ²<1 indicates noise or nonlinearity. Divergence between the
    two curves shows where the model's nonlinearity differs from the pedal's.

    Uses scipy.signal.coherence with nperseg=4096.

    Look for: regions where target coherence is high but predicted coherence
    is low (model is missing a deterministic component of the pedal's output).

    Args:
        dry: Dry input signal.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(dry), len(target), len(predicted))
    d, t, p = dry[:n], target[:n], predicted[:n]

    f_t, Cxy_t = _coherence(d, t, fs=sr, nperseg=4096)
    f_p, Cxy_p = _coherence(d, p, fs=sr, nperseg=4096)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        mask = (f_t >= 20) & (f_t <= 20_000)
        ax.plot(f_t[mask], Cxy_t[mask], color=C_TARGET, lw=1.4, label="Target coherence")
        ax.plot(f_p[mask], Cxy_p[mask], color=C_PRED,   lw=1.4, ls="--",
                label="Predicted coherence")
        ax.axhline(0.9, color="white", lw=0.5, ls=":", alpha=0.4, label="γ² = 0.9 ref")
        ax.set_ylabel("γ²  (magnitude-squared coherence)")
        ax.set_title("Coherence  dry→target  vs  dry→predicted")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        _log_freq_axis(ax, 20, 20_000)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_spectrogram_overlay(
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Side-by-side mel spectrograms: target, predicted, and difference.

    Three panels: target mel spectrogram, predicted mel spectrogram, and
    difference (target - predicted) with a diverging colormap centred at zero.
    Uses 128 mel bins, 50–8000 Hz, Hann window.

    Look for: red/blue regions in the difference panel → systematic over/
    under-prediction of energy at specific time-frequency locations.

    Args:
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    def _mels(sig: np.ndarray):
        S = librosa.feature.melspectrogram(
            y=sig.astype(np.float32), sr=sr, n_mels=128,
            hop_length=512, fmin=50, fmax=8_000,
        )
        return librosa.power_to_db(S, ref=np.max)

    n = min(len(target), len(predicted))
    S_t = _mels(target[:n])
    S_p = _mels(predicted[:n])
    S_d = S_t - S_p

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, S, cmap, label, clim in [
            (axes[0], S_t, "magma",  "Target",         (None, None)),
            (axes[1], S_p, "magma",  "Predicted",       (None, None)),
            (axes[2], S_d, "RdBu_r", "Difference (dB)", None),
        ]:
            kw = dict(y_axis="mel", x_axis="time", ax=ax,
                      fmin=50, fmax=8_000, sr=sr, hop_length=512, cmap=cmap)
            if clim is not None:
                kw["vmin"], kw["vmax"] = clim
            elif label.startswith("Diff"):
                v = float(np.percentile(np.abs(S_d), 98))
                kw["vmin"], kw["vmax"] = -v, v
            img = librosa.display.specshow(S, **kw)
            fig.colorbar(img, ax=ax, format="%+2.0f dB", fraction=0.046, pad=0.04)
            ax.set_title(label)

        fig.suptitle("Mel Spectrogram Comparison  (128 mel bins, 50–8 kHz)",
                     fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 4. NONLINEAR-SPECIFIC ─────────────────────────────────────────────────────

def plot_harmonic_profile(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    fundamentals: list[float] | None = None,
    drive_levels: list[float] | None = None,
    model_fns: dict[str, Callable] | None = None,
    save_path=None,
) -> plt.Figure:
    """Grouped bar chart: harmonics 2–8 normalised to harmonic 1.

    For each (fundamental, drive_level) pair, synthesises a test sine and
    runs it through the model (or estimates from pre-computed arrays). Plots
    harmonics 2–8 relative to harmonic 1, target vs predicted side by side.

    With model_fns = {"target": fn_t, "predicted": fn_p}: synthesises test
    tones, runs through callables — accurate for any drive level.
    Without model_fns: uses the pre-computed arrays, finding windows where
    the fundamental is dominant at approximately the target level. Falls back
    to the full signal if no matching windows exist.

    Look for: model producing too few odd harmonics (sounds clean), wrong
    harmonic ratios (wrong clipping curve), or even harmonics where the
    pedal produces only odd ones (asymmetry mismatch).

    Args:
        dry: Dry input signal.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        fundamentals: Fundamental frequencies to test (Hz).
            Default: [110, 220, 440, 880].
        drive_levels: Input levels in dBFS. Default: [-20, -12, -6, -3].
        model_fns: Optional dict {"target": callable, "predicted": callable}.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    if fundamentals is None:
        fundamentals = [110, 220, 440, 880]
    if drive_levels is None:
        drive_levels = [-20, -12, -6, -3]

    n_f = len(fundamentals)
    n_d = len(drive_levels)
    n_harm = 8  # harmonics 1..8; bars are 2..8

    # Build (f0, level) → harmonics[0..7] for target and predicted
    harms_t: dict[tuple, np.ndarray] = {}
    harms_p: dict[tuple, np.ndarray] = {}

    for f0 in fundamentals:
        for ldb in drive_levels:
            amp = 10.0 ** (ldb / 20.0)
            key = (f0, ldb)

            if model_fns is not None:
                dur = max(0.1, 10 / f0)
                x = _synth_sine(f0, sr, dur, amp)
                harms_t[key] = _harmonic_amps(model_fns["target"](x).astype(np.float32),
                                               sr, f0, n_harm)
                harms_p[key] = _harmonic_amps(model_fns["predicted"](x).astype(np.float32),
                                               sr, f0, n_harm)
            else:
                # Find windows in dry near (f0, level) via bandpass energy
                sos = _bandpass_sos(f0, sr)
                dry_bp = sosfilt(sos, dry.astype(np.float64))
                win_n  = min(int(sr * 0.05), len(dry) // 8)
                wins_t, wins_p = [], []
                for i in range(0, len(dry_bp) - win_n, win_n):
                    rms = float(np.sqrt(np.mean(dry_bp[i:i + win_n] ** 2)))
                    if rms < 1e-6:
                        continue
                    if abs(20.0 * np.log10(rms) - ldb) < 4.0:
                        wins_t.append(target[i:i + win_n])
                        wins_p.append(predicted[i:i + win_n])
                if wins_t:
                    t_buf = np.concatenate(wins_t)
                    p_buf = np.concatenate(wins_p)
                else:
                    t_buf, p_buf = target, predicted
                harms_t[key] = _harmonic_amps(t_buf, sr, f0, n_harm)
                harms_p[key] = _harmonic_amps(p_buf, sr, f0, n_harm)

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(n_f, n_d, figsize=(3.8 * n_d, 3.5 * n_f),
                                  sharex=True, sharey=False)
        axes = np.atleast_2d(axes)

        x_bars = np.arange(n_harm - 1)  # harmonics 2..n_harm
        bar_w  = 0.38

        for fi, f0 in enumerate(fundamentals):
            for di, ldb in enumerate(drive_levels):
                ax  = axes[fi, di]
                key = (f0, ldb)
                ht  = harms_t[key]
                hp  = harms_p[key]
                # Normalise to harmonic 1
                ht_norm = ht[1:] / (ht[0] + 1e-10)
                hp_norm = hp[1:] / (hp[0] + 1e-10)
                ax.bar(x_bars - bar_w / 2, ht_norm, bar_w,
                       color=C_TARGET, alpha=0.85, label="Target")
                ax.bar(x_bars + bar_w / 2, hp_norm, bar_w,
                       color=C_PRED,   alpha=0.85, label="Predicted")
                ax.set_xticks(x_bars)
                ax.set_xticklabels([f"H{k}" for k in range(2, n_harm + 1)], fontsize=7)
                ax.set_title(f"{int(f0)} Hz  /  {ldb} dBFS", fontsize=9)
                if di == 0:
                    ax.set_ylabel("Hk / H1", fontsize=8)
                if fi == 0 and di == 0:
                    ax.legend(fontsize=8)

        fig.suptitle("Harmonic Profile  (H2–H8 relative to H1)", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_imd(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    pairs: list[tuple[float, float]] | None = None,
    model_fns: dict[str, Callable] | None = None,
    save_path=None,
) -> plt.Figure:
    """Output spectrum for two-tone excitations with IMD product markers.

    For each (f1, f2) pair, runs the signal through target and predicted
    (or finds matching windows in pre-computed arrays), plots the output
    spectrum from DC to 4·max(f1, f2), and marks IMD products with dashed
    lines: f2−f1, 2f1−f2, 2f2−f1, f1+f2, 2f1+f2, 2f2+f1.

    Look for: IMD product levels that differ between target and predicted —
    this is a key audible quality indicator for fuzz/overdrive.

    Args:
        dry: Dry input signal.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        pairs: List of (f1, f2) pairs. Default: [(220, 330), (440, 550), (880, 1100)].
        model_fns: Optional dict {"target": callable, "predicted": callable}.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    if pairs is None:
        pairs = [(220, 330), (440, 550), (880, 1100)]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(len(pairs), 2,
                                  figsize=(14, 3.8 * len(pairs)))
        if len(pairs) == 1:
            axes = axes[np.newaxis, :]

        for pi, (f1, f2) in enumerate(pairs):
            imd_freqs = {
                "f2−f1":   abs(f2 - f1),
                "2f1−f2":  abs(2 * f1 - f2),
                "2f2−f1":  abs(2 * f2 - f1),
                "f1+f2":   f1 + f2,
                "2f1+f2":  2 * f1 + f2,
                "2f2+f1":  2 * f2 + f1,
            }
            fmax_plot = 4 * max(f1, f2)
            amp = 0.35  # per tone; combined peak ≈ 0.70

            if model_fns is not None:
                dur = 1.0
                x  = _synth_sine(f1, sr, dur, amp) + _synth_sine(f2, sr, dur, amp)
                sigs = {
                    "Target":    (model_fns["target"](x.copy()).astype(np.float32),    C_TARGET),
                    "Predicted": (model_fns["predicted"](x.copy()).astype(np.float32), C_PRED),
                }
            else:
                # Use full pre-computed arrays as a proxy
                sigs = {
                    "Target":    (target,    C_TARGET),
                    "Predicted": (predicted, C_PRED),
                }

            for col, (label, (sig, color)) in enumerate(sigs.items()):
                ax = axes[pi, col]
                win = sig * np.hanning(len(sig))
                X   = np.abs(rfft(win.astype(np.float64)))
                freqs = rfftfreq(len(sig), 1.0 / sr)
                X_db  = 20.0 * np.log10(X / (len(sig) / 2.0) + 1e-10)

                mask = freqs <= fmax_plot
                ax.plot(freqs[mask], X_db[mask], color=color, lw=0.8)
                ax.axvline(f1, color=C_DRY, lw=0.8, ls=":", alpha=0.6)
                ax.axvline(f2, color=C_DRY, lw=0.8, ls=":", alpha=0.6)

                for imd_label, imd_f in imd_freqs.items():
                    if 0 < imd_f <= fmax_plot:
                        ax.axvline(imd_f, color=C_ERR, lw=0.7, ls="--", alpha=0.7)
                        ax.text(imd_f, ax.get_ylim()[1] if ax.get_ylim()[1] > -10 else -5,
                                imd_label, color=C_ERR, fontsize=6, rotation=90,
                                va="top", ha="right")

                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude (dBFS)")
                ax.set_title(f"{label}  —  {int(f1)}+{int(f2)} Hz")
                ax.set_xlim(0, fmax_plot)
                ax.set_ylim(-90, 5)

        fig.suptitle("Intermodulation Distortion  (dashed = expected IMD products)",
                     fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_level_dependent_fr(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    levels: list[float] | None = None,
    save_path=None,
) -> plt.Figure:
    """Transfer function magnitude at multiple input levels.

    Groups signal windows by their RMS level and computes a separate transfer
    function for each group. A linear system has all curves coinciding; a fuzz
    pedal fans out dramatically — high-level inputs are clipped, changing the
    effective frequency response.

    Colormap: blue (quiet) → red (loud).

    Look for: fan-out at high levels (nonlinear), spectral tilt changes,
    and whether the model's level-dependent curves match the target's.

    Args:
        dry: Dry input signal.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        levels: Input level bins in dBFS. Default: [-40, -20, -12, -6, -3, 0].
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    if levels is None:
        levels = [-40, -20, -12, -6, -3, 0]

    n = min(len(dry), len(target), len(predicted))
    win_n = min(int(sr * 0.5), n // 8)  # 0.5 s windows
    hop_n = win_n // 2

    # Bucket windows by RMS level (±4 dB tolerance around each level bin)
    buckets_t: dict[float, list[np.ndarray]] = {ldb: [] for ldb in levels}
    buckets_p: dict[float, list[np.ndarray]] = {ldb: [] for ldb in levels}

    for i in range(0, n - win_n, hop_n):
        rms = float(np.sqrt(np.mean(dry[i:i + win_n] ** 2)))
        if rms < 1e-6:
            continue
        ldb_win = 20.0 * np.log10(rms)
        best = min(levels, key=lambda l: abs(l - ldb_win))
        if abs(best - ldb_win) <= 4.0:
            buckets_t[best].append(target[i:i + win_n].astype(np.float64))
            buckets_p[best].append(dry[i:i + win_n].astype(np.float64))

    cmap = plt.cm.coolwarm  # blue=quiet, red=loud

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        for ax, buckets, title in [
            (axes[0], buckets_t, "Target"),
            (axes[1], buckets_p, "Predicted"),
        ]:
            for i_ldb, ldb in enumerate(levels):
                if not buckets[ldb]:
                    continue
                dry_cat  = np.concatenate([dry[j * hop_n:j * hop_n + win_n]
                                            for j in range(len(buckets[ldb]))])
                wet_cat  = np.concatenate(buckets[ldb])
                nc = min(len(dry_cat), len(wet_cat))
                H = _wiener_tf(dry_cat[:nc], wet_cat[:nc])
                freqs = rfftfreq(nc, 1.0 / sr)
                mag = 20.0 * np.log10(np.abs(H) + 1e-12)
                mask = (freqs >= 30) & (freqs <= 18_000)
                color = cmap(i_ldb / max(len(levels) - 1, 1))
                ax.plot(freqs[mask], mag[mask],
                        color=color, lw=1.2, alpha=0.85, label=f"{ldb} dBFS")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title(f"Level-dependent FR — {title}")
            ax.set_ylim(-30, 20)
            ax.legend(fontsize=8, loc="upper right")
            _log_freq_axis(ax, 30, 18_000)

        fig.suptitle("Level-Dependent Frequency Response", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_nonlinear_frequency_map(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    n_tones: int = 16,
    model_fns: dict[str, Callable] | None = None,
    save_path=None,
) -> plt.Figure:
    """Energy routing heatmap: input tone → output frequency.

    Runs each of n_tones logarithmically-spaced input tones individually,
    computes the output power spectrum, and stacks into a 2-D matrix.
    On the heatmap: X = output frequency (log), Y = input frequency (log).
    A linear system has energy only on the diagonal; fuzz spawns harmonic
    columns for each input tone.

    With model_fns = {"target": fn, "predicted": fn}: synthesises sines and
    runs through callables — exact.
    Without model_fns: uses bandpass isolation on pre-computed arrays —
    approximate but useful when capture data contains broadband content.

    Args:
        dry: Dry input signal.
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        n_tones: Number of logarithmically-spaced input tones (80–8000 Hz).
        model_fns: Optional {"target": callable, "predicted": callable}.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    f_lo, f_hi = 80.0, 8_000.0
    tones = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), n_tones))
    amp   = 0.5
    dur   = 0.5       # 500 ms per tone is enough

    n_out = 256
    out_freqs = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), n_out))

    mat_t = np.zeros((n_tones, n_out))
    mat_p = np.zeros((n_tones, n_out))

    for i, f_in in enumerate(tones):
        if model_fns is not None:
            x = _synth_sine(f_in, sr, dur, amp)
            sig_t = model_fns["target"](x.copy()).astype(np.float32)
            sig_p = model_fns["predicted"](x.copy()).astype(np.float32)
            for j, (sig, mat) in enumerate([(sig_t, mat_t), (sig_p, mat_p)]):
                n_sig = len(sig)
                win   = sig * np.hanning(n_sig)
                ps    = np.abs(rfft(win.astype(np.float64))) ** 2
                freqs = rfftfreq(n_sig, 1.0 / sr)
                # Bin into n_out log-spaced output frequency bins
                for bi in range(n_out):
                    fl = out_freqs[bi] * (0.85 if bi == 0 else out_freqs[bi - 1] / out_freqs[bi] ** 0.5)
                    fh = out_freqs[bi] * (1.15 if bi == n_out - 1 else out_freqs[bi + 1] / out_freqs[bi] ** 0.5)
                    m  = (freqs >= fl) & (freqs < fh)
                    mat[i, bi] = float(np.sum(ps[m])) if m.any() else 0.0
        else:
            # Bandpass-isolate the input tone in the pre-computed dry signal
            sos      = _bandpass_sos(f_in, sr, width=0.15)
            dry_bp   = sosfilt(sos, dry.astype(np.float64))
            energy   = dry_bp ** 2
            smooth_n = max(1, int(sr * 0.05))
            energy_s = np.convolve(energy, np.ones(smooth_n) / smooth_n, mode="same")
            thr      = np.percentile(energy_s, 85) if energy_s.max() > 0 else 1.0
            active   = np.where(energy_s > thr)[0]
            if active.size < 512:
                continue
            fft_n = 4096
            hann  = np.hanning(fft_n)
            ps_t_acc = np.zeros(fft_n // 2 + 1)
            ps_p_acc = np.zeros(fft_n // 2 + 1)
            count = 0
            for j in range(0, len(active) - fft_n, fft_n):
                s = active[j]
                if s + fft_n > len(target):
                    break
                ps_t_acc += np.abs(rfft(target[s:s + fft_n] * hann)) ** 2
                ps_p_acc += np.abs(rfft(predicted[s:s + fft_n] * hann)) ** 2
                count += 1
            if count == 0:
                continue
            freqs = rfftfreq(fft_n, 1.0 / sr)
            for mat, ps_acc in [(mat_t, ps_t_acc), (mat_p, ps_p_acc)]:
                for bi in range(n_out):
                    fl = out_freqs[bi] * 0.85
                    fh = out_freqs[bi] * 1.15
                    m  = (freqs >= fl) & (freqs < fh)
                    mat[i, bi] = float(np.sum(ps_acc[m]) / count) if m.any() else 0.0

    # Convert to dB
    mat_t_db = 10.0 * np.log10(mat_t + 1e-20)
    mat_p_db = 10.0 * np.log10(mat_p + 1e-20)

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        tick_pos  = np.linspace(0, n_out - 1, 5).astype(int)
        tick_labs = [f"{int(out_freqs[k] / 1000)}k" if out_freqs[k] >= 1000
                     else str(int(out_freqs[k])) for k in tick_pos]

        for ax, mat_db, title in [
            (axes[0], mat_t_db, "Target"),
            (axes[1], mat_p_db, "Predicted"),
        ]:
            im = ax.imshow(mat_db, origin="lower", aspect="auto",
                           cmap="inferno", vmin=-80, vmax=0)
            fig.colorbar(im, ax=ax, label="Power (dB)", fraction=0.046, pad=0.04)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labs)
            ax.set_yticks(np.linspace(0, n_tones - 1, 6).astype(int))
            ax.set_yticklabels([f"{int(tones[k])} Hz"
                                 for k in np.linspace(0, n_tones - 1, 6).astype(int)])
            ax.set_xlabel("Output frequency")
            ax.set_ylabel("Input tone")
            ax.set_title(f"Nonlinear Frequency Map — {title}")

        fig.suptitle("Energy routing: input tone → output spectrum\n"
                     "(off-diagonal = nonlinear frequency generation)", fontsize=12)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 5. PERCEPTUAL / RESIDUAL ──────────────────────────────────────────────────

def plot_error_spectrogram(
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Spectrogram of the error signal with time-averaged error power spectrum.

    Left panel: mel spectrogram of error = target − predicted (log magnitude).
    Right panel: time-averaged error power spectrum on a log frequency axis.
    Shows *when* and *at which frequencies* the model fails.

    Look for: persistent bright regions at specific frequencies (systematic
    spectral bias), or time-localised bursts (transient response failure).

    Args:
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(target), len(predicted))
    error = (target[:n] - predicted[:n]).astype(np.float32)

    S_err = librosa.feature.melspectrogram(
        y=error, sr=sr, n_mels=128, hop_length=512, fmin=50, fmax=8_000,
    )
    S_db = librosa.power_to_db(S_err, ref=np.max)

    # Time-averaged error power spectrum (using full rfft)
    X_err = np.abs(rfft(error.astype(np.float64))) ** 2
    freqs = rfftfreq(n, 1.0 / sr)
    X_db  = 10.0 * np.log10(X_err / (n / 2.0) ** 2 + 1e-20)

    with plt.style.context(_STYLE):
        fig = plt.figure(figsize=(15, 5))
        gs  = fig.add_gridspec(1, 3, width_ratios=[2, 1, 0.05])
        ax_sg = fig.add_subplot(gs[0])
        ax_sp = fig.add_subplot(gs[1])

        img = librosa.display.specshow(
            S_db, y_axis="mel", x_axis="time", ax=ax_sg,
            sr=sr, hop_length=512, fmin=50, fmax=8_000,
            cmap="hot",
        )
        ax_sg.set_title("Error spectrogram  (target − predicted)")
        fig.colorbar(img, ax=ax_sg, format="%+2.0f dB", fraction=0.046, pad=0.04)

        mask = (freqs >= 30) & (freqs <= 18_000)
        ax_sp.plot(X_db[mask], freqs[mask], color=C_ERR, lw=0.9)
        ax_sp.set_ylabel("Frequency (Hz)")
        ax_sp.set_xlabel("Power (dBFS)")
        ax_sp.set_yscale("log")
        yticks = [t for t in _LOG_TICKS if 30 <= t <= 18_000]
        ax_sp.set_yticks(yticks)
        ax_sp.set_yticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in yticks])
        ax_sp.set_ylim(30, 18_000)
        ax_sp.set_title("Time-averaged error spectrum")

        fig.suptitle("Error Spectrogram Analysis", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_aweighted_error(
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Error spectrum with A-weighting — shows where audible errors concentrate.

    Three panels on a shared log-frequency axis:
      1. Unweighted error spectrum |FFT(target) - FFT(predicted)|² in dB
      2. A-weighted error spectrum (IEC 61672)
      3. A-weighting curve itself for reference

    A-weighting matches our ears' reduced sensitivity to low and very high
    frequencies. A model may have high unweighted error at 60 Hz but low
    A-weighted error — meaning the error is inaudible in practice.

    Args:
        target: Pedal wet output.
        predicted: Model output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(target), len(predicted))
    t = target[:n].astype(np.float64)
    p = predicted[:n].astype(np.float64)

    E  = rfft(t) - rfft(p)
    E2 = np.abs(E) ** 2
    freqs = rfftfreq(n, 1.0 / sr)

    aw  = _aweight_db(freqs)
    aw_lin = 10.0 ** (aw / 10.0)
    E2_aw = E2 * aw_lin

    err_db    = 10.0 * np.log10(E2 / (n / 2.0) ** 2 + 1e-20)
    err_aw_db = 10.0 * np.log10(E2_aw / (n / 2.0) ** 2 + 1e-20)

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        mask = (freqs >= 20) & (freqs <= 20_000)

        axes[0].plot(freqs[mask], err_db[mask], color=C_ERR, lw=0.8)
        axes[0].set_ylabel("Power (dBFS)")
        axes[0].set_title("Unweighted error spectrum  |FFT(target) − FFT(predicted)|²")
        axes[0].set_ylim(-120, 0)

        axes[1].plot(freqs[mask], err_aw_db[mask], color=C_PRED, lw=0.8)
        axes[1].set_ylabel("Power (dBA)")
        axes[1].set_title("A-weighted error spectrum")
        axes[1].set_ylim(-120, 0)

        axes[2].plot(freqs[mask], aw[mask], color=C_DRY, lw=1.2)
        axes[2].axhline(0, color="white", lw=0.4, ls="--", alpha=0.3)
        axes[2].set_ylabel("A-weighting (dB)")
        axes[2].set_title("IEC 61672 A-weighting curve  (0 dB at 1 kHz)")
        _log_freq_axis(axes[2], 20, 20_000)

        fig.suptitle("A-Weighted Error Analysis", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── PEDAL CHARACTERIZATION — dry vs wet analysis ──────────────────────────────
# These functions characterize a physical pedal from dry/wet capture pairs.
# They do not require a model — just dry input and the pedal's wet output.

def plot_waveform_morphology(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    n_panels: int = 5,
    save_path=None,
) -> plt.Figure:
    """Five representative 20 ms snapshots showing how the pedal transforms waveforms.

    Automatically selects: quietest, median, loudest sustained passages
    (by 50 ms RMS), fastest transient (largest sample-to-sample delta in a
    50 ms window), and longest decay tail (200 ms window 100 ms after the
    largest onset).

    Look for: clipping shape (rounded vs square), asymmetry between positive
    and negative halves, transient shaping, sustain/compression.

    Args:
        dry: Dry input, shape (N,), float32.
        wet: Pedal wet output, same shape.
        sr: Sample rate in Hz.
        n_panels: Number of panels (1–5, more are ignored).
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    win_50ms = int(sr * 0.05)
    snap_n   = int(sr * 0.02)  # 20 ms display window
    hop      = win_50ms // 4
    n        = min(len(dry), len(wet))

    # Compute RMS and delta for each 50ms window
    rms_list, delta_list, win_starts = [], [], []
    for i in range(0, n - win_50ms, hop):
        block = dry[i:i + win_50ms]
        rms_list.append(float(np.sqrt(np.mean(block ** 2))))
        delta_list.append(float(np.max(np.abs(np.diff(block)))))
        win_starts.append(i)

    rms_arr   = np.array(rms_list)
    delta_arr = np.array(delta_list)
    starts    = np.array(win_starts)

    p10_thr = np.percentile(rms_arr[rms_arr > 1e-5], 10)
    p50_thr = np.percentile(rms_arr[rms_arr > 1e-5], 50)
    p90_thr = np.percentile(rms_arr[rms_arr > 1e-5], 90)

    def _pick(mask: np.ndarray) -> int:
        valid = starts[mask]
        return int(valid[len(valid) // 2]) if len(valid) else int(starts[len(starts) // 2])

    # Quiet, median, loud
    s_quiet  = _pick(rms_arr <= p10_thr)
    s_median = _pick(np.abs(rms_arr - p50_thr) < np.std(rms_arr) * 0.3)
    s_loud   = _pick(rms_arr >= p90_thr)

    # Fastest transient
    s_trans  = int(starts[np.argmax(delta_arr)])

    # Decay tail: 100 ms after the loudest onset
    onset_candidates = starts[rms_arr >= p90_thr]
    if len(onset_candidates):
        s_decay = int(onset_candidates[0]) + int(sr * 0.1)
        s_decay = min(s_decay, n - int(sr * 0.2) - 1)
    else:
        s_decay = int(n * 0.7)

    labels_starts = [
        ("Quiet (p10)",    s_quiet),
        ("Median (p50)",   s_median),
        ("Loud (p90)",     s_loud),
        ("Fastest trans.", s_trans),
        ("Decay tail",     s_decay),
    ][:n_panels]

    t_ms = np.arange(snap_n) / sr * 1000.0

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, len(labels_starts),
                                  figsize=(4.2 * len(labels_starts), 4),
                                  sharey=True)
        axes = np.atleast_1d(axes)

        for ax, (label, start) in zip(axes, labels_starts):
            end = min(start + snap_n, n)
            d = dry[start:end]
            w = wet[start:end]
            t = t_ms[:len(d)]
            rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(dry[start:start + win_50ms] ** 2))) + 1e-10)
            ax.plot(t, d, color=C_DRY,    lw=1.2, label="Dry")
            ax.plot(t, w, color=C_TARGET, lw=1.2, label="Wet")
            ax.axhline(0, color="white", lw=0.3, alpha=0.3)
            ax.set_title(f"{label}\n{rms_db:.1f} dBFS", fontsize=9)
            ax.set_xlabel("ms")
            if ax is axes[0]:
                ax.set_ylabel("Amplitude")
                ax.legend(fontsize=8)

        fig.suptitle("Waveform Morphology — 20 ms snapshots at key moments", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_gain_reduction(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    window_ms: float = 5.0,
    save_path=None,
) -> plt.Figure:
    """RMS envelope and gain reduction over the full signal.

    Three panels (shared time axis):
      1. Dry RMS envelope in dBFS (blue)
      2. Wet RMS envelope in dBFS (green)
      3. Gain reduction = wet_dBFS − dry_dBFS (gold), with 0 dB reference.
         Positive regions (expansion) shaded green; negative (compression) red.

    Look for: dynamic compression at high levels (gain reduction spikes down),
    expansion at low levels (pedal boosts quiet passages), or a flat line
    (volume-unity static clipper like a Big Muff).

    Args:
        dry: Dry input, shape (N,), float32.
        wet: Pedal wet output.
        sr: Sample rate in Hz.
        window_ms: RMS sliding window length in milliseconds.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    win_n = int(sr * window_ms / 1000)
    hop_n = win_n // 4
    n     = min(len(dry), len(wet))

    # Vectorised sliding RMS
    times, rms_d, rms_w = [], [], []
    for i in range(0, n - win_n, hop_n):
        times.append((i + win_n / 2) / sr)
        rms_d.append(float(np.sqrt(np.mean(dry[i:i + win_n] ** 2))))
        rms_w.append(float(np.sqrt(np.mean(wet[i:i + win_n] ** 2))))

    t   = np.array(times)
    d_db = 20.0 * np.log10(np.maximum(rms_d, 1e-10))
    w_db = 20.0 * np.log10(np.maximum(rms_w, 1e-10))
    gr   = w_db - d_db

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)

        axes[0].plot(t, d_db, color=C_DRY,    lw=0.8)
        axes[0].set_ylabel("dBFS")
        axes[0].set_title("Dry RMS envelope")
        axes[0].set_ylim(-70, 5)

        axes[1].plot(t, w_db, color=C_TARGET, lw=0.8)
        axes[1].set_ylabel("dBFS")
        axes[1].set_title("Wet RMS envelope")
        axes[1].set_ylim(-70, 5)

        axes[2].plot(t, gr, color=C_ERR, lw=0.8)
        axes[2].axhline(0, color="white", lw=0.5, ls="--", alpha=0.4)
        axes[2].fill_between(t, 0, gr, where=(gr > 0), color="#00FF88", alpha=0.25, label="Expansion")
        axes[2].fill_between(t, gr, 0, where=(gr < 0), color="#FF6B6B", alpha=0.25, label="Compression")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("dB")
        axes[2].set_title(f"Gain reduction  (wet − dry dBFS,  {window_ms:.0f} ms window)")
        axes[2].legend(fontsize=9)

        fig.suptitle("Dynamic Gain Analysis", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_envelope_comparison(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Instantaneous (Hilbert) and macro (RMS) envelopes for dry and wet.

    Two panels:
      1. Instantaneous envelope via scipy.signal.hilbert for the first 2 s.
      2. Macro RMS envelope (20 ms window) for the full signal.
    Annotated with attack time (10%→90% peak) and release time (90%→10% peak)
    for both dry and wet, measured from the largest transient.

    Look for: fuzz pedals often have faster attack and longer release than the
    dry signal. A slow Hilbert envelope rise = capacitor charging.

    Args:
        dry: Dry input, shape (N,), float32.
        wet: Pedal wet output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    from scipy.signal import hilbert

    n = min(len(dry), len(wet))

    # Instantaneous envelope (first 2 s)
    clip2s  = min(n, int(sr * 2.0))
    env_d_inst = np.abs(hilbert(dry[:clip2s].astype(np.float64))).astype(np.float32)
    env_w_inst = np.abs(hilbert(wet[:clip2s].astype(np.float64))).astype(np.float32)
    t_inst = np.arange(clip2s) / sr

    # Macro RMS envelope (20 ms window)
    win_20ms = int(sr * 0.02)
    hop_20ms = win_20ms // 4
    t_mac, rms_d_mac, rms_w_mac = [], [], []
    for i in range(0, n - win_20ms, hop_20ms):
        t_mac.append((i + win_20ms / 2) / sr)
        rms_d_mac.append(float(np.sqrt(np.mean(dry[i:i + win_20ms] ** 2))))
        rms_w_mac.append(float(np.sqrt(np.mean(wet[i:i + win_20ms] ** 2))))
    t_mac    = np.array(t_mac)
    rms_d_m  = np.array(rms_d_mac)
    rms_w_m  = np.array(rms_w_mac)

    def _attack_release(env: np.ndarray, t: np.ndarray):
        pk_idx = int(np.argmax(env))
        pk     = env[pk_idx]
        # Attack: scan back from peak to find 10%
        lo, hi = 0.1 * pk, 0.9 * pk
        atk_idx = pk_idx
        for j in range(pk_idx, 0, -1):
            if env[j] < lo:
                atk_idx = j
                break
        # Release: scan forward from peak to find 10%
        rel_idx = pk_idx
        for j in range(pk_idx, len(env)):
            if env[j] < lo:
                rel_idx = j
                break
        atk_ms = (t[pk_idx] - t[atk_idx]) * 1000
        rel_ms = (t[rel_idx] - t[pk_idx]) * 1000
        return float(atk_ms), float(rel_ms), pk_idx

    atk_d, rel_d, pk_d = _attack_release(rms_d_m, t_mac)
    atk_w, rel_w, pk_w = _attack_release(rms_w_m, t_mac)

    with plt.style.context(_STYLE):
        fig, (ax_i, ax_m) = plt.subplots(2, 1, figsize=(13, 7))

        ax_i.plot(t_inst, env_d_inst, color=C_DRY,    lw=0.8, label="Dry (Hilbert)")
        ax_i.plot(t_inst, env_w_inst, color=C_TARGET, lw=0.8, label="Wet (Hilbert)")
        ax_i.set_xlabel("Time (s)")
        ax_i.set_ylabel("Amplitude")
        ax_i.set_title("Instantaneous envelope (Hilbert)  — first 2 s")
        ax_i.legend(fontsize=9)

        ax_m.plot(t_mac, rms_d_m, color=C_DRY,    lw=0.9, label="Dry RMS (20 ms)")
        ax_m.plot(t_mac, rms_w_m, color=C_TARGET, lw=0.9, label="Wet RMS (20 ms)")
        ax_m.axvline(t_mac[pk_d], color=C_DRY,    lw=0.7, ls=":", alpha=0.7)
        ax_m.axvline(t_mac[pk_w], color=C_TARGET, lw=0.7, ls=":", alpha=0.7)
        ax_m.set_xlabel("Time (s)")
        ax_m.set_ylabel("Amplitude")
        ax_m.set_title(
            f"Macro RMS envelope  |  "
            f"Dry: atk {atk_d:.0f} ms, rel {rel_d:.0f} ms  "
            f"|  Wet: atk {atk_w:.0f} ms, rel {rel_w:.0f} ms"
        )
        ax_m.legend(fontsize=9)

        fig.suptitle("Envelope Comparison", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_phase_portrait(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    n_points: int = 20_000,
    save_path=None,
) -> plt.Figure:
    """Phase portrait: x[n] vs x[n-1] density plot for dry and wet.

    Uses 200×200 histogram2d with log-scaled density. For a sine wave this
    traces an ellipse; a hard clipper traces a rounded rectangle; a fuzz
    pedal is somewhere between. Comparing dry and wet shows how the pedal
    modifies the phase-space trajectory.

    Look for: wet portrait with flat top/bottom (hard clipping), expanded
    horizontal extent (compression), or irregular density (noise/chaos).

    Args:
        dry: Dry input, shape (N,), float32.
        wet: Pedal wet output.
        sr: Sample rate in Hz (unused; kept for API consistency).
        n_points: Subsampled point count for the histogram.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(dry), len(wet))
    idx = np.round(np.linspace(1, n - 1, min(n_points, n - 1))).astype(int)

    bins = 200
    rng_xy = [[-1.05, 1.05], [-1.05, 1.05]]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, sig, color, label in [
            (axes[0], dry, C_DRY,    "Dry"),
            (axes[1], wet, C_TARGET, "Wet"),
        ]:
            x_now  = sig[idx]
            x_prev = sig[idx - 1]
            H, xe, ye = np.histogram2d(x_now, x_prev, bins=bins, range=rng_xy)
            log_H = np.log1p(H)
            im = ax.imshow(
                log_H.T, origin="lower", extent=[-1.05, 1.05, -1.05, 1.05],
                aspect="equal", cmap="inferno",
                vmin=0, vmax=log_H.max(),
            )
            fig.colorbar(im, ax=ax, label="log(1 + count)", fraction=0.046, pad=0.04)
            ax.set_xlabel("x[n]")
            ax.set_ylabel("x[n-1]")
            ax.set_title(f"Phase portrait — {label}")

        fig.suptitle("Phase Portrait  (log density, 200×200 bins)", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_odd_even_harmonic_ratio(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    fundamentals: list[float] | None = None,
    levels_dbfs: list[float] | None = None,
    wet_fn: Callable | None = None,
    save_path=None,
) -> plt.Figure:
    """Odd/even harmonic ratio heatmap across frequency and drive level.

    For each (fundamental, level) pair measures harmonics 2–8 of the wet
    signal and computes:
        ratio = odd_power / (odd_power + even_power)
    where odd = H3² + H5² + H7² and even = H2² + H4² + H6² + H8².
    A ratio of 1.0 = purely odd harmonics (Big Muff ideal);
    0.0 = purely even (tube warmth); 0.5 = balanced.

    With wet_fn provided: synthesises test sines, runs through callable.
    Without wet_fn: searches captured arrays for matching-level windows.

    Look for: >0.8 odd ratio = characteristic fuzz symmetry;
    frequency dependence of ratio reveals circuit resonances.

    Args:
        dry: Dry input signal.
        wet: Pedal wet output.
        sr: Sample rate in Hz.
        fundamentals: Fundamental frequencies (Hz). Default: [110, 220, 440, 880].
        levels_dbfs: Input levels (dBFS). Default: [-30, -20, -12, -6, -3, 0].
        wet_fn: Optional callable (x: np.ndarray) -> np.ndarray for synthesised test.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    if fundamentals is None:
        fundamentals = [110, 220, 440, 880]
    if levels_dbfs is None:
        levels_dbfs = [-30, -20, -12, -6, -3, 0]

    n_f = len(fundamentals)
    n_l = len(levels_dbfs)
    ratio_mat = np.full((n_f, n_l), np.nan)

    for fi, f0 in enumerate(fundamentals):
        sos = _bandpass_sos(f0, sr)
        dry_bp = sosfilt(sos, dry.astype(np.float64))
        win_n  = min(int(sr * 0.05), len(dry) // 8)

        for li, ldb in enumerate(levels_dbfs):
            amp = 10.0 ** (ldb / 20.0)

            if wet_fn is not None:
                dur = max(0.1, 10 / f0)
                x   = _synth_sine(f0, sr, dur, amp)
                sig = wet_fn(x).astype(np.float32)
            else:
                # Find capture windows near (f0, level)
                wins = []
                for i in range(0, len(dry_bp) - win_n, win_n):
                    rms = float(np.sqrt(np.mean(dry_bp[i:i + win_n] ** 2)))
                    if rms < 1e-6:
                        continue
                    if abs(20.0 * np.log10(rms) - ldb) < 4.0:
                        wins.append(wet[i:i + win_n])
                sig = np.concatenate(wins) if wins else wet

            h = _harmonic_amps(sig, sr, f0, n_harmonics=8)
            h2 = h ** 2
            odd  = h2[2] + h2[4] + h2[6]          # H3, H5, H7
            even = h2[1] + h2[3] + h2[5] + h2[7]  # H2, H4, H6, H8
            denom = odd + even
            ratio_mat[fi, li] = float(odd / denom) if denom > 1e-20 else np.nan

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, max(4, n_f * 1.4)))
        im = ax.imshow(ratio_mat, aspect="auto", cmap="RdBu_r",
                       vmin=0.0, vmax=1.0, origin="lower")
        fig.colorbar(im, ax=ax, label="Odd / (Odd + Even)  [0=even, 1=odd]",
                     fraction=0.046, pad=0.04)

        ax.set_xticks(range(n_l))
        ax.set_xticklabels([f"{l}" for l in levels_dbfs], fontsize=9)
        ax.set_yticks(range(n_f))
        ax.set_yticklabels([f"{int(f)} Hz" for f in fundamentals], fontsize=9)
        ax.set_xlabel("Input level (dBFS)")
        ax.set_ylabel("Fundamental frequency")
        ax.set_title("Odd/Even Harmonic Ratio  (blue=even, red=odd; fuzz ideal ≈ 0.8+)")

        for fi in range(n_f):
            for li in range(n_l):
                v = ratio_mat[fi, li]
                if not np.isnan(v):
                    ax.text(li, fi, f"{v:.2f}", ha="center", va="center", fontsize=8)

        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_frequency_smearing_matrix(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    n_tones: int = 20,
    wet_fn: Callable | None = None,
    save_path=None,
) -> plt.Figure:
    """Frequency routing heatmap for a pedal: input tone → output spectrum.

    Rows = input tone frequency (log), columns = output frequency (log).
    For a linear system energy stays on the diagonal; fuzz spawns harmonic
    columns above the diagonal. Subharmonics appear below the diagonal.

    With wet_fn: synthesises tones, runs through callable (exact).
    Without wet_fn: uses bandpass isolation on captured arrays (approximate).

    Look for: bright off-diagonal columns at 2×, 3×, 5× input frequency —
    characteristic of hard-clipping fuzz.

    Args:
        dry: Dry input signal.
        wet: Pedal wet output.
        sr: Sample rate in Hz.
        n_tones: Logarithmically-spaced input tones (80–8000 Hz).
        wet_fn: Optional callable (x: np.ndarray) -> np.ndarray.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    f_lo, f_hi = 80.0, min(8_000.0, sr * 0.4)
    tones   = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), n_tones))
    amp     = 10.0 ** (-12.0 / 20.0)   # -12 dBFS
    tone_dur = 0.5                       # 500 ms per tone

    out_f_lo, out_f_hi = 80.0, min(16_000.0, sr * 0.49)
    n_out = 120
    out_bins = np.exp(np.linspace(np.log(out_f_lo), np.log(out_f_hi), n_out))
    mat  = np.zeros((n_tones, n_out))

    fft_n = 4096
    hann  = np.hanning(fft_n)
    freqs_fft = rfftfreq(fft_n, 1.0 / sr)

    for i, f_in in enumerate(tones):
        if wet_fn is not None:
            x   = _synth_sine(f_in, sr, tone_dur, amp)
            sig = wet_fn(x).astype(np.float32)
            # Average FFT over non-overlapping windows
            ps = np.zeros(fft_n // 2 + 1)
            count = 0
            for j in range(0, len(sig) - fft_n, fft_n):
                ps += np.abs(rfft(sig[j:j + fft_n] * hann)) ** 2
                count += 1
            ps = ps / max(count, 1)
        else:
            sos      = _bandpass_sos(f_in, sr, width=0.15)
            dry_bp   = sosfilt(sos, dry.astype(np.float64))
            energy_s = np.convolve(dry_bp ** 2, np.ones(fft_n) / fft_n, mode="same")
            thr      = max(np.percentile(energy_s, 85), 1e-12)
            active   = np.where(energy_s > thr)[0]
            ps = np.zeros(fft_n // 2 + 1)
            count = 0
            for j in range(0, len(active) - fft_n, fft_n):
                s = active[j]
                if s + fft_n > len(wet):
                    break
                ps += np.abs(rfft(wet[s:s + fft_n] * hann)) ** 2
                count += 1
            ps = ps / max(count, 1)

        # Bin into output frequency grid
        for bi in range(n_out):
            fl = out_bins[bi] / 1.15
            fh = out_bins[bi] * 1.15
            m  = (freqs_fft >= fl) & (freqs_fft < fh)
            mat[i, bi] = float(ps[m].sum()) if m.any() else 0.0

    mat_db = 10.0 * np.log10(mat + 1e-20)

    tick_in_idx  = np.linspace(0, n_tones - 1, min(8, n_tones)).astype(int)
    tick_out_idx = np.linspace(0, n_out - 1, 8).astype(int)

    def _hz_label(f: float) -> str:
        return f"{int(f // 1000)}k" if f >= 1_000 else str(int(f))

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(13, 7))
        im = ax.imshow(mat_db, aspect="auto", origin="lower",
                       cmap="magma", vmin=-80, vmax=0)
        fig.colorbar(im, ax=ax, label="Power (dB)", fraction=0.046, pad=0.04)

        ax.set_xticks(tick_out_idx)
        ax.set_xticklabels([_hz_label(out_bins[k]) for k in tick_out_idx])
        ax.set_yticks(tick_in_idx)
        ax.set_yticklabels([_hz_label(tones[k]) for k in tick_in_idx])
        ax.set_xlabel("Output frequency")
        ax.set_ylabel("Input tone frequency")
        ax.set_title("Frequency Smearing Matrix  (bright off-diagonal = harmonic generation)")

        # Diagonal: output = input (fundamental)
        diag_x = np.interp(tones, out_bins, np.arange(n_out))
        ax.plot(diag_x, np.arange(n_tones), color="white", lw=0.8, ls="--", alpha=0.5,
                label="f_out = f_in (linear)")
        ax.legend(fontsize=8, loc="upper left")

        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_coherence_nonlinearity(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    save_path=None,
) -> plt.Figure:
    """Coherence and nonlinear power spectrum of a pedal.

    Two panels:
      1. γ²(f) between dry and wet. Regions where γ² < 0.9 are shaded red —
         these frequencies contain energy generated by nonlinearity.
      2. Nonlinearity spectrum: (1 − γ²) × S_ww(f) — the power in the wet
         signal not linearly predictable from the dry signal, in dB.

    Look for: large red regions in the guitar frequency band (80–8 kHz)
    confirm strong nonlinearity; peaks in the nonlinearity spectrum show
    which frequencies the fuzz generates most harmonics at.

    Args:
        dry: Dry input, shape (N,), float32.
        wet: Pedal wet output.
        sr: Sample rate in Hz.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    n = min(len(dry), len(wet))
    f_coh, Cxy = _coherence(dry[:n], wet[:n], fs=sr, nperseg=4096)

    # Wet power spectrum for nonlinearity spectrum
    from scipy.signal import welch as _welch
    f_w, S_ww = _welch(wet[:n].astype(np.float64), fs=sr, nperseg=4096)

    nl_spec = (1.0 - np.clip(Cxy, 0, 1)) * S_ww
    nl_db   = 10.0 * np.log10(nl_spec + 1e-20)
    S_ww_db = 10.0 * np.log10(S_ww + 1e-20)

    with plt.style.context(_STYLE):
        fig, (ax_c, ax_nl) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        mask = (f_coh >= 20) & (f_coh <= 20_000)

        ax_c.plot(f_coh[mask], Cxy[mask], color=C_TARGET, lw=1.4,
                  label="γ²  dry→wet")
        ax_c.axhline(0.9, color="white", lw=0.6, ls="--", alpha=0.5, label="γ² = 0.9")
        ax_c.fill_between(f_coh[mask], 0, Cxy[mask],
                           where=(Cxy[mask] < 0.9), color="#FF6B6B", alpha=0.25,
                           label="Nonlinear region")
        ax_c.set_ylabel("γ²")
        ax_c.set_title("Magnitude-Squared Coherence  (red = nonlinear frequency content)")
        ax_c.set_ylim(0, 1.05)
        ax_c.legend(fontsize=9)

        ax_nl.plot(f_w[mask], nl_db[mask], color=C_ERR, lw=1.0,
                   label="Nonlinear power  (1−γ²)·Sww")
        ax_nl.plot(f_w[mask], S_ww_db[mask], color=C_TARGET, lw=0.7, alpha=0.5,
                   label="Total wet power  Sww")
        ax_nl.set_ylabel("Power (dB)")
        ax_nl.set_title("Nonlinearity Spectrum  — where in frequency the pedal generates NL content")
        ax_nl.legend(fontsize=9)
        _log_freq_axis(ax_nl, 20, 20_000)

        fig.suptitle("Coherence Nonlinearity Analysis", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dynamic_transfer_curve(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    window_ms: float = 10.0,
    stride_ms: float = 1.0,
    save_path=None,
) -> plt.Figure:
    """Time-evolving transfer curve as a 2-D heatmap.

    Divides the signal into overlapping windows, bins each window's samples
    by input amplitude, and records the mean output. Stacks into a matrix:
    rows = time, columns = input amplitude bins, colour = mean output.
    Reveals whether the clipping shape changes over time (germanium transistor
    warm-up, capacitor charging during sustain, etc.).

    A second panel shows the static (time-averaged) transfer curve.

    Look for: row colour shifting over time (dynamic saturation); heatmap
    banding that widens at high amplitudes = hard clipping onset.

    Args:
        dry: Dry input, shape (N,), float32.
        wet: Pedal wet output.
        sr: Sample rate in Hz.
        window_ms: Window length in milliseconds.
        stride_ms: Window stride in milliseconds.
        save_path: Optional PNG save path.

    Returns:
        matplotlib Figure.
    """
    win_n    = int(sr * window_ms / 1000)
    stride_n = max(1, int(sr * stride_ms / 1000))
    n        = min(len(dry), len(wet))
    n_bins   = 100

    amp_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    amp_centres = 0.5 * (amp_edges[:-1] + amp_edges[1:])

    times, mat = [], []
    for i in range(0, n - win_n, stride_n):
        d = dry[i:i + win_n].astype(np.float64)
        w = wet[i:i + win_n].astype(np.float64)
        sums   = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)
        bi = np.clip(np.searchsorted(amp_edges[1:], d), 0, n_bins - 1)
        np.add.at(sums,   bi, w)
        np.add.at(counts, bi, 1)
        with np.errstate(invalid="ignore"):
            row = np.where(counts > 0, sums / counts, np.nan)
        mat.append(row)
        times.append((i + win_n / 2) / sr)

    if not mat:
        with plt.style.context(_STYLE):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Signal too short", ha="center", va="center",
                    transform=ax.transAxes)
        _save(fig, save_path)
        return fig

    mat_arr = np.array(mat)  # (n_windows, n_bins)
    t_arr   = np.array(times)

    # Static (time-averaged) transfer curve
    static_t = np.nanmean(mat_arr, axis=0)

    vabs = float(np.nanpercentile(np.abs(mat_arr), 97))

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                                  gridspec_kw={"width_ratios": [3, 1]})

        im = axes[0].imshow(
            mat_arr.T, origin="lower", aspect="auto",
            extent=[t_arr[0], t_arr[-1], -1, 1],
            cmap="RdBu_r", vmin=-vabs, vmax=vabs,
        )
        fig.colorbar(im, ax=axes[0], label="Mean output amplitude",
                     fraction=0.046, pad=0.04)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Input amplitude")
        axes[0].set_title(f"Dynamic transfer curve  ({window_ms:.0f} ms window, "
                           f"{stride_ms:.0f} ms stride)")
        diag = np.array([-1, 1])
        axes[0].plot([t_arr[0], t_arr[-1]], [0, 0], color="white", lw=0.4, alpha=0.3)

        axes[1].plot(static_t, amp_centres, color=C_TARGET, lw=1.5)
        axes[1].plot(amp_centres, amp_centres, color="white", lw=0.5, ls="--",
                     alpha=0.3, label="Linear")
        axes[1].set_xlabel("Mean output")
        axes[1].set_ylabel("Input amplitude")
        axes[1].set_title("Static (time-avg)")
        axes[1].set_xlim(-1.1, 1.1)
        axes[1].set_ylim(-1.1, 1.1)
        axes[1].set_aspect("equal")

        fig.suptitle("Dynamic Transfer Curve Analysis", fontsize=13)
        fig.tight_layout()
    _save(fig, save_path)
    return fig


def characterize_pedal(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    output_dir: Path | str,
    pedal_name: str = "pedal",
    wet_fn: Callable | None = None,
    dry_name: str = "dry",
    wet_name: str = "wet",
) -> dict[str, plt.Figure]:
    """Run all pedal characterization plots, save PNGs, print a summary.

    Output files: {output_dir}/{pedal_name}_{plot_name}.png

    Summary scalar statistics printed:
      - Mean coherence across 80–8 kHz
      - Odd/even harmonic ratio at 440 Hz, -6 dBFS
      - Attack / release times (dry vs wet)
      - Peak gain reduction (dB)

    Args:
        dry: Dry input, shape (N,), float32.
        wet: Pedal wet output.
        sr: Sample rate in Hz.
        output_dir: Directory to write PNGs.
        pedal_name: Prefix for output filenames.
        wet_fn: Optional callable (x: np.ndarray) -> np.ndarray for functions
            that can use a model callable instead of captured arrays.
        dry_name: Human-readable label for the dry signal (stamped on each figure).
        wet_name: Human-readable label for the wet signal.

    Returns:
        Dict mapping plot_name → Figure for successfully generated plots.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _path(name: str) -> Path:
        return out / f"{pedal_name}_{name}.png"

    tasks: list[tuple[str, Callable]] = [
        ("waveform_morphology",
         lambda: plot_waveform_morphology(dry, wet, sr,
                                           save_path=_path("waveform_morphology"))),
        ("gain_reduction",
         lambda: plot_gain_reduction(dry, wet, sr,
                                      save_path=_path("gain_reduction"))),
        ("envelope_comparison",
         lambda: plot_envelope_comparison(dry, wet, sr,
                                           save_path=_path("envelope_comparison"))),
        ("phase_portrait",
         lambda: plot_phase_portrait(dry, wet, sr,
                                      save_path=_path("phase_portrait"))),
        ("odd_even_harmonic_ratio",
         lambda: plot_odd_even_harmonic_ratio(dry, wet, sr, wet_fn=wet_fn,
                                               save_path=_path("odd_even_harmonic_ratio"))),
        ("frequency_smearing_matrix",
         lambda: plot_frequency_smearing_matrix(dry, wet, sr, wet_fn=wet_fn,
                                                 save_path=_path("frequency_smearing_matrix"))),
        ("coherence_nonlinearity",
         lambda: plot_coherence_nonlinearity(dry, wet, sr,
                                              save_path=_path("coherence_nonlinearity"))),
        ("dynamic_transfer_curve",
         lambda: plot_dynamic_transfer_curve(dry, wet, sr,
                                              save_path=_path("dynamic_transfer_curve"))),
    ]

    results: dict[str, plt.Figure] = {}
    col_w = 32

    print(f"\n{'─' * 67}")
    print(f"  Pedal characterization for '{pedal_name}'")
    print(f"{'─' * 67}")

    for name, fn in tasks:
        try:
            fig = fn()
            _stamp_figure(fig, pedal_name, dry_name, wet_name)
            _save(fig, _path(name))
            results[name] = fig
            status = "OK"
            detail = str(_path(name).name)
        except Exception as exc:
            status = "FAILED"
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            plt.close("all")

        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark}  {name:<{col_w}}  {status:<7}  {detail}")

    print(f"{'─' * 67}")
    print(f"  Generated {len(results)}/{len(tasks)} plots → {out}/")

    # ── Key scalar summary ────────────────────────────────────────────────
    print(f"\n  Key statistics for '{pedal_name}':")
    n = min(len(dry), len(wet))
    try:
        f_c, Cxy = _coherence(dry[:n], wet[:n], fs=sr, nperseg=4096)
        mask_g = (f_c >= 80) & (f_c <= 8_000)
        print(f"    Mean coherence 80–8k Hz : {float(np.mean(Cxy[mask_g])):.3f}")
    except Exception as e:
        print(f"    Mean coherence          : n/a ({e})")

    try:
        sos_440 = _bandpass_sos(440.0, sr)
        dry_bp = sosfilt(sos_440, dry.astype(np.float64))
        win_n = min(int(sr * 0.05), len(dry) // 8)
        ldb = -6.0
        wins = []
        for i in range(0, len(dry_bp) - win_n, win_n):
            rms = float(np.sqrt(np.mean(dry_bp[i:i + win_n] ** 2)))
            if rms > 1e-6 and abs(20.0 * np.log10(rms) - ldb) < 4.0:
                wins.append(wet[i:i + win_n])
        sig_440 = np.concatenate(wins) if wins else wet
        h = _harmonic_amps(sig_440, sr, 440.0, 8)
        h2 = h ** 2
        odd = h2[2] + h2[4] + h2[6]
        even = h2[1] + h2[3] + h2[5] + h2[7]
        denom = odd + even
        ratio = float(odd / denom) if denom > 1e-20 else float("nan")
        print(f"    Odd/even ratio 440 Hz   : {ratio:.3f}  (fuzz ideal > 0.8)")
    except Exception as e:
        print(f"    Odd/even ratio          : n/a ({e})")

    try:
        from scipy.signal import hilbert
        win_20ms = int(sr * 0.02)
        hop_20ms = win_20ms // 4
        t_m, rd, rw = [], [], []
        for i in range(0, n - win_20ms, hop_20ms):
            t_m.append((i + win_20ms / 2) / sr)
            rd.append(float(np.sqrt(np.mean(dry[i:i + win_20ms] ** 2))))
            rw.append(float(np.sqrt(np.mean(wet[i:i + win_20ms] ** 2))))
        t_m = np.array(t_m)
        rd_arr = np.array(rd)
        rw_arr = np.array(rw)

        def _atk_rel(env, t):
            pk = env.max()
            pk_i = int(np.argmax(env))
            atk_i = pk_i
            for j in range(pk_i, 0, -1):
                if env[j] < 0.1 * pk:
                    atk_i = j; break
            rel_i = pk_i
            for j in range(pk_i, len(env)):
                if env[j] < 0.1 * pk:
                    rel_i = j; break
            return (t[pk_i] - t[atk_i]) * 1000, (t[rel_i] - t[pk_i]) * 1000

        atk_d, rel_d = _atk_rel(rd_arr, t_m)
        atk_w, rel_w = _atk_rel(rw_arr, t_m)
        print(f"    Attack  dry/wet         : {atk_d:.0f} ms / {atk_w:.0f} ms")
        print(f"    Release dry/wet         : {rel_d:.0f} ms / {rel_w:.0f} ms")
    except Exception as e:
        print(f"    Attack/release          : n/a ({e})")

    try:
        win_5ms = int(sr * 0.005)
        gr_min = 0.0
        for i in range(0, n - win_5ms, win_5ms):
            rd = float(np.sqrt(np.mean(dry[i:i + win_5ms] ** 2)))
            rw = float(np.sqrt(np.mean(wet[i:i + win_5ms] ** 2)))
            if rd > 1e-6:
                gr = 20 * np.log10(rw / (rd + 1e-10))
                if gr < gr_min:
                    gr_min = gr
        print(f"    Peak gain reduction     : {gr_min:.1f} dB")
    except Exception as e:
        print(f"    Peak gain reduction     : n/a ({e})")

    print()
    return results


# ── Updated run_all_diagnostics with mode support ─────────────────────────────

def run_all_diagnostics(
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    output_dir: Path | str,
    model_name: str = "model",
    model_fn: Callable | None = None,
    model_fns: dict[str, Callable] | None = None,
    mode: str = "model",
    dry_name: str = "dry",
    wet_name: str = "wet",
) -> dict[str, plt.Figure]:
    """Run every diagnostic plot, save PNGs, return a dict of figures.

    Each function is wrapped in a try/except so a single failure does not
    abort the entire suite. A summary table is printed to stdout.
    Every figure is stamped with dry_name, wet_name, model_name and today's date.

    Output files: {output_dir}/{model_name}_{plot_name}.png

    Args:
        dry: Dry input signal, shape (N,), float32.
        target: Pedal wet output (ground truth).
        predicted: Model output.
        sr: Sample rate in Hz.
        output_dir: Directory to write PNGs.
        model_name: Prefix for output filenames and summary table.
        model_fn: Optional model callable for plot_describing_function.
        model_fns: Optional {"target": fn, "predicted": fn} dict for
            plot_harmonic_profile, plot_imd, plot_nonlinear_frequency_map.
        mode: One of:
            'model'  — model evaluation plots (default, existing behaviour)
            'pedal'  — pedal characterization plots (dry vs wet)
            'both'   — all plots
        dry_name: Human-readable label for the dry signal (stamped on each figure).
        wet_name: Human-readable label for the wet/target signal.

    Returns:
        Dict mapping plot_name → Figure for successfully generated plots.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _path(name: str) -> Path:
        return out / f"{model_name}_{name}.png"

    model_tasks: list[tuple[str, Callable]] = [
        ("static_transfer_curve",
         lambda: plot_static_transfer_curve(dry, target, predicted, sr,
                                             _path("static_transfer_curve"))),
        ("describing_function",
         lambda: plot_describing_function(dry, target, predicted, sr,
                                           model_fn=model_fn,
                                           save_path=_path("describing_function"))),
        ("lag_error",
         lambda: plot_lag_error(dry, target, predicted, sr,
                                 save_path=_path("lag_error"))),
        ("impulse_response",
         lambda: plot_impulse_response(dry, target, predicted, sr,
                                        save_path=_path("impulse_response"))),
        ("step_response",
         lambda: plot_step_response(dry, target, predicted, sr,
                                     save_path=_path("step_response"))),
        ("transfer_function",
         lambda: plot_transfer_function(dry, target, predicted, sr,
                                         save_path=_path("transfer_function"))),
        ("group_delay",
         lambda: plot_group_delay(dry, target, predicted, sr,
                                   save_path=_path("group_delay"))),
        ("coherence",
         lambda: plot_coherence(dry, target, predicted, sr,
                                 save_path=_path("coherence"))),
        ("spectrogram_overlay",
         lambda: plot_spectrogram_overlay(target, predicted, sr,
                                           save_path=_path("spectrogram_overlay"))),
        ("harmonic_profile",
         lambda: plot_harmonic_profile(dry, target, predicted, sr,
                                        model_fns=model_fns,
                                        save_path=_path("harmonic_profile"))),
        ("imd",
         lambda: plot_imd(dry, target, predicted, sr,
                           model_fns=model_fns, save_path=_path("imd"))),
        ("level_dependent_fr",
         lambda: plot_level_dependent_fr(dry, target, predicted, sr,
                                          save_path=_path("level_dependent_fr"))),
        ("nonlinear_frequency_map",
         lambda: plot_nonlinear_frequency_map(dry, target, predicted, sr,
                                               model_fns=model_fns,
                                               save_path=_path("nonlinear_frequency_map"))),
        ("error_spectrogram",
         lambda: plot_error_spectrogram(target, predicted, sr,
                                         save_path=_path("error_spectrogram"))),
        ("aweighted_error",
         lambda: plot_aweighted_error(target, predicted, sr,
                                       save_path=_path("aweighted_error"))),
    ]

    pedal_tasks: list[tuple[str, Callable]] = [
        ("waveform_morphology",
         lambda: plot_waveform_morphology(dry, target, sr,
                                           save_path=_path("waveform_morphology"))),
        ("gain_reduction",
         lambda: plot_gain_reduction(dry, target, sr,
                                      save_path=_path("gain_reduction"))),
        ("envelope_comparison",
         lambda: plot_envelope_comparison(dry, target, sr,
                                           save_path=_path("envelope_comparison"))),
        ("phase_portrait",
         lambda: plot_phase_portrait(dry, target, sr,
                                      save_path=_path("phase_portrait"))),
        ("odd_even_harmonic_ratio",
         lambda: plot_odd_even_harmonic_ratio(dry, target, sr,
                                               save_path=_path("odd_even_harmonic_ratio"))),
        ("frequency_smearing_matrix",
         lambda: plot_frequency_smearing_matrix(dry, target, sr,
                                                 save_path=_path("frequency_smearing_matrix"))),
        ("coherence_nonlinearity",
         lambda: plot_coherence_nonlinearity(dry, target, sr,
                                              save_path=_path("coherence_nonlinearity"))),
        ("dynamic_transfer_curve",
         lambda: plot_dynamic_transfer_curve(dry, target, sr,
                                              save_path=_path("dynamic_transfer_curve"))),
    ]

    if mode == "model":
        tasks = model_tasks
    elif mode == "pedal":
        tasks = pedal_tasks
    elif mode == "both":
        tasks = model_tasks + pedal_tasks
    else:
        raise ValueError(f"mode must be 'model', 'pedal', or 'both'; got {mode!r}")

    results: dict[str, plt.Figure] = {}
    col_w = 32

    print(f"\n{'─' * 67}")
    print(f"  Diagnostics for '{model_name}'  (mode={mode})")
    print(f"{'─' * 67}")

    for name, fn in tasks:
        try:
            fig = fn()
            _stamp_figure(fig, model_name, dry_name, wet_name)
            _save(fig, _path(name))
            results[name] = fig
            status = "OK"
            detail = str(_path(name).name)
        except Exception as exc:
            status = "FAILED"
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            plt.close("all")

        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark}  {name:<{col_w}}  {status:<7}  {detail}")

    print(f"{'─' * 67}")
    print(f"  Generated {len(results)}/{len(tasks)} plots → {out}/\n")
    return results


# ── __main__ smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _SR  = 48_000
    _DUR = 5.0
    _rng = np.random.default_rng(42)
    _N   = int(_SR * _DUR)
    _t   = np.arange(_N) / _SR

    dry_sig = (
        0.45 * np.sin(2 * np.pi * 220 * _t)
        + 0.30 * np.sin(2 * np.pi * 440 * _t)
        + 0.15 * np.sin(2 * np.pi * 880 * _t)
        + 0.10 * _rng.standard_normal(_N).astype(np.float32)
    ).astype(np.float32)
    dry_sig = np.clip(dry_sig, -1.0, 1.0)

    # Pedal simulation: tanh clipper
    wet_sig = np.tanh(3.0 * dry_sig).astype(np.float32)

    # Model approximation (slightly different)
    predicted_sig = np.clip(
        (np.tanh(2.6 * dry_sig) + 0.05 * dry_sig ** 2).astype(np.float32), -1, 1
    )

    import matplotlib
    matplotlib.use("Agg")

    out_dir = Path("./diagnostics_test_output")
    out_dir.mkdir(exist_ok=True)

    print(f"Dry  peak: {20*np.log10(np.max(np.abs(dry_sig))  +1e-10):.1f} dBFS")
    print(f"Wet  peak: {20*np.log10(np.max(np.abs(wet_sig))  +1e-10):.1f} dBFS")
    print(f"Pred peak: {20*np.log10(np.max(np.abs(predicted_sig))+1e-10):.1f} dBFS")

    # ── mode='model': existing 15 evaluation plots ─────────────────────
    figs_m = run_all_diagnostics(
        dry_sig, wet_sig, predicted_sig, _SR,
        output_dir=out_dir, model_name="tanh_model", mode="model",
    )
    assert len(figs_m) == 15, f"Expected 15 model plots, got {len(figs_m)}"

    # ── mode='pedal': 8 characterization plots ─────────────────────────
    figs_p = run_all_diagnostics(
        dry_sig, wet_sig, predicted_sig, _SR,
        output_dir=out_dir, model_name="tanh_pedal", mode="pedal",
    )
    assert len(figs_p) == 8, f"Expected 8 pedal plots, got {len(figs_p)}"

    # ── standalone characterize_pedal ──────────────────────────────────
    figs_c = characterize_pedal(
        dry_sig, wet_sig, _SR,
        output_dir=out_dir, pedal_name="tanh_char",
        wet_fn=lambda x: np.tanh(3.0 * x).astype(np.float32),
    )
    assert len(figs_c) == 8, f"Expected 8 char plots, got {len(figs_c)}"

    total = len(figs_m) + len(figs_p) + len(figs_c)
    print(f"\nAll smoke tests passed — {total}/31 total plots generated → {out_dir}/")
    sys.exit(0)

    sys.exit(0 if len(figs) == 15 else 1)
