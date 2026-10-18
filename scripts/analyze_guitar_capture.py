"""Analyse a raw dry/wet guitar capture — waveform and frequency comparison.

No manifest required: works on any matched dry/wet WAV pair captured at the
same time with no alignment offset (as verified by xcorr analysis).

Usage:
    uv run scripts/analyze_guitar_capture.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DRY_WAV = ROOT / "recordings/data/notaklon/captures/Notaklon - 01 Start - guitar dry.wav"
WET_WAV = ROOT / "recordings/data/notaklon/captures/Notaklon - 01 Start - guitar wet.wav"
OUT_DIR = ROOT / "outputs/notaklon_guitar_capture"

SR = 48_000

DRY_COLOR = "#1565C0"
WET_COLOR = "#C62828"


# ─── DSP helpers ──────────────────────────────────────────────────────────────

def _smooth_log_freq(
    freqs: np.ndarray,
    vals: np.ndarray,
    frac: float = 1 / 6,
    n_out: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """1/6-octave smoothing in log-frequency space (from analyze_train_signal_sections)."""
    f_out = np.logspace(
        np.log10(max(float(freqs[1]), 15.0)),
        np.log10(min(float(freqs[-1]), 22050.0)),
        n_out,
    )
    v_out = np.full(n_out, np.nan)
    half = frac / 2.0
    for i, f in enumerate(f_out):
        mask = (freqs >= f * 2 ** (-half)) & (freqs <= f * 2 ** half)
        if mask.any():
            v_out[i] = float(np.mean(vals[mask]))
    return f_out, v_out


def _fft_mag_db(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-signal FFT magnitude in dBFS, 1/6-oct smoothed."""
    n = len(sig)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mag = np.abs(np.fft.rfft(sig.astype(np.float64)))
    mag_db = 20.0 * np.log10(mag / n + 1e-15)
    mask = (freqs >= 15.0) & (freqs <= SR / 2 * 0.95)
    return _smooth_log_freq(freqs[mask], mag_db[mask])


def _transfer_db(dry: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FFT-ratio transfer function (dB), 1/6-oct smoothed (from analyze_treble_sweep)."""
    n = min(len(dry), len(wet))
    D = np.fft.rfft(dry[:n].astype(np.float64))
    W = np.fft.rfft(wet[:n].astype(np.float64))
    H = W / (D + 1e-30)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    db = 20.0 * np.log10(np.abs(H) + 1e-12)
    mask = (freqs >= 15.0) & (freqs <= SR / 2 * 0.95)
    return _smooth_log_freq(freqs[mask], db[mask])


def _find_active_start(sig: np.ndarray, dur_s: float, hop_s: float = 1.0) -> int:
    """Return sample index of the loudest hop-aligned window of length dur_s."""
    win = int(dur_s * SR)
    hop = int(hop_s * SR)
    best_rms, best_idx = 0.0, 0
    for i in range(0, len(sig) - win, hop):
        rms = float(np.sqrt(np.mean(sig[i:i + win] ** 2)))
        if rms > best_rms:
            best_rms, best_idx = rms, i
    return best_idx


def _log_fmt(x: float, _: object) -> str:
    return f"{int(x)}" if x < 1000 else f"{int(x) // 1000}k"


# ─── Plots ─────────────────────────────────────────────────────────────────────

def plot_waveform_overview(dry: np.ndarray, wet: np.ndarray, out_dir: Path) -> Path:
    """01: Full 85-second waveform with independent Y-axes."""
    t = np.arange(len(dry)) / SR

    fig, ax1 = plt.subplots(figsize=(16, 4))
    ax2 = ax1.twinx()

    ax1.plot(t, dry, color=DRY_COLOR, lw=0.3, alpha=0.8, label="Dry")
    ax2.plot(t, wet, color=WET_COLOR, lw=0.3, alpha=0.65, label="Wet")

    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Dry amplitude", color=DRY_COLOR, fontsize=11)
    ax2.set_ylabel("Wet amplitude", color=WET_COLOR, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=DRY_COLOR)
    ax2.tick_params(axis="y", labelcolor=WET_COLOR)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10)
    ax1.set_title(
        "Full waveform — Notaklon guitar capture (dry vs wet, independent Y-axes)",
        fontsize=12, fontweight="bold",
    )
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    out = out_dir / "01_waveform_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_waveform_zoom(dry: np.ndarray, wet: np.ndarray, out_dir: Path) -> Path:
    """02: 3-second zoom on the loudest passage, independent Y-axes."""
    dur_s = 1.0
    start = _find_active_start(dry, dur_s)
    n = int(dur_s * SR)
    d_seg = dry[start: start + n]
    w_seg = wet[start: start + n]
    t_ms = np.arange(n) / SR * 1000

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    ax1.plot(t_ms, d_seg, color=DRY_COLOR, lw=0.9, alpha=0.85, label="Dry")
    ax2.plot(t_ms, w_seg, color=WET_COLOR, lw=0.9, alpha=0.75, label="Wet")

    ax1.set_xlabel("Time (ms)", fontsize=11)
    ax1.set_ylabel("Dry amplitude", color=DRY_COLOR, fontsize=11)
    ax2.set_ylabel("Wet amplitude", color=WET_COLOR, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=DRY_COLOR)
    ax2.tick_params(axis="y", labelcolor=WET_COLOR)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10)
    t0 = start / SR
    ax1.set_title(
        f"Waveform zoom — loudest 1 s (t = {t0:.1f}–{t0 + dur_s:.1f} s)",
        fontsize=12, fontweight="bold",
    )
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    out = out_dir / "02_waveform_zoom.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_peak_zoom(dry: np.ndarray, wet: np.ndarray, out_dir: Path) -> Path:
    """03: 200 ms window centred on the peak loudness sample of the dry signal."""
    half = int(0.050 * SR)   # 50 ms each side → 100 ms total
    peak_idx = int(np.argmax(np.abs(dry)))
    start = max(0, peak_idx - half)
    end   = min(len(dry), peak_idx + half)
    d_seg = dry[start:end]
    w_seg = wet[start:end]
    t_ms  = (np.arange(len(d_seg)) - (peak_idx - start)) / SR * 1000

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    ax1.plot(t_ms, d_seg, color=DRY_COLOR, lw=1.0, alpha=0.9, label="Dry")
    ax2.plot(t_ms, w_seg, color=WET_COLOR,  lw=1.0, alpha=0.8, label="Wet")
    ax1.axvline(0, color="#555555", lw=0.8, ls="--", alpha=0.6, label="Peak (dry)")

    ax1.set_xlabel("Time relative to peak (ms)", fontsize=11)
    ax1.set_ylabel("Dry amplitude", color=DRY_COLOR, fontsize=11)
    ax2.set_ylabel("Wet amplitude", color=WET_COLOR, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=DRY_COLOR)
    ax2.tick_params(axis="y", labelcolor=WET_COLOR)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10)
    ax1.set_title(
        f"Peak loudness zoom — 100 ms window centred on dry peak (t = {peak_idx/SR:.3f} s)",
        fontsize=12, fontweight="bold",
    )
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    out = out_dir / "03_peak_zoom.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_absolute_spectra(dry: np.ndarray, wet: np.ndarray, out_dir: Path) -> Path:
    """03: Two-panel spectra — absolute levels and RMS-normalised shape comparison."""
    f_d, db_d = _fft_mag_db(dry)
    f_w, db_w = _fft_mag_db(wet)

    # RMS offset to align gain references for the normalised panel
    rms_dry = float(20 * np.log10(np.sqrt(np.mean(dry.astype(np.float64) ** 2)) + 1e-15))
    rms_wet = float(20 * np.log10(np.sqrt(np.mean(wet.astype(np.float64) ** 2)) + 1e-15))
    gain_offset = rms_wet - rms_dry   # dB to add to dry to match wet level

    fig, (ax_abs, ax_norm) = plt.subplots(2, 1, figsize=(14, 11))
    fig.suptitle(
        "Magnitude spectra — Notaklon guitar capture (1/6-oct smoothed)",
        fontsize=13, fontweight="bold",
    )

    def _draw(ax: plt.Axes, d_vals: np.ndarray, w_vals: np.ndarray, title: str, ylabel: str) -> None:
        valid_d = ~np.isnan(d_vals)
        valid_w = ~np.isnan(w_vals)
        ax.semilogx(f_d[valid_d], d_vals[valid_d], color=DRY_COLOR, lw=1.8, alpha=0.9, label="Dry")
        ax.semilogx(f_w[valid_w], w_vals[valid_w], color=WET_COLOR, lw=1.8, alpha=0.9, label="Wet (Notaklon)")
        ax.set_xlim(20, 20_000)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_log_fmt))
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

    _draw(ax_abs, db_d, db_w,
          "Absolute levels — offset = Notaklon output gain boost",
          "Magnitude (dBFS)")

    _draw(ax_norm, db_d + gain_offset, db_w,
          f"Level-normalised — dry shifted +{gain_offset:.1f} dB (RMS-matched) — shape difference only",
          "Magnitude (dBFS, RMS-normalised)")

    fig.tight_layout()
    out = out_dir / "04_absolute_spectra.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_transfer_function(dry: np.ndarray, wet: np.ndarray, out_dir: Path) -> Path:
    """04: Approximate wet/dry transfer function — two panels: full signal and active 20 s window.

    Guitar is not white noise so bins with low dry energy produce noisy ratios.
    The 1/6-oct smoothing partially mitigates this but treat as indicative only.
    """
    start = _find_active_start(dry, 20.0, hop_s=5.0)
    n20 = int(20.0 * SR)

    f_full, tf_full = _transfer_db(dry, wet)
    f_win, tf_win   = _transfer_db(dry[start: start + n20], wet[start: start + n20])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        "Approximate transfer function: Wet ÷ Dry (1/6-oct smoothed)\n"
        "Note: guitar ≠ white noise — sparse harmonics make this noisy outside played frequencies",
        fontsize=12, fontweight="bold",
    )

    for ax, f, tf, title in [
        (ax1, f_full, tf_full, "Full 85 s signal"),
        (ax2, f_win,  tf_win,  f"Active 20 s window (t = {start/SR:.0f}–{(start+n20)/SR:.0f} s)"),
    ]:
        valid = ~np.isnan(tf)
        ax.semilogx(f[valid], tf[valid], color="#2E7D32", lw=1.8)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlim(20, 20_000)
        ax.set_ylim(-20, 40)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_log_fmt))
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("H(f)  [dB]", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "05_transfer_function.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_spectrograms(dry: np.ndarray, wet: np.ndarray, out_dir: Path) -> Path:
    """05: Side-by-side log-frequency spectrograms, first 30 s."""
    from scipy.signal import spectrogram as _sg

    n30 = min(int(30 * SR), len(dry))
    nperseg, noverlap = 2048, 1536

    fig, (ax_d, ax_w) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    fig.suptitle("Spectrograms — first 30 s", fontsize=13, fontweight="bold")

    for ax, sig, title, cmap in [
        (ax_d, dry[:n30], "Dry",             "Blues"),
        (ax_w, wet[:n30], "Wet (Notaklon)",  "Reds"),
    ]:
        f, t, Sxx = _sg(sig.astype(np.float64), fs=SR,
                        nperseg=nperseg, noverlap=noverlap)
        Sxx_db = 10.0 * np.log10(Sxx + 1e-30)
        vmin = np.percentile(Sxx_db, 10)
        vmax = np.percentile(Sxx_db, 99)
        ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap=cmap,
                      vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_yscale("log")
        ax.set_ylim(50, 20_000)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_log_fmt))
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    out = out_dir / "06_spectrograms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading audio…")
    dry, sr = sf.read(DRY_WAV, dtype="float32", always_2d=False)
    wet, _  = sf.read(WET_WAV, dtype="float32", always_2d=False)
    assert sr == SR, f"Expected {SR} Hz, got {sr}"

    for name, sig in [("dry", dry), ("wet", wet)]:
        peak = float(20 * np.log10(np.max(np.abs(sig)) + 1e-12))
        rms  = float(20 * np.log10(np.sqrt(np.mean(sig ** 2)) + 1e-12))
        print(f"  {name}: {len(sig)/SR:.1f} s  peak={peak:.1f} dBFS  rms={rms:.1f} dBFS")

    print("\nGenerating plots…")
    plots = [
        plot_waveform_overview(dry, wet, OUT_DIR),
        plot_waveform_zoom(dry, wet, OUT_DIR),
        plot_peak_zoom(dry, wet, OUT_DIR),
        plot_absolute_spectra(dry, wet, OUT_DIR),
        plot_transfer_function(dry, wet, OUT_DIR),
        plot_spectrograms(dry, wet, OUT_DIR),
    ]

    print(f"\nAll plots written to {OUT_DIR}/")
    for p in plots:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
