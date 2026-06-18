#!/usr/bin/env python3
"""Per-section analysis of a train signal through one or more processing chains.

Compares a dry reference against any number of wet captures across all section
types defined in a train_signal JSON manifest.

Usage:
    uv run python scripts/analyze_train_signal_sections.py \\
        --dry  data/signals/train_signal_v2.wav \\
        --wet  "data/signals/train_signal_v2_reamp only.wav:Reamp Only" \\
        --wet  "results/pedalode/GE-7_v1/GE7 - flat.wav:GE-7 Flat" \\
        --output outputs/ge7_flat_analysis/

    # Manifest is inferred from the dry filename if not given:
    # train_signal_v2.wav  →  train_signal_v2.json  (same directory)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

matplotlib.use("Agg")

# ─── Colour palette (up to 6 wet files) ──────────────────────────────────────

_PALETTE = [
    "#1565C0",  # blue
    "#2E7D32",  # green
    "#C62828",  # red
    "#F57F17",  # amber
    "#6A1B9A",  # purple
    "#00695C",  # teal
]
_DRY_COLOR = "#888888"
_DRY_LABEL = "Dry"


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-section analysis of a train signal through processing chains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry", required=True, metavar="WAV",
                   help="Dry (unprocessed) reference WAV.")
    p.add_argument("--manifest", default=None, metavar="JSON",
                   help="Train signal JSON manifest. Inferred from --dry if omitted.")
    p.add_argument("--wet", action="append", default=[], metavar="WAV[:LABEL]",
                   help="Wet capture WAV, optionally followed by :Label. Repeat for multiple.")
    p.add_argument("--output", default=None, metavar="DIR",
                   help="Output directory. Defaults to outputs/<dry_stem>_analysis/.")
    return p.parse_args(argv)


def _infer_manifest(dry_path: Path) -> Path:
    candidate = dry_path.with_suffix(".json")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find manifest at {candidate}. Pass --manifest explicitly."
    )


# ─── Audio loading ────────────────────────────────────────────────────────────


def _load_signals(
    dry_path: Path,
    wet_paths: list[Path],
) -> tuple[dict[str, np.ndarray], int]:
    """Load all files, resampling dry to match wet SR if they differ.

    Returns:
        sigs: dict key → float32 mono array, all at the same sample rate.
        sr: common sample rate in Hz.
    """
    dry_raw, sr_dry = sf.read(str(dry_path), dtype="float32")
    dry_raw = dry_raw[:, 0] if dry_raw.ndim > 1 else dry_raw

    wet_data: list[tuple[np.ndarray, int]] = []
    for wp in wet_paths:
        w, sr_w = sf.read(str(wp), dtype="float32")
        wet_data.append((w[:, 0] if w.ndim > 1 else w, sr_w))

    sr_wet_values = {sr_w for _, sr_w in wet_data}
    if len(sr_wet_values) > 1:
        raise ValueError(f"Wet files have mismatched sample rates: {sr_wet_values}")
    sr = next(iter(sr_wet_values)) if wet_data else sr_dry

    if sr_dry != sr:
        from math import gcd
        g = gcd(sr_dry, sr)
        dry_raw = resample_poly(dry_raw, sr // g, sr_dry // g).astype(np.float32)
        print(f"[load] Resampled dry {sr_dry} → {sr} Hz")

    sigs: dict[str, np.ndarray] = {"dry": dry_raw}
    for i, (w, _) in enumerate(wet_data):
        sigs[f"wet_{i}"] = w
    return sigs, sr


# ─── Alignment ───────────────────────────────────────────────────────────────


def _find_lag(dry: np.ndarray, wet: np.ndarray, sr: int) -> int:
    """Return samples to drop from wet start to align with dry.

    Uses the signal region just after the 0.5 s leading silence.
    Positive → wet starts later than dry. Negative → wet starts earlier.
    """
    offset = int(0.5 * sr)
    n = min(int(2.0 * sr), len(dry) - offset, len(wet) - offset)
    if n <= 0:
        return 0
    chunk = dry[offset: offset + n].astype(np.float64)
    search = wet[: offset + n + int(0.1 * sr)].astype(np.float64)
    xcorr = correlate(search, chunk, mode="valid")
    best = int(np.argmax(np.abs(xcorr)))
    return best - offset


def _align_and_trim(
    sigs: dict[str, np.ndarray],
    wet_keys: list[str],
    sr: int,
) -> tuple[dict[str, np.ndarray], float]:
    """Align each wet file to dry independently; trim all to shortest length."""
    out: dict[str, np.ndarray] = {"dry": sigs["dry"]}
    for key in wet_keys:
        lag = _find_lag(sigs["dry"], sigs[key], sr)
        w = sigs[key]
        out[key] = w[lag:] if lag >= 0 else np.pad(w, (-lag, 0))
        print(f"[align] {key}: lag = {lag} samples ({lag / sr * 1000:.2f} ms)")
    n_out = min(len(v) for v in out.values())
    trimmed = {k: v[:n_out] for k, v in out.items()}
    dur = n_out / sr
    print(f"[align] Common length: {dur:.1f} s")
    return trimmed, dur


# ─── DSP helpers ─────────────────────────────────────────────────────────────


def _db(x: float, floor: float = -120.0) -> float:
    return max(20.0 * np.log10(max(float(x), 1e-10)), floor)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _section_slice(sec: dict, sr: int) -> slice:
    return slice(int(sec["start_s"] * sr), int(sec["end_s"] * sr))


def _smooth_log_freq(
    freqs: np.ndarray,
    vals: np.ndarray,
    frac: float = 1 / 6,
    n_out: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """1/6-octave smoothing in log-frequency space."""
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


def _compute_thd_pct(seg: np.ndarray, f0: float, sr: int, n_harm: int = 12) -> float:
    """THD (%) of seg at fundamental f0, harmonics 2..n_harm."""
    win = np.hanning(len(seg))
    mag = np.abs(np.fft.rfft(seg * win))
    fres = sr / len(seg)

    def _peak(f_target: float) -> float:
        idx = np.clip(int(round(f_target / fres)), 0, len(mag) - 1)
        return float(np.max(mag[max(0, idx - 5): idx + 6]))

    fund = _peak(f0)
    if fund < 1e-10:
        return 0.0
    return 100.0 * np.sqrt(sum(_peak(f0 * n) ** 2 for n in range(2, n_harm + 1))) / fund


# ─── Stepped-tone helpers ─────────────────────────────────────────────────────


def _grouped_stepped(sections: list[dict]) -> dict[float, list[dict]]:
    by_freq: dict[float, list[dict]] = {}
    for s in sections:
        if s["type"] == "stepped_sine_tone":
            f = float(s["params"]["freq_hz"])
            by_freq.setdefault(f, []).append(s)
    return {
        f: sorted(v, key=lambda s: s["amplitude_dbfs"])
        for f, v in sorted(by_freq.items())
    }


def _captured_freqs(by_freq: dict[float, list[dict]], dur: float) -> list[float]:
    return [f for f, secs in by_freq.items() if secs[0]["start_s"] < dur]


# ─── Plot 1: Section timeline ─────────────────────────────────────────────────


def _plot_timeline(sections: list[dict], dur: float, plot_dir: Path) -> Path:
    type_colors = {
        "log_sine_sweep":    "#1A237E",
        "stepped_sine_tone": "#1B5E20",
        "pink_noise":        "#E65100",
        "white_noise":       "#BF360C",
        "amplitude_ramp":    "#4A148C",
        "impulse_train":     "#880E4F",
        "am_chirp":          "#006064",
        "decaying_plucks":   "#3E2723",
        "two_tone_sweep":    "#33691E",
        "multitone":         "#0D47A1",
        "layered":           "#37474F",
    }
    fig, ax = plt.subplots(figsize=(16, 3))
    seen: set[str] = set()
    for sec in sections:
        t = sec["type"]
        label = t.replace("_", " ") if t not in seen else None
        seen.add(t)
        ax.barh(0, sec["end_s"] - sec["start_s"], left=sec["start_s"],
                height=0.65, color=type_colors.get(t, "#999999"), alpha=0.9, label=label)
    ax.axvline(dur, color="#C62828", lw=2.5, ls="--", label=f"Capture end ({dur:.0f} s)")
    ax.set_xlim(0, sections[-1]["end_s"])
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_yticks([])
    ax.set_title("Signal sections — red dashed = capture end", fontsize=12, fontweight="bold")
    ax.legend(ncol=6, fontsize=8.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out = plot_dir / "01_overview_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Plot 2: Sweep frequency response ────────────────────────────────────────


def _plot_sweep_response(
    sigs: dict[str, np.ndarray],
    sections: list[dict],
    wet_keys: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    sr: int,
    plot_dir: Path,
) -> Path:
    sweep_secs = [s for s in sections if s["type"] == "log_sine_sweep"]
    n_sweeps = len(sweep_secs)
    fig, axes = plt.subplots(1, n_sweeps, figsize=(6 * n_sweeps, 5),
                             sharey=True, squeeze=False)
    nyq = sr / 2
    for ax, sec in zip(axes[0], sweep_secs):
        sl = _section_slice(sec, sr)
        dry_seg = sigs["dry"][sl].astype(np.float64)
        freqs = np.fft.rfftfreq(len(dry_seg), 1.0 / sr)
        dry_mag = np.abs(np.fft.rfft(dry_seg)) + 1e-15
        for key in wet_keys:
            wet_mag = np.abs(np.fft.rfft(sigs[key][sl].astype(np.float64))) + 1e-15
            ratio_db = 20.0 * np.log10(wet_mag / dry_mag)
            mask = (freqs >= 15.0) & (freqs <= nyq * 0.95)
            f_s, r_s = _smooth_log_freq(freqs[mask], ratio_db[mask])
            valid = ~np.isnan(r_s)
            ax.semilogx(f_s[valid], r_s[valid], color=colors[key], lw=1.8, label=labels[key])
        ax.axhline(0, color="k", lw=0.7, alpha=0.4)
        ax.set_title(f"Sweep {sec['amplitude_dbfs']:+.0f} dBFS", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(20, nyq)
        ax.set_ylim(-50, 25)
        ax.grid(True, which="both", alpha=0.25)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}" if x < 1000 else f"{int(x // 1000)}k")
        )
    axes[0, 0].set_ylabel("H(f)  [dB re. dry]")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Frequency response: Wet ÷ Dry  (log sine sweep sections)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "02_sweep_frequency_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Plot 3: Stepped-tone gain curves ────────────────────────────────────────


def _plot_stepped_gain_curves(
    sigs: dict[str, np.ndarray],
    sections: list[dict],
    wet_keys: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    sr: int,
    dur: float,
    plot_dir: Path,
) -> Path:
    by_freq = _grouped_stepped(sections)
    cap_freq = _captured_freqs(by_freq, dur)
    ncols = 3
    nrows = max(1, (len(cap_freq) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                             constrained_layout=True, squeeze=False)
    all_axes = list(axes.flat)
    for ax, freq in zip(all_axes, cap_freq):
        for key in wet_keys:
            in_db, out_db = [], []
            for s in by_freq[freq]:
                if s["start_s"] >= dur:
                    continue
                sl = _section_slice(s, sr)
                lo = (sl.stop - sl.start) * 3 // 10
                in_db.append(_db(_rms(sigs["dry"][sl][lo:])))
                out_db.append(_db(_rms(sigs[key][sl][lo:])))
            ax.plot(in_db, out_db, "o-", color=colors[key], lw=1.8, ms=6, label=labels[key])
        ax.set_title(f"{freq:.1f} Hz", fontweight="bold")
        ax.set_xlabel("Input RMS (dBFS)")
        ax.set_ylabel("Output RMS (dBFS)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5)
    for ax in all_axes[len(cap_freq):]:
        ax.set_visible(False)
    fig.suptitle("Stepped tones — output vs input level", fontsize=13, fontweight="bold")
    out = plot_dir / "03_stepped_gain_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 4: THD vs input level ──────────────────────────────────────────────


def _plot_stepped_thd(
    sigs: dict[str, np.ndarray],
    sections: list[dict],
    wet_keys: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    sr: int,
    dur: float,
    plot_dir: Path,
) -> Path:
    by_freq = _grouped_stepped(sections)
    cap_freq = _captured_freqs(by_freq, dur)
    ncols = 3
    nrows = max(1, (len(cap_freq) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                             constrained_layout=True, squeeze=False)
    all_axes = list(axes.flat)
    for ax, freq in zip(all_axes, cap_freq):
        for key in wet_keys:
            in_db, thd = [], []
            for s in by_freq[freq]:
                if s["start_s"] >= dur:
                    continue
                sl = _section_slice(s, sr)
                lo = (sl.stop - sl.start) * 3 // 10
                in_db.append(_db(_rms(sigs["dry"][sl][lo:])))
                thd.append(_compute_thd_pct(sigs[key][sl][lo:], freq, sr))
            ax.plot(in_db, thd, "o-", color=colors[key], lw=1.8, ms=6, label=labels[key])
        ax.set_title(f"{freq:.1f} Hz", fontweight="bold")
        ax.set_xlabel("Input RMS (dBFS)")
        ax.set_ylabel("THD (%)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5)
    for ax in all_axes[len(cap_freq):]:
        ax.set_visible(False)
    fig.suptitle("Stepped tones — THD vs input level", fontsize=13, fontweight="bold")
    out = plot_dir / "04_stepped_thd.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 5: Harmonic spectra ─────────────────────────────────────────────────


def _plot_harmonic_spectra(
    sigs: dict[str, np.ndarray],
    sections: list[dict],
    wet_keys: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    sr: int,
    dur: float,
    plot_dir: Path,
) -> Path:
    by_freq = _grouped_stepped(sections)
    want = [82.4, 110.0, 220.0, 330.0]
    to_plot = [f for f in want if f in by_freq
               and any(s["start_s"] < dur for s in by_freq[f])][:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True, squeeze=False)
    f_max = min(8000.0, sr / 2 * 0.9)
    for ax, freq in zip(axes.flat, to_plot):
        secs = [s for s in by_freq[freq] if s["start_s"] < dur]
        sec = secs[-1]
        sl = _section_slice(sec, sr)
        lo = (sl.stop - sl.start) * 3 // 10
        for key in ["dry"] + wet_keys:
            seg = sigs[key][sl][lo:]
            win = np.hanning(len(seg))
            mag = np.abs(np.fft.rfft(seg * win))
            mag_db = 20.0 * np.log10(mag / (len(seg) * 0.5) + 1e-10)
            freqs_ax = np.fft.rfftfreq(len(seg), 1.0 / sr)
            mask = freqs_ax <= f_max
            c = _DRY_COLOR if key == "dry" else colors[key]
            lbl = _DRY_LABEL if key == "dry" else labels[key]
            ax.plot(freqs_ax[mask], mag_db[mask], color=c,
                    lw=0.9 if key == "dry" else 1.5, alpha=0.75, label=lbl)
        for n_h in range(1, 12):
            f_h = freq * n_h
            if f_h > f_max:
                break
            ax.axvline(f_h, color="#FF6F00", lw=0.5, ls=":", alpha=0.5)
            ax.text(f_h, -18, f"×{n_h}", fontsize=6, color="#FF6F00", ha="center", va="top")
        ax.set_title(f"f₀ = {freq:.1f} Hz  @  {sec['amplitude_dbfs']:+.0f} dBFS",
                     fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Level (dBFS)")
        ax.set_xlim(0, f_max)
        ax.set_ylim(-110, 5)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(to_plot):]:
        ax.set_visible(False)
    fig.suptitle("Harmonic spectra — dry vs wet at peak stepped-tone level",
                 fontsize=13, fontweight="bold")
    out = plot_dir / "05_harmonic_spectra.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 6: Waveform snippets ────────────────────────────────────────────────


def _plot_stepped_waveforms(
    sigs: dict[str, np.ndarray],
    sections: list[dict],
    wet_keys: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    sr: int,
    dur: float,
    plot_dir: Path,
) -> Path:
    by_freq = _grouped_stepped(sections)
    want = [82.4, 220.0, 330.0, 440.0]
    to_plot = [f for f in want if f in by_freq
               and any(s["start_s"] < dur for s in by_freq[f])][:4]
    all_keys = ["dry"] + wet_keys
    n_rows, n_cols = len(to_plot), len(all_keys)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             constrained_layout=True, sharey="row", squeeze=False)
    n_snippet = int(0.05 * sr)
    t_ms = np.arange(n_snippet) / sr * 1000.0
    for row, freq in enumerate(to_plot):
        secs = [s for s in by_freq[freq] if s["start_s"] < dur]
        sec = secs[-1]
        sl = _section_slice(sec, sr)
        mid = (sl.stop - sl.start) // 2
        for col, key in enumerate(all_keys):
            ax = axes[row, col]
            seg = sigs[key][sl][mid: mid + n_snippet]
            c = _DRY_COLOR if key == "dry" else colors[key]
            lbl = _DRY_LABEL if key == "dry" else labels[key]
            ax.plot(t_ms, seg, color=c, lw=0.8)
            if row == 0:
                ax.set_title(lbl, fontweight="bold", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{freq:.0f} Hz\n{sec['amplitude_dbfs']:+.0f} dBFS", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Time (ms)")
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, 50)
    fig.suptitle("Stepped tone waveforms — 50 ms excerpt at peak input level",
                 fontsize=13, fontweight="bold")
    out = plot_dir / "06_stepped_waveforms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 7: Absolute magnitude spectra ──────────────────────────────────────


def _plot_absolute_spectra(
    sigs: dict[str, np.ndarray],
    sections: list[dict],
    wet_keys: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    sr: int,
    plot_dir: Path,
) -> Path:
    """Absolute FFT magnitude of every signal overlaid, one panel per sweep level."""
    sweep_secs = [s for s in sections if s["type"] == "log_sine_sweep"]
    n_sweeps = len(sweep_secs)
    fig, axes = plt.subplots(1, n_sweeps, figsize=(6 * n_sweeps, 5),
                             sharey=True, squeeze=False)
    nyq = sr / 2
    for ax, sec in zip(axes[0], sweep_secs):
        sl = _section_slice(sec, sr)
        n = sl.stop - sl.start
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mask = (freqs >= 15.0) & (freqs <= nyq * 0.95)

        dry_mag = np.abs(np.fft.rfft(sigs["dry"][sl].astype(np.float64)))
        dry_db = 20.0 * np.log10(dry_mag / n + 1e-15)
        f_s, v_s = _smooth_log_freq(freqs[mask], dry_db[mask])
        valid = ~np.isnan(v_s)
        ax.semilogx(f_s[valid], v_s[valid], color=_DRY_COLOR, lw=1.5, label=_DRY_LABEL)

        for key in wet_keys:
            wet_mag = np.abs(np.fft.rfft(sigs[key][sl].astype(np.float64)))
            wet_db = 20.0 * np.log10(wet_mag / n + 1e-15)
            f_s, v_s = _smooth_log_freq(freqs[mask], wet_db[mask])
            valid = ~np.isnan(v_s)
            ax.semilogx(f_s[valid], v_s[valid], color=colors[key], lw=1.5, label=labels[key])

        ax.set_title(f"Sweep {sec['amplitude_dbfs']:+.0f} dBFS", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(20, nyq)
        ax.grid(True, which="both", alpha=0.25)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}" if x < 1000 else f"{int(x // 1000)}k")
        )
    axes[0, 0].set_ylabel("Magnitude (dBFS, 1/6-oct smoothed)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Absolute magnitude spectra — all signals (log sine sweep sections)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "07_absolute_spectra.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Plot 8: Phase response ───────────────────────────────────────────────────


def _plot_sweep_phase(
    sigs: dict[str, np.ndarray],
    sections: list[dict],
    wet_keys: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    sr: int,
    plot_dir: Path,
) -> Path:
    """Unwrapped phase of H(f) = Wet / Dry for each sweep section."""
    sweep_secs = [s for s in sections if s["type"] == "log_sine_sweep"]
    n_sweeps = len(sweep_secs)
    fig, axes = plt.subplots(1, n_sweeps, figsize=(6 * n_sweeps, 5),
                             sharey=True, squeeze=False)
    nyq = sr / 2
    for ax, sec in zip(axes[0], sweep_secs):
        sl = _section_slice(sec, sr)
        n = sl.stop - sl.start
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        # Restrict to where the sweep has meaningful energy and avoid Nyquist noise
        mask = (freqs >= 15.0) & (freqs <= nyq * 0.95)

        dry_fft = np.fft.rfft(sigs["dry"][sl].astype(np.float64))
        dry_mag = np.abs(dry_fft)
        # Build a signal-presence mask: bins where dry is at least -60 dB of peak
        peak = np.max(dry_mag)
        snr_mask = dry_mag >= peak * 10 ** (-60 / 20)

        for key in wet_keys:
            wet_fft = np.fft.rfft(sigs[key][sl].astype(np.float64))
            H = wet_fft / (dry_fft + 1e-15)
            phase_raw = np.angle(H)  # radians, wrapped
            phase_unwrapped = np.unwrap(phase_raw)  # radians, continuous
            phase_deg = np.degrees(phase_unwrapped)
            combined = mask & snr_mask
            ax.semilogx(freqs[combined], phase_deg[combined],
                        color=colors[key], lw=1.5, label=labels[key])

        ax.axhline(0, color="k", lw=0.7, alpha=0.4)
        ax.set_title(f"Sweep {sec['amplitude_dbfs']:+.0f} dBFS", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(20, nyq)
        ax.grid(True, which="both", alpha=0.25)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}" if x < 1000 else f"{int(x // 1000)}k")
        )
    axes[0, 0].set_ylabel("Phase [degrees]")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Phase response: ∠(Wet / Dry)  (log sine sweep sections)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "08_phase_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    dry_path = Path(args.dry)
    if not dry_path.exists():
        raise FileNotFoundError(f"Dry file not found: {dry_path}")

    manifest_path = Path(args.manifest) if args.manifest else _infer_manifest(dry_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    wet_paths: list[Path] = []
    wet_labels: list[str] = []
    for entry in args.wet:
        if ":" in entry:
            p_str, lbl = entry.rsplit(":", 1)
        else:
            p_str, lbl = entry, Path(entry).stem
        wet_paths.append(Path(p_str))
        wet_labels.append(lbl)

    if not wet_paths:
        raise ValueError("Provide at least one --wet file.")

    for wp in wet_paths:
        if not wp.exists():
            raise FileNotFoundError(f"Wet file not found: {wp}")

    out_dir = (Path(args.output) if args.output
               else Path("outputs") / f"{dry_path.stem}_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    wet_keys = [f"wet_{i}" for i in range(len(wet_paths))]
    colors = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(wet_keys)}
    labels = dict(zip(wet_keys, wet_labels))

    print("[load] Reading audio files …")
    sigs, sr = _load_signals(dry_path, wet_paths)
    for k, v in sigs.items():
        lbl = _DRY_LABEL if k == "dry" else labels[k]
        print(f"  {lbl:<22} {len(v) / sr:.1f} s  ({len(v)} samples @ {sr} Hz)")

    sigs, dur = _align_and_trim(sigs, wet_keys, sr)

    with open(manifest_path) as fh:
        meta = json.load(fh)
    sections = meta["sections"]

    captured = [s for s in sections if s["start_s"] < dur]
    types_captured = {s["type"] for s in captured}
    print(f"\n[info] Capture covers {dur:.1f} s → "
          f"{len(captured)}/{len(sections)} sections, {len(types_captured)} section types")

    print(f"\n[plot] Writing to {out_dir}/")
    saved: list[Path] = []
    saved.append(_plot_timeline(sections, dur, out_dir))
    saved.append(_plot_sweep_response(sigs, sections, wet_keys, colors, labels, sr, out_dir))
    saved.append(_plot_stepped_gain_curves(sigs, sections, wet_keys, colors, labels, sr, dur, out_dir))
    saved.append(_plot_stepped_thd(sigs, sections, wet_keys, colors, labels, sr, dur, out_dir))
    saved.append(_plot_harmonic_spectra(sigs, sections, wet_keys, colors, labels, sr, dur, out_dir))
    saved.append(_plot_stepped_waveforms(sigs, sections, wet_keys, colors, labels, sr, dur, out_dir))
    saved.append(_plot_absolute_spectra(sigs, sections, wet_keys, colors, labels, sr, out_dir))
    saved.append(_plot_sweep_phase(sigs, sections, wet_keys, colors, labels, sr, out_dir))

    print(f"[done] {len(saved)} plots saved:")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
