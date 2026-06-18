#!/usr/bin/env python3
"""Analyse a single GE-7 preset hardware capture against bypass and expected model.

Produces seven plots in outputs/ge7_captures/<preset>/ and a summary JSON.

Usage:
    uv run python scripts/analyze_ge7_capture.py --preset preset_smile
    uv run python scripts/analyze_ge7_capture.py --preset preset_frown
"""

import argparse
import json
import sys
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import soundfile as sf
from scipy.signal import correlate, freqz, resample_poly

ROOT = Path(__file__).resolve().parent.parent

DRY_WAV      = ROOT / "data/signals/train_signal_v2.wav"
DRY_MANIFEST = ROOT / "data/signals/train_signal_v2.json"
DRY_CACHE    = ROOT / "data/signals/train_signal_v2_48k.npy"  # cached downsample
BYPASS_WAV   = ROOT / "data/signals/train_signal_v2_reamp only.wav"
FLAT_WAV     = ROOT / "data/ge7/captures/preset_flat/GE7-Flat2 - flat 2.wav"
CAPTURES_DIR = ROOT / "data/ge7/captures"
PRESETS_DIR  = ROOT / "data/ge7/presets"
OUT_BASE     = ROOT / "outputs/ge7_captures"

SR_WET = 48_000  # all wet captures at 48 kHz
RATIO  = 0.5     # dry is 96 kHz

# GE-7 band centres and their slider keys
GE7_BANDS = [
    (100,  "hz_100"),
    (200,  "hz_200"),
    (400,  "hz_400"),
    (800,  "hz_800"),
    (1600, "hz_1600"),
    (3200, "hz_3200"),
    (6400, "hz_6400"),
]

BLUE   = "#2196F3"
RED    = "#F44336"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
GREY   = "#9E9E9E"


# ─── helpers ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", required=True, help="Preset dir name, e.g. preset_smile")
    return p.parse_args()


def _load_dry() -> tuple[np.ndarray, np.ndarray]:
    """Load dry signal and its 48 kHz downsampled version, using a cached .npy if available."""
    dry, sr_dry = sf.read(DRY_WAV, always_2d=False)
    if DRY_CACHE.exists():
        dry_ds = np.load(DRY_CACHE)
    else:
        print("  Resampling dry signal (first run, will be cached)…")
        g      = gcd(SR_WET, int(sr_dry))
        dry_ds = resample_poly(dry, SR_WET // g, int(sr_dry) // g).astype(np.float32)
        np.save(DRY_CACHE, dry_ds)
        print(f"  Cached to {DRY_CACHE}")
    return dry, dry_ds


def _load_all(preset: str):
    wet_path = CAPTURES_DIR / preset / "wet_train_signal_v2.wav"
    if not wet_path.exists():
        sys.exit(f"Capture not found: {wet_path}")

    dry, dry_ds = _load_dry()
    bypass, _ = sf.read(BYPASS_WAV, always_2d=False)
    flat2,  _ = sf.read(FLAT_WAV,   always_2d=False)
    wet,    _ = sf.read(wet_path,   always_2d=False)

    with open(DRY_MANIFEST) as f:
        manifest = json.load(f)
    with open(PRESETS_DIR / f"{preset}.json") as f:
        preset_def = json.load(f)

    return dry, dry_ds, bypass, flat2, wet, manifest, preset_def


def _find_lag(dry_ds: np.ndarray, wet: np.ndarray) -> int:
    """Cross-correlate a 2-second chunk to find sample lag (wet relative to dry_ds)."""
    offset = int(0.5 * SR_WET)
    n      = min(int(2 * SR_WET), len(dry_ds) - offset, len(wet) - offset)
    chunk  = dry_ds[offset: offset + n].astype(np.float64)
    search = wet[: offset + n + int(0.1 * SR_WET)].astype(np.float64)
    best   = int(np.argmax(np.abs(correlate(search, chunk, mode="valid"))))
    return best - offset


def _wet_window(sig: np.ndarray, lag: int, sec: dict) -> np.ndarray:
    a = lag + int(sec["start_sample"] * RATIO)
    b = lag + int(sec["end_sample"]   * RATIO)
    b = min(b, len(sig))
    return sig[max(0, a): b]


def _smooth_log(freqs: np.ndarray, db: np.ndarray, frac: float = 1 / 6) -> np.ndarray:
    """1/N-octave smoothing in log-frequency space.

    Vectorised: converts to log2 then applies a sliding uniform-window average
    via cumsum, which is O(n) rather than the naive O(n²) per-bin loop.
    """
    if len(freqs) == 0:
        return db.copy()
    log_f  = np.log2(np.maximum(freqs, 1e-9))
    half_w = frac / 2.0          # half-window in octaves
    out    = np.empty_like(db)
    # cumsum trick: for a sorted log-freq array we binary-search window edges
    cs     = np.concatenate(([0.0], np.cumsum(db)))
    cnt    = np.arange(1, len(db) + 1, dtype=np.float64)
    cs_cnt = np.concatenate(([0], cnt))
    lo_idx = np.searchsorted(log_f, log_f - half_w, side="left")
    hi_idx = np.searchsorted(log_f, log_f + half_w, side="right")
    n_bins = hi_idx - lo_idx
    out    = (cs[hi_idx] - cs[lo_idx]) / np.maximum(n_bins, 1)
    return out


def _expected_fr(sliders: dict, n: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """Expected magnitude response (dB) from slider settings using peaking EQ biquads."""
    Q  = 1.4
    H  = np.ones(n // 2 + 1, dtype=np.complex128)
    for fc, key in GE7_BANDS:
        gain_db = sliders.get(key, 0.0)
        if abs(gain_db) < 0.01:
            continue
        A      = 10 ** (gain_db / 40)
        w0     = 2 * np.pi * fc / SR_WET
        alpha  = np.sin(w0) / (2 * Q)
        b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
        _, h = freqz(b / a[0], a / a[0], worN=n // 2 + 1, fs=SR_WET)
        H *= h
    level_db = sliders.get("level", 0.0)
    H       *= 10 ** (level_db / 20)
    f_axis   = np.fft.rfftfreq(n, 1 / SR_WET)
    return f_axis, 20 * np.log10(np.abs(H) + 1e-15)


def _sweep_transfer(dry_seg: np.ndarray, wet_seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H(f) via FFT ratio of matching-length segments."""
    n   = min(len(dry_seg), len(wet_seg))
    D   = np.fft.rfft(dry_seg[:n])
    W   = np.fft.rfft(wet_seg[:n])
    H   = W / (D + 1e-30)
    f   = np.fft.rfftfreq(n, 1 / SR_WET)
    return f, H


# ─── plot functions ───────────────────────────────────────────────────────────

def _plot_lag_correlation(dry_ds, bypass, flat2, wet, lags, out_dir: Path, preset_name: str) -> Path:
    """01: Cross-correlation curves showing alignment for all three captures."""
    offset = int(0.5 * SR_WET)
    n      = int(2 * SR_WET)
    chunk  = dry_ds[offset: offset + n].astype(np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), squeeze=False)
    fig.suptitle(f"Lag / Alignment — {preset_name}", fontsize=13, fontweight="bold")

    signals   = [bypass, flat2, wet]
    names     = ["Bypass (reamp only)", "Flat (GE-7 flat)", f"{preset_name} (GE-7)"]
    colors    = [GREY, GREEN, BLUE]
    lag_vals  = [lags["bypass"], lags["flat2"], lags["wet"]]

    for ax, sig, name, color, lag in zip(axes[:, 0], signals, names, colors, lag_vals):
        search = sig[: offset + n + int(0.1 * SR_WET)].astype(np.float64)
        xcorr  = correlate(search, chunk, mode="valid")
        t_ms   = (np.arange(len(xcorr)) - offset) * 1000 / SR_WET

        ax.plot(t_ms, xcorr / np.max(np.abs(xcorr)), color=color, lw=0.8)
        ax.axvline(lag * 1000 / SR_WET, color="red", lw=1.5, ls="--",
                   label=f"lag = {lag} smp  ({lag * 1000 / SR_WET:.2f} ms)")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Norm. xcorr")
        ax.legend(fontsize=9)
        ax.set_xlim(-5, 5)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Lag (ms)")
    fig.tight_layout()
    out = out_dir / "01_lag_correlation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_frequency_response(dry_ds, bypass, flat2, wet, lags, manifest, sliders, out_dir: Path, preset_name: str) -> Path:
    """02: EQ frequency response — preset / flat2 (both at same playback level, offset cancels)."""
    sweep = next(s for s in manifest["sections"] if s["label"] == "sweep_mid")

    def _seg(sig, lag):
        a = lag + int(sweep["start_sample"] * RATIO)
        b = lag + int(sweep["end_sample"]   * RATIO)
        return sig[max(0, a): min(b, len(sig))]

    dry_seg  = _seg(dry_ds, 0)
    bp_seg   = _seg(bypass, lags["bypass"])
    fl_seg   = _seg(flat2,  lags["flat2"])
    wet_seg  = _seg(wet,    lags["wet"])

    f_bp,  H_bp  = _sweep_transfer(dry_seg, bp_seg)
    f_fl,  H_fl  = _sweep_transfer(dry_seg, fl_seg)
    f_wet, H_wet = _sweep_transfer(dry_seg, wet_seg)

    # Primary: preset relative to flat2 (same playback level → offset cancels → true EQ curve)
    H_ge7_true = H_wet / (H_fl + 1e-30)
    # Secondary: bypass relative to flat2 (shows reamp frequency tilt for reference)
    H_bp_vs_fl = H_bp  / (H_fl + 1e-30)

    mask   = (f_wet > 20) & (f_wet < 22000)
    f_plot = f_wet[mask]

    ge7_db    = _smooth_log(f_plot, 20 * np.log10(np.abs(H_ge7_true[mask]) + 1e-15))
    bp_fl_db  = _smooth_log(f_plot, 20 * np.log10(np.abs(H_bp_vs_fl[mask]) + 1e-15))

    f_exp, exp_db = _expected_fr(sliders)
    exp_mask = (f_exp > 20) & (f_exp < 22000)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.plot(f_plot,          bp_fl_db,           color=GREY,   lw=1.2, alpha=0.7, ls="--", label="Bypass vs flat (reamp tilt)")
    ax.plot(f_exp[exp_mask], exp_db[exp_mask],   color=ORANGE, lw=2.0, ls="--",            label="Expected (Ge7Sim model)")
    ax.plot(f_plot,          ge7_db,             color=BLUE,   lw=2.0,                     label=f"Measured — {preset_name}")

    for fc, _ in GE7_BANDS:
        ax.axvline(fc, color=GREY, lw=0.6, ls=":")
        ax.annotate(f"{fc:,}", xy=(fc, -19), fontsize=7, ha="center", va="bottom", color=GREY)

    ax.set_xscale("log")
    ax.set_xlim(20, 22000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain relative to flat preset (dB)")
    ax.set_title(f"Frequency Response — {preset_name}  (1/6-oct smoothed, sweep_mid)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-20, 20)
    fig.tight_layout()
    out = out_dir / "02_frequency_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_linearity(dry, dry_ds, bypass, flat2, wet, lags, manifest, out_dir: Path, preset_name: str) -> Path:
    """03: GE-7 gain (preset / flat2) vs amplitude level — flat means linear, slope means saturation."""
    tones      = [s for s in manifest["sections"] if s["type"] == "stepped_sine_tone"]
    levels     = sorted({s["amplitude_dbfs"] for s in tones})
    tone_freqs = sorted({int(s["label"].split("_")[2][1:]) for s in tones})

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    fig.suptitle(
        f"Linearity Check — {preset_name}  (GE-7 gain = preset/flat, should be flat across levels)",
        fontsize=13, fontweight="bold"
    )
    axes_flat = list(axes.flat)

    for idx, freq in enumerate(tone_freqs):
        ax = axes_flat[idx]
        lvls, g_rel = [], []

        for level in levels:
            sec = next((s for s in tones if int(s["label"].split("_")[2][1:]) == freq
                        and s["amplitude_dbfs"] == level), None)
            if sec is None:
                continue
            ds = sec["start_sample"]; de = sec["end_sample"]
            d_pk  = float(np.max(np.abs(dry[ds:de])))
            fl_pk = float(np.max(np.abs(flat2[lags["flat2"] + int(ds*RATIO): lags["flat2"] + int(de*RATIO)])))
            wt_pk = float(np.max(np.abs(wet  [lags["wet"]   + int(ds*RATIO): lags["wet"]   + int(de*RATIO)])))
            if d_pk > 0 and fl_pk > 0:
                g_fl  = 20 * np.log10(fl_pk / d_pk + 1e-12)
                g_wet = 20 * np.log10(wt_pk / d_pk + 1e-12)
                g_rel.append(g_wet - g_fl)   # preset minus flat → pure GE-7 EQ change
                lvls.append(level)

        ax.axhline(0, color=GREEN, lw=1.0, ls="--", alpha=0.6)
        ax.plot(lvls, g_rel, "o-", color=BLUE, lw=1.8, ms=5)
        valid = g_rel[1:] if len(g_rel) > 1 else g_rel  # skip -24 (noisy)
        spread = max(valid) - min(valid) if valid else 0
        ax.set_title(f"{freq} Hz  (spread={spread:.1f} dB)", fontsize=9)
        ax.set_xlabel("Input level (dBFS)", fontsize=8)
        ax.set_ylabel("Preset − flat (dB)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_ylim(-20, 20)

    for ax in axes_flat[len(tone_freqs):]:
        ax.set_visible(False)

    fig.tight_layout()
    out = out_dir / "03_linearity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_gain_by_freq(dry, bypass, flat2, wet, lags, manifest, sliders, out_dir: Path, preset_name: str) -> Path:
    """04: Gain vs frequency from stepped tones at -18 dBFS, compared to expected."""
    tones  = [s for s in manifest["sections"] if s["type"] == "stepped_sine_tone"
              and s["amplitude_dbfs"] == -18]
    tone_freqs = sorted({int(s["label"].split("_")[2][1:]) for s in tones})

    f_meas, g_bp, g_fl, g_wet = [], [], [], []
    for freq in tone_freqs:
        sec = next((s for s in tones if int(s["label"].split("_")[2][1:]) == freq), None)
        if sec is None:
            continue
        ds = sec["start_sample"]; de = sec["end_sample"]
        d_pk  = float(np.max(np.abs(dry[ds:de])))
        bp_pk = float(np.max(np.abs(bypass[lags["bypass"] + int(ds*RATIO): lags["bypass"] + int(de*RATIO)])))
        fl_pk = float(np.max(np.abs(flat2 [lags["flat2"]  + int(ds*RATIO): lags["flat2"]  + int(de*RATIO)])))
        wt_pk = float(np.max(np.abs(wet   [lags["wet"]    + int(ds*RATIO): lags["wet"]    + int(de*RATIO)])))
        f_meas.append(freq)
        g_bp.append(20 * np.log10(bp_pk / d_pk + 1e-12))
        g_fl.append(20 * np.log10(fl_pk / d_pk + 1e-12))
        g_wet.append(20 * np.log10(wt_pk / d_pk + 1e-12))

    # GE-7 response: preset / flat2 (both at same playback level — offset cancels)
    ge7_db  = [w - f for w, f in zip(g_wet, g_fl)]

    # Expected from sliders (Ge7Sim absolute response ≈ preset vs flat since flat ≈ 0 dB in model)
    f_exp, exp_db_arr = _expected_fr(sliders, n=16384)
    exp_at_tones = [float(exp_db_arr[np.argmin(np.abs(f_exp - fr))]) for fr in f_meas]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.plot(f_meas, exp_at_tones, "s--", color=ORANGE, lw=1.5, ms=7, label="Expected (Ge7Sim)")
    ax.plot(f_meas, ge7_db,       "o-",  color=BLUE,   lw=2.0, ms=9, label=f"Measured — {preset_name}")

    # Error: measured vs expected
    errors = [m - e for m, e in zip(ge7_db, exp_at_tones)]
    for f, g, e in zip(f_meas, ge7_db, errors):
        ax.annotate(f"{e:+.1f}", xy=(f, g), xytext=(0, 8), textcoords="offset points",
                    fontsize=8, ha="center", color=RED)

    ax.set_xscale("log")
    ax.set_xticks(f_meas)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain relative to flat preset (dB)")
    ax.set_title(f"Stepped-Tone Gain @ −18 dBFS — {preset_name}  (red = measured − expected)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-20, 20)
    fig.tight_layout()
    out = out_dir / "04_gain_by_frequency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_waveform_sections(dry, bypass, wet, lags, manifest, out_dir: Path, preset_name: str) -> Path:
    """05: Waveform comparison dry vs wet for representative stepped-tone sections."""
    target_freqs  = [82, 440, 10560]
    target_levels = [-18, -12]

    pairs = []
    for freq in target_freqs:
        for level in target_levels:
            sec = next((s for s in manifest["sections"]
                        if s["type"] == "stepped_sine_tone"
                        and int(s["label"].split("_")[2][1:]) == freq
                        and s["amplitude_dbfs"] == level), None)
            if sec:
                pairs.append((sec, freq, level))

    n_panels = len(pairs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels), squeeze=False)
    fig.suptitle(f"Waveform Comparison — {preset_name}", fontsize=13, fontweight="bold")

    show_s = 0.05  # show 50 ms

    for ax, (sec, freq, level) in zip(axes[:, 0], pairs):
        ds = sec["start_sample"]; de = sec["end_sample"]

        # Dry (downsampled 2x for display at same rate)
        dry_ds_disp = dry[ds::2]
        n_show = int(show_s * SR_WET)
        t = np.arange(n_show) / SR_WET * 1000

        dry_win = dry_ds_disp[:n_show]
        a_wet   = lags["wet"]    + int(ds * RATIO)
        a_bp    = lags["bypass"] + int(ds * RATIO)
        wet_win = wet   [a_wet: a_wet + n_show]
        bp_win  = bypass[a_bp:  a_bp  + n_show]

        ax.plot(t, dry_win[:len(t)], color=GREY,  lw=0.8, alpha=0.7, label="Dry")
        ax.plot(t, bp_win[:len(t)],  color=GREEN, lw=0.9, alpha=0.8, label="Bypass")
        ax.plot(t, wet_win[:len(t)], color=BLUE,  lw=1.2,            label=preset_name)
        ax.set_title(f"{freq} Hz  @ {level} dBFS", fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        if ax == axes[0, 0]:
            ax.legend(fontsize=9, loc="upper right")

    axes[-1, 0].set_xlabel("Time (ms)")
    fig.tight_layout()
    out = out_dir / "05_waveform_sections.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_phase_response(dry_ds, bypass, wet, lags, manifest, out_dir: Path, preset_name: str) -> Path:
    """06: Unwrapped phase of GE-7 relative to bypass (removes reamp phase contribution)."""
    sweep = next(s for s in manifest["sections"] if s["label"] == "sweep_mid")

    def _seg(sig, lag):
        a = lag + int(sweep["start_sample"] * RATIO)
        b = lag + int(sweep["end_sample"]   * RATIO)
        return sig[max(0, a): min(b, len(sig))]

    dry_seg = _seg(dry_ds, 0)
    bp_seg  = _seg(bypass, lags["bypass"])
    wet_seg = _seg(wet,    lags["wet"])

    f_bp,  H_bp  = _sweep_transfer(dry_seg, bp_seg)
    f_wet, H_wet = _sweep_transfer(dry_seg, wet_seg)

    H_rel   = H_wet / (H_bp + 1e-30)
    phase   = np.unwrap(np.angle(H_rel)) * 180 / np.pi

    # SNR mask: only plot bins where dry has decent energy
    dry_fft   = np.abs(np.fft.rfft(dry_seg[:min(len(dry_seg), len(bp_seg))]))
    snr_mask  = dry_fft > dry_fft.max() * 10 ** (-50 / 20)
    freq_mask = (f_wet > 30) & (f_wet < 20000) & snr_mask

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(f_wet[freq_mask], _smooth_log(f_wet[freq_mask], phase[freq_mask], frac=1/4),
            color=BLUE, lw=1.8, label=f"{preset_name} relative to bypass")
    ax.axhline(0, color="black", lw=0.7, ls=":")

    for fc, _ in GE7_BANDS:
        ax.axvline(fc, color=GREY, lw=0.6, ls=":")

    ax.set_xscale("log")
    ax.set_xlim(30, 20000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase relative to bypass (degrees)")
    ax.set_title(f"Phase Response — {preset_name}  (GE-7 relative to bypass, 1/4-oct smoothed)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "06_phase_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_level_sweep(dry, bypass, flat2, wet, lags, manifest, out_dir: Path, preset_name: str) -> Path:
    """07: For each amplitude level, how does gain vary across frequency — linearity heat view."""
    tones  = [s for s in manifest["sections"] if s["type"] == "stepped_sine_tone"]
    levels = sorted({s["amplitude_dbfs"] for s in tones})
    tone_freqs = sorted({int(s["label"].split("_")[2][1:]) for s in tones})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    fig.suptitle(f"Gain vs Frequency at Each Amplitude Level — {preset_name}", fontsize=13, fontweight="bold")

    cmap = plt.cm.plasma
    colors = [cmap(i / (len(levels) - 1)) for i in range(len(levels))]

    for ax_idx, (sig, lag, title) in enumerate([
        (wet,   lags["wet"],   preset_name),
        (flat2, lags["flat2"], "Flat (reference)"),
    ]):
        ax = axes[0, ax_idx]
        for level, color in zip(levels, colors):
            f_pts, g_pts = [], []
            for freq in tone_freqs:
                sec = next((s for s in tones if int(s["label"].split("_")[2][1:]) == freq
                            and s["amplitude_dbfs"] == level), None)
                if sec is None:
                    continue
                ds = sec["start_sample"]; de = sec["end_sample"]
                d_pk  = float(np.max(np.abs(dry[ds:de])))
                bp_pk = float(np.max(np.abs(bypass[lags["bypass"] + int(ds*RATIO): lags["bypass"] + int(de*RATIO)])))
                wt_pk = float(np.max(np.abs(sig[lag + int(ds*RATIO): lag + int(de*RATIO)])))
                fl_pk = float(np.max(np.abs(flat2[lags["flat2"] + int(ds*RATIO): lags["flat2"] + int(de*RATIO)])))
                if d_pk > 0 and fl_pk > 0:
                    f_pts.append(freq)
                    # Use flat2 as reference (same playback level as preset; bypass saturated at high levels)
                    g_pts.append(20 * np.log10(wt_pk / d_pk + 1e-12) - 20 * np.log10(fl_pk / d_pk + 1e-12))

            ax.plot(f_pts, g_pts, "o-", color=color, lw=1.5, ms=5,
                    label=f"{level} dBFS")

        ax.axhline(0, color="black", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(tone_freqs)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain vs flat preset (dB)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-20, 20)

    fig.tight_layout()
    out = out_dir / "07_gain_level_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _build_summary(dry, bypass, flat2, wet, lags, manifest, sliders, preset_name: str) -> dict:
    """Compute key metrics for the summary JSON and results doc."""
    tones = [s for s in manifest["sections"] if s["type"] == "stepped_sine_tone"
             and s["amplitude_dbfs"] == -18]
    tone_freqs = sorted({int(s["label"].split("_")[2][1:]) for s in tones})

    measured, expected, errors = [], [], []
    f_exp, exp_db_arr = _expected_fr(sliders)

    for freq in tone_freqs:
        sec = next((s for s in tones if int(s["label"].split("_")[2][1:]) == freq), None)
        if sec is None:
            continue
        ds = sec["start_sample"]; de = sec["end_sample"]
        d_pk  = float(np.max(np.abs(dry[ds:de])))
        fl_pk = float(np.max(np.abs(flat2[lags["flat2"] + int(ds*RATIO): lags["flat2"] + int(de*RATIO)])))
        wt_pk = float(np.max(np.abs(wet  [lags["wet"]   + int(ds*RATIO): lags["wet"]   + int(de*RATIO)])))
        # Use flat2 as reference (same playback level → offset cancels → true GE-7 EQ response)
        m_db  = 20 * np.log10(wt_pk / d_pk + 1e-12) - 20 * np.log10(fl_pk / d_pk + 1e-12)
        e_db  = float(exp_db_arr[np.argmin(np.abs(f_exp - freq))])
        measured.append(m_db)
        expected.append(e_db)
        errors.append(m_db - e_db)

    # Linearity: spread of (preset/flat) gain across levels at 440 Hz
    linearity_sections = [s for s in manifest["sections"]
                          if s["type"] == "stepped_sine_tone"
                          and "f440" in s["label"]
                          and s["amplitude_dbfs"] in [-18, -12, -6, -3]]
    lin_gains = []
    for sec in linearity_sections:
        ds = sec["start_sample"]; de = sec["end_sample"]
        d_pk  = float(np.max(np.abs(dry[ds:de])))
        fl_pk = float(np.max(np.abs(flat2[lags["flat2"] + int(ds*RATIO): lags["flat2"] + int(de*RATIO)])))
        wt_pk = float(np.max(np.abs(wet  [lags["wet"]   + int(ds*RATIO): lags["wet"]   + int(de*RATIO)])))
        if d_pk > 0 and fl_pk > 0:
            lin_gains.append(
                20 * np.log10(wt_pk / d_pk + 1e-12) - 20 * np.log10(fl_pk / d_pk + 1e-12)
            )

    return {
        "preset":          preset_name,
        "lags":            lags,
        "tone_freqs_hz":   tone_freqs,
        "measured_ge7_db": [round(v, 2) for v in measured],
        "expected_ge7_db": [round(v, 2) for v in expected],
        "error_db":        [round(v, 2) for v in errors],
        "max_error_db":    round(float(np.max(np.abs(errors))), 2),
        "rms_error_db":    round(float(np.sqrt(np.mean(np.array(errors) ** 2))), 2),
        "linearity_spread_db": round(float(np.max(lin_gains) - np.min(lin_gains)), 2) if lin_gains else None,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = _parse_args()
    preset      = args.preset
    preset_name = preset.replace("preset_", "")

    REF_FADER_DB = -10.0  # flat2 was captured at this playback level

    print(f"Loading signals for {preset}...")
    dry, dry_ds, bypass, flat2, wet, manifest, preset_def = _load_all(preset)
    sliders = preset_def["sliders"]

    # Compensate for non-standard DAW playback level relative to flat2 reference
    preset_fader_db = float(preset_def.get("fader_db", REF_FADER_DB))
    fader_correction_db = REF_FADER_DB - preset_fader_db
    if abs(fader_correction_db) > 0.1:
        print(f"  Fader correction: {fader_correction_db:+.1f} dB "
              f"(preset at {preset_fader_db} dB, ref at {REF_FADER_DB} dB)")
        wet = wet * (10.0 ** (fader_correction_db / 20.0))

    print("Computing lags...")
    lags = {
        "bypass": _find_lag(dry_ds, bypass),
        "flat2":  _find_lag(dry_ds, flat2),
        "wet":    _find_lag(dry_ds, wet),
    }
    for k, v in lags.items():
        print(f"  {k}: {v} samples  ({v * 1000 / SR_WET:.2f} ms)")

    out_dir = OUT_BASE / preset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plots = [
        _plot_lag_correlation(dry_ds, bypass, flat2, wet, lags, out_dir, preset_name),
        _plot_frequency_response(dry_ds, bypass, flat2, wet, lags, manifest, sliders, out_dir, preset_name),
        _plot_linearity(dry, dry_ds, bypass, flat2, wet, lags, manifest, out_dir, preset_name),
        _plot_gain_by_freq(dry, bypass, flat2, wet, lags, manifest, sliders, out_dir, preset_name),
        _plot_waveform_sections(dry, bypass, wet, lags, manifest, out_dir, preset_name),
        _plot_phase_response(dry_ds, bypass, wet, lags, manifest, out_dir, preset_name),
        _plot_level_sweep(dry, bypass, flat2, wet, lags, manifest, out_dir, preset_name),
    ]

    print("Computing summary metrics...")
    summary = _build_summary(dry, bypass, flat2, wet, lags, manifest, sliders, preset_name)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs written to {out_dir}/")
    for p in plots:
        print(f"  {p.name}")
    print(f"  {summary_path.name}")
    print(f"\nKey metrics:")
    print(f"  Max freq-response error vs Ge7Sim: {summary['max_error_db']:.2f} dB")
    print(f"  RMS freq-response error:           {summary['rms_error_db']:.2f} dB")
    print(f"  Linearity spread (440 Hz, -18 to -3 dBFS): {summary['linearity_spread_db']:.2f} dB")


if __name__ == "__main__":
    main()
