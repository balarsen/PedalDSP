#!/usr/bin/env python3
"""Measure each GE-7 slider's actual centre frequency and bandwidth.

For each single-slider capture (one band at +10 dB, all others flat) computes
H(f) = slider_wet / flat2_wet and finds the true peak frequency.  Produces one
overlay plot comparing all 7 bands plus individual per-band plots.

Usage:
    uv run python scripts/analyze_single_sliders.py
"""

import json
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
DRY_CACHE    = ROOT / "data/signals/train_signal_v2_48k.npy"
FLAT_WAV     = ROOT / "data/ge7/captures/preset_flat/GE7-Flat2 - flat 2.wav"
SLIDER_DIR   = ROOT / "data/ge7/captures/single_slider"
OUT_DIR      = ROOT / "outputs/ge7_captures/single_slider"

SR_WET = 48_000
RATIO  = 0.5  # dry 96 kHz, wet 48 kHz

# Labelled centre freqs, slider filenames, plot colours
BANDS: list[tuple[int, str, str]] = [
    (100,  "slider_100hz.wav",  "#E53935"),  # red
    (200,  "slider_200hz.wav",  "#FF7043"),  # deep-orange
    (400,  "slider_400hz.wav",  "#FDD835"),  # yellow
    (800,  "slider_800hz.wav",  "#43A047"),  # green
    (1600, "slider_1600hz.wav", "#00ACC1"),  # cyan
    (3200, "slider_3200hz.wav", "#1E88E5"),  # blue
    (6400, "slider_6400hz.wav", "#8E24AA"),  # purple
]

SLIDER_GAIN_DB = 10.0  # all band-slider captures use +10 dB

# LEVEL-slider captures: (label, filename, gain_db, colour)
LEVEL_FILES: list[tuple[str, str, float, str]] = [
    ("LEVEL +10 dB", "slider_level_p10db.wav", +10.0, "#00BCD4"),
    ("LEVEL -10 dB", "slider_level_m10db.wav", -10.0, "#607D8B"),
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _find_lag(dry_ds: np.ndarray, wet: np.ndarray) -> int:
    """Cross-correlate a 2-second chunk; returns lag (negative = wet leads dry)."""
    offset = int(0.5 * SR_WET)
    n      = min(int(2 * SR_WET), len(dry_ds) - offset, len(wet) - offset)
    chunk  = dry_ds[offset: offset + n].astype(np.float64)
    search = wet[: offset + n + int(0.1 * SR_WET)].astype(np.float64)
    best   = int(np.argmax(np.abs(correlate(search, chunk, mode="valid"))))
    return best - offset


def _wet_window(sig: np.ndarray, lag: int, sec: dict) -> np.ndarray:
    a = lag + int(sec["start_sample"] * RATIO)
    b = lag + int(sec["end_sample"]   * RATIO)
    return sig[max(0, a): min(b, len(sig))]


def _smooth_log(freqs: np.ndarray, db: np.ndarray, frac: float = 1 / 6) -> np.ndarray:
    """1/N-octave smoothing — O(n log n) via binary-search cumsum."""
    if len(freqs) == 0:
        return db.copy()
    log_f  = np.log2(np.maximum(freqs, 1e-9))
    half_w = frac / 2.0
    cs     = np.concatenate(([0.0], np.cumsum(db)))
    lo_idx = np.searchsorted(log_f, log_f - half_w, side="left")
    hi_idx = np.searchsorted(log_f, log_f + half_w, side="right")
    return (cs[hi_idx] - cs[lo_idx]) / np.maximum(hi_idx - lo_idx, 1)


def _sweep_transfer(dry_seg: np.ndarray, wet_seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(dry_seg), len(wet_seg))
    D = np.fft.rfft(dry_seg[:n])
    W = np.fft.rfft(wet_seg[:n])
    return np.fft.rfftfreq(n, 1 / SR_WET), W / (D + 1e-30)


def _expected_peaking(fc_hz: int, gain_db: float, n: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """Ge7Sim peaking EQ biquad at Q=1.4 for a single band."""
    Q  = 1.4
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc_hz / SR_WET
    al = np.sin(w0) / (2 * Q)
    b  = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A])
    a  = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    f, h = freqz(b / a[0], a / a[0], worN=n // 2 + 1, fs=SR_WET)
    return f, 20 * np.log10(np.abs(h) + 1e-15)


def _find_peak(freqs: np.ndarray, db_smooth: np.ndarray,
               fc_nominal: int) -> tuple[float, float]:
    """Return (actual_peak_hz, peak_db) within ±1.5 octaves of the nominal centre."""
    lo = fc_nominal / (2 ** 1.5)
    hi = fc_nominal * (2 ** 1.5)
    m  = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return float(fc_nominal), float(db_smooth.max())
    idx = np.argmax(db_smooth[m])
    return float(freqs[m][idx]), float(db_smooth[m][idx])


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dry and flat2 (use cached 48k downsample if available)
    dry, sr_dry = sf.read(DRY_WAV, always_2d=False)
    if DRY_CACHE.exists():
        dry_ds = np.load(DRY_CACHE)
    else:
        print("Resampling dry signal (first run, will be cached)…")
        g      = gcd(SR_WET, int(sr_dry))
        dry_ds = resample_poly(dry, SR_WET // g, int(sr_dry) // g).astype(np.float32)
        np.save(DRY_CACHE, dry_ds)
    flat2, _ = sf.read(FLAT_WAV, always_2d=False)

    with open(DRY_MANIFEST) as f:
        manifest = json.load(f)

    # Use sweep_mid (-12 dBFS) — best SNR for +10 dB boost without clip
    sweep_mid = next(s for s in manifest["sections"] if s["label"] == "sweep_mid")

    lag_flat2 = _find_lag(dry_ds, flat2)
    fl_seg    = _wet_window(flat2, lag_flat2, sweep_mid)
    dry_seg   = dry_ds[int(sweep_mid["start_sample"] * RATIO): int(sweep_mid["end_sample"] * RATIO)]
    print(f"flat2 lag: {lag_flat2} samples")

    results: list[dict] = []

    for fc_nominal, fname, colour in BANDS:
        wav_path = SLIDER_DIR / fname
        if not wav_path.exists():
            print(f"MISSING: {wav_path}")
            continue
        wet, _ = sf.read(wav_path, always_2d=False)
        lag    = _find_lag(dry_ds, wet)
        print(f"  {fc_nominal:5d} Hz lag: {lag:+d} samples")
        sl_seg = _wet_window(wet, lag, sweep_mid)

        # H(f) = slider / flat2 at corresponding dry positions
        n       = min(len(dry_seg), len(fl_seg), len(sl_seg))
        F       = np.fft.rfftfreq(n, 1 / SR_WET)
        H_sl_fl = np.fft.rfft(sl_seg[:n]) / (np.fft.rfft(fl_seg[:n]) + 1e-30)
        db_raw  = 20 * np.log10(np.abs(H_sl_fl) + 1e-15)

        freq_mask = (F >= 30) & (F <= 18000)
        f_plot    = F[freq_mask]
        db_smooth = _smooth_log(f_plot, db_raw[freq_mask])

        f_peak, db_peak = _find_peak(f_plot, db_smooth, fc_nominal)
        f_exp, db_exp   = _expected_peaking(fc_nominal, SLIDER_GAIN_DB)

        results.append({
            "fc_nominal_hz": fc_nominal,
            "fc_actual_hz":  round(f_peak, 1),
            "peak_db":       round(db_peak, 2),
            "expected_peak_db": round(float(db_exp.max()), 2),
            "lag_samples":   lag,
            "colour":        colour,
            "f_plot":        f_plot,
            "db_smooth":     db_smooth,
            "f_exp":         f_exp,
            "db_exp":        db_exp,
        })

    # LEVEL slider captures — flat gain curve, no band peak to find
    level_results: list[dict] = []
    for label, fname, gain_db, colour in LEVEL_FILES:
        wav_path = SLIDER_DIR / fname
        if not wav_path.exists():
            print(f"MISSING level file: {wav_path}")
            continue
        wet, _ = sf.read(wav_path, always_2d=False)
        lag    = _find_lag(dry_ds, wet)
        print(f"  {label} lag: {lag:+d} samples")
        sl_seg = _wet_window(wet, lag, sweep_mid)

        n      = min(len(dry_seg), len(fl_seg), len(sl_seg))
        F      = np.fft.rfftfreq(n, 1 / SR_WET)
        H      = np.fft.rfft(sl_seg[:n]) / (np.fft.rfft(fl_seg[:n]) + 1e-30)
        db_raw = 20 * np.log10(np.abs(H) + 1e-15)

        freq_mask = (F >= 30) & (F <= 18000)
        f_plot    = F[freq_mask]
        db_smooth = _smooth_log(f_plot, db_raw[freq_mask])
        mean_gain = float(np.mean(db_smooth[(f_plot >= 200) & (f_plot <= 5000)]))

        level_results.append({
            "label":         label,
            "gain_db_set":   gain_db,
            "gain_db_meas":  round(mean_gain, 2),
            "colour":        colour,
            "f_plot":        f_plot,
            "db_smooth":     db_smooth,
        })

    _plot_overlay(results, level_results)
    _plot_individual(results)
    _save_summary(results, level_results)
    print(f"\nOutputs written to {OUT_DIR}/")


def _plot_overlay(results: list[dict], level_results: list[dict]) -> None:
    """All 7 band responses + LEVEL slider captures on one figure."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for r in results:
        fc   = r["fc_nominal_hz"]
        ax.plot(r["f_plot"], r["db_smooth"],
                color=r["colour"], lw=2.0, label=f'{fc} Hz')

    # LEVEL slider — flat gain curves shown as dashed thin lines
    for lr in level_results:
        ax.plot(lr["f_plot"], lr["db_smooth"],
                color=lr["colour"], lw=1.4, ls="--",
                label=f'{lr["label"]} (meas {lr["gain_db_meas"]:+.1f} dB)')

    # Nominal centre frequencies — vertical dashed lines
    for r in results:
        fc = r["fc_nominal_hz"]
        ax.axvline(fc, color=r["colour"], ls="--", lw=0.8, alpha=0.5)

    # Actual peaks — annotated
    for r in results:
        fc_act = r["fc_actual_hz"]
        db_pk  = r["peak_db"]
        fc_nom = r["fc_nominal_hz"]
        cent_err = 1200 * np.log2(fc_act / fc_nom) if fc_act > 0 else 0
        label = (f'{fc_act:.0f} Hz\n({cent_err:+.0f} ¢)')
        ax.annotate(label, xy=(fc_act, db_pk),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                    color=r["colour"],
                    arrowprops=dict(arrowstyle="-", color=r["colour"], lw=0.6))

    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.set_xscale("log")
    ax.set_xlim(30, 18000)
    ax.set_ylim(-4, 16)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain relative to flat (dB)")
    ax.set_title("GE-7 Single-Slider Capture — All 7 Bands (+10 dB each)\n"
                 "Dashed lines = labelled centre freq  |  Annotations = actual measured peak")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_slider_overlay.png", dpi=150)
    plt.close(fig)
    print("  01_slider_overlay.png")


def _plot_individual(results: list[dict]) -> None:
    """Per-band plot comparing measured vs Ge7Sim biquad response."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, r in enumerate(results):
        ax  = axes_flat[i]
        fc  = r["fc_nominal_hz"]
        col = r["colour"]

        ax.plot(r["f_exp"], r["db_exp"], color="grey", lw=1.2, ls="--",
                label="Ge7Sim (Q=1.4)")
        ax.plot(r["f_plot"], r["db_smooth"], color=col, lw=2.0,
                label="Measured")

        ax.axvline(fc, color="grey", ls=":", lw=0.8)
        ax.axvline(r["fc_actual_hz"], color=col, ls=":", lw=1.0)
        ax.axhline(0, color="black", lw=0.5)

        fc_act = r["fc_actual_hz"]
        cent_err = 1200 * np.log2(fc_act / fc) if fc_act > 0 else 0
        ax.set_title(f"{fc} Hz band\npeak: {fc_act:.0f} Hz ({cent_err:+.0f} ¢)",
                     fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(30, 22000)
        ax.set_ylim(-4, 16)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused subplot (3×3 = 9 slots, 7 bands)
    for j in range(len(results), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("GE-7 Individual Slider Responses: Measured vs Ge7Sim (Q=1.4)",
                 fontsize=12, y=1.01)
    fig.supxlabel("Frequency (Hz)", y=-0.01)
    fig.supylabel("Gain relative to flat (dB)", x=-0.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_slider_individual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  02_slider_individual.png")


def _save_summary(results: list[dict], level_results: list[dict]) -> None:
    bands = []
    for r in results:
        fc_nom = r["fc_nominal_hz"]
        fc_act = r["fc_actual_hz"]
        cent_err = 1200 * np.log2(fc_act / fc_nom) if fc_act > 0 else 0
        bands.append({
            "fc_nominal_hz":    fc_nom,
            "fc_actual_hz":     fc_act,
            "cent_error":       round(cent_err, 1),
            "measured_peak_db": r["peak_db"],
            "expected_peak_db": r["expected_peak_db"],
        })
    level = [{"label": lr["label"], "gain_set_db": lr["gain_db_set"],
              "gain_measured_db": lr["gain_db_meas"]} for lr in level_results]
    with open(OUT_DIR / "slider_summary.json", "w") as f:
        json.dump({"bands": bands, "level": level}, f, indent=2)
    print("  slider_summary.json")

    print("\nBand centre frequency accuracy:")
    print(f"  {'Nominal':>8}  {'Actual':>8}  {'Error':>8}  {'MeasPeak':>9}  {'ExpPeak':>8}")
    for s in bands:
        print(f"  {s['fc_nominal_hz']:>8}  {s['fc_actual_hz']:>8.1f}  "
              f"{s['cent_error']:>+8.1f}¢  {s['measured_peak_db']:>8.2f} dB  "
              f"{s['expected_peak_db']:>7.2f} dB")
    print("\nLEVEL slider calibration:")
    for lv in level:
        print(f"  {lv['label']:20s}: set={lv['gain_set_db']:+.0f} dB  meas={lv['gain_measured_db']:+.2f} dB")


if __name__ == "__main__":
    main()
