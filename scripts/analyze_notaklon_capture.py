"""Analyse all Notaklon hardware captures and produce diagnostic plots.

Generates plots in outputs/notaklon_captures/ comparing all gain settings,
treble variation, Shamrock mod, and the bypass (reamp-only) reference.

Usage:
    uv run scripts/analyze_notaklon_capture.py
"""

import json
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


DRY_NPY      = ROOT / "recordings/data/signals/train_signal_v2_48k.npy"   # 48 kHz cached, loads in < 1 s
DRY_WAV      = ROOT / "recordings/data/notaklon/Notaklon - train_signal_v2_48k.wav"   # fallback
MANIFEST     = ROOT / "recordings/data/signals/train_signal_v2.json"
CAPTURES_DIR = ROOT / "recordings/data/notaklon/captures"
OUT_DIR      = ROOT / "outputs/notaklon_captures"

SR    = 48_000
RATIO = 0.5          # manifest sample_numbers at 96 kHz → divide by 2 for 48 kHz

# All captures: (dir, label, color, lag_smp)
# Lags measured individually via xcorr on sweep_low (all ≈47 smp = ADC buffer)
CAPTURES: list[tuple[str, str, str, int]] = [
    ("bypass",           "bypass",   "#9E9E9E", 45),  # reamp-only reference
    ("V12-T12-G9-False", "G9",       "#2196F3", 47),
    ("V12-T12-G10-False","G10",      "#FF9800", 47),
    ("V12-T8-G11-False", "G11-T8",   "#9C27B0", 48),  # treble cut
    ("V12-T12-G11-False","G11",      "#4CAF50", 47),
    ("V12-T12-G12-False","G12",      "#F44336", 46),
    ("V12-T12-G12-True", "G12-S",    "#795548", 47),  # G12 + Shamrock
    # V10-T10 set — different volume/treble; not comparable to V12 set on level plots
    ("V10-T10-G1-False", "G1",       "#E91E63", 47),  # gain past noon (1 o'clock)
]

# Subsets used in different comparisons
GAIN_SUBSET    = ["G9", "G10", "G11", "G12"]          # gain sweep, same V/T settings
TREBLE_SUBSET  = ["G11", "G11-T8"]                    # treble comparison
SHAMROCK_SUBSET= ["G12", "G12-S"]                     # Shamrock on vs off

def _captures_by_label(*labels: str) -> list[tuple[str, str, str, int]]:
    return [c for c in CAPTURES if c[1] in labels]


# ─── helpers ─────────────────────────────────────────────────────────────────

def _load_wet(preset: str) -> np.ndarray:
    path = CAPTURES_DIR / preset / "wet_train_signal_v2_48k.wav"
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return audio


def _load_params(preset: str) -> dict:
    return json.loads((CAPTURES_DIR / preset / "params.json").read_text())


def _seg48(sig: np.ndarray, sec: dict, lag: int = 47) -> np.ndarray:
    """Extract a signal window for a manifest section, applying the hardware lag."""
    a = lag + int(sec["start_sample"] * RATIO)
    b = lag + int(sec["end_sample"]   * RATIO)
    return sig[max(0, a): min(b, len(sig))]


def _seg48_dry(sig: np.ndarray, sec: dict) -> np.ndarray:
    """Extract from the dry signal (no lag offset)."""
    a = int(sec["start_sample"] * RATIO)
    b = int(sec["end_sample"]   * RATIO)
    return sig[max(0, a): min(b, len(sig))]


def _transfer_db(dry_seg: np.ndarray, wet_seg: np.ndarray, frac: float = 1 / 6) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed FFT-ratio magnitude response (dB), 1/N-octave smoothed."""
    n = min(len(dry_seg), len(wet_seg))
    D = np.fft.rfft(dry_seg[:n])
    W = np.fft.rfft(wet_seg[:n])
    H = W / (D + 1e-30)
    f = np.fft.rfftfreq(n, 1 / SR)
    db = 20 * np.log10(np.abs(H) + 1e-12)
    # 1/N-octave smoothing
    mask = f > 0
    log_f = np.log2(np.maximum(f, 1e-9))
    half_w = frac / 2
    cs = np.concatenate(([0.0], np.cumsum(db)))
    lo = np.searchsorted(log_f, log_f - half_w, side="left")
    hi = np.searchsorted(log_f, log_f + half_w, side="right")
    db_smooth = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return f, db_smooth


def _tone_peak_db(sig: np.ndarray, sec: dict, lag: int = 47) -> float:
    chunk = _seg48(sig, sec, lag)
    if len(chunk) == 0:
        return -80.0
    return float(20 * np.log10(np.max(np.abs(chunk)) + 1e-12))


def _harmonic_powers(sig: np.ndarray, sec: dict, freq_hz: float, lag: int = 47, n_harmonics: int = 6) -> list[float]:
    """Peak power (dB) at fundamental + harmonics from a pure-tone section."""
    chunk = _seg48(sig, sec, lag).astype(np.float64)
    if len(chunk) < 256:
        return [-80.0] * n_harmonics
    N = len(chunk)
    freqs = np.fft.rfftfreq(N, 1 / SR)
    spectrum = np.abs(np.fft.rfft(chunk)) * 2 / N
    powers = []
    for k in range(1, n_harmonics + 1):
        target = freq_hz * k
        if target > SR / 2:
            powers.append(-80.0)
            continue
        idx = int(np.argmin(np.abs(freqs - target)))
        width = max(1, int(SR / N * 3))   # ±3-bin window
        lo, hi = max(0, idx - width), min(len(spectrum), idx + width + 1)
        pk = np.max(spectrum[lo:hi])
        powers.append(float(20 * np.log10(pk + 1e-12)))
    return powers



# ─── plot functions ───────────────────────────────────────────────────────────

def _plot_lag(dry: np.ndarray, captures: list[tuple], manifest: dict, out_dir: Path) -> Path:
    """01: Cross-correlation curves confirming hardware lag for each capture."""
    sec = next(s for s in manifest["sections"] if s["label"] == "sweep_low")
    s0 = int(sec["start_sample"] * RATIO)
    s1 = int(sec["end_sample"]   * RATIO)
    n  = min(s1 - s0, int(5 * SR))

    d_chunk = dry[s0: s0 + n].astype(np.float64)
    max_lag = int(0.01 * SR)

    from scipy.signal import correlate as _correlate
    fig, axes = plt.subplots(len(captures), 1, figsize=(12, 3 * len(captures)), squeeze=False)
    fig.suptitle("Lag / Alignment — Notaklon captures (sweep_low, −24 dBFS)", fontsize=13, fontweight="bold")

    for ax, (_, label, color, lag, w) in zip(axes[:, 0], captures):
        w_chunk = w[s0: s0 + n].astype(np.float64)
        xcorr = _correlate(d_chunk, w_chunk, mode="full")
        centre = len(xcorr) // 2
        snippet = xcorr[centre - max_lag: centre + max_lag + 1]
        lags_ms = np.arange(-max_lag, max_lag + 1) / SR * 1000

        ax.plot(lags_ms, snippet / (np.max(np.abs(snippet)) + 1e-12), color=color, lw=0.9, label=label)
        ax.axvline(lag / SR * 1000, color="red", ls="--", lw=1.5, label=f"lag = {lag} smp ({lag/SR*1000:.2f} ms)")
        ax.set_ylabel("Norm. xcorr")
        ax.legend(fontsize=9)
        ax.set_xlim(-5, 5)
        ax.grid(True, alpha=0.3)
        ax.set_title(label, fontsize=10)

    axes[-1, 0].set_xlabel("Lag (ms)")
    fig.tight_layout()
    p = out_dir / "01_lag_alignment.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# captures arg is list of (dir, label, color, lag, wet_array)
CaptureList = list[tuple[str, str, str, int, np.ndarray]]


def _plot_linear_fr(dry: np.ndarray, captures: CaptureList, manifest: dict, out_dir: Path) -> Path:
    """02: Linear-regime FR — all captures on one axes, bypass shown as grey reference."""
    sec = next(s for s in manifest["sections"] if s["label"] == "sweep_low")
    dry_seg = _seg48_dry(dry, sec)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color="k", lw=0.7, ls=":")

    for _, label, color, lag, w in captures:
        wet_seg = _seg48(w, sec, lag)
        f, db = _transfer_db(dry_seg, wet_seg, frac=1 / 6)
        mask = (f > 20) & (f < 20000)
        lw = 1.5 if label == "bypass" else 2.0
        ls = "--" if label == "bypass" else "-"
        ax.plot(f[mask], db[mask], color=color, lw=lw, ls=ls, alpha=0.85, label=label)

    ax.set_xscale("log")
    ax.set_xlim(20, 20000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain vs dry (dB)")
    ax.set_title("Linear-regime frequency response — sweep_low (−24 dBFS, 1/6-oct smoothed)\n"
                 "Absolute level includes Radial ProRMP reamp tilt (subtract bypass to isolate pedal)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-40, 20)
    fig.tight_layout()
    p = out_dir / "02_linear_freq_response.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_pedal_only_fr(dry: np.ndarray, captures: CaptureList, manifest: dict, out_dir: Path) -> Path:
    """03: Pedal-only FR — each capture normalised by bypass to remove reamp tilt."""
    sec = next(s for s in manifest["sections"] if s["label"] == "sweep_low")
    dry_seg = _seg48_dry(dry, sec)

    bypass_cap = next((c for c in captures if c[1] == "bypass"), None)
    if bypass_cap is None:
        return out_dir / "03_pedal_only_fr.png"  # skip if no bypass

    _, _, _, bp_lag, bp_w = bypass_cap
    bp_seg = _seg48(bp_w, sec, bp_lag)
    f_bp, db_bp = _transfer_db(dry_seg, bp_seg, frac=1 / 6)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color="k", lw=0.7, ls=":")

    for _, label, color, lag, w in captures:
        if label == "bypass":
            continue
        wet_seg = _seg48(w, sec, lag)
        f, db = _transfer_db(dry_seg, wet_seg, frac=1 / 6)
        pedal_db = db - db_bp   # remove reamp tilt
        mask = (f > 30) & (f < 18000)
        ax.plot(f[mask], pedal_db[mask], color=color, lw=2.0, alpha=0.85, label=label)

    ax.set_xscale("log")
    ax.set_xlim(30, 18000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Pedal gain (dB, bypass-normalised)")
    ax.set_title("Pedal-only frequency response — sweep_low, bypass-normalised (1/6-oct smoothed)\n"
                 "Reamp tilt subtracted: shows pure Notaklon EQ character in linear regime",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-25, 20)
    fig.tight_layout()
    p = out_dir / "03_pedal_only_fr.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_driven_fr(dry: np.ndarray, captures: CaptureList, manifest: dict, out_dir: Path) -> Path:
    """04: Driven FR — sweep_high (−3 dBFS), all gain settings."""
    sec = next(s for s in manifest["sections"] if s["label"] == "sweep_high")
    dry_seg = _seg48_dry(dry, sec)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color="k", lw=0.7, ls=":")

    for _, label, color, lag, w in captures:
        wet_seg = _seg48(w, sec, lag)
        f, db = _transfer_db(dry_seg, wet_seg, frac=1 / 3)
        mask = (f > 20) & (f < 20000)
        lw = 1.5 if label == "bypass" else 2.0
        ls = "--" if label == "bypass" else "-"
        ax.plot(f[mask], db[mask], color=color, lw=lw, ls=ls, alpha=0.85, label=label)

    ax.set_xscale("log")
    ax.set_xlim(20, 20000)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain vs dry (dB)")
    ax.set_title("Driven frequency response — sweep_high (−3 dBFS, 1/3-oct smoothed)\n"
                 "FFT-ratio on nonlinear signal: shows average spectral shaping, not true LTI",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-40, 20)
    fig.tight_layout()
    p = out_dir / "04_driven_freq_response.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_saturation(dry: np.ndarray, captures: CaptureList, manifest: dict, out_dir: Path) -> Path:
    """05: Output peak vs input level — saturation curves for gain sweep only."""
    tones = [s for s in manifest["sections"] if s["type"] == "stepped_sine_tone"]
    levels = sorted({s["amplitude_dbfs"] for s in tones})
    target_freqs = [220, 440, 880]
    gain_caps = [c for c in captures if c[1] in GAIN_SUBSET + ["bypass"]]

    fig, axes = plt.subplots(1, len(target_freqs), figsize=(15, 5), squeeze=False)
    fig.suptitle("Saturation curves — output peak vs input level", fontsize=13, fontweight="bold")

    for ax_idx, freq in enumerate(target_freqs):
        ax = axes[0, ax_idx]
        ax.plot(levels, levels, "k--", lw=0.8, alpha=0.4, label="Linear (1:1)")

        for _, label, color, lag, w in gain_caps:
            out_levels = []
            for lvl in levels:
                sec = next((s for s in tones
                            if int(s["params"]["freq_hz"]) == freq
                            and s["amplitude_dbfs"] == lvl), None)
                if sec is None:
                    out_levels.append(np.nan)
                    continue
                out_levels.append(_tone_peak_db(w, sec, lag))
            lw = 1.5 if label == "bypass" else 1.8
            ls = "--" if label == "bypass" else "-"
            ax.plot(levels, out_levels, "o" if label != "bypass" else "s",
                    color=color, lw=lw, ls=ls, ms=6, label=label)

        ax.set_xlabel("Input level (dBFS)")
        ax.set_ylabel("Output peak (dBFS)")
        ax.set_title(f"{freq} Hz", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-26, -1)

    fig.tight_layout()
    p = out_dir / "05_saturation_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_harmonics(dry: np.ndarray, captures: CaptureList, manifest: dict, out_dir: Path) -> Path:
    """06: Harmonic content at 440 Hz for each gain and input level."""
    tones = [s for s in manifest["sections"] if s["type"] == "stepped_sine_tone"]
    levels = sorted({s["amplitude_dbfs"] for s in tones})
    freq = 440
    n_harmonics = 5
    gain_caps = [c for c in captures if c[1] in GAIN_SUBSET]

    n_lvl = len(levels)
    fig, axes = plt.subplots(len(gain_caps), n_lvl,
                             figsize=(4 * n_lvl, 4 * len(gain_caps)), squeeze=False)
    fig.suptitle(f"Harmonic distortion — {freq} Hz tones at each gain × level", fontsize=13, fontweight="bold")

    x_pos = np.arange(1, n_harmonics + 1)
    harm_labels = [f"H{k}" for k in range(1, n_harmonics + 1)]

    for row, (_, g_label, color, lag, w) in enumerate(gain_caps):
        for col, lvl in enumerate(levels):
            ax = axes[row, col]
            sec = next((s for s in tones
                        if int(s["params"]["freq_hz"]) == freq
                        and s["amplitude_dbfs"] == lvl), None)
            if sec is None:
                ax.set_visible(False)
                continue
            pows = _harmonic_powers(w, sec, freq, lag, n_harmonics=n_harmonics)
            fund = pows[0]
            rel = [p - fund for p in pows]
            ax.bar(x_pos, rel, color=color, alpha=0.75, width=0.6)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(harm_labels, fontsize=8)
            ax.set_ylim(-80, 5)
            ax.set_ylabel("dB rel. H1", fontsize=8)
            ax.set_title(f"{g_label}  @{lvl} dBFS\nH1={fund:.1f} dBFS", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(labelsize=8)

    fig.tight_layout()
    p = out_dir / "06_harmonic_distortion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_waveforms(dry: np.ndarray, captures: CaptureList, manifest: dict, out_dir: Path) -> Path:
    """07: Waveform overlays — gain sweep at three drive levels (440 Hz tones)."""
    targets = [
        ("stepped_tone_f440_l-12", "440 Hz @ −12 dBFS"),
        ("stepped_tone_f440_l-6",  "440 Hz @ −6 dBFS"),
        ("stepped_tone_f440_l-3",  "440 Hz @ −3 dBFS"),
    ]
    sec_map = {s["label"]: s for s in manifest["sections"]}
    gain_caps = [c for c in captures if c[1] in GAIN_SUBSET + ["bypass"]]
    show_s = 0.03

    fig, axes = plt.subplots(len(targets), 1, figsize=(14, 3.5 * len(targets)), squeeze=False)
    fig.suptitle("Waveform comparison — dry vs captures (440 Hz tones)", fontsize=13, fontweight="bold")

    n_show = int(show_s * SR)
    t_ms = np.arange(n_show) / SR * 1000

    for ax, (sec_label, title) in zip(axes[:, 0], targets):
        if sec_label not in sec_map:
            ax.set_visible(False)
            continue
        sec = sec_map[sec_label]
        d_seg = _seg48_dry(dry, sec)[:n_show]
        ax.plot(t_ms[:len(d_seg)], d_seg, color="#9E9E9E", lw=0.9, alpha=0.8, label="Dry")
        for _, g_label, color, lag, w in gain_caps:
            w_seg = _seg48(w, sec, lag)[:n_show]
            lw = 1.2 if g_label == "bypass" else 1.5
            ls = "--" if g_label == "bypass" else "-"
            ax.plot(t_ms[:len(w_seg)], w_seg, color=color, lw=lw, ls=ls, alpha=0.85, label=g_label)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (ms)")
    fig.tight_layout()
    p = out_dir / "07_waveform_overlay.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_treble_shamrock(dry: np.ndarray, captures: CaptureList, manifest: dict, out_dir: Path) -> Path:
    """08: Treble cut vs noon, and Shamrock on vs off — bypass-normalised FR."""
    sec_low = next(s for s in manifest["sections"] if s["label"] == "sweep_low")
    dry_seg = _seg48_dry(dry, sec_low)

    bypass_cap = next((c for c in captures if c[1] == "bypass"), None)
    if bypass_cap is None:
        return out_dir / "08_treble_shamrock.png"
    _, _, _, bp_lag, bp_w = bypass_cap
    bp_seg = _seg48(bp_w, sec_low, bp_lag)
    _, db_bp = _transfer_db(dry_seg, bp_seg, frac=1 / 6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Treble cut and Shamrock mod — pedal-only FR (bypass-normalised, sweep_low)",
                 fontsize=13, fontweight="bold")

    # Treble comparison
    for cap_label in ["G11", "G11-T8"]:
        cap = next((c for c in captures if c[1] == cap_label), None)
        if cap is None:
            continue
        _, label, color, lag, w = cap
        wet_seg = _seg48(w, sec_low, lag)
        f, db = _transfer_db(dry_seg, wet_seg, frac=1 / 6)
        pedal_db = db - db_bp
        mask = (f > 30) & (f < 18000)
        ax1.plot(f[mask], pedal_db[mask], color=color, lw=2.0, label=label)

    ax1.axhline(0, color="k", lw=0.7, ls=":")
    ax1.set_xscale("log"); ax1.set_xlim(30, 18000)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("Pedal gain (dB)")
    ax1.set_title("Treble: noon (G11) vs 8 o'clock cut (G11-T8)", fontsize=10)
    ax1.legend(fontsize=10); ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(-25, 20)

    # Shamrock comparison — use sweep_high for driven character
    sec_high = next(s for s in manifest["sections"] if s["label"] == "sweep_high")
    dry_seg_h = _seg48_dry(dry, sec_high)
    bp_seg_h = _seg48(bp_w, sec_high, bp_lag)
    _, db_bp_h = _transfer_db(dry_seg_h, bp_seg_h, frac=1 / 3)

    for cap_label in ["G12", "G12-S"]:
        cap = next((c for c in captures if c[1] == cap_label), None)
        if cap is None:
            continue
        _, label, color, lag, w = cap
        wet_seg_h = _seg48(w, sec_high, lag)
        f, db = _transfer_db(dry_seg_h, wet_seg_h, frac=1 / 3)
        pedal_db = db - db_bp_h
        mask = (f > 30) & (f < 18000)
        ax2.plot(f[mask], pedal_db[mask], color=color, lw=2.0, label=label)

    ax2.axhline(0, color="k", lw=0.7, ls=":")
    ax2.set_xscale("log"); ax2.set_xlim(30, 18000)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Pedal gain vs bypass (dB)")
    ax2.set_title("Shamrock mod: off (G12) vs on (G12-S) — sweep_high −3 dBFS", fontsize=10)
    ax2.legend(fontsize=10); ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(-25, 20)

    fig.tight_layout()
    p = out_dir / "08_treble_shamrock.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_gain_vs_freq(dry: np.ndarray, captures: CaptureList, manifest: dict, out_dir: Path) -> Path:
    """09: Gain vs frequency for each level — gain sweep only."""
    tones = [s for s in manifest["sections"] if s["type"] == "stepped_sine_tone"]
    levels = sorted({s["amplitude_dbfs"] for s in tones})
    tone_freqs = sorted({int(s["params"]["freq_hz"]) for s in tones})
    gain_caps = [c for c in captures if c[1] in GAIN_SUBSET]

    fig, axes = plt.subplots(len(gain_caps), 1,
                             figsize=(13, 4 * len(gain_caps)), squeeze=False)
    fig.suptitle("Gain vs frequency at each input level (colour = level)", fontsize=13, fontweight="bold")

    cmap = plt.cm.plasma
    level_colors = [cmap(i / max(len(levels) - 1, 1)) for i in range(len(levels))]

    for ax, (_, g_label, _, lag, w) in zip(axes[:, 0], gain_caps):
        for lvl, lcol in zip(levels, level_colors):
            f_pts, g_pts = [], []
            for freq in tone_freqs:
                sec = next((s for s in tones
                            if int(s["params"]["freq_hz"]) == freq
                            and s["amplitude_dbfs"] == lvl), None)
                if sec is None:
                    continue
                d_pk = float(np.max(np.abs(_seg48_dry(dry, sec))) + 1e-12)
                w_pk = float(np.max(np.abs(_seg48(w, sec, lag))) + 1e-12)
                f_pts.append(freq)
                g_pts.append(20 * np.log10(w_pk / d_pk))
            ax.plot(f_pts, g_pts, "o-", color=lcol, lw=1.5, ms=5, label=f"{lvl} dBFS")

        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(tone_freqs)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_ylabel("Gain vs dry (dB)")
        ax.set_title(g_label, fontsize=11)
        ax.legend(fontsize=7, ncol=len(levels), loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-40, 20)

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    p = out_dir / "09_gain_vs_freq.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    manifest = json.loads(MANIFEST.read_text())

    print("Loading signals…")
    if DRY_NPY.exists():
        dry = np.load(DRY_NPY).astype(np.float32)
        print(f"  dry: {len(dry)/SR:.1f} s @ {SR} Hz  (npy cache)")
    else:
        dry, dry_sr = sf.read(str(DRY_WAV), dtype="float32", always_2d=False)
        if dry_sr != SR:
            raise ValueError(f"Dry WAV: expected {SR} Hz, got {dry_sr}")
        print(f"  dry: {len(dry)/SR:.1f} s @ {SR} Hz  (wav)")

    n_dry = len(dry)
    loaded: CaptureList = []
    for cap_dir, label, color, lag in CAPTURES:
        w = _load_wet(cap_dir)[:n_dry]
        peak = float(20 * np.log10(np.max(np.abs(w)) + 1e-12))
        print(f"  {label:10s}  lag={lag:+3d} smp  peak={peak:.1f} dBFS")
        loaded.append((cap_dir, label, color, lag, w))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots…")
    plots = [
        _plot_lag(dry, loaded, manifest, OUT_DIR),
        _plot_linear_fr(dry, loaded, manifest, OUT_DIR),
        _plot_pedal_only_fr(dry, loaded, manifest, OUT_DIR),
        _plot_driven_fr(dry, loaded, manifest, OUT_DIR),
        _plot_saturation(dry, loaded, manifest, OUT_DIR),
        _plot_harmonics(dry, loaded, manifest, OUT_DIR),
        _plot_waveforms(dry, loaded, manifest, OUT_DIR),
        _plot_treble_shamrock(dry, loaded, manifest, OUT_DIR),
        _plot_gain_vs_freq(dry, loaded, manifest, OUT_DIR),
    ]

    print(f"\nAll plots written to {OUT_DIR}/")
    for p in plots:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
