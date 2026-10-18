#!/usr/bin/env python3
"""Daisy hardware vs KlonSim simulation diagnostic analysis.

Answers the question: is the gap between Daisy and the sim due to the
algorithm (firmware/program) or the analog chain (re-amp, cables, ADC/DAC)?

Conclusion telegraphed by the numbers:
  PC model ESR ≈ 0.002 for every section type → algorithm is correct.
  Daisy ESR ≈ 1.1–1.6, level offset +6 to +11 dB → calibration is wrong.

Output: outputs/train_signal_daisy_test/section_analysis/  (plots 07-11)

Run from repo root:
    uv run python scripts/analyze_daisy_vs_sim.py
"""

from __future__ import annotations

from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import correlate, hilbert

# ─── Paths & constants ────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "recordings/data/signals/train_signal_v1.json"
OUTDIR   = ROOT / "outputs/train_signal_daisy_test"
PLOT_DIR = OUTDIR / "section_analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SR = 48_000

FILES: dict[str, Path] = {
    "dry":         OUTDIR / "train_signal_v1_48kHz.wav",
    "klonsim_ref": OUTDIR / "train_signal_v1_klonsim_wet_48kHz.wav",
    "pc_model":    OUTDIR / "train_signal_v1_model_wet_48kHz.wav",
    "daisy_hw":    OUTDIR / "train_signal_v1_48kHz-thru-daisy-wet.wav",
}

COLORS = {
    "dry":         "#888888",
    "klonsim_ref": "#1565C0",
    "pc_model":    "#2E7D32",
    "daisy_hw":    "#C62828",
}
LABELS = {
    "dry":         "Dry",
    "klonsim_ref": "KlonSim ref",
    "pc_model":    "PC ODE model",
    "daisy_hw":    "Daisy hardware",
}

TYPE_COLORS = {
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


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_and_align() -> tuple[dict[str, np.ndarray], list[dict]]:
    sigs: dict[str, np.ndarray] = {}
    for k, p in FILES.items():
        a, sr = sf.read(p, dtype="float32")
        if sr != SR:
            raise ValueError(f"{p.name}: expected {SR} Hz got {sr}")
        sigs[k] = a[:, 0] if a.ndim > 1 else a

    n     = 2 * SR
    corr  = correlate(sigs["daisy_hw"][:n].astype(np.float64),
                      sigs["dry"][:n].astype(np.float64), mode="full")
    lag   = int(np.argmax(corr)) - (n - 1)
    print(f"[align] lag = {lag} samples ({lag / SR * 1000:.2f} ms)")

    daisy_al = (sigs["daisy_hw"][lag:] if lag >= 0
                else np.pad(sigs["daisy_hw"], (-lag, 0)))
    n_out = len(daisy_al)
    for k in sigs:
        sigs[k] = sigs[k][:n_out]
    sigs["daisy_hw"] = daisy_al

    with open(JSON_PATH) as f:
        meta = json.load(f)
    return sigs, meta["sections"]


def esr(ref: np.ndarray, pred: np.ndarray) -> float:
    e = ref.astype(np.float64) - pred.astype(np.float64)
    return float(np.sum(e ** 2) / (np.sum(ref.astype(np.float64) ** 2) + 1e-10))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-10))


def section_slice(sec: dict, skip_frac: float = 0.30) -> tuple[int, int]:
    lo = int(sec["start_s"] * SR)
    hi = int(sec["end_s"] * SR)
    skip = int((hi - lo) * skip_frac)
    return lo + skip, hi


# ─── Plot 7: ESR per section (bar chart) ─────────────────────────────────────

def plot_per_section_esr(sigs: dict[str, np.ndarray],
                         sections: list[dict]) -> Path:
    n_out = len(sigs["klonsim_ref"])

    xs, esr_pc, esr_d, lvl_off, colors, xtypes = [], [], [], [], [], []
    for i, sec in enumerate(sections):
        lo, hi = section_slice(sec)
        if hi > n_out or (hi - lo) < 100:
            continue
        ref = sigs["klonsim_ref"][lo:hi]
        pc  = sigs["pc_model"][lo:hi]
        d   = sigs["daisy_hw"][lo:hi]
        xs.append(sec["start_s"])
        esr_pc.append(esr(ref, pc))
        esr_d.append(esr(ref, d))
        rms_ref = rms(ref); rms_d = rms(d)
        lvl_off.append(db(rms_ref / rms_d) if rms_d > 1e-10 else 0.0)
        colors.append(TYPE_COLORS.get(sec["type"], "#999999"))
        xtypes.append(sec["type"])

    xs_arr   = np.array(xs)
    esr_pc   = np.array(esr_pc)
    esr_d    = np.array(esr_d)
    lvl_off  = np.array(lvl_off)

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                             constrained_layout=True)

    # Top: ESR comparison (log scale)
    ax = axes[0]
    ax.scatter(xs_arr, esr_d,  s=18, c=colors, alpha=0.85, label="Daisy hardware", zorder=3)
    ax.scatter(xs_arr, esr_pc, s=14, c="#888888", alpha=0.5, label="PC ODE model", zorder=2,
               marker="^")
    ax.axhline(1.0, color="#C62828", lw=1.2, ls="--", alpha=0.7, label="ESR = 1 (error = signal)")
    ax.set_ylabel("ESR  (lower = better)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 3.0)
    ax.set_title("Per-section ESR: PC ODE model (grey) vs Daisy hardware (coloured by type)",
                 fontweight="bold")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9)

    # Middle: Daisy level offset per section
    ax = axes[1]
    ax.scatter(xs_arr, lvl_off, s=18, c=colors, alpha=0.85, zorder=3)
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    mean_off = float(np.median(lvl_off))
    ax.axhline(mean_off, color="#FF6F00", lw=1.5, ls="--",
               label=f"Median offset = {mean_off:+.1f} dB")
    ax.set_ylabel("Daisy level offset (dBFS)\n[ref RMS − Daisy RMS]")
    ax.set_title("Daisy output is systematically quiet — all section types affected equally",
                 fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    # Bottom: section type colour legend
    ax = axes[2]
    seen: set[str] = set()
    for i, (x, t, c) in enumerate(zip(xs_arr, xtypes, colors)):
        lbl = t.replace("_", " ") if t not in seen else None
        seen.add(t)
        ax.barh(0, sections[i]["end_s"] - sections[i]["start_s"],
                left=sections[i]["start_s"], height=0.65,
                color=c, alpha=0.85, label=lbl)
    ax.set_xlim(0, sections[-1]["end_s"])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.legend(ncol=6, fontsize=7.5, loc="upper right")

    out = PLOT_DIR / "07_per_section_esr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 8: Amplitude ramp transfer curves ───────────────────────────────────
# Input amplitude vs output amplitude, extracted via Hilbert envelope.
# Shows the static nonlinearity (tanh knee) for all signals.
# Mismatch here = tanh at wrong operating point = kInputGain wrong.

def plot_ramp_transfer(sigs: dict[str, np.ndarray],
                       sections: list[dict]) -> Path:
    ramp_secs = [s for s in sections if s["type"] == "amplitude_ramp"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    for ax, sec in zip(axes.flat, ramp_secs):
        lo = int(sec["start_s"] * SR)
        hi = int(sec["end_s"] * SR)
        dry_seg = sigs["dry"][lo:hi]

        # Hilbert envelope of dry gives the instantaneous input amplitude
        env_in = np.abs(hilbert(dry_seg.astype(np.float64)))

        for key in ["klonsim_ref", "pc_model", "daisy_hw"]:
            wet_seg  = sigs[key][lo:hi]
            env_out  = np.abs(hilbert(wet_seg.astype(np.float64)))
            # Smooth both envelopes over 5 cycles
            f0   = float(sec["params"]["freq_hz"])
            win  = max(1, int(5 * SR / f0))
            kernel = np.ones(win) / win
            sm_in  = np.convolve(env_in,  kernel, mode="same")
            sm_out = np.convolve(env_out, kernel, mode="same")
            # Trim edges
            edge = win
            ax.plot(sm_in[edge:-edge], sm_out[edge:-edge],
                    color=COLORS[key], lw=1.5, label=LABELS[key], alpha=0.85)

        f0 = float(sec["params"]["freq_hz"])
        ax.set_title(f"Amplitude ramp — {f0:.0f} Hz\n"
                     f"({sec['params']['amp_start_dbfs']:+.0f} → "
                     f"{sec['params']['amp_end_dbfs']:+.0f} dBFS)",
                     fontweight="bold")
        ax.set_xlabel("Input amplitude (linear)")
        ax.set_ylabel("Output amplitude (linear)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Transfer curve (static nonlinearity)  —  input amplitude vs output amplitude\n"
        "If Daisy curve is shifted LEFT, kInputGain needs to be higher.",
        fontsize=12, fontweight="bold",
    )
    out = PLOT_DIR / "08_ramp_transfer_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 9: Waveform detail at 440 Hz ────────────────────────────────────────
# 440 Hz is ABOVE HPF2 (350 Hz) so the fundamental survives the filter chain.
# This is the cleanest direct comparison of waveform shape between signals.

def plot_440hz_detail(sigs: dict[str, np.ndarray],
                      sections: list[dict]) -> Path:
    # Find the 440 Hz stepped tone at -3 dBFS (peak level)
    sec = next(s for s in sections
               if s["type"] == "stepped_sine_tone"
               and abs(s["params"]["freq_hz"] - 440.0) < 1
               and abs(s["amplitude_dbfs"] - (-3.0)) < 1)

    lo = int(sec["start_s"] * SR)
    hi = int(sec["end_s"] * SR)
    n  = hi - lo
    mid = lo + n // 2

    # 5-cycle snippet at 440 Hz
    n_cycles = 5
    n_snip   = int(n_cycles * SR / 440.0)
    t_ms     = np.arange(n_snip) / SR * 1000.0

    # Phase portrait: daisy vs klonsim_ref for the full section (normalised)
    ref_full = sigs["klonsim_ref"][lo + n // 4: hi]
    d_full   = sigs["daisy_hw"][lo + n // 4: hi]
    # Normalise each to unit RMS for shape comparison
    ref_n = ref_full / (np.sqrt(np.mean(ref_full ** 2)) + 1e-10)
    d_n   = d_full   / (np.sqrt(np.mean(d_full   ** 2)) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Left: 5-cycle waveform overlay
    ax = axes[0]
    for key in ["klonsim_ref", "pc_model", "daisy_hw"]:
        seg = sigs[key][mid: mid + n_snip]
        # Normalise each signal for shape comparison
        scale = rms(seg) + 1e-10
        ax.plot(t_ms, seg / scale, color=COLORS[key], lw=1.8,
                label=LABELS[key], alpha=0.9)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("440 Hz @ −3 dBFS: 5-cycle waveform  (all RMS-normalised)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Right: phase portrait (klonsim_ref vs daisy)  — RMS normalised
    ax = axes[1]
    # Downsample for scatter speed
    step = max(1, len(ref_n) // 5000)
    ax.scatter(ref_n[::step], d_n[::step],
               s=2, alpha=0.3, color="#C62828", rasterized=True)
    lim = max(np.abs(ref_n).max(), np.abs(d_n).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0, alpha=0.5,
            label="Perfect match (y=x)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("KlonSim ref  (normalised)")
    ax.set_ylabel("Daisy hardware  (normalised)")
    ax.set_title("Phase portrait: Daisy vs KlonSim ref at 440 Hz\n"
                 "Diagonal line = perfect match; spread = character mismatch",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    out = PLOT_DIR / "09_440hz_detail.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 10: Impulse response comparison ────────────────────────────────────
# Extract the first clear impulse from the impulse_train section.
# Compares the tail shape (filter ringing) across all signals.

def plot_impulse_response(sigs: dict[str, np.ndarray],
                          sections: list[dict]) -> Path:
    sec = next(s for s in sections if s["type"] == "impulse_train")
    lo  = int(sec["start_s"] * SR)
    hi  = int(sec["end_s"] * SR)

    # Find first impulse in dry signal (within first 100 ms of section)
    search_end = lo + int(0.1 * SR)
    dry_chunk  = np.abs(sigs["dry"][lo:search_end])
    peak_idx   = int(np.argmax(dry_chunk)) + lo

    # Extract 60 ms window: 5 ms before, 55 ms after
    pre_ms  = 5
    post_ms = 55
    win_lo  = peak_idx - int(pre_ms * SR / 1000)
    win_hi  = peak_idx + int(post_ms * SR / 1000)
    t_ms    = (np.arange(win_hi - win_lo) - int(pre_ms * SR / 1000)) / SR * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True,
                             sharey=False)

    impulse_spacing_ms = 50  # first spacing = 50 ms
    n_impulses = 3  # show first 3

    for col, key in enumerate(["klonsim_ref", "pc_model", "daisy_hw"]):
        ax = axes[0, col if col < 2 else 1]  # only 3 signals
        ax = axes.flat[col]
        ax.plot(t_ms, sigs[key][win_lo:win_hi],
                color=COLORS[key], lw=1.2, label=LABELS[key])
        ax.axvline(0, color="#FF6F00", lw=1.0, ls=":", alpha=0.7)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title(LABELS[key], fontweight="bold", color=COLORS[key])
        ax.grid(True, alpha=0.2)

    # Bottom-right: overlay all three (RMS-normalised for shape comparison)
    ax = axes.flat[3]
    for key in ["klonsim_ref", "pc_model", "daisy_hw"]:
        seg   = sigs[key][win_lo:win_hi]
        scale = max(float(np.max(np.abs(seg))), 1e-10)
        ax.plot(t_ms, seg / scale, color=COLORS[key], lw=1.5,
                label=LABELS[key], alpha=0.85)
    ax.axvline(0, color="#FF6F00", lw=1.0, ls=":", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("Overlay (peak-normalised)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Impulse response — first impulse from impulse_train section\n"
        f"(impulse at t=0, amplitude {sec['amplitude_dbfs']:+.0f} dBFS)",
        fontsize=12, fontweight="bold",
    )
    out = PLOT_DIR / "10_impulse_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 11: Decaying pluck spectrogram ──────────────────────────────────────

def plot_pluck_spectrogram(sigs: dict[str, np.ndarray],
                           sections: list[dict]) -> Path:
    sec = next(s for s in sections if s["type"] == "decaying_plucks")
    lo  = int(sec["start_s"] * SR)
    hi  = int(sec["end_s"] * SR)

    keys   = ["dry", "klonsim_ref", "pc_model", "daisy_hw"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    for ax, key in zip(axes.flat, keys):
        seg = sigs[key][lo:hi]
        ax.specgram(seg, NFFT=2048, Fs=SR, noverlap=1800,
                    cmap="inferno", vmin=-90, vmax=-20, scale="dB")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 8000)
        ax.set_title(LABELS[key], fontweight="bold", color=COLORS[key])
        ax.set_xlim(0, (hi - lo) / SR)

    fig.suptitle("Decaying plucks — spectrogram comparison\n"
                 "(7 guitar strings, τ=0.4 s decay, −6 dBFS peak)",
                 fontsize=12, fontweight="bold")
    out = PLOT_DIR / "11_pluck_spectrogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Plot 12: Summary — ESR & level offset by section type ───────────────────

def plot_type_summary(sigs: dict[str, np.ndarray],
                      sections: list[dict]) -> Path:
    n_out = len(sigs["klonsim_ref"])
    by_type: dict[str, dict] = defaultdict(lambda: {"esr_pc": [], "esr_d": [], "lvl": []})

    for sec in sections:
        lo, hi = section_slice(sec)
        if hi > n_out or (hi - lo) < 100:
            continue
        ref = sigs["klonsim_ref"][lo:hi]
        pc  = sigs["pc_model"][lo:hi]
        d   = sigs["daisy_hw"][lo:hi]
        t   = sec["type"]
        by_type[t]["esr_pc"].append(esr(ref, pc))
        by_type[t]["esr_d"].append(esr(ref, d))
        rms_d = rms(d)
        if rms_d > 1e-10:
            by_type[t]["lvl"].append(db(rms(ref) / rms_d))

    types     = sorted(by_type.keys())
    x         = np.arange(len(types))
    mean_esr_pc = [float(np.mean(by_type[t]["esr_pc"])) for t in types]
    mean_esr_d  = [float(np.mean(by_type[t]["esr_d"]))  for t in types]
    med_lvl     = [float(np.median(by_type[t]["lvl"]))  for t in types]
    colors_t    = [TYPE_COLORS.get(t, "#999999") for t in types]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)

    # Top: ESR by type
    ax = axes[0]
    w = 0.35
    bars_pc = ax.bar(x - w/2, mean_esr_pc, w, color="#AAAAAA", label="PC ODE model",
                     zorder=3, edgecolor="white")
    bars_d  = ax.bar(x + w/2, mean_esr_d,  w, color=colors_t,  label="Daisy hardware",
                     zorder=3, edgecolor="white")
    ax.axhline(1.0, color="#C62828", lw=1.5, ls="--", alpha=0.8,
               label="ESR = 1  (error ≥ signal)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=8)
    ax.set_ylabel("Mean ESR vs KlonSim ref")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 4.0)
    ax.set_title("ESR by section type: PC model ≈ 0 everywhere; Daisy ≈ 1 everywhere",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2, axis="y")

    # Bottom: Level offset by type
    ax = axes[1]
    bars = ax.bar(x, med_lvl, color=colors_t, edgecolor="white", zorder=3)
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    overall_med = float(np.median([v for t in types for v in by_type[t]["lvl"]]))
    ax.axhline(overall_med, color="#FF6F00", lw=1.8, ls="--",
               label=f"Overall median = {overall_med:+.1f} dB")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=8)
    ax.set_ylabel("Median level offset (dB)\n[ref RMS − Daisy RMS]")
    ax.set_title("Daisy output level deficit by section type  "
                 "(uniform offset → single calibration error)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate recommended kInputGain
    new_gain = 4.270 * (10 ** (overall_med / 20.0))
    ax.text(0.98, 0.95,
            f"Current kInputGain = 4.270\n"
            f"Suggested = {new_gain:.1f}  (+{overall_med:.1f} dB)",
            transform=ax.transAxes, fontsize=9.5, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    out = PLOT_DIR / "12_type_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[load] Reading and aligning signals …")
    sigs, sections = load_and_align()

    print("\n[plot] Generating diagnostic plots …")
    saved = []

    p = plot_per_section_esr(sigs, sections)
    saved.append(p); print(f"  {p.name}")

    p = plot_ramp_transfer(sigs, sections)
    saved.append(p); print(f"  {p.name}")

    p = plot_440hz_detail(sigs, sections)
    saved.append(p); print(f"  {p.name}")

    p = plot_impulse_response(sigs, sections)
    saved.append(p); print(f"  {p.name}")

    p = plot_pluck_spectrogram(sigs, sections)
    saved.append(p); print(f"  {p.name}")

    p = plot_type_summary(sigs, sections)
    saved.append(p); print(f"  {p.name}")

    print(f"\n[done] {len(saved)} plots in {PLOT_DIR}/")


if __name__ == "__main__":
    main()
