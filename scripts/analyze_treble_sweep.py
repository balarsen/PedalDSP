"""JHS Notaklon — treble knob sweep, frequency response overlay.

Compares 4 short captures (65 s, opening sweep section of train_signal_v2) at
G12 / V12 with treble ∈ {T6, T9, T12, T3}.  Produces a single bypass-normalised
pedal-only FR plot in the style of ge7_captures/single_slider/01_slider_overlay.

Treble positions (6→4 o'clock dial, _oclock_to_norm(h) = (h-6)/10):
    T6  → norm 0.0 → −8.0 dB (maximum cut)
    T9  → norm 0.3 → −4.0 dB cut
    T12 → norm 0.6 →  0.0 dB (noon, flat)
    T3  → norm 0.9 → +6.0 dB boost   (3 o'clock past noon = hour 15)

Usage:
    uv run scripts/analyze_treble_sweep.py
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

from pedal_model.capture.hardware_delay import estimate_lag

SR       = 48_000
N_SHORT  = int(65 * SR)        # first 65 s — opening sweep section
N_USE    = N_SHORT - 60        # FR window: leave headroom for max lag (~50 smp)

DRY_NPY    = ROOT / "recordings/data/signals/train_signal_v2_48k.npy"
BYPASS_WAV = ROOT / "recordings/data/notaklon/captures/bypass/wet_train_signal_v2_48k.wav"
SWEEP_DIR  = ROOT / "recordings/data/notaklon/captures/treble_sweep"
OUT_DIR    = ROOT / "outputs/notaklon_treble_sweep"

# (label, subdir, color, treble_db_expected)
# Ordered darkest → brightest
TREBLE_CAPS: list[tuple[str, str, str, float]] = [
    ("T6",  "V12-T6-G12-False",  "#5C6BC0", -8.0),
    ("T9",  "V12-T9-G12-False",  "#42A5F5", -4.0),
    ("T12", "V12-T12-G12-False", "#26A69A",  0.0),
    ("T3",  "V12-T3-G12-False",  "#EF5350", +6.0),
]


def _transfer_db(
    dry_seg: np.ndarray,
    wet_seg: np.ndarray,
    frac: float = 1 / 6,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT magnitude ratio with 1/N-octave cumsum smoothing."""
    n = min(len(dry_seg), len(wet_seg))
    D = np.fft.rfft(dry_seg[:n])
    W = np.fft.rfft(wet_seg[:n])
    H = W / (D + 1e-30)
    f = np.fft.rfftfreq(n, 1 / SR)
    db = 20.0 * np.log10(np.abs(H) + 1e-12)
    log_f = np.log2(np.maximum(f, 1e-9))
    half_w = frac / 2
    cs = np.concatenate(([0.0], np.cumsum(db)))
    lo = np.searchsorted(log_f, log_f - half_w, side="left")
    hi = np.searchsorted(log_f, log_f + half_w, side="right")
    return f, (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)


def _load_aligned(path: Path, dry_win: np.ndarray, dry_full: np.ndarray) -> tuple[np.ndarray, int]:
    """Load WAV, estimate lag against full dry, return (aligned window, lag)."""
    raw, _ = sf.read(path, dtype="float32", always_2d=False)
    lag = estimate_lag(dry_full, raw.astype(np.float64), SR,
                       start_s=0.5, end_s=min(20.0, len(raw) / SR - 1.0))
    n = len(dry_win)
    aligned = raw[lag: lag + n].astype(np.float64)
    return aligned, lag


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dry_full = np.load(DRY_NPY)
    dry_win  = dry_full[:N_USE].astype(np.float64)   # common FR window

    # ── bypass reference ─────────────────────────────────────────────────────
    bp_aligned, bp_lag = _load_aligned(BYPASS_WAV, dry_win, dry_full)
    print(f"bypass  lag={bp_lag:+4d} smp")
    f_bp, db_bp = _transfer_db(dry_win, bp_aligned)

    # ── per-treble FRs ───────────────────────────────────────────────────────
    curves: list[tuple[str, str, float, np.ndarray, np.ndarray]] = []
    for label, subdir, color, treble_db_exp in TREBLE_CAPS:
        wet, lag = _load_aligned(SWEEP_DIR / subdir / "wet_01_start_48k.wav", dry_win, dry_full)
        peak_dbfs = float(20 * np.log10(np.abs(wet).max() + 1e-12))
        rms_dbfs  = float(20 * np.log10(np.sqrt(np.mean(wet**2)) + 1e-12))
        print(f"{label:4s}    lag={lag:+4d} smp  peak={peak_dbfs:.1f}  rms={rms_dbfs:.1f}  (treble_db≈{treble_db_exp:+.1f})")
        f_wet, db_wet = _transfer_db(dry_win, wet)
        # pedal-only: wet relative to bypass (same freq grid — same n)
        db_pedal = db_wet - db_bp
        curves.append((label, color, treble_db_exp, f_wet, db_pedal))

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    # Compute T12 baseline for relative annotation
    _, _, _, f_ref, db_ref = curves[2]  # T12 is index 2

    for label, color, treble_db_exp, f, db in curves:
        mask = (f > 30) & (f < 22_000)
        f_m, db_m = f[mask], db[mask]
        ax.plot(f_m, db_m, color=color, lw=2.2,
                label=f"{label}  (treble ≈ {treble_db_exp:+.0f} dB shelf)")

        # Annotate at 10 kHz — curves are well-separated there
        f_ann = 10_000
        y_ann = float(np.interp(f_ann, f_m, db_m))
        y_ref = float(np.interp(f_ann, f_ref[mask], db_ref[mask]))
        delta = y_ann - y_ref
        delta_str = "" if abs(delta) < 0.2 else f"\n({delta:+.1f} dB vs T12)"
        ax.annotate(
            f"{label}: {y_ann:+.1f} dB{delta_str}",
            xy=(f_ann, y_ann),
            xytext=(f_ann * 0.55, y_ann),
            fontsize=8.5, color=color, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
        )

    ax.axhline(0, color="#9E9E9E", lw=1.2, ls="--", label="0 dB  =  bypass (reamp only)")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: (f"{int(x)}" if x < 1000 else f"{int(x)//1000}k")))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(40, 20_000)
    ax.set_ylim(-25, 10)
    ax.axhline(-3, color="k", lw=0.6, ls=":", alpha=0.4)   # −3 dB reference
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Pedal gain vs bypass (dB)", fontsize=11)
    ax.set_title(
        "JHS Notaklon — Treble knob sweep (G12, V12, bypass-normalised)\n"
        "4 short captures · opening 65 s sweep · 1/6-oct smoothed",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    out_path = OUT_DIR / "01_treble_sweep_fr.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
