#!/usr/bin/env python3
"""Compare KlonSim reference, PC model wet, and Daisy hardware wet.

Loads the dry recording and up to three wet versions, resamples all to a common
sample rate, time-aligns the Daisy capture via cross-correlation, then runs the
full metrics suite on every pair and saves a comprehensive plot suite.

All plotting is delegated to ``pedal_model.utils.plotting`` comparison functions;
this script handles I/O, alignment, metrics, and saving figures to disk.

Multi-panel compound figures (fewer files, richer panels):
    waveform_error_panel.png      — dry + overlaid + per-pred overlay & error
    spectrum_analysis_panel.png   — averaged spectrum | transfer function | coherence
    phase_analysis_panel.png      — phase response | group delay
    rms_envelopes.png             — rolling RMS for all signals
    mel_spectrograms.png          — side-by-side mel spectrograms
    phase_portraits.png           — Lissajous density portraits
    error_distributions.png       — error histogram + scatter density per pred
    harmonic_fingerprint.png      — H1–H5 bar chart
    radar_chart.png               — normalised metrics spider
    sliding_esr.png               — ESR in 1 s windows
    metrics_bar.png               — raw metric values per pair

Comparison pairs:
    ref  vs pc    — how well does the PC ODE model match KlonSim?
    ref  vs daisy — how well does the Daisy hardware match KlonSim?
    pc   vs daisy — how much does the hardware differ from the PC model?

Usage:
    uv run python scripts/compare_ode_daisy.py \\
        --dry   outputs/2026-06-09_medley_mono/2026-06-09_medley_mono__dry.wav \\
        --ref   results/pedalode/klonsim_v1/staged/audio/2026-06-09_medley_mono__klonsim_wet.wav \\
        --pc    results/pedalode/klonsim_v1/staged/audio/2026-06-09_medley_mono__model_wet.wav \\
        --daisy results/pedalode/klonsim_v1/staged/audio/2026-06-09_medley_mono__daisy_wet.wav
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from pedal_model.metrics.suite import compute_all_metrics
from pedal_model.utils.resampling import resample_audio
from pedal_model.utils.plotting import (
    compare_waveform_error_panel,
    compare_spectrum_analysis_panel,
    compare_phase_analysis_panel,
    compare_rms_envelopes,
    compare_mel_spectrograms,
    compare_phase_portraits,
    compare_error_distributions,
    compare_harmonic_fingerprint,
    compare_radar_chart,
    compare_sliding_esr,
)


# ── Script-local utilities ────────────────────────────────────────────────────


def _load(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(sr)


def _align_to_dry(
    dry: np.ndarray, wet: np.ndarray, max_lag_samples: int = 8192
) -> np.ndarray:
    """Shift wet to align with dry via cross-correlation (corrects ADC/DAC latency).

    Args:
        dry: Reference dry signal, shape (N,).
        wet: Wet signal to align, shape (M,).
        max_lag_samples: Maximum lag to search in either direction.

    Returns:
        Aligned wet signal, same length as dry.
    """
    n = min(len(dry), len(wet), 4 * max_lag_samples)
    corr = np.correlate(wet[:n], dry[:n], mode="full")
    lags = np.arange(-(n - 1), n)
    mask = np.abs(lags) <= max_lag_samples
    best_lag = int(lags[mask][np.argmax(np.abs(corr[mask]))])
    print(f"[compare]   best lag: {best_lag} samples ({best_lag/48000*1000:.2f} ms)")

    if best_lag >= 0:
        aligned = wet[best_lag:]
    else:
        aligned = np.concatenate([np.zeros(-best_lag, dtype=np.float32), wet])

    N = len(dry)
    if len(aligned) >= N:
        return aligned[:N]
    return np.concatenate([aligned, np.zeros(N - len(aligned), dtype=np.float32)])


def _run_metrics(
    label: str,
    target: np.ndarray,
    predicted: np.ndarray,
    dry: np.ndarray,
    sr: int,
) -> dict[str, float]:
    N = min(len(target), len(predicted), len(dry))
    result = compute_all_metrics(
        target[:N].astype(np.float32),
        predicted[:N].astype(np.float32),
        dry[:N].astype(np.float32),
        sr,
    )
    print(f"\n  [{label}]")
    for k, v in result.items():
        print(f"    {k:<20} {v:.6f}")
    return result


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_metrics_bar(results: dict[str, dict[str, float]], out_dir: Path) -> None:
    """Bar chart of raw metric values across comparison pairs."""
    keys = ["ESR", "null_depth_dB", "STFT", "FR_err_dB", "THD_err", "HP_sim", "MCD"]
    pairs = list(results.keys())
    x = np.arange(len(keys))
    width = 0.8 / len(pairs)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, pair in enumerate(pairs):
        vals = [results[pair].get(k, float("nan")) for k in keys]
        ax.bar(x + i * width, vals, width, label=pair, alpha=0.85)
    ax.set_xticks(x + width * (len(pairs) - 1) / 2)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Metrics by Comparison Pair")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, out_dir / "metrics_bar.png")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry",   type=Path, required=True, help="Dry guitar WAV.")
    p.add_argument("--ref",   type=Path, required=True, help="KlonSim reference wet WAV.")
    p.add_argument("--pc",    type=Path, required=True, help="PC ODE model wet WAV.")
    p.add_argument("--daisy", type=Path, default=None,
                   help="Daisy hardware wet WAV (optional unless --require-all).")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/pedalode/klonsim_v1/staged/daisy_comparison"))
    p.add_argument("--compare-sr", type=int, default=48_000,
                   help="Common sample rate for comparison (default: 48000).")
    p.add_argument("--require-all", action="store_true",
                   help="Error and stop if any file (including --daisy) is missing.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip RMS normalization (level differences will affect all metrics).")
    args = p.parse_args()

    if args.require_all and args.daisy is None:
        print("ERROR: --require-all set but --daisy was not provided.", file=sys.stderr)
        sys.exit(1)

    required = [args.dry, args.ref, args.pc]
    if args.daisy is not None:
        required.append(args.daisy)
    for path in required:
        if not path.is_file():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csr = args.compare_sr
    print(f"[compare] Comparison SR: {csr} Hz")

    # Load and resample all signals to common SR
    dry_raw, dry_sr = _load(args.dry)
    ref_raw, ref_sr = _load(args.ref)
    pc_raw,  pc_sr  = _load(args.pc)

    dry = resample_audio(dry_raw, dry_sr, csr)
    ref = resample_audio(ref_raw, ref_sr, csr)
    pc  = resample_audio(pc_raw,  pc_sr,  csr)

    N = min(len(dry), len(ref), len(pc))
    dry, ref, pc = dry[:N], ref[:N], pc[:N]

    signals: dict[str, np.ndarray] = {"KlonSim ref": ref, "PC model": pc}

    if args.daisy is not None:
        daisy_raw, daisy_sr = _load(args.daisy)
        daisy = resample_audio(daisy_raw, daisy_sr, csr)
        print(f"[compare] Aligning Daisy capture (native SR: {daisy_sr} Hz) …")
        daisy = _align_to_dry(dry, daisy)
        signals["Daisy hw"] = daisy

    ref_key = "KlonSim ref"
    ref = signals[ref_key]

    # ── RMS normalization ─────────────────────────────────────────────────────
    rms_ref = float(np.sqrt(np.mean(ref ** 2)))
    level_offsets: dict[str, float] = {}
    for label, sig in signals.items():
        if label == ref_key:
            level_offsets[label] = 0.0
            continue
        rms_sig = float(np.sqrt(np.mean(sig ** 2)))
        offset_db = 20.0 * np.log10(rms_ref / (rms_sig + 1e-10))
        level_offsets[label] = offset_db
        if not args.no_normalize:
            signals[label] = sig * float(10.0 ** (offset_db / 20.0))

    normalize_tag = "disabled (--no-normalize)" if args.no_normalize else "on"
    print(f"\n[compare] RMS normalization: {normalize_tag}")
    for label, db in level_offsets.items():
        applied = "" if args.no_normalize or label == ref_key else "  → applied"
        print(f"  {label:<20} {db:+.1f} dB{applied}")

    # ── Signal diagnostics ────────────────────────────────────────────────────
    print("\n[compare] Signal diagnostics …")
    rms_dry = float(np.sqrt(np.mean(dry ** 2)))
    print(f"  dry:                {20.0 * np.log10(rms_dry + 1e-10):+.1f} dBFS RMS  "
          f"  peak={20.0 * np.log10(float(np.max(np.abs(dry))) + 1e-10):+.1f} dBFS")
    for label, sig in signals.items():
        rms_sig = float(np.sqrt(np.mean(sig ** 2)))
        peak_sig = float(np.max(np.abs(sig)))
        corr_dry = float(np.corrcoef(sig[:N], dry[:N])[0, 1])
        corr_ref = float(np.corrcoef(sig[:N], ref[:N])[0, 1]) if label != ref_key else 1.0
        tag = "← ref" if label == ref_key else ""
        print(f"  {label:<20} {20.0 * np.log10(rms_sig + 1e-10):+.1f} dBFS RMS  "
              f"peak={20.0 * np.log10(peak_sig + 1e-10):+.1f} dBFS  "
              f"corr(dry)={corr_dry:+.3f}  corr(ref)={corr_ref:+.3f}  {tag}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print(f"\n[compare] Running metrics ({N / csr:.1f} s at {csr} Hz) …")

    def _pair_metrics(
        label: str, target: np.ndarray, predicted: np.ndarray, pred_label: str
    ) -> dict[str, float]:
        result = _run_metrics(label, target, predicted, dry, csr)
        return {"level_offset_dB": level_offsets.get(pred_label, 0.0), **result}

    all_results: dict[str, dict[str, float]] = {}
    all_results["ref vs pc"] = _pair_metrics(
        "KlonSim ref vs PC model", ref, signals["PC model"], "PC model")
    if "Daisy hw" in signals:
        all_results["ref vs daisy"] = _pair_metrics(
            "KlonSim ref vs Daisy hw", ref, signals["Daisy hw"], "Daisy hw")
        all_results["pc vs daisy"] = _pair_metrics(
            "PC model vs Daisy hw", signals["PC model"], signals["Daisy hw"], "Daisy hw")

    csv_path = args.out_dir / "metrics_table.csv"
    metric_keys = list(next(iter(all_results.values())).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair"] + metric_keys)
        writer.writeheader()
        for pair, metrics in all_results.items():
            writer.writerow({"pair": pair, **metrics})
    print(f"\n  Saved: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[compare] Generating plots …")

    _save(
        compare_waveform_error_panel(dry, signals, ref_key, csr),
        args.out_dir / "waveform_error_panel.png",
    )
    _save(
        compare_spectrum_analysis_panel(dry, signals, ref_key, csr),
        args.out_dir / "spectrum_analysis_panel.png",
    )
    _save(
        compare_phase_analysis_panel(dry, signals, ref_key, csr),
        args.out_dir / "phase_analysis_panel.png",
    )
    _save(
        compare_rms_envelopes(signals, csr),
        args.out_dir / "rms_envelopes.png",
    )
    _save(
        compare_mel_spectrograms(signals, csr),
        args.out_dir / "mel_spectrograms.png",
    )
    _save(
        compare_phase_portraits(signals, ref_key),
        args.out_dir / "phase_portraits.png",
    )
    _save(
        compare_error_distributions(signals, ref_key),
        args.out_dir / "error_distributions.png",
    )
    _save(
        compare_harmonic_fingerprint(dry, signals, csr),
        args.out_dir / "harmonic_fingerprint.png",
    )
    _save(
        compare_radar_chart(all_results),
        args.out_dir / "radar_chart.png",
    )
    _save(
        compare_sliding_esr(signals, ref_key, csr),
        args.out_dir / "sliding_esr.png",
    )
    _plot_metrics_bar(all_results, args.out_dir)

    print(f"\n[compare] Done — 11 plots + CSV in {args.out_dir}/")

    import subprocess
    win_path = subprocess.check_output(
        ["wslpath", "-w", str(args.out_dir.resolve())], text=True
    ).strip()
    subprocess.Popen(["explorer.exe", win_path])


if __name__ == "__main__":
    main()
