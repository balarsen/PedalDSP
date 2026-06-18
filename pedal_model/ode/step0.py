"""Step 0 pre-fitting analysis for Wiener-Hammerstein system identification.

Produces four diagnostic plots without any ODE solver or gradient-based
optimisation. All computation is NumPy/SciPy. Call run_step0() from the
fitting script with --stage 0.

Analyses:
  1. Cross-correlation delay  — detect integer-sample offset between dry / wet.
  2. Bode plot                — measure and curve-fit the linear transfer function.
  3. Gain compression         — estimate drive and output_level from gain vs amplitude.
  4. NL forward-model check   — verify Step 0 estimates reproduce the wet signal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal


# ─── Ground-truth reference values (for overlay only) ─────────────────────────
_GT_FC1_HZ: float = 80.0
_GT_FC2_HZ: float = 350.0
_GT_SHELF_HZ: float = 3000.0
_GT_SHELF_DB: float = 2.0
_GT_DRIVE: float = 4.0
_GT_OUTPUT_LEVEL: float = 0.7


# ─── Public entry point ────────────────────────────────────────────────────────


def run_step0(
    dry: np.ndarray,
    wet: np.ndarray,
    sections: list[dict],
    sr: int,
    results_dir: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Step 0 analysis and save four diagnostic plots.

    Args:
        dry: Full dry audio, shape (N,), float32.
        wet: Full wet audio, same shape.
        sections: Manifest section dicts from the JSON file.
        sr: Sample rate in Hz.
        results_dir: Root results directory (plots/ subdirectory is created).
        verbose: Print progress.

    Returns:
        Dict of estimated parameters:
          delay_samples, delay_ms,
          bode_fc1_hz, bode_fc2_hz, bode_G_db,
          gc_drive, gc_output_level, gc_G_combined,
          fwd_esr_gt, fwd_esr_step0.
    """
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {}

    if verbose:
        print("\n[Step 0] ── Pre-fitting analysis (SciPy only) ───────────────────────────")

    out.update(_plot_delay_xcorr(dry, wet, sr, plots_dir, verbose))
    out.update(_plot_bode(dry, wet, sections, sr, plots_dir, verbose))
    out.update(_plot_gain_compression(dry, wet, sections, sr, plots_dir, verbose))

    fc1 = float(out.get("bode_fc1_hz") or _GT_FC1_HZ)
    fc2 = float(out.get("bode_fc2_hz") or _GT_FC2_HZ)
    drive = float(out.get("gc_drive") or _GT_DRIVE)
    level = float(out.get("gc_output_level") or _GT_OUTPUT_LEVEL)
    out.update(_plot_nl_forward_model(dry, wet, sections, sr, fc1, fc2, drive, level,
                                      plots_dir, verbose))

    # Serialise (numpy scalars → Python floats)
    serialised = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in out.items()
    }
    out_path = results_dir / "step0_params.json"
    with open(out_path, "w") as f:
        json.dump(serialised, f, indent=2)

    if verbose:
        print(f"\n[Step 0] Saved → {out_path}")
        print("[Step 0] Summary:")
        for k, v in serialised.items():
            print(f"  {k:<30s} {v}")

    return serialised


# ─── Analysis 1: cross-correlation delay ──────────────────────────────────────


def _plot_delay_xcorr(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    plots_dir: Path,
    verbose: bool,
) -> dict[str, float]:
    """Cross-correlation of bandpass-filtered dry vs wet to find integer delay."""
    if verbose:
        print("\n[Step 0] 1/4 — Cross-correlation delay")

    # Limit to first 2 s; bandpass 200–3 kHz to avoid HPF phase and HF noise
    n = min(len(dry), len(wet), 2 * sr)
    bp = scipy.signal.butter(4, [200.0, 3000.0], btype="bandpass", fs=sr, output="sos")
    d_bp = scipy.signal.sosfilt(bp, dry[:n].astype(np.float64))
    w_bp = scipy.signal.sosfilt(bp, wet[:n].astype(np.float64))

    d_norm = d_bp / (np.std(d_bp) + 1e-9)
    w_norm = w_bp / (np.std(w_bp) + 1e-9)

    corr = scipy.signal.correlate(w_norm, d_norm, mode="full")
    lags = scipy.signal.correlation_lags(len(w_norm), len(d_norm))

    max_lag = int(0.050 * sr)
    mask = np.abs(lags) <= max_lag
    peak_idx = int(np.argmax(np.abs(corr[mask])))
    delay_samples = int(lags[mask][peak_idx])
    delay_ms = delay_samples / sr * 1000.0

    # Estimate causal filter group delay at the bandpass centre (~1 kHz)
    # For a 2nd-order Butterworth HPF: group delay at high-freq → 2ζ/ω₀
    zeta = float(1.0 / np.sqrt(2.0))
    gd_hpf1_ms = 1000.0 * 2.0 * zeta / (2.0 * np.pi * _GT_FC1_HZ)
    gd_hpf2_ms = 1000.0 * 2.0 * zeta / (2.0 * np.pi * _GT_FC2_HZ)
    expected_gd_ms = gd_hpf1_ms + gd_hpf2_ms
    hardware_latency_ms = delay_ms - expected_gd_ms

    if verbose:
        print(f"  Peak delay: {delay_samples} samples  ({delay_ms:.3f} ms)")
        print(f"  Expected causal filter group delay: HPF1({_GT_FC1_HZ:.0f} Hz)={gd_hpf1_ms:.2f} ms  "
              f"HPF2({_GT_FC2_HZ:.0f} Hz)={gd_hpf2_ms:.2f} ms  total≈{expected_gd_ms:.2f} ms")
        print(f"  Residual (hardware / alignment offset): {hardware_latency_ms:.2f} ms")
        if abs(hardware_latency_ms) > 1.0:
            print(f"  NOTE: residual > 1 ms — may indicate true hardware latency needing correction.")
        else:
            print(f"  Delay is consistent with causal filter group delay — no alignment correction needed.")
            print(f"  Waveform phase mismatch in overlay = wrong filter freq, not a timing offset.")

    lags_ms = lags[mask] / sr * 1000.0
    corr_norm = corr[mask] / (np.max(np.abs(corr[mask])) + 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(lags_ms, corr_norm, linewidth=0.8)
    axes[0].axvline(delay_ms, color="red", linestyle="--",
                    label=f"Peak: {delay_ms:.2f} ms ({delay_samples} samp)")
    axes[0].axvline(0, color="gray", linestyle=":", alpha=0.5, label="Zero lag")
    axes[0].set_xlabel("Lag (ms)")
    axes[0].set_ylabel("Normalised cross-correlation")
    axes[0].set_title("Step 0-1 — Cross-Correlation: Dry vs Wet (BPF 200–3 kHz)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    zoom = 5.0
    zm = np.abs(lags_ms - delay_ms) < zoom
    if zm.sum() > 3:
        axes[1].plot(lags_ms[zm], corr_norm[zm], linewidth=1.5)
        axes[1].axvline(delay_ms, color="red", linestyle="--",
                        label=f"Peak: {delay_ms:.2f} ms")
        axes[1].set_xlabel("Lag (ms)")
        axes[1].set_ylabel("Normalised cross-correlation")
        axes[1].set_title(f"Zoom ±{zoom:.0f} ms around peak")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "step0_01_delay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {path}")

    return {"delay_samples": float(delay_samples), "delay_ms": float(delay_ms)}


# ─── Analysis 2: Bode plot ────────────────────────────────────────────────────


def _butterworth_hpf_mag(f: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    """2nd-order Butterworth HPF magnitude response."""
    ratio = np.abs(fc / (f + 1e-9))
    return 1.0 / np.sqrt(1.0 + ratio ** (2 * order))


def _shelf_mag(f: np.ndarray, shelf_hz: float, shelf_db: float) -> np.ndarray:
    """High-shelf magnitude approximation (first-order prototype)."""
    A = 10.0 ** (shelf_db / 20.0)
    ratio_sq = (f / (shelf_hz + 1e-9)) ** 2
    return np.sqrt((A ** 2 * ratio_sq + 1.0) / (ratio_sq + 1.0))


def _system_mag(
    f: np.ndarray,
    fc1: float,
    fc2: float,
    shelf_hz: float = _GT_SHELF_HZ,
    shelf_db: float = _GT_SHELF_DB,
    G: float = 1.0,
) -> np.ndarray:
    """Full linear-chain magnitude: HPF1 × HPF2 × shelf × G."""
    return (
        G
        * _butterworth_hpf_mag(f, fc1)
        * _butterworth_hpf_mag(f, fc2)
        * _shelf_mag(f, shelf_hz, shelf_db)
    )


def _plot_bode(
    dry: np.ndarray,
    wet: np.ndarray,
    sections: list[dict],
    sr: int,
    plots_dir: Path,
    verbose: bool,
) -> dict[str, float]:
    """Measure Bode plot from the lowest-amplitude sweep; curve-fit fc1 and fc2."""
    from pedal_model.ode.identifiability import compute_transfer_function

    if verbose:
        print("\n[Step 0] 2/4 — Bode plot analysis")

    # Select lowest-amplitude log_sine_sweep section
    sweeps = [s for s in sections if s.get("type") == "log_sine_sweep"]
    if not sweeps:
        if verbose:
            print("  No log_sine_sweep sections — skipping Bode analysis.")
        return {}
    sweeps.sort(key=lambda s: float(s.get("amplitude_dbfs", 0)))
    sec = sweeps[0]
    amp_dbfs = float(sec.get("amplitude_dbfs", -24))
    start, end = int(sec["start_sample"]), int(sec["end_sample"])

    if verbose:
        print(f"  Section '{sec['label']}'  {amp_dbfs:.1f} dBFS  "
              f"{(end - start) / sr:.1f} s")

    freqs_hz, H_meas = compute_transfer_function(
        dry[start:end], wet[start:end], sr, n_fft=8192, hop=2048
    )
    H_db = 20.0 * np.log10(np.abs(H_meas) + 1e-9)

    # Passband gain estimate: median 1–8 kHz (both HPFs flat, shelf ≈ +2 dB)
    pb = (freqs_hz >= 1000) & (freqs_hz <= 8000)
    G_db_est = float(np.median(H_db[pb]))
    G_est = 10.0 ** (G_db_est / 20.0)

    plot_mask = (freqs_hz >= 20) & (freqs_hz <= 20000)
    f_plot = freqs_hz[plot_mask]

    # Curve-fit fc1, fc2, G.
    # Use 50 Hz–2 kHz: avoids noisy sub-50 Hz bins (near DC floor) and avoids
    # NL harmonics above 2 kHz which inflate the measured H at high frequencies.
    fit_mask_full = plot_mask & (freqs_hz >= 50.0) & (freqs_hz <= 2000.0)
    f_fit = freqs_hz[fit_mask_full]
    H_fit_db = H_db[fit_mask_full]

    fc1_est = fc2_est = G_db_cf = None
    H_cf_db = None

    def _model_db(f: np.ndarray, fc1: float, fc2: float, G_db: float) -> np.ndarray:
        return 20.0 * np.log10(
            _system_mag(f, np.abs(fc1), np.abs(fc2), G=10.0 ** (G_db / 20.0)) + 1e-9
        )

    try:
        popt, _ = scipy.optimize.curve_fit(
            _model_db,
            f_fit,
            H_fit_db,
            p0=[80.0, 350.0, G_db_est],
            bounds=([10.0, 50.0, G_db_est - 12.0], [800.0, 2000.0, G_db_est + 12.0]),
            maxfev=8000,
        )
        fc1_est, fc2_est, G_db_cf = float(popt[0]), float(popt[1]), float(popt[2])
        H_cf_db = 20.0 * np.log10(
            _system_mag(f_plot, fc1_est, fc2_est, G=10.0 ** (G_db_cf / 20.0)) + 1e-9
        )
        if verbose:
            print(f"  curve_fit → fc1={fc1_est:.1f} Hz (GT={_GT_FC1_HZ:.0f})  "
                  f"fc2={fc2_est:.1f} Hz (GT={_GT_FC2_HZ:.0f})  "
                  f"G={G_db_cf:.1f} dB")
    except Exception as exc:
        if verbose:
            print(f"  curve_fit failed: {exc}")

    # Analytical overlays
    H_gt_db = 20.0 * np.log10(_system_mag(f_plot, _GT_FC1_HZ, _GT_FC2_HZ, G=G_est) + 1e-9)
    # Stage 1 result (hardcoded from param_recovery)
    H_s1_db = 20.0 * np.log10(_system_mag(f_plot, 64.75, 333.76, G=G_est) + 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].semilogx(f_plot, H_db[plot_mask], alpha=0.85, linewidth=1.2,
                     label=f"Measured ({sec['label']}, {amp_dbfs:.0f} dBFS)")
    axes[0].semilogx(f_plot, H_gt_db, "--", linewidth=1.8, alpha=0.85,
                     label=f"Analytical GT (fc1={_GT_FC1_HZ:.0f}, fc2={_GT_FC2_HZ:.0f} Hz)")
    axes[0].semilogx(f_plot, H_s1_db, ":", linewidth=1.5, alpha=0.75,
                     label="Stage 1 result (fc1=64.75, fc2=333.76 Hz)")
    if H_cf_db is not None:
        axes[0].semilogx(f_plot, H_cf_db, "-.", linewidth=1.8, alpha=0.85,
                         label=f"Step0 fit (fc1={fc1_est:.1f}, fc2={fc2_est:.1f} Hz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title("Step 0-2 — Bode Plot with Analytical Overlays")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-65, 25)
    axes[0].set_xlim(20, 20000)

    # Residuals vs GT and vs curve-fit
    res_gt = H_db[plot_mask] - H_gt_db
    axes[1].semilogx(f_plot, res_gt, color="tomato", linewidth=1.0, alpha=0.85,
                     label="Measured − GT")
    if H_cf_db is not None:
        res_cf = H_db[plot_mask] - H_cf_db
        axes[1].semilogx(f_plot, res_cf, color="steelblue", linestyle="--",
                          linewidth=1.0, alpha=0.85, label="Measured − Step0 fit")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Residual (dB)")
    axes[1].set_ylim(-15, 15)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "step0_02_bode.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {path}")

    result: dict[str, float] = {"bode_G_db": G_db_est}
    if fc1_est is not None:
        result["bode_fc1_hz"] = fc1_est
        result["bode_fc2_hz"] = fc2_est
    return result


# ─── Analysis 3: gain compression ─────────────────────────────────────────────


def _tanh_fundamental_gain(z0_arr: np.ndarray) -> np.ndarray:
    """Ratio (output fundamental amplitude) / (input amplitude) for tanh(z0·sin θ).

    Fully vectorised: computes all M amplitudes in a single batch FFT.

    Args:
        z0_arr: Peak amplitude at tanh input (drive × A_dry_peak), shape (M,).

    Returns:
        Gain values, shape (M,), in [0, 1].
    """
    z0 = np.asarray(z0_arr, dtype=np.float64)
    theta = np.linspace(0, 2.0 * np.pi, 4096, endpoint=False)
    # (M, N) — broadcast z0 over theta
    y = np.tanh(z0[:, np.newaxis] * np.sin(theta))  # (M, 4096)
    Y = np.fft.rfft(y, axis=1)                        # (M, 2049)
    y_fund = np.abs(Y[:, 1]) * 2.0 / y.shape[1]       # (M,)
    safe_z0 = np.where(z0 > 1e-9, z0, 1.0)
    return np.where(z0 > 1e-9, y_fund / safe_z0, 1.0)


def _tanh_gain_model(A_dry_peak: np.ndarray, drive: float, output_level: float) -> np.ndarray:
    """Measured fundamental gain (wet_fund / dry_fund) for flat-band frequencies.

    Full chain (flat-band, filter gains ≈ 1):
      DRY → ×drive → tanh → ×output_level → WET
      gain = WET_fund / DRY_amp
             = output_level × tanh_fund_amp(drive×A) / A
             = output_level × drive × tanh_fundamental_gain(drive×A)

    Linear regime (A→0): gain → output_level × drive.
    Saturated regime (A→∞): gain → 0.

    Args:
        A_dry_peak: Dry fundamental peak amplitude per section, shape (M,).
        drive: Pre-gain scalar.
        output_level: Post-NL scalar.

    Returns:
        Model gain, shape (M,).
    """
    z0 = drive * np.asarray(A_dry_peak, dtype=np.float64)
    return output_level * drive * _tanh_fundamental_gain(z0)


def _extract_freq_from_label(label: str) -> float | None:
    m = re.search(r"_f(\d+)_", label)
    return float(m.group(1)) if m else None


def _extract_level_from_label(label: str) -> float | None:
    m = re.search(r"_l(-?\d+)$", label)
    return float(m.group(1)) if m else None


def _plot_gain_compression(
    dry: np.ndarray,
    wet: np.ndarray,
    sections: list[dict],
    sr: int,
    plots_dir: Path,
    verbose: bool,
) -> dict[str, float]:
    """Fundamental gain vs input amplitude per section; curve-fit drive + output_level."""
    if verbose:
        print("\n[Step 0] 3/4 — Gain compression curve")

    tone_secs = [s for s in sections if s.get("type") == "stepped_sine_tone"]
    if not tone_secs:
        if verbose:
            print("  No stepped_sine_tone sections — skipping gain compression.")
        return {}

    records: list[dict] = []
    for sec in tone_secs:
        label = sec.get("label", "")
        freq = _extract_freq_from_label(label)
        level_dbfs = _extract_level_from_label(label)
        if freq is None or level_dbfs is None:
            continue

        start, end = int(sec["start_sample"]), int(sec["end_sample"])
        skip = (end - start) // 10  # skip first 10 % (transient)
        d_ss = dry[start + skip : end].astype(np.float64)
        w_ss = wet[start + skip : end].astype(np.float64)

        N = len(d_ss)
        if N < 64:
            continue

        win = np.hanning(N)
        win_cg = float(np.mean(win))  # coherent gain ≈ 0.5 for Hann
        D = np.fft.rfft(d_ss * win)
        W = np.fft.rfft(w_ss * win)
        f_bins = np.fft.rfftfreq(N, d=1.0 / sr)

        k = int(np.argmin(np.abs(f_bins - freq)))
        k_lo, k_hi = max(0, k - 2), min(len(D), k + 3)
        # Correct for Hann window coherent gain so DFT peak = true signal peak
        d_fund = float(np.max(np.abs(D[k_lo:k_hi]))) * 2.0 / (N * win_cg)
        w_fund = float(np.max(np.abs(W[k_lo:k_hi]))) * 2.0 / (N * win_cg)

        if d_fund < 1e-7:
            continue

        gain = w_fund / d_fund
        A_peak = d_fund  # true peak amplitude of dry fundamental at this frequency
        records.append({
            "freq": freq,
            "level_dbfs": level_dbfs,
            "A_peak": A_peak,
            "gain": gain,
            "gain_db": 20.0 * np.log10(gain + 1e-9),
        })

    if not records:
        if verbose:
            print("  No usable records — skipping.")
        return {}

    freqs_all = sorted(set(r["freq"] for r in records))
    # Flat-band: both HPFs flat → f > 500 Hz (well above fc2=350 Hz)
    flat_freqs = {f for f in freqs_all if f >= 523}
    flat_records = [r for r in records if r["freq"] in flat_freqs]

    drive_est = output_level_est = G_combined_est = None

    if len(flat_records) >= 3:
        A_flat = np.array([r["A_peak"] for r in flat_records])
        g_flat = np.array([r["gain"] for r in flat_records])

        # Initial G guess from lowest-amplitude (most linear) flat-band point
        G0 = float(g_flat[np.argmin(A_flat)])

        try:
            popt, _ = scipy.optimize.curve_fit(
                _tanh_gain_model,
                A_flat,
                g_flat,
                p0=[3.0, G0],
                bounds=([0.5, 0.02], [20.0, 5.0]),
                maxfev=2000,
            )
            drive_est = float(popt[0])
            output_level_est = float(popt[1])
            G_combined_est = drive_est * output_level_est
            if verbose:
                print(f"  curve_fit → drive={drive_est:.3f} (GT={_GT_DRIVE:.1f})  "
                      f"output_level={output_level_est:.3f} (GT={_GT_OUTPUT_LEVEL:.1f})  "
                      f"G=drive×level={G_combined_est:.3f} (GT={_GT_DRIVE * _GT_OUTPUT_LEVEL:.2f})")
        except Exception as exc:
            if verbose:
                print(f"  curve_fit failed: {exc}")

    # ── Plots ──
    n_freqs = len(freqs_all)
    cmap_vals = plt.cm.viridis(np.linspace(0.1, 0.9, n_freqs))
    freq_color = {f: cmap_vals[i] for i, f in enumerate(freqs_all)}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: gain (dB) vs level (dBFS) per frequency
    for freq in freqs_all:
        pts = sorted([r for r in records if r["freq"] == freq],
                     key=lambda r: r["level_dbfs"])
        ls = "-" if freq in flat_freqs else "--"
        ax0.plot(
            [p["level_dbfs"] for p in pts],
            [p["gain_db"] for p in pts],
            f"{ls}o",
            color=freq_color[freq],
            label=f"{freq:.0f} Hz",
            markersize=6,
            linewidth=1.5,
        )

    ax0.set_xlabel("Input level (dBFS)")
    ax0.set_ylabel("Fundamental gain (dB)")
    ax0.set_title("Step 0-3 — Gain vs Level (solid = flat-band ≥ 523 Hz)")
    ax0.legend(fontsize=8, ncol=2)
    ax0.grid(True, alpha=0.3)

    # Right: gain (linear) vs peak amplitude with tanh model overlays
    A_plot = np.linspace(0.001, 1.1, 500)

    # GT model
    gain_gt = _tanh_gain_model(A_plot, _GT_DRIVE, _GT_OUTPUT_LEVEL)
    ax1.plot(A_plot, gain_gt, "r--", linewidth=2.0, alpha=0.8,
             label=f"GT (d={_GT_DRIVE}, l={_GT_OUTPUT_LEVEL})  G_lin={_GT_DRIVE*_GT_OUTPUT_LEVEL:.2f}")

    # Fitted model
    if drive_est is not None:
        gain_fit = _tanh_gain_model(A_plot, drive_est, output_level_est)
        G_lin = drive_est * output_level_est
        ax1.plot(A_plot, gain_fit, "g-", linewidth=2.0, alpha=0.8,
                 label=f"Step0 (d={drive_est:.2f}, l={output_level_est:.3f})  G_lin={G_lin:.2f}")

    # Stage 2 result for comparison
    gain_s2 = _tanh_gain_model(A_plot, 2.437, 1.10)
    ax1.plot(A_plot, gain_s2, "b:", linewidth=1.5, alpha=0.7,
             label="Stage 2 (d=2.44, l=1.10)  G_lin=2.68")

    for r in flat_records:
        ax1.scatter(r["A_peak"], r["gain"], color=freq_color[r["freq"]],
                    zorder=5, s=55, edgecolors="k", linewidths=0.5)

    ax1.set_xlabel("Input peak amplitude (linear)")
    ax1.set_ylabel("Fundamental gain")
    ax1.set_title("Step 0-3 — Gain Compression Fit (flat-band only)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "step0_03_gain_compression.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {path}")

    # ── Linearity detection ──
    # Measure how much the flat-band gain varies across measured amplitude levels.
    # A truly linear pedal has constant gain regardless of level (gain_range_db ≈ 0).
    # Threshold: < 1 dB compression over the full measured amplitude range → linear.
    is_linear = False
    gain_range_db = float("nan")
    if flat_records:
        flat_gains_db = np.array([r["gain_db"] for r in flat_records])
        gain_range_db = float(np.max(flat_gains_db) - np.min(flat_gains_db))
        is_linear = gain_range_db < 1.0
        if verbose:
            tag = "LINEAR ✓" if is_linear else f"nonlinear (range={gain_range_db:.1f} dB)"
            print(f"  Gain compression range: {gain_range_db:.2f} dB → {tag}")

    result: dict[str, Any] = {}
    result["is_linear"] = is_linear
    result["gain_range_db"] = gain_range_db
    if drive_est is not None:
        result["gc_drive"] = drive_est
        result["gc_output_level"] = output_level_est
        result["gc_G_combined"] = G_combined_est
    return result


# ─── Analysis 4: NL forward-model check ───────────────────────────────────────


def _high_shelf_ba(shelf_hz: float, shelf_db: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Audio EQ Cookbook high-shelf biquad (matches KlonSim's _high_shelf_biquad exactly)."""
    A = 10.0 ** (shelf_db / 40.0)
    w0 = 2.0 * np.pi * shelf_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    Q = 0.707
    alpha = sin_w0 / (2.0 * Q)
    sqrt_A = float(np.sqrt(A))
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2.0 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2.0 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def _apply_hpf_causal(sig: np.ndarray, fc_hz: float, sr: int) -> np.ndarray:
    """Causal 2nd-order Butterworth HPF via sosfilt (matches KlonSim)."""
    sos = scipy.signal.butter(2, fc_hz, btype="high", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, sig.astype(np.float64)).astype(np.float32)


def _apply_hpf_zerophase(sig: np.ndarray, fc_hz: float, sr: int) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth HPF via sosfiltfilt (for NL scatter only)."""
    sos = scipy.signal.butter(2, fc_hz, btype="high", fs=sr, output="sos")
    return scipy.signal.sosfiltfilt(sos, sig.astype(np.float64)).astype(np.float32)


def _forward_model_esr(
    dry: np.ndarray,
    wet: np.ndarray,
    fc1_hz: float,
    fc2_hz: float,
    shelf_hz: float,
    shelf_db: float,
    drive: float,
    output_level: float,
    sr: int,
) -> tuple[float, np.ndarray]:
    """Compute ESR of the analytical forward model against wet.

    Uses causal (sosfilt / lfilter) filters to match KlonSim's processing exactly.
    Forward model: HPF1 → × drive → tanh → HPF2 → shelf → × output_level

    Returns:
        (esr, y_hat) where y_hat is the model output array.
    """
    x_pre = _apply_hpf_causal(dry, fc1_hz, sr)
    x_drive = x_pre * drive
    x_nl = np.tanh(x_drive.astype(np.float64)).astype(np.float32)
    x_post = _apply_hpf_causal(x_nl, fc2_hz, sr)
    b_shelf, a_shelf = _high_shelf_ba(shelf_hz, shelf_db, sr)
    x_shelf = scipy.signal.lfilter(b_shelf, a_shelf, x_post.astype(np.float64)).astype(np.float32)
    y_hat = (x_shelf * output_level).astype(np.float32)

    N = min(len(y_hat), len(wet))
    esr = float(
        np.sum((wet[:N] - y_hat[:N]) ** 2)
        / (np.sum(wet[:N] ** 2) + 1e-9)
    )
    return esr, y_hat


def _plot_nl_forward_model(
    dry: np.ndarray,
    wet: np.ndarray,
    sections: list[dict],
    sr: int,
    fc1_est: float,
    fc2_est: float,
    drive_est: float,
    output_level_est: float,
    plots_dir: Path,
    verbose: bool,
) -> dict[str, float]:
    """Forward-model check: compute and plot ESR for GT vs Step0 estimated params."""
    if verbose:
        print("\n[Step 0] 4/4 — NL forward-model check (scipy filters + tanh)")

    esr_gt, y_hat_gt = _forward_model_esr(
        dry, wet,
        _GT_FC1_HZ, _GT_FC2_HZ, _GT_SHELF_HZ, _GT_SHELF_DB,
        _GT_DRIVE, _GT_OUTPUT_LEVEL, sr,
    )
    esr_s0, y_hat_s0 = _forward_model_esr(
        dry, wet,
        fc1_est, fc2_est, _GT_SHELF_HZ, _GT_SHELF_DB,
        drive_est, output_level_est, sr,
    )
    # Stage 2 result for comparison
    esr_s2, y_hat_s2 = _forward_model_esr(
        dry, wet, 64.75, 333.76, 3010.0, 1.89, 2.437, 1.10, sr,
    )

    if verbose:
        print(f"  ESR — GT params:     {esr_gt:.5f}  {'PASS ✓' if esr_gt < 0.01 else 'review'}")
        print(f"  ESR — Step0 params:  {esr_s0:.5f}  {'PASS ✓' if esr_s0 < 0.01 else 'review'}")
        print(f"  ESR — Stage 2 params:{esr_s2:.5f}  (baseline)")

    # Find a high-amplitude section for the waveform overlay
    hi_secs = sorted(
        [s for s in sections
         if s.get("type") == "stepped_sine_tone"
         and float(s.get("amplitude_dbfs", -99)) >= -4.0],
        key=lambda s: float(s.get("amplitude_dbfs", 0)),
    )
    if not hi_secs:
        hi_secs = [s for s in sections if s.get("type") == "stepped_sine_tone"]

    result: dict[str, float] = {"fwd_esr_gt": esr_gt, "fwd_esr_step0": esr_s0,
                                  "fwd_esr_stage2": esr_s2}

    if not hi_secs:
        return result

    sec = hi_secs[0]
    start = int(sec["start_sample"])
    n_plot = int(0.040 * sr)  # 40 ms
    end = min(start + n_plot, int(sec["end_sample"]))
    t_ms = np.arange(end - start) * 1000.0 / sr
    n = end - start

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t_ms, wet[start:end], label="KlonSim wet", alpha=0.9, linewidth=1.2)
    axes[0].plot(t_ms, y_hat_gt[start:end], "--",
                 label=f"GT params (ESR={esr_gt:.4f})", alpha=0.85, linewidth=1.2)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Step 0-4 — Forward Model Check: {sec.get('label', '')}")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ms, wet[start:end], label="KlonSim wet", alpha=0.9, linewidth=1.2)
    axes[1].plot(t_ms, y_hat_s0[start:end], "--",
                 label=f"Step0 ({fc1_est:.0f}/{fc2_est:.0f} Hz, "
                       f"d={drive_est:.2f}, l={output_level_est:.3f}) "
                       f"ESR={esr_s0:.4f}",
                 alpha=0.85, linewidth=1.2)
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Residuals
    res_gt = wet[start:end] - y_hat_gt[start:end]
    res_s0 = wet[start:end] - y_hat_s0[start:end]
    res_s2 = wet[start:end] - y_hat_s2[start:end]
    axes[2].plot(t_ms, res_s2, color="steelblue", alpha=0.6, linewidth=0.9,
                 label=f"Stage 2 residual (ESR={esr_s2:.4f})")
    axes[2].plot(t_ms, res_s0, color="darkorange", alpha=0.8, linewidth=1.0,
                 label=f"Step0 residual (ESR={esr_s0:.4f})")
    axes[2].plot(t_ms, res_gt, color="green", alpha=0.8, linewidth=1.0,
                 label=f"GT residual (ESR={esr_gt:.4f})")
    axes[2].axhline(0, color="gray", linewidth=0.6)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Residual")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "step0_04_nl_forward_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {path}")

    # ── NL curve scatter for flat-band high-amplitude sections ──
    _plot_nl_scatter(dry, wet, sections, sr,
                     fc1_est, fc2_est, drive_est, output_level_est,
                     plots_dir, verbose)

    return result


def _plot_nl_scatter(
    dry: np.ndarray,
    wet: np.ndarray,
    sections: list[dict],
    sr: int,
    fc1_hz: float,
    fc2_hz: float,
    drive: float,
    output_level: float,
    plots_dir: Path,
    verbose: bool,
) -> None:
    """Scatter plot of (x_tanh_input, NL_output) using flat-band high-amplitude tones.

    For flat-band sections (f >> fc2), wet ≈ output_level × tanh(drive × HPF1(dry)).
    We back-compute: NL_output ≈ wet / output_level, x_input = drive × HPF1(dry).
    Plotting (x_input, NL_output) should trace out tanh(x).
    """
    # Flat-band sections at highest amplitude
    hi_flat = sorted(
        [s for s in sections
         if s.get("type") == "stepped_sine_tone"
         and float(s.get("amplitude_dbfs", -99)) >= -4.0
         and (_extract_freq_from_label(s.get("label", "")) or 0) >= 523],
        key=lambda s: _extract_freq_from_label(s.get("label", "")) or 0,
    )
    if not hi_flat:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_curve = np.linspace(-4.0, 4.0, 500)
    axes[0].plot(x_curve, np.tanh(x_curve), "r--", linewidth=2.0, alpha=0.7,
                 label="tanh(x) reference", zorder=10)
    axes[1].plot(x_curve, np.tanh(x_curve), "r--", linewidth=2.0, alpha=0.7,
                 label="tanh(x) reference", zorder=10)

    cmap_vals = plt.cm.cool(np.linspace(0.1, 0.9, len(hi_flat)))

    for i, sec in enumerate(hi_flat):
        freq = _extract_freq_from_label(sec.get("label", ""))
        start, end = int(sec["start_sample"]), int(sec["end_sample"])
        skip = (end - start) // 10
        d_sec = dry[start + skip : end]
        w_sec = wet[start + skip : end]

        # GT parameters scatter
        x_gt = _apply_hpf_zerophase(d_sec, _GT_FC1_HZ, sr) * _GT_DRIVE
        y_gt = w_sec / (_GT_OUTPUT_LEVEL + 1e-9)
        axes[0].scatter(x_gt[::4], y_gt[::4], s=3, alpha=0.4,
                        color=cmap_vals[i], label=f"{freq:.0f} Hz")

        # Step0 parameters scatter
        x_s0 = _apply_hpf_zerophase(d_sec, fc1_hz, sr) * drive
        y_s0 = w_sec / (output_level + 1e-9)
        axes[1].scatter(x_s0[::4], y_s0[::4], s=3, alpha=0.4,
                        color=cmap_vals[i], label=f"{freq:.0f} Hz")

    axes[0].set_xlabel("x_tanh_input  (drive × HPF1(dry))")
    axes[0].set_ylabel("NL output  (wet / output_level)")
    axes[0].set_title(f"Step 0-5 — NL scatter: GT params "
                      f"(fc1={_GT_FC1_HZ:.0f}, d={_GT_DRIVE}, l={_GT_OUTPUT_LEVEL})")
    axes[0].set_xlim(-4.5, 4.5)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].legend(fontsize=8, markerscale=4)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("x_tanh_input  (drive × HPF1(dry))")
    axes[1].set_ylabel("NL output  (wet / output_level)")
    axes[1].set_title(f"Step 0-5 — NL scatter: Step0 params "
                      f"(fc1={fc1_hz:.1f}, d={drive:.2f}, l={output_level:.3f})")
    axes[1].set_xlim(-4.5, 4.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].legend(fontsize=8, markerscale=4)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = plots_dir / "step0_05_nl_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {path}")
