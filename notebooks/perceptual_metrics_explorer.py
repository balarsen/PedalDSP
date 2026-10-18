# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy",
#   "scipy",
#   "matplotlib",
#   "seaborn",
#   "pandas",
#   "soundfile",
# ]
# ///
"""Perceptual metrics explorer — diagnostic visualization for all six weighted spectral metrics.

Run with:
    marimo run  notebooks/perceptual_metrics_explorer.py
    marimo edit notebooks/perceptual_metrics_explorer.py
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import scipy.signal as sig
    import marimo as mo

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from pedal_model.metrics.perceptual import (
        mel_axis_weighted,
        equal_loudness_weighted,
        masking_weighted,
        signal_adaptive_weighted,
        guitar_prior_weighted,
        harmonic_weighted,
        composite_perceptual,
    )
    from pedal_model.eval.results import ZooResults

    BG = "#1a1a2e"
    C_TARGET  = "#00FF88"
    C_PRED    = "#FF6B6B"
    C_DRY     = "#4FC3F7"
    C_ERROR   = "#FFD700"
    C_EVEN    = "#4FC3F7"
    C_ODD     = "#FF6B6B"
    C_NEUTRAL = "#888888"

    def dark_fig(nrows=1, ncols=1, figsize=None):
        """Create a dark-background figure with styled axes."""
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (12, 4 * nrows))
        fig.patch.set_facecolor(BG)
        for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
            ax.set_facecolor(BG)
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")
        return fig, axes

    def dark_ax(ax):
        """Apply dark style to an existing Axes."""
        ax.set_facecolor(BG)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        return ax

    print("Imports OK")
    return (
        BG,
        C_ERROR,
        C_EVEN,
        C_NEUTRAL,
        C_ODD,
        C_PRED,
        C_TARGET,
        PROJECT_ROOT,
        Path,
        ZooResults,
        composite_perceptual,
        dark_ax,
        dark_fig,
        equal_loudness_weighted,
        guitar_prior_weighted,
        harmonic_weighted,
        masking_weighted,
        mel_axis_weighted,
        mo,
        mticker,
        np,
        pd,
        plt,
        sig,
        signal_adaptive_weighted,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Perceptual Metrics Explorer

    Six weighted spectral error metrics that catch distinct perceptual failure modes.
    Each score is **lower = better**. Explore model predictions vs the target signal
    and see where errors are perceptually salient.
    """)
    return


@app.cell
def _(PROJECT_ROOT, mo):
    results_path_input = mo.ui.text(
        value=str(
            PROJECT_ROOT / "results" / "data" / "klon_sim" / "klon_sim_v1" / "results.pkl"
        ),
        label="ZooResults pickle path (leave blank for synthetic signals)",
        full_width=True,
    )
    results_path_input
    return (results_path_input,)


@app.cell
def _(Path, ZooResults, mo, np, results_path_input, sig):
    rpath = Path(results_path_input.value)

    if rpath.exists():
        zoo = ZooResults.load(rpath)
        sr = zoo.sr
        dry_full    = zoo.dry_val
        target_full = zoo.wet_val
        model_names = list(zoo.models.keys())
        preds       = {k: zoo.models[k].prediction for k in model_names}
        source_label = f"ZooResults: {len(model_names)} models, sr={sr} Hz"
    else:
        sr = 48_000
        N  = sr * 4
        t  = np.arange(N) / sr
        dry_full    = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        target_full = np.clip(dry_full * 8.0, -0.9, 0.9).astype(np.float32)
        sos_lp = sig.butter(4, 2000 / (sr / 2), btype="low",  output="sos")
        sos_hp = sig.butter(4,  500 / (sr / 2), btype="high", output="sos")
        lp = sig.sosfilt(sos_lp, target_full).astype(np.float32)
        hp = sig.sosfilt(sos_hp, target_full).astype(np.float32)
        model_names = ["lp_filtered", "hp_filtered"]
        preds       = {"lp_filtered": lp, "hp_filtered": hp}
        source_label = "Synthetic: 220 Hz sine → clipped target | LP / HP preds"

    mo.md(f"**Loaded:** {source_label}")
    return dry_full, model_names, preds, sr, target_full


@app.cell
def _(mo, model_names):
    model_selector = mo.ui.dropdown(
        options=model_names,
        value=model_names[0],
        label="Model to inspect",
    )
    model_selector
    return (model_selector,)


@app.cell
def _(dry_full, mo, model_selector, np, preds, sr, target_full):
    MAX_SAMPLES = int(sr * 10)
    dry    = dry_full[:MAX_SAMPLES].astype(np.float32)
    target = target_full[:MAX_SAMPLES].astype(np.float32)
    pred   = preds[model_selector.value][:MAX_SAMPLES].astype(np.float32)
    mo.md(
        f"**Active model:** `{model_selector.value}` | "
        f"{len(dry)/1e3:.1f}k samples @ {sr} Hz"
    )
    return dry, pred, target


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2 — All six metrics
    """)
    return


@app.cell
def _(
    composite_perceptual,
    dry,
    equal_loudness_weighted,
    guitar_prior_weighted,
    harmonic_weighted,
    masking_weighted,
    mel_axis_weighted,
    mo,
    pred,
    signal_adaptive_weighted,
    sr,
    target,
):
    results = {
        "mel_axis_weighted":        mel_axis_weighted(pred, target, dry, sr=sr),
        "equal_loudness_weighted":  equal_loudness_weighted(pred, target, dry, sr=sr),
        "masking_weighted":         masking_weighted(pred, target, dry, sr=sr),
        "signal_adaptive_weighted": signal_adaptive_weighted(pred, target, dry, sr=sr),
        "guitar_prior_weighted":    guitar_prior_weighted(pred, target, dry, sr=sr),
        "harmonic_weighted":        harmonic_weighted(pred, target, dry, sr=sr),
    }
    comp = composite_perceptual(pred, target, dry, sr=sr)

    notes_map = {
        "mel_axis_weighted":        "axis warping only",
        "equal_loudness_weighted":  "sensitivity weighting",
        "masking_weighted":         f"masked_fraction={results['masking_weighted']['metadata']['masked_fraction']:.0%}",
        "signal_adaptive_weighted": "signal-energy weighted",
        "guitar_prior_weighted":    "guitar register prior",
        "harmonic_weighted":        f"f0={results['harmonic_weighted']['metadata']['f0_hz']:.1f} Hz, N=12 harmonics",
    }
    rows = [
        {
            "Metric": name,
            "Score": f"{res['score']:.4f}",
            "Unit": res["unit"],
            "Notes": notes_map[name],
        }
        for name, res in results.items()
    ]
    rows.append({
        "Metric": "composite",
        "Score": f"{comp['composite_score']:.4f}",
        "Unit": "z-score",
        "Notes": "equal weights",
    })

    mo.vstack([mo.md("### Summary"), mo.ui.table(data=rows)])
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 3 — Per-metric visualization
    """)
    return


@app.cell
def _(C_ERROR, dark_fig, mo, np, plt, results):
    r1 = results["mel_axis_weighted"]
    em  = r1["arrays"]["error_map"]
    pbe = r1["arrays"]["per_band_error"]
    mf  = np.array(r1["arrays"]["mel_freqs"])
    n_mels = len(mf)

    fig1, (ax1l, ax1r) = dark_fig(1, 2, figsize=(14, 5))
    fig1.suptitle("3.1 — Mel-Axis Weighted Distance", color="white", fontsize=13)

    im1 = ax1l.imshow(
        em, aspect="auto", origin="lower", cmap="inferno",
        extent=[0, em.shape[1], 0, n_mels],
    )
    plt.colorbar(im1, ax=ax1l, label="|error| dB", fraction=0.046, pad=0.04)
    tick_idx = np.linspace(0, n_mels - 1, 8, dtype=int)
    ax1l.set_yticks(tick_idx)
    ax1l.set_yticklabels([f"{mf[i]:.0f}" for i in tick_idx])
    ax1l.set_xlabel("Frame")
    ax1l.set_ylabel("Frequency (Hz)")
    ax1l.set_title(f"2D error map  (score={r1['score']:.3f} dB)")

    ax1r.barh(np.arange(n_mels), pbe, color=C_ERROR, alpha=0.85)
    tick_idx2 = np.linspace(0, n_mels - 1, 10, dtype=int)
    ax1r.set_yticks(tick_idx2)
    ax1r.set_yticklabels([f"{mf[i]:.0f} Hz" for i in tick_idx2])
    ax1r.set_xlabel("Mean |error| dB")
    ax1r.set_ylabel("Mel band")
    ax1r.set_title("Per-band mean error")

    fig1.tight_layout()
    mo.mpl.interactive(fig1)
    return


@app.cell
def _(C_ERROR, C_NEUTRAL, C_PRED, dark_fig, mo, mticker, np, results):
    r2 = results["equal_loudness_weighted"]
    freqs2  = np.array(r2["arrays"]["freqs_hz"])
    a_db    = r2["arrays"]["a_weights_db"]
    wt_err2 = r2["arrays"]["per_bin_weighted_error"]
    raw2    = r2["arrays"]["raw_error_per_bin"]

    fig2, (ax2l, ax2r) = dark_fig(1, 2, figsize=(14, 5))
    fig2.suptitle("3.2 — Equal Loudness (A-Weighting)", color="white", fontsize=13)

    ax2l.semilogx(freqs2[1:], a_db[1:], color=C_PRED, linewidth=2)
    ax2l.axvspan(82, 6000, alpha=0.12, color=C_PRED, label="Guitar range 82–6k Hz")
    ax2l.set_xlabel("Frequency (Hz)")
    ax2l.set_ylabel("A-weighting (dB)")
    ax2l.set_title("IEC 61672 A-weighting curve")
    ax2l.legend(facecolor="#2a2a4a", labelcolor="white")
    ax2l.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    ax2r.semilogx(freqs2[1:], raw2[1:],    color=C_NEUTRAL, linewidth=1,
                  alpha=0.6, label="Raw error")
    ax2r.semilogx(freqs2[1:], wt_err2[1:], color=C_ERROR, linewidth=2,
                  label="A-weighted error")
    ax2r.set_xlabel("Frequency (Hz)")
    ax2r.set_ylabel("Mean magnitude error")
    ax2r.set_title(f"Per-bin error  (score={r2['score']:.4f})")
    ax2r.legend(facecolor="#2a2a4a", labelcolor="white")
    ax2r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    fig2.tight_layout()
    mo.mpl.interactive(fig2)
    return


@app.cell
def _(BG, C_ERROR, C_PRED, C_TARGET, dark_ax, mo, mticker, np, plt, results):
    r3   = results["masking_weighted"]
    md3  = r3["metadata"]
    arr3 = r3["arrays"]
    freqs3   = np.array(arr3["freqs_hz"])
    m_thr    = np.array(md3["mean_masking_threshold"])
    m_err    = np.array(md3["mean_error"])
    vis_map  = arr3["visibility_map"]
    mfrac    = md3["masked_fraction"]

    dB = lambda x: 20 * np.log10(np.maximum(x, 1e-10))

    fig3 = plt.figure(figsize=(15, 8))
    fig3.patch.set_facecolor(BG)
    gs3 = fig3.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    ax3t  = dark_ax(fig3.add_subplot(gs3[0, :]))
    ax3bl = dark_ax(fig3.add_subplot(gs3[1, 0]))
    ax3br = dark_ax(fig3.add_subplot(gs3[1, 1]))
    fig3.suptitle("3.3 — Masking-Weighted Visibility", color="white", fontsize=13)

    ax3t.semilogx(freqs3[1:], dB(m_thr)[1:], color=C_TARGET, linewidth=2,
                  label="Masking threshold")
    ax3t.semilogx(freqs3[1:], dB(m_err)[1:], color=C_ERROR, linewidth=1.5,
                  linestyle="--", label="Mean error")
    ax3t.fill_between(
        freqs3[1:], dB(m_thr)[1:], dB(m_err)[1:],
        where=dB(m_err)[1:] > dB(m_thr)[1:],
        color=C_PRED, alpha=0.35, label="Audible (above threshold)",
    )
    ax3t.set_xlabel("Frequency (Hz)")
    ax3t.set_ylabel("dBFS")
    ax3t.set_title(
        f"Masking threshold vs error  |  score={r3['score']:.4f}  masked={mfrac:.0%}"
    )
    ax3t.legend(facecolor="#2a2a4a", labelcolor="white")
    ax3t.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    im3 = ax3bl.imshow(
        np.log1p(vis_map), aspect="auto", origin="lower", cmap="inferno",
        extent=[0, vis_map.shape[1], 0, vis_map.shape[0]],
    )
    plt.colorbar(im3, ax=ax3bl, label="log(1+visibility)", fraction=0.046, pad=0.04)
    ax3bl.set_xlabel("Frame")
    ax3bl.set_ylabel("Freq bin")
    ax3bl.set_title("2D visibility map (freq × time)")

    ratio3 = (m_err / (m_thr + 1e-10)).ravel()
    bins3  = np.linspace(0, min(ratio3.max(), 5), 60)
    ax3br.hist(ratio3, bins=bins3, color=C_ERROR, edgecolor="none", alpha=0.85)
    ax3br.axvline(1.0, color=C_PRED, linewidth=2, linestyle="--", label="Threshold=1")
    ax3br.set_xlabel("error / masking threshold")
    ax3br.set_ylabel("Freq bins")
    ax3br.set_title(f"Error/threshold histogram  masked={mfrac:.0%}")
    ax3br.legend(facecolor="#2a2a4a", labelcolor="white")

    mo.mpl.interactive(fig3)
    return


@app.cell
def _(C_ERROR, C_PRED, C_TARGET, dark_fig, mo, mticker, np, results):
    r4 = results["signal_adaptive_weighted"]
    freqs4   = np.array(r4["arrays"]["freqs_hz"])
    pow_db4  = r4["arrays"]["mean_target_power_db"]
    wt_err4  = r4["arrays"]["mean_weighted_error"]
    snr_db4  = r4["arrays"]["effective_snr_db"]

    fig4, (ax4l, ax4r) = dark_fig(1, 2, figsize=(14, 5))
    fig4.suptitle("3.4 — Signal-Adaptive Weighted Error", color="white", fontsize=13)

    ax4l2 = ax4l.twinx()
    ax4l2.set_facecolor("#1a1a2e")
    ax4l2.tick_params(colors="white")
    ax4l2.yaxis.label.set_color(C_ERROR)

    ax4l.semilogx(freqs4[1:], np.array(pow_db4)[1:], color=C_TARGET, linewidth=2,
                  label="Target power (dB)")
    ax4l2.semilogx(freqs4[1:], np.array(wt_err4)[1:], color=C_ERROR, linewidth=1.5,
                   linestyle="--", label="Weighted error")
    ax4l.set_xlabel("Frequency (Hz)")
    ax4l.set_ylabel("Power (dBFS)", color=C_TARGET)
    ax4l2.set_ylabel("Weighted error", color=C_ERROR)
    ax4l.set_title(f"Signal power vs weighted error  (score={r4['score']:.4f})")
    ax4l.tick_params(axis="y", colors=C_TARGET)
    ax4l.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    lines_l, labs_l = ax4l.get_legend_handles_labels()
    lines_r, labs_r = ax4l2.get_legend_handles_labels()
    ax4l.legend(lines_l + lines_r, labs_l + labs_r,
                facecolor="#2a2a4a", labelcolor="white")

    snr4 = np.array(snr_db4)
    ax4r.semilogx(freqs4[1:], snr4[1:], color=C_PRED, linewidth=2)
    ax4r.axhline(0, color="white", linewidth=1, linestyle="--",
                 label="0 dB: error = signal")
    ax4r.fill_between(freqs4[1:], snr4[1:], 0,
                      where=snr4[1:] < 0, color=C_PRED, alpha=0.2,
                      label="Error > signal")
    ax4r.set_xlabel("Frequency (Hz)")
    ax4r.set_ylabel("Effective SNR (dB)")
    ax4r.set_title("Effective SNR per bin")
    ax4r.legend(facecolor="#2a2a4a", labelcolor="white")
    ax4r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    fig4.tight_layout()
    mo.mpl.interactive(fig4)
    return


@app.cell
def _(C_ERROR, C_NEUTRAL, C_PRED, dark_fig, mo, mticker, np, results):
    r5 = results["guitar_prior_weighted"]
    freqs5    = np.array(r5["arrays"]["freqs_hz"])
    prior_db5 = np.array(r5["arrays"]["prior_weights_db"])
    wt_err5   = np.array(r5["arrays"]["per_bin_weighted_error"])
    raw5      = np.array(r5["arrays"]["raw_error_per_bin"])

    fig5, (ax5l, ax5r) = dark_fig(1, 2, figsize=(14, 5))
    fig5.suptitle("3.5 — Guitar Prior Weighted Error", color="white", fontsize=13)

    ax5l.semilogx(freqs5[1:], prior_db5[1:], color=C_PRED, linewidth=2.5)
    for bp_hz, bp_db, bp_lbl in [
        (80, 0, "80 Hz"), (3000, 0, "3 kHz"),
        (5000, 3, "+3dB"), (10000, -6, "−6dB"),
    ]:
        ax5l.scatter([bp_hz], [bp_db], color=C_ERROR, s=60, zorder=5)
        ax5l.annotate(bp_lbl, (bp_hz, bp_db), textcoords="offset points",
                      xytext=(5, 4), color="white", fontsize=8)
    ax5l.axhline(0, color="#555555", linewidth=0.8, linestyle=":")
    ax5l.set_xlabel("Frequency (Hz)")
    ax5l.set_ylabel("Prior (dB)")
    ax5l.set_title("Fixed guitar spectral prior")
    ax5l.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    f5 = freqs5[1:]
    we5 = wt_err5[1:]
    re5 = raw5[1:]
    mask_hi  = (f5 >= 80)  & (f5 <= 3000)
    mask_med = (f5 > 3000) & (f5 <= 10000)
    mask_lo  = ~(mask_hi | mask_med)

    ax5r.semilogx(f5[mask_lo],  we5[mask_lo],  color=C_NEUTRAL,  linewidth=1.5,
                  label="Low (<80 Hz, >10 kHz)")
    ax5r.semilogx(f5[mask_med], we5[mask_med], color="#FFA500",  linewidth=1.5,
                  label="Medium (3–10 kHz)")
    ax5r.semilogx(f5[mask_hi],  we5[mask_hi],  color=C_PRED,     linewidth=2,
                  label="High (80–3 kHz)")
    ax5r.semilogx(f5, re5, color="#555555", linewidth=0.8, linestyle="--",
                  label="Raw error")
    ax5r.set_xlabel("Frequency (Hz)")
    ax5r.set_ylabel("Weighted error")
    ax5r.set_title(f"Per-bin error  (score={r5['score']:.4f})")
    ax5r.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax5r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    fig5.tight_layout()
    mo.mpl.interactive(fig5)
    return


@app.cell
def _(
    BG,
    C_ERROR,
    C_EVEN,
    C_ODD,
    C_PRED,
    C_TARGET,
    dark_ax,
    mo,
    np,
    plt,
    results,
):
    from matplotlib.patches import Patch

    r6 = results["harmonic_weighted"]
    md6  = r6["metadata"]
    arr6 = r6["arrays"]
    freqs6  = np.array(arr6["freqs_hz"])
    wt_mask = arr6["weight_mask"]
    h_freqs = md6["harmonic_freqs"]
    t_amps  = np.array(md6["target_harmonic_amps"])
    p_amps  = np.array(md6["pred_harmonic_amps"])
    err_db  = np.array(md6["harmonic_amp_error_db"])
    f0_6    = md6["f0_hz"]
    nh      = len(h_freqs)

    fig6 = plt.figure(figsize=(15, 9))
    fig6.patch.set_facecolor(BG)
    gs6 = fig6.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax6tl = dark_ax(fig6.add_subplot(gs6[0, 0]))
    ax6tr = dark_ax(fig6.add_subplot(gs6[0, 1]))
    ax6b  = dark_ax(fig6.add_subplot(gs6[1, :]))
    fig6.suptitle(
        f"3.6 — Harmonic Weighted Error  |  f0={f0_6:.1f} Hz, N={nh} harmonics",
        color="white", fontsize=13,
    )

    ax6tl.semilogx(freqs6[1:], wt_mask[1:], color=C_ERROR, linewidth=2)
    for i_h, fh in enumerate(h_freqs):
        ax6tl.axvline(fh, color="#555555", linewidth=0.8, linestyle=":")
        if i_h < 12:
            ax6tl.text(fh * 1.03, ax6tl.get_ylim()[1] * 0.9,
                       f"H{i_h+1}", color="white", fontsize=7)
    ax6tl.set_xlabel("Frequency (Hz)")
    ax6tl.set_ylabel("Weight")
    ax6tl.set_title("Harmonic weight mask")
    from matplotlib.ticker import FuncFormatter as _FF
    ax6tl.xaxis.set_major_formatter(_FF(lambda x, _: f"{x:.0f}"))

    x6 = np.arange(nh)
    w6 = 0.35
    t_db6 = 20 * np.log10(t_amps + 1e-10)
    p_db6 = 20 * np.log10(p_amps + 1e-10)
    ax6tr.bar(x6 - w6/2, t_db6, w6, color=C_TARGET, label="Target",    alpha=0.85)
    ax6tr.bar(x6 + w6/2, p_db6, w6, color=C_PRED,   label="Predicted", alpha=0.85)
    ax6tr.set_xticks(x6)
    ax6tr.set_xticklabels([f"H{i+1}" for i in range(nh)], rotation=45, fontsize=8)
    ax6tr.set_ylabel("Amplitude (dBFS)")
    ax6tr.set_title("Per-harmonic amplitude comparison")
    ax6tr.legend(facecolor="#2a2a4a", labelcolor="white")
    ax6tr.text(0.02, 0.05,
               f"Even err: {md6['even_harmonic_error_db']:.1f} dB\n"
               f"Odd  err: {md6['odd_harmonic_error_db']:.1f} dB",
               transform=ax6tr.transAxes, color="white", fontsize=8,
               verticalalignment="bottom", bbox=dict(facecolor="#2a2a4a", alpha=0.7))

    bar_colors = [C_EVEN if (i % 2 == 1) else C_ODD for i in range(nh)]
    ax6b.bar(x6, err_db, color=bar_colors, alpha=0.85, edgecolor="none")
    ax6b.axhline(0, color="white", linewidth=0.8)
    for thr_db, ls in [(1, "--"), (3, ":"), (6, "-.")]:
        ax6b.axhline( thr_db, color="#888888", linewidth=0.6, linestyle=ls)
        ax6b.axhline(-thr_db, color="#888888", linewidth=0.6, linestyle=ls)
    ax6b.set_xticks(x6)
    ax6b.set_xticklabels([f"H{i+1}" for i in range(nh)], rotation=45, fontsize=8)
    ax6b.set_ylabel("Amplitude error (dB)")
    ax6b.set_title("Per-harmonic error  (blue=even, red=odd)")
    ax6b.legend(
        handles=[
            Patch(color=C_EVEN, label="Even harmonics (warmth)"),
            Patch(color=C_ODD,  label="Odd harmonics (harshness)"),
        ],
        facecolor="#2a2a4a", labelcolor="white",
    )

    mo.mpl.interactive(fig6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 4 — Composite radar chart
    """)
    return


@app.cell
def _(
    dry_full,
    equal_loudness_weighted,
    guitar_prior_weighted,
    harmonic_weighted,
    masking_weighted,
    mel_axis_weighted,
    model_names,
    np,
    preds,
    signal_adaptive_weighted,
    sr,
    target_full,
):
    MAX_RADAR = int(sr * 10)
    dry_r    = dry_full[:MAX_RADAR].astype(float)
    target_r = target_full[:MAX_RADAR].astype(float)

    metric_fns = [
        mel_axis_weighted,
        equal_loudness_weighted,
        masking_weighted,
        signal_adaptive_weighted,
        guitar_prior_weighted,
        harmonic_weighted,
    ]
    metric_labels_short = [
        "mel\naxis", "A-\nwtd", "mask\ning",
        "sig\nadapt", "guitar\nprior", "harm\nonic",
    ]

    all_model_scores = {}
    for mn in model_names:
        p_r = preds[mn][:MAX_RADAR].astype(float)
        scores_mn = [fn(p_r, target_r, dry_r, sr=sr)["score"] for fn in metric_fns]
        all_model_scores[mn] = np.array(scores_mn)
    return all_model_scores, metric_labels_short


@app.cell
def _(BG, all_model_scores, metric_labels_short, mo, np, plt):
    all_arr  = np.array(list(all_model_scores.values()))  # (n_models, 6)
    mn_s = all_arr.min(axis=0)
    mx_s = all_arr.max(axis=0)
    rng_s = mx_s - mn_s + 1e-10

    N_spokes = len(metric_labels_short)
    angles   = np.linspace(0, 2 * np.pi, N_spokes, endpoint=False).tolist()
    angles  += angles[:1]

    fig7, ax7 = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    fig7.patch.set_facecolor(BG)
    ax7.set_facecolor(BG)
    ax7.tick_params(colors="white")
    ax7.set_thetagrids(
        np.degrees(angles[:-1]), metric_labels_short, color="white", fontsize=10
    )
    ax7.yaxis.set_tick_params(labelcolor="#888888")
    for sp in ax7.spines.values():
        sp.set_edgecolor("#555555")

    cmap7 = plt.cm.tab10
    for i, (mn, scores_mn) in enumerate(all_model_scores.items()):
        norm = (scores_mn - mn_s) / rng_s
        vals = norm.tolist() + norm[:1].tolist()
        color7 = cmap7(i / max(len(all_model_scores) - 1, 1))
        ax7.plot(angles, vals, linewidth=2, color=color7, label=mn)
        ax7.fill(angles, vals, alpha=0.06, color=color7)

    ax7.set_ylim(0, 1)
    ax7.set_title(
        "Perceptual metrics radar\n(0=best, 1=worst per metric)",
        color="white", pad=20,
    )
    ax7.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
               facecolor="#2a2a4a", labelcolor="white", fontsize=8)

    mo.mpl.interactive(fig7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5 — Metric correlation (multi-model)
    """)
    return


@app.cell
def _(BG, all_model_scores, metric_labels_short, mo, pd, plt):
    if len(all_model_scores) < 2:
        mo.md("_Need ≥ 2 models for correlation analysis._")
    else:
        import matplotlib.colors as _mc
        short_labels = [s.replace("\n", " ") for s in metric_labels_short]
        df_corr = pd.DataFrame(
            {k: v for k, v in all_model_scores.items()},
            index=short_labels,
        ).T.corr()

        cmap_rdgn = _mc.LinearSegmentedColormap.from_list(
            "rdgn", ["#FF6B6B", "#1a1a2e", "#00FF88"]
        )
        fig8, ax8 = plt.subplots(figsize=(7, 6))
        fig8.patch.set_facecolor(BG)
        ax8.set_facecolor(BG)
        ax8.tick_params(colors="white")
        for sp in ax8.spines.values():
            sp.set_edgecolor("#555555")
        im8 = ax8.imshow(df_corr.values, cmap=cmap_rdgn, vmin=-1, vmax=1)
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
        ax8.set_xticks(range(len(short_labels)))
        ax8.set_yticks(range(len(short_labels)))
        ax8.set_xticklabels(short_labels, rotation=45, ha="right", color="white")
        ax8.set_yticklabels(short_labels, color="white")
        for i in range(len(short_labels)):
            for j in range(len(short_labels)):
                ax8.text(j, i, f"{df_corr.iloc[i,j]:.2f}",
                         ha="center", va="center", fontsize=9, color="white")
        ax8.set_title("Metric correlation matrix", color="white")
        ax8.xaxis.label.set_color("white")
        ax8.yaxis.label.set_color("white")
        ax8.title.set_color("white")
        fig8.tight_layout()
        mo.mpl.interactive(fig8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 6 — Big Muff / TCN diagnostic

    Validates that the metrics catch the known failure mode: a model that sounds
    **too clean** (linear EQ approximation) vs one that captures clipping character.

    Correct result: `linear_approx` should score **worse** than `clipped` on
    `harmonic_weighted`, `masking_weighted`, and `signal_adaptive_weighted`.
    """)
    return


@app.cell
def _(
    dry_full,
    equal_loudness_weighted,
    guitar_prior_weighted,
    harmonic_weighted,
    masking_weighted,
    mel_axis_weighted,
    mo,
    np,
    sig,
    signal_adaptive_weighted,
    sr,
    target_full,
):
    N6 = int(sr * 4)
    dry6    = dry_full[:N6].astype(np.float32)
    target6 = target_full[:N6].astype(np.float32)

    sos_lp6 = sig.butter(4, 3000 / (sr / 2), btype="low", output="sos")
    lin_approx = sig.sosfilt(sos_lp6, dry6).astype(np.float32)
    scale6 = float(np.sqrt(np.mean(target6**2))) / (
        float(np.sqrt(np.mean(lin_approx**2))) + 1e-10
    )
    lin_approx = np.clip(lin_approx * scale6, -1, 1).astype(np.float32)

    clipped6 = np.clip(dry6 * 6.0, -0.5, 0.5)
    sos_hp6  = sig.butter(2, 100 / (sr / 2), btype="high", output="sos")
    clipped6 = sig.sosfilt(sos_hp6, clipped6).astype(np.float32)

    diag_fns = [
        ("mel_axis",        mel_axis_weighted),
        ("equal_loudness",  equal_loudness_weighted),
        ("masking",         masking_weighted),
        ("signal_adaptive", signal_adaptive_weighted),
        ("guitar_prior",    guitar_prior_weighted),
        ("harmonic",        harmonic_weighted),
    ]
    diag_rows = []
    for name6, fn6 in diag_fns:
        r_lin  = fn6(lin_approx, target6, dry6, sr=sr)
        r_clip = fn6(clipped6,   target6, dry6, sr=sr)
        diag_rows.append({
            "Metric":        name6,
            "linear_approx": f"{r_lin['score']:.4f}",
            "clipped":       f"{r_clip['score']:.4f}",
            "linear worse?": "✓" if r_lin["score"] > r_clip["score"] else "✗",
        })

    n_correct = sum(1 for r in diag_rows if r["linear worse?"] == "✓")
    mo.vstack([
        mo.md(
            f"**{n_correct}/{len(diag_fns)} metrics** correctly score "
            f"`linear_approx` worse than `clipped`."
        ),
        mo.ui.table(data=diag_rows),
    ])
    return


if __name__ == "__main__":
    app.run()
