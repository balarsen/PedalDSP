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
#   "librosa",
# ]
# ///
"""KlonSim combined diagnostics — 02_diagnostics + selected perceptual plots.

21 plots across 7 sections.  Replaces both source notebooks for day-to-day use.

Run with:
    marimo run  notebooks/notaklon/03_combined_diagnostics.py
    marimo edit notebooks/notaklon/03_combined_diagnostics.py
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
    import seaborn as sns
    import librosa
    import librosa.display
    import marimo as mo

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from pedal_model.eval.results import ZooResults
    from pedal_model.signals.manifest import Manifest
    from pedal_model.metrics.perceptual import (
        mel_axis_weighted,
        equal_loudness_weighted,
        masking_weighted,
        signal_adaptive_weighted,
        guitar_prior_weighted,
        harmonic_weighted,
    )
    from pedal_model.eval.diagnostics import (
        dark_ax,
        dark_fig,
        BG, C_TARGET, C_PRED, C_ERROR, C_EVEN, C_ODD, C_NEUTRAL,
        extract_drive_level,
    )

    RESULTS_PATH = PROJECT_ROOT / "results" / "data" / "klon_sim" / "klon_sim_v1" / "results.pkl"
    zoo = ZooResults.load(RESULTS_PATH)
    manifest = Manifest(PROJECT_ROOT / "recordings" / "data" / "signals" / "val_signal_v1.json")

    sr = zoo.sr
    MODEL_NAMES = list(zoo.models.keys())

    _cmap = plt.cm.tab20
    COLOR_MAP = {name: _cmap(i / max(len(MODEL_NAMES) - 1, 1)) for i, name in enumerate(MODEL_NAMES)}

    SECTION_BOUNDS = {s.label: (s.start_sample, s.end_sample) for s in manifest.sections}
    SECTION_TYPES  = {s.label: s.type for s in manifest.sections}
    SECTION_LABELS = [s.label for s in manifest.sections]

    # Tone sections sorted ascending by drive level so THD x-axis reads low → high
    TONE_LABELS = sorted(
        [s.label for s in manifest.sections_of_type("stepped_sine_tone")],
        key=lambda lb: (extract_drive_level(lb) or 0),
    )

    # Default interactive section = highest-drive tone section (most interesting)
    default_section = TONE_LABELS[-1] if TONE_LABELS else (SECTION_LABELS[0] if SECTION_LABELS else "")

    # Full val signals for perceptual metrics
    dry_full    = zoo.dry_val
    target_full = zoo.wet_val
    preds       = {k: zoo.models[k].prediction for k in MODEL_NAMES}

    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
    print(f"Loaded: {len(MODEL_NAMES)} models, {len(SECTION_LABELS)} sections, sr={sr} Hz")
    print(f"Tone sections: {len(TONE_LABELS)}, others: {len(SECTION_LABELS) - len(TONE_LABELS)}")
    return (
        BG,
        COLOR_MAP,
        C_ERROR,
        C_EVEN,
        C_NEUTRAL,
        C_ODD,
        C_PRED,
        C_TARGET,
        MODEL_NAMES,
        SECTION_BOUNDS,
        SECTION_LABELS,
        SECTION_TYPES,
        TONE_LABELS,
        dark_ax,
        default_section,
        dry_full,
        equal_loudness_weighted,
        extract_drive_level,
        guitar_prior_weighted,
        harmonic_weighted,
        librosa,
        manifest,
        masking_weighted,
        mel_axis_weighted,
        mo,
        mticker,
        np,
        pd,
        plt,
        preds,
        sig,
        signal_adaptive_weighted,
        sr,
        target_full,
        zoo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # KlonSim Combined Diagnostics

    Merges `02_diagnostics` with selected perceptual-error plots from
    `perceptual_metrics_explorer`.  **21 plots** across 7 sections.

    | # | Plot | Section |
    |---|------|---------|
    | 1.1 | RMS envelope | 1.0 – Signal overview |
    | 1.2 | Summary metrics heatmap | 1.0 |
    | 1.3 | ESR by section type | 1.0 |
    | 2.1 | Waveform overlay + residual | 2.0 – Waveform / spectral |
    | 2.2 | FFT spectrum comparison | 2.0 |
    | 2.3 | Spectrogram grid | 2.0 |
    | 3.1 | ESR vs null-test depth | 3.0 – Model ranking |
    | 3.2 | ESR vs STFT loss | 3.0 |
    | 3.3 | ESR vs drive level | 3.0 |
    | 3.4 | ESR vs FR_err_dB | 3.0 |
    | 4.1 | THD analysis — tone sections | 4.0 – Harmonic / THD |
    | 4.2 | THD_pattern_dist vs HP_sim scatter | 4.0 |
    | 4.3 | Per-harmonic amplitude comparison | 4.0 |
    | 5.1 | Masking threshold vs error | 5.0 – Perceptual error |
    | 5.2 | A-weighted error spectrum | 5.0 |
    | 5.3 | Perceptual radar — 5 models | 5.0 |
    | 6.1 | Clipping discriminator (synthetic) | 6.0 – Self-test |
    | 7.1 | KlonDDSP circuit fraction | 7.0 – Circuit / timing |
    | 7.2 | Timing chart (neural models) | 7.0 |
    | 7.3 | Raw per-section metrics table | 7.0 |
    | 7.4 | Aggregated summary table | 7.0 |

    Plots 1.1–4.2, 7.1–7.4 come from `02_diagnostics`.
    Plots 4.3–6.1 come from `perceptual_metrics_explorer`.
    Plots 2.1, 2.2, 2.3, 7.2 have been modified (default section / neural-only filter).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1.0 — Signal and model overview
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1.1 — RMS envelope
    """)
    return


@app.cell
def _(MODEL_NAMES, SECTION_TYPES, manifest, mo, np, plt, sr, zoo):
    from pedal_model.utils.plotting import plot_rms_envelope

    # Determine best model by mean ESR across all sections
    _best_ov = min(
        MODEL_NAMES,
        key=lambda n: np.nanmean([
            v.get("ESR", np.nan)
            for v in zoo.models[n].metrics_per_section.values()
        ]),
    )
    # Map ESR → band color (alpha applied via axvspan)
    def _esr_color(esr: float) -> str:
        if esr < 0.01:   return "#00FF88"   # green  — essentially perfect
        if esr < 0.1:    return "#FFD700"   # yellow — good
        if esr < 1.0:    return "#FF8C00"   # orange — marginal
        return           "#FF6B6B"          # red    — worse than silence

    _sections_ov = [(sec.start_sample, sec.end_sample) for sec in manifest.sections]
    fig_ov = plot_rms_envelope(zoo.dry_val, zoo.wet_val, sr, sections=_sections_ov)
    _ax_d_ov, _ax_w_ov = fig_ov.axes[0], fig_ov.axes[1]

    _seen_ov: set[str] = set()
    for _sec_ov in manifest.sections:
        _lb_ov  = _sec_ov.label
        _t0_ov  = _sec_ov.start_sample / sr
        _t1_ov  = _sec_ov.end_sample   / sr
        _esr_ov = zoo.models[_best_ov].metrics_per_section.get(_lb_ov, {}).get("ESR", float("nan"))
        _col_ov = _esr_color(_esr_ov if np.isfinite(_esr_ov) else 2.0)

        # Colored band on both panels
        _ax_d_ov.axvspan(_t0_ov, _t1_ov, alpha=0.15, color=_col_ov, lw=0)
        _ax_w_ov.axvspan(_t0_ov, _t1_ov, alpha=0.15, color=_col_ov, lw=0)

        # Short section-type label at top of wet panel (deduplicate per type)
        _type_short = SECTION_TYPES.get(_lb_ov, "").split("_")[0]
        _ax_w_ov.text(
            (_t0_ov + _t1_ov) / 2, _ax_w_ov.get_ylim()[1] * 0.95,
            _type_short, ha="center", va="top", fontsize=6, color="gray",
            clip_on=True,
        )

    # Legend for color bands
    _esr_legend = [
        plt.Rectangle((0, 0), 1, 1, fc="#00FF88", alpha=0.4, label="ESR < 0.01"),
        plt.Rectangle((0, 0), 1, 1, fc="#FFD700", alpha=0.4, label="ESR 0.01–0.1"),
        plt.Rectangle((0, 0), 1, 1, fc="#FF8C00", alpha=0.4, label="ESR 0.1–1.0"),
        plt.Rectangle((0, 0), 1, 1, fc="#FF6B6B", alpha=0.4, label="ESR > 1.0"),
    ]
    _ax_w_ov.legend(handles=_esr_legend, fontsize=7, loc="upper right",
                    title=f"Background: {_best_ov} ESR", title_fontsize=7)
    mo.mpl.interactive(fig_ov)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1.2 — Summary metrics heatmap (mean across all sections)
    """)
    return


@app.cell
def _(
    MODEL_NAMES,
    dry_full,
    equal_loudness_weighted,
    guitar_prior_weighted,
    harmonic_weighted,
    masking_weighted,
    mel_axis_weighted,
    np,
    preds,
    signal_adaptive_weighted,
    sr,
    target_full,
):
    # Compute perceptual scores for all models — ~17s one-time
    # Cached in perc_cache; re-running this cell re-computes.
    _PERC_N = int(sr * 10)
    _dry_pc  = dry_full[:_PERC_N].astype(np.float32)
    _tgt_pc  = target_full[:_PERC_N].astype(np.float32)
    _perc_fns = [
        ("mel",       mel_axis_weighted),
        ("eq_loud",   equal_loudness_weighted),
        ("masking",   masking_weighted),
        ("sig_adpt",  signal_adaptive_weighted),
        ("gtr_prior", guitar_prior_weighted),
        ("harmonic",  harmonic_weighted),
    ]
    perc_cache: dict = {}
    for _mn_pc in MODEL_NAMES:
        _pr_pc = preds[_mn_pc][:_PERC_N].astype(np.float32)
        perc_cache[_mn_pc] = {
            k: fn(_pr_pc, _tgt_pc, _dry_pc, sr=sr)["score"]
            for k, fn in _perc_fns
        }
    print(f"Perceptual cache ready for {len(perc_cache)} models")
    return (perc_cache,)


@app.cell
def _(MODEL_NAMES, mo, perc_cache: dict, zoo):
    from pedal_model.utils.plotting import plot_metrics_heatmap
    from pedal_model.eval.diagnostics import build_perceptual_heatmap_df

    _DISPLAY_METRICS = [
        "ESR", "null_depth_dB", "STFT", "FR_err_dB", "THD_pattern_dist", "HP_sim",
        "mel", "eq_loud", "masking", "sig_adpt", "gtr_prior", "harmonic",
    ]
    df_heat = build_perceptual_heatmap_df(zoo, MODEL_NAMES, perc_cache, _DISPLAY_METRICS)
    fig_heat = plot_metrics_heatmap(df_heat, metric_cols=_DISPLAY_METRICS, figsize=(16, 5))
    mo.mpl.interactive(fig_heat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1.3 — ESR per section (grouped by section type)
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, SECTION_TYPES, mo, pd, zoo):
    from pedal_model.utils.plotting import plot_esr_by_section_type

    _rows_esr = []
    for _name in MODEL_NAMES:
        for _sec, _metrics in zoo.models[_name].metrics_per_section.items():
            _rows_esr.append({
                "model":   _name,
                "section": _sec,
                "type":    SECTION_TYPES.get(_sec, "unknown"),
                "ESR":     float(_metrics.get("ESR", float("nan"))),
            })
    df_esr = pd.DataFrame(_rows_esr)
    fig_esr = plot_esr_by_section_type(df_esr, MODEL_NAMES, color_map=COLOR_MAP, figsize=(16, 5))
    mo.mpl.interactive(fig_esr)
    return


@app.cell
def _(MODEL_NAMES, mo, zoo):
    from pedal_model.eval.diagnostics import compute_viable_models as _compute_viable

    _vm = _compute_viable(zoo, MODEL_NAMES)

    def _fmt_list(entries):
        return ", ".join(f"{n} (ESR={e:.3f})" for n, e, _ in entries)

    _any_viable = len(_vm["viable"]) > 0
    _verdict_md = f"""
    **Viable models** (mean ESR < 1.0, null_depth > 0 dB):
    {_fmt_list(_vm['viable']) or '*(none)*'}

    **Marginal** (mean ESR < 1.0, null_depth ≤ 0 dB):
    {_fmt_list(_vm['marginal']) or '*(none)*'}

    **Not viable** (mean ESR ≥ 1.0):
    {_fmt_list(_vm['not_viable']) or '*(none)*'}
    """
    mo.callout(mo.md(_verdict_md), kind="success" if _any_viable else "warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2.0 — Waveform and spectral views
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2.1 — Waveform overlay
    """)
    return


@app.cell
def _(MODEL_NAMES, SECTION_LABELS, default_section, mo):
    _wf_default_models = [m for m in ["tcn", "fir", "wiener_hammerstein"] if m in MODEL_NAMES]
    if not _wf_default_models:
        _wf_default_models = MODEL_NAMES[:3]

    wf_section    = mo.ui.dropdown(SECTION_LABELS, value=default_section, label="Section")
    wf_models     = mo.ui.multiselect(MODEL_NAMES, value=_wf_default_models,
                                      label="Models to overlay")
    wf_window_ms  = mo.ui.slider(10, 200, value=40, step=5, label="Window (ms)")
    wf_offset_ms  = mo.ui.slider(0, 500, value=0,  step=5, label="Offset into section (ms)")
    mo.hstack([wf_section, wf_models, wf_window_ms, wf_offset_ms])
    return wf_models, wf_offset_ms, wf_section, wf_window_ms


@app.cell
def _(
    COLOR_MAP,
    SECTION_BOUNDS,
    mo,
    np,
    plt,
    sr,
    wf_models,
    wf_offset_ms,
    wf_section,
    wf_window_ms,
    zoo,
):
    _sec   = wf_section.value
    _s0, _e0 = SECTION_BOUNDS[_sec]
    _off   = int(wf_offset_ms.value * sr / 1000)
    _win   = int(wf_window_ms.value  * sr / 1000)
    _start = min(_s0 + _off, _e0 - _win)
    _end   = min(_start + _win, _e0)

    _dry_w = zoo.dry_val[_start:_end]
    _wet_w = zoo.wet_val[_start:_end]
    _t_ms  = 1000.0 * np.arange(len(_dry_w)) / sr

    fig_wf, (ax_wf, ax_res) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    ax_wf.plot(_t_ms, _dry_w, color="gray",  lw=0.7, alpha=0.6, label="dry")
    ax_wf.plot(_t_ms, _wet_w, color="black", lw=1.0, label="target")
    for _nm in wf_models.value:
        _pred_wf = zoo.models[_nm].prediction[_start:_end]
        ax_wf.plot(_t_ms, _pred_wf, lw=0.8, alpha=0.8, color=COLOR_MAP[_nm], label=_nm)
    ax_wf.set_ylabel("amplitude")
    ax_wf.set_title(
        f"Waveform overlay — {_sec}  [{wf_window_ms.value} ms @ offset {wf_offset_ms.value} ms]"
    )
    ax_wf.legend(fontsize=8)

    for _nm in wf_models.value:
        _pred_wf = zoo.models[_nm].prediction[_start:_end]
        ax_res.plot(_t_ms, _wet_w - _pred_wf, lw=0.7, alpha=0.8,
                    color=COLOR_MAP[_nm], label=_nm)
    ax_res.axhline(0, color="black", lw=0.5, ls="--")
    ax_res.set_ylabel("residual (target − pred)")
    ax_res.set_xlabel("time within window (ms)")
    ax_res.set_title("Residual (target − prediction)  (lower is better)")
    ax_res.legend(fontsize=8)

    plt.tight_layout()
    mo.mpl.interactive(fig_wf)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2.2 — FFT spectrum comparison — 4-section ladder
    """)
    return


@app.cell
def _(
    COLOR_MAP,
    MODEL_NAMES,
    SECTION_BOUNDS,
    SECTION_LABELS,
    SECTION_TYPES,
    TONE_LABELS,
    dark_ax,
    extract_drive_level,
    mo,
    np,
    plt,
    sr,
    zoo,
):
    # Best 3 models by mean ESR (computed dynamically)
    _esr_means_fft = {
        n: float(np.nanmean([
            v.get("ESR", np.nan)
            for v in zoo.models[n].metrics_per_section.values()
        ]))
        for n in MODEL_NAMES
    }
    _best3_fft = sorted(MODEL_NAMES, key=lambda n: _esr_means_fft[n])[:3]

    # 4-section ladder: lowest drive, highest drive, first pluck, first sweep
    _ladder_fft = [
        min(TONE_LABELS, key=lambda l: (extract_drive_level(l) or 0)),
        max(TONE_LABELS, key=lambda l: (extract_drive_level(l) or 0)),
        next((l for l in SECTION_LABELS if "pluck" in l), SECTION_LABELS[0]),
        next((l for l in SECTION_LABELS if "sweep" in l), SECTION_LABELS[-1]),
    ]

    fig_fft, axes_fft = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig_fft.patch.set_facecolor("#1a1a2e")

    for _ax_fft, _sec_l in zip(axes_fft, _ladder_fft):
        dark_ax(_ax_fft)
        _s_l, _e_l = SECTION_BOUNDS[_sec_l]
        _n_l  = _e_l - _s_l
        _freq_fft = np.fft.rfftfreq(_n_l, 1.0 / sr)
        _mask_fft = (_freq_fft >= 20) & (_freq_fft <= sr / 2)

        # Target
        _X_tgt = np.abs(np.fft.rfft(zoo.wet_val[_s_l:_e_l].astype(np.float64)))
        _db_tgt = 20 * np.log10(np.maximum(_X_tgt[_mask_fft] * 2 / _n_l, 1e-10))
        _db_tgt = np.maximum(_db_tgt, -100.0)
        _ax_fft.plot(_freq_fft[_mask_fft], _db_tgt, color="#00FF88", lw=1.2, label="target", zorder=5)

        # Best 3 models
        for _nm_l in _best3_fft:
            _X_m = np.abs(np.fft.rfft(zoo.models[_nm_l].prediction[_s_l:_e_l].astype(np.float64)))
            _db_m = 20 * np.log10(np.maximum(_X_m[_mask_fft] * 2 / _n_l, 1e-10))
            _db_m = np.maximum(_db_m, -100.0)
            _ax_fft.plot(_freq_fft[_mask_fft], _db_m, color=COLOR_MAP[_nm_l], lw=0.9,
                         alpha=0.85, label=_nm_l)

        _sec_type_l = SECTION_TYPES.get(_sec_l, "")
        _ax_fft.set_title(f"{_sec_l}  [{_sec_type_l}]", color="white", fontsize=9)
        _ax_fft.set_ylabel("dBFS", color="white", fontsize=8)
        _ax_fft.set_ylim(-100, 5)
        _ax_fft.set_xscale("log")
        _ax_fft.set_xlim(20, sr / 2)
        _ax_fft.legend(fontsize=7, loc="upper right")

    axes_fft[-1].set_xlabel("Frequency (Hz)", color="white")
    fig_fft.suptitle("FFT magnitude ladder — target + best-3 models by ESR",
                     color="white", fontsize=11)
    plt.tight_layout()
    mo.mpl.interactive(fig_fft)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2.3 — Spectrogram: target + error (best model)
    """)
    return


@app.cell
def _(SECTION_LABELS, default_section, mo):
    spec_section = mo.ui.dropdown(SECTION_LABELS, value=default_section,
                                  label="Section for spectrogram")
    spec_section
    return (spec_section,)


@app.cell
def _(
    BG,
    MODEL_NAMES,
    SECTION_BOUNDS,
    dark_ax,
    librosa,
    mo,
    np,
    plt,
    spec_section,
    sr,
    zoo,
):
    _sec_sp   = spec_section.value
    _s_sp, _e_sp = SECTION_BOUNDS[_sec_sp]

    # Best model by mean ESR
    _best_sp = min(
        MODEL_NAMES,
        key=lambda n: float(np.nanmean([
            v.get("ESR", np.nan)
            for v in zoo.models[n].metrics_per_section.values()
        ])),
    )

    _tgt_sp  = zoo.wet_val[_s_sp:_e_sp].astype(np.float32)
    _pred_sp = zoo.models[_best_sp].prediction[_s_sp:_e_sp].astype(np.float32)
    _err_sp  = _tgt_sp - _pred_sp

    _n_fft_sp, _hop_sp = 2048, 512

    _S_tgt = librosa.amplitude_to_db(
        np.abs(librosa.stft(_tgt_sp, n_fft=_n_fft_sp, hop_length=_hop_sp)),
        ref=1.0,
    )
    # Error spectrogram in dB of error amplitude
    _S_err_lin = np.abs(librosa.stft(_err_sp, n_fft=_n_fft_sp, hop_length=_hop_sp))
    _S_err_db  = 20 * np.log10(np.maximum(_S_err_lin, 1e-10))

    fig_spec, (ax_tgt_sp, ax_err_sp) = plt.subplots(2, 1, figsize=(16, 8))
    fig_spec.patch.set_facecolor(BG)
    dark_ax(ax_tgt_sp)
    dark_ax(ax_err_sp)

    _img_tgt = librosa.display.specshow(
        _S_tgt, sr=sr, hop_length=_hop_sp, y_axis="log", x_axis="time",
        ax=ax_tgt_sp, cmap="viridis", vmin=-80, vmax=0,
    )
    fig_spec.colorbar(_img_tgt, ax=ax_tgt_sp, format="%+.0f dBFS", fraction=0.03, pad=0.01)
    ax_tgt_sp.set_title(f"Target spectrogram — {_sec_sp}", color="white")

    _img_err = librosa.display.specshow(
        _S_err_db, sr=sr, hop_length=_hop_sp, y_axis="log", x_axis="time",
        ax=ax_err_sp, cmap="RdBu_r", vmin=-30, vmax=30,
    )
    _cb_err = fig_spec.colorbar(_img_err, ax=ax_err_sp, fraction=0.03, pad=0.01)
    _cb_err.set_label("error (dB)", color="white")
    _cb_err.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(_cb_err.ax.axes, "yticklabels"), color="white")
    ax_err_sp.set_title(f"Error spectrogram — {_sec_sp} | error = {_best_sp} prediction",
                        color="white")

    plt.tight_layout()
    mo.mpl.interactive(fig_spec)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3.0 — Model ranking and diagnostic scatters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3.1 — ESR vs null-test depth
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, zoo):
    from pedal_model.eval.diagnostics import (
        compute_viable_models as _cvm31,
        plot_esr_null_scatter as _pens31,
    )

    _vm31 = _cvm31(zoo, MODEL_NAMES)
    _viable_names31 = [n for n, _, _ in _vm31["viable"]]
    fig_mv = _pens31(zoo, MODEL_NAMES, color_map=COLOR_MAP, viable_models=_viable_names31)
    mo.mpl.interactive(fig_mv)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3.2 — ESR vs STFT loss
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, zoo):
    from pedal_model.eval.diagnostics import plot_esr_stft_scatter as _pess32

    fig_es = _pess32(zoo, MODEL_NAMES, color_map=COLOR_MAP)
    mo.mpl.interactive(fig_es)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3.3 — ESR vs drive level

    The Klon's gain stage is drive-dependent: at high drive levels the circuit saturates
    harder, producing a richer harmonic spectrum that is more difficult to model.
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, extract_drive_level, mo, np, plt, zoo):
    from pedal_model.eval.diagnostics import compute_viable_models as _cvm33

    _vm33 = _cvm33(zoo, MODEL_NAMES)
    _viable33 = [n for n, _, _ in _vm33["viable"]]

    fig_dr, ax_dr = plt.subplots(figsize=(12, 7))
    for _name in _viable33:
        _drives, _esrs_dr = [], []
        for _sec, _metrics in zoo.models[_name].metrics_per_section.items():
            _d = extract_drive_level(_sec)
            _e = _metrics.get("ESR", float("nan"))
            if _d is not None and np.isfinite(_e):
                _drives.append(_d)
                _esrs_dr.append(min(_e, 5.0))
        if not _drives:
            continue
        _unique = sorted(set(_drives))
        _means = [float(np.mean([_esrs_dr[i] for i, x in enumerate(_drives) if x == u]))
                  for u in _unique]
        ax_dr.plot(_unique, _means, marker="o", ms=8, lw=2,
                   color=COLOR_MAP[_name])
        # Right-side label at the last point
        ax_dr.text(_unique[-1] + 0.5, _means[-1], _name,
                   va="center", ha="left", fontsize=8, color=COLOR_MAP[_name])

    ax_dr.axhline(1.0, color="gray", ls="--", lw=0.8, label="ESR=1 (worse than silence)")
    ax_dr.axhline(0.01, color="green", ls="--", lw=0.8, label="ESR=0.01 (good fit target)")
    ax_dr.set_xlabel("Drive level (dBFS)")
    ax_dr.set_ylabel("mean ESR per drive level  ←  better")
    ax_dr.set_title(
        f"ESR vs drive level — viable models only"
        f" ({len(_viable33)} of {len(MODEL_NAMES)})"
    )
    ax_dr.legend(fontsize=8)
    plt.tight_layout()
    mo.mpl.interactive(fig_dr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3.4 — ESR vs FR_err_dB
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, np, plt, zoo):
    fig_ef, ax_ef = plt.subplots(figsize=(10, 7))
    for _name in MODEL_NAMES:
        _esrs_ef, _frs = [], []
        for _m in zoo.models[_name].metrics_per_section.values():
            _e = _m.get("ESR", float("nan"))
            _f = _m.get("FR_err_dB", float("nan"))
            if np.isfinite(_e) and np.isfinite(_f):
                _esrs_ef.append(min(_e, 5.0))
                _frs.append(min(_f, 80.0))
        ax_ef.scatter(_esrs_ef, _frs, s=20, alpha=0.6, color=COLOR_MAP[_name], label=_name)
        if _esrs_ef:
            ax_ef.scatter([float(np.mean(_esrs_ef))], [float(np.mean(_frs))],
                          s=120, marker="D", color=COLOR_MAP[_name],
                          edgecolors="black", lw=1, zorder=5)
    ax_ef.axvline(1.0, color="gray", ls="--", lw=0.8, label="ESR=1")
    ax_ef.set_xlabel("ESR (clipped at 5)  ← better")
    ax_ef.set_ylabel("FR error dB (clipped at 80)  ← better")
    ax_ef.set_title("ESR vs FR error — low ESR + high FR_err = right nonlinearity, wrong EQ")
    ax_ef.legend(fontsize=7)
    plt.tight_layout()
    mo.mpl.interactive(fig_ef)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4.0 — Harmonic and THD analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4.1 — THD analysis — tone sections (x-axis sorted by drive level)
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, TONE_LABELS, mo, np, plt, zoo):
    if not TONE_LABELS:
        mo.md("*No tone sections found in results.*")
    else:
        _first_m = list(zoo.models.values())[0]
        _thd_target = {
            _sec: _first_m.metrics_per_section[_sec].get("THD_target", float("nan"))
            for _sec in TONE_LABELS if _sec in _first_m.metrics_per_section
        }

        _n_tone = len(TONE_LABELS)
        _x_t = np.arange(_n_tone)
        _w_t = 0.7 / (len(MODEL_NAMES) + 1)

        fig_thd, (ax_thd, ax_thdp) = plt.subplots(2, 1, figsize=(18, 10))

        for _i, _name in enumerate(MODEL_NAMES):
            _vals_t = [
                zoo.models[_name].metrics_per_section.get(_sec, {}).get("THD_pred", float("nan"))
                for _sec in TONE_LABELS
            ]
            ax_thd.bar(_x_t + _i * _w_t - 0.35, _vals_t, _w_t,
                       label=_name, color=COLOR_MAP[_name], alpha=0.8)

        _t_vals = [_thd_target.get(_sec, float("nan")) for _sec in TONE_LABELS]
        ax_thd.plot(_x_t, _t_vals, color="black", lw=2,
                    drawstyle="steps-mid", label="THD_target")
        ax_thd.set_xticks(_x_t)
        ax_thd.set_xticklabels(TONE_LABELS, rotation=40, ha="right", fontsize=7)
        ax_thd.set_ylabel("THD")
        ax_thd.set_title("THD (predicted vs target) per tone section")
        ax_thd.legend(fontsize=7, ncol=4)

        for _i, _name in enumerate(MODEL_NAMES):
            _vals_tp = [
                zoo.models[_name].metrics_per_section.get(_sec, {}).get("THD_pattern_dist", float("nan"))
                for _sec in TONE_LABELS
            ]
            ax_thdp.bar(_x_t + _i * _w_t - 0.35, _vals_tp, _w_t,
                        label=_name, color=COLOR_MAP[_name], alpha=0.8)
        ax_thdp.set_xticks(_x_t)
        ax_thdp.set_xticklabels(TONE_LABELS, rotation=40, ha="right", fontsize=7)
        ax_thdp.set_ylabel("THD pattern distance")
        ax_thdp.set_title("Harmonic pattern distance per tone section — lower is better")
        ax_thdp.legend(fontsize=7, ncol=4)

        plt.tight_layout()
        mo.mpl.interactive(fig_thd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4.2 — THD_pattern_dist vs HP_sim
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, TONE_LABELS, mo, np, plt, zoo):
    if not TONE_LABELS:
        mo.md("*No tone sections — harmonic scatter not available.*")
    else:
        fig_hpsc, ax_hpsc = plt.subplots(figsize=(9, 7))
        for _name in MODEL_NAMES:
            _dists, _sims = [], []
            for _sec in TONE_LABELS:
                _m = zoo.models[_name].metrics_per_section.get(_sec, {})
                _d = _m.get("THD_pattern_dist", float("nan"))
                _h = _m.get("HP_sim", float("nan"))
                if np.isfinite(_d) and np.isfinite(_h):
                    _dists.append(_d)
                    _sims.append(_h)
            ax_hpsc.scatter(_dists, _sims, s=25, alpha=0.7, color=COLOR_MAP[_name], label=_name)
            if _dists:
                ax_hpsc.scatter([float(np.mean(_dists))], [float(np.mean(_sims))],
                                s=120, marker="D", color=COLOR_MAP[_name],
                                edgecolors="black", lw=1, zorder=5)
        ax_hpsc.set_xlabel("THD pattern distance  ← better")
        ax_hpsc.set_ylabel("HP_sim (harmonic profile cosine similarity)  → better")
        ax_hpsc.set_title("Harmonic shape vs profile similarity (tone sections)\n"
                          "models on the diagonal = metrics agree; "
                          "off-diagonal = discrepancy")
        ax_hpsc.legend(fontsize=7)
        plt.tight_layout()
        mo.mpl.interactive(fig_hpsc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4.3 — Per-harmonic amplitude comparison

    Use the dropdowns below to select a model and a tone section.
    The same selection drives plots 14 and 15.
    """)
    return


@app.cell
def _(MODEL_NAMES, SECTION_LABELS, TONE_LABELS, default_section, mo):
    _default_perc_model = "tcn" if "tcn" in MODEL_NAMES else MODEL_NAMES[0]
    _perc_sec_options   = TONE_LABELS if TONE_LABELS else SECTION_LABELS
    _perc_sec_default   = (
        default_section if default_section in _perc_sec_options else _perc_sec_options[0]
    )

    perc_model_sel = mo.ui.dropdown(
        options=MODEL_NAMES,
        value=_default_perc_model,
        label="Model (plots 13-15)",
    )
    perc_sec_sel = mo.ui.dropdown(
        options=_perc_sec_options,
        value=_perc_sec_default,
        label="Section",
    )
    mo.hstack([perc_model_sel, perc_sec_sel])
    return perc_model_sel, perc_sec_sel


@app.cell
def _(
    SECTION_BOUNDS,
    dry_full,
    equal_loudness_weighted,
    harmonic_weighted,
    masking_weighted,
    np,
    perc_model_sel,
    perc_sec_sel,
    preds,
    sr,
    target_full,
):
    _MAX_PERC = int(sr * 10)
    _ps, _pe = SECTION_BOUNDS[perc_sec_sel.value]
    _pe = min(_pe, _ps + _MAX_PERC)

    _dry_p    = dry_full[_ps:_pe].astype(np.float32)
    _target_p = target_full[_ps:_pe].astype(np.float32)
    _pred_p   = preds[perc_model_sel.value][_ps:_pe].astype(np.float32)

    harm_res = harmonic_weighted(_pred_p, _target_p, _dry_p, sr=sr)
    mask_res = masking_weighted(_pred_p, _target_p, _dry_p, sr=sr)
    eq_res   = equal_loudness_weighted(_pred_p, _target_p, _dry_p, sr=sr)
    return eq_res, harm_res, mask_res


@app.cell
def _(
    BG,
    C_ERROR,
    C_EVEN,
    C_ODD,
    C_PRED,
    C_TARGET,
    dark_ax,
    harm_res,
    mo,
    np,
    perc_model_sel,
    perc_sec_sel,
    plt,
):
    from matplotlib.patches import Patch as _Patch

    _md13  = harm_res["metadata"]
    _h_freqs = _md13["harmonic_freqs"]
    _t_amps  = np.array(_md13["target_harmonic_amps"])
    _p_amps  = np.array(_md13["pred_harmonic_amps"])
    _err_db  = np.array(_md13["harmonic_amp_error_db"])
    _f0_13   = _md13["f0_hz"]
    _nh      = len(_h_freqs)

    fig13 = plt.figure(figsize=(14, 7))
    fig13.patch.set_facecolor(BG)
    _gs13 = fig13.add_gridspec(2, 1, hspace=0.45)
    _ax13t = dark_ax(fig13.add_subplot(_gs13[0]))
    _ax13b = dark_ax(fig13.add_subplot(_gs13[1]))
    fig13.suptitle(
        f"Plot 13 — Per-Harmonic Amplitude  |  {perc_model_sel.value}"
        f"  /  {perc_sec_sel.value}  |  f0={_f0_13:.1f} Hz",
        color="white", fontsize=11,
    )

    _x13 = np.arange(_nh)
    _w13 = 0.35
    _t_db13 = 20 * np.log10(_t_amps + 1e-10)
    _p_db13 = 20 * np.log10(_p_amps + 1e-10)
    _ax13t.bar(_x13 - _w13/2, _t_db13, _w13, color=C_TARGET, label="Target",    alpha=0.85)
    _ax13t.bar(_x13 + _w13/2, _p_db13, _w13, color=C_PRED,   label="Predicted", alpha=0.85)
    _ax13t.set_xticks(_x13)
    _ax13t.set_xticklabels([f"H{i+1}" for i in range(_nh)], rotation=45, fontsize=8)
    _ax13t.set_ylabel("Amplitude (dBFS)")
    _ax13t.set_title("Per-harmonic amplitude comparison")
    _ax13t.legend(facecolor="#2a2a4a", labelcolor="white")
    _ax13t.text(
        0.02, 0.05,
        f"Even err: {_md13['even_harmonic_error_db']:.1f} dB\n"
        f"Odd  err: {_md13['odd_harmonic_error_db']:.1f} dB",
        transform=_ax13t.transAxes, color="white", fontsize=8,
        verticalalignment="bottom", bbox=dict(facecolor="#2a2a4a", alpha=0.7),
    )

    _bar_colors13 = [C_EVEN if (i % 2 == 1) else C_ODD for i in range(_nh)]
    _ax13b.bar(_x13, _err_db, color=_bar_colors13, alpha=0.85, edgecolor="none")
    _ax13b.axhline(0, color="white", linewidth=0.8)
    for _thr13, _ls13 in [(1, "--"), (3, ":"), (6, "-.")]:
        _ax13b.axhline( _thr13, color="#888888", linewidth=0.6, linestyle=_ls13)
        _ax13b.axhline(-_thr13, color="#888888", linewidth=0.6, linestyle=_ls13)
    _ax13b.set_xticks(_x13)
    _ax13b.set_xticklabels([f"H{i+1}" for i in range(_nh)], rotation=45, fontsize=8)
    _ax13b.set_ylabel("Amplitude error (dB)")
    _ax13b.set_title("Per-harmonic error  (blue=even, red=odd)")
    _ax13b.legend(
        handles=[
            _Patch(color=C_EVEN, label="Even harmonics (warmth)"),
            _Patch(color=C_ODD,  label="Odd harmonics (harshness)"),
        ],
        facecolor="#2a2a4a", labelcolor="white",
    )
    _ax13b.text(
        0.02, 0.95,
        f"score={harm_res['score']:.4f} {harm_res['unit']}",
        transform=_ax13b.transAxes, color=C_ERROR, fontsize=9,
        verticalalignment="top", bbox=dict(facecolor="#2a2a4a", alpha=0.7),
    )

    mo.mpl.interactive(fig13)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5.0 — Perceptual error analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5.1 — Masking threshold vs error
    """)
    return


@app.cell
def _(
    BG,
    C_ERROR,
    C_PRED,
    C_TARGET,
    dark_ax,
    mask_res,
    mo,
    mticker,
    np,
    perc_model_sel,
    perc_sec_sel,
    plt,
):
    _md14  = mask_res["metadata"]
    _arr14 = mask_res["arrays"]
    _freqs14 = np.array(_arr14["freqs_hz"])
    _m_thr14 = np.array(_md14["mean_masking_threshold"])
    _m_err14 = np.array(_md14["mean_error"])
    _mfrac14 = _md14["masked_fraction"]

    def _dB14(x):
        return 20 * np.log10(np.maximum(x, 1e-10))

    fig14 = plt.figure(figsize=(14, 5))
    fig14.patch.set_facecolor(BG)
    ax14 = dark_ax(fig14.add_subplot(111))

    ax14.semilogx(_freqs14[1:], _dB14(_m_thr14)[1:], color=C_TARGET, linewidth=2,
                  label="Masking threshold")
    ax14.semilogx(_freqs14[1:], _dB14(_m_err14)[1:], color=C_ERROR, linewidth=1.5,
                  linestyle="--", label="Mean error")
    ax14.fill_between(
        _freqs14[1:], _dB14(_m_thr14)[1:], _dB14(_m_err14)[1:],
        where=_dB14(_m_err14)[1:] > _dB14(_m_thr14)[1:],
        color=C_PRED, alpha=0.35, label="Audible (above threshold)",
    )
    ax14.set_xlabel("Frequency (Hz)")
    ax14.set_ylabel("dBFS")
    ax14.set_title(
        f"Plot 14 — Masking threshold vs error  |  "
        f"{perc_model_sel.value}  /  {perc_sec_sel.value}\n"
        f"score={mask_res['score']:.4f}   "
        f"{_mfrac14:.0%} of error is masked (inaudible)"
    )
    ax14.legend(facecolor="#2a2a4a", labelcolor="white")
    ax14.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    mo.mpl.interactive(fig14)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5.2 — A-weighted error spectrum
    """)
    return


@app.cell
def _(
    BG,
    C_ERROR,
    C_NEUTRAL,
    dark_ax,
    eq_res,
    mo,
    mticker,
    np,
    perc_model_sel,
    perc_sec_sel,
    plt,
):
    _arr15   = eq_res["arrays"]
    _freqs15 = np.array(_arr15["freqs_hz"])
    _wt_err15 = np.array(_arr15["per_bin_weighted_error"])
    _raw15    = np.array(_arr15["raw_error_per_bin"])

    fig15 = plt.figure(figsize=(14, 5))
    fig15.patch.set_facecolor(BG)
    ax15 = dark_ax(fig15.add_subplot(111))

    ax15.semilogx(_freqs15[1:], _raw15[1:],    color=C_NEUTRAL, linewidth=1,
                  alpha=0.6, label="Raw error")
    ax15.semilogx(_freqs15[1:], _wt_err15[1:], color=C_ERROR, linewidth=2,
                  label="A-weighted error")
    ax15.set_xlabel("Frequency (Hz)")
    ax15.set_ylabel("Mean magnitude error")
    ax15.set_title(
        f"Plot 15 — A-Weighted Error Spectrum  |  "
        f"{perc_model_sel.value}  /  {perc_sec_sel.value}\n"
        f"score={eq_res['score']:.4f}"
    )
    ax15.legend(facecolor="#2a2a4a", labelcolor="white")
    ax15.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    mo.mpl.interactive(fig15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5.3 — Perceptual radar chart — 5 models

    Models: `tcn`, `fir`, `wiener_hammerstein`, `lstm`, `klon_ddsp`
    (subset shown only; uses full 10 s of val signal).
    """)
    return


@app.cell
def _(
    MODEL_NAMES,
    dry_full,
    equal_loudness_weighted,
    guitar_prior_weighted,
    harmonic_weighted,
    masking_weighted,
    mel_axis_weighted,
    np,
    preds,
    signal_adaptive_weighted,
    sr,
    target_full,
):
    _RADAR_SUBSET = ["tcn", "fir", "wiener_hammerstein", "lstm", "klon_ddsp"]
    RADAR_MODELS  = [m for m in _RADAR_SUBSET if m in MODEL_NAMES]

    _MAX_RADAR = int(sr * 10)
    _dry_r    = dry_full[:_MAX_RADAR].astype(np.float32)
    _target_r = target_full[:_MAX_RADAR].astype(np.float32)

    _radar_fns = [
        mel_axis_weighted, equal_loudness_weighted, masking_weighted,
        signal_adaptive_weighted, guitar_prior_weighted, harmonic_weighted,
    ]
    radar_labels = [
        "mel\naxis", "A-\nwtd", "mask\ning",
        "sig\nadapt", "guitar\nprior", "harm\nonic",
    ]

    radar_scores = {}
    for _mn_r in RADAR_MODELS:
        _pr = preds[_mn_r][:_MAX_RADAR].astype(np.float32)
        radar_scores[_mn_r] = np.array(
            [fn(_pr, _target_r, _dry_r, sr=sr)["score"] for fn in _radar_fns]
        )
    return RADAR_MODELS, radar_labels, radar_scores


@app.cell
def _(BG, COLOR_MAP, RADAR_MODELS, mo, np, plt, radar_labels, radar_scores):
    _all_arr_r = np.array(list(radar_scores.values()))
    _mn_r = _all_arr_r.min(axis=0)
    _mx_r = _all_arr_r.max(axis=0)
    _rng_r = _mx_r - _mn_r + 1e-10

    _N_sp   = len(radar_labels)
    _angles = np.linspace(0, 2 * np.pi, _N_sp, endpoint=False).tolist()
    _angles += _angles[:1]

    fig16, ax16 = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    fig16.patch.set_facecolor(BG)
    ax16.set_facecolor(BG)
    ax16.tick_params(colors="white")
    ax16.set_thetagrids(np.degrees(_angles[:-1]), radar_labels, color="white", fontsize=10)
    ax16.yaxis.set_tick_params(labelcolor="#888888")
    for _sp in ax16.spines.values():
        _sp.set_edgecolor("#555555")

    for _mn_r2 in RADAR_MODELS:
        _sc_r = radar_scores[_mn_r2]
        _norm_r = (_sc_r - _mn_r) / _rng_r
        _vals_r = _norm_r.tolist() + _norm_r[:1].tolist()
        _c_r = COLOR_MAP.get(_mn_r2, "gray")
        ax16.plot(_angles, _vals_r, linewidth=2, color=_c_r, label=_mn_r2)
        ax16.fill(_angles, _vals_r, alpha=0.06, color=_c_r)

    ax16.set_ylim(0, 1)
    ax16.set_title(
        "Plot 16 — Perceptual Metrics Radar\n(0=best per metric, 1=worst per metric)",
        color="white", pad=20,
    )
    ax16.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                facecolor="#2a2a4a", labelcolor="white", fontsize=8)

    mo.mpl.interactive(fig16)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6.0 — Perceptual metric self-test

    ## 6.1 — Clipping discriminator

    Validates that the metrics catch the known failure mode: a model that sounds
    **too clean** vs one that captures clipping character.

    Two **hand-built synthetic signals** are compared (neither is a zoo model):
    - **`linear_approx`** — 3 kHz low-pass linear EQ applied to dry signal, RMS-scaled to match target level
    - **`clipped`** — dry signal hard-clipped at 6× gain (±0.5), then 100 Hz high-pass filtered

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
    _N17    = int(sr * 4)
    _dry17    = dry_full[:_N17].astype(np.float32)
    _target17 = target_full[:_N17].astype(np.float32)

    _sos_lp17 = sig.butter(4, 3000 / (sr / 2), btype="low", output="sos")
    _lin17    = sig.sosfilt(_sos_lp17, _dry17).astype(np.float32)
    _sc17     = (float(np.sqrt(np.mean(_target17 ** 2)))
                 / (float(np.sqrt(np.mean(_lin17 ** 2))) + 1e-10))
    _lin17    = np.clip(_lin17 * _sc17, -1, 1).astype(np.float32)

    _clp17    = np.clip(_dry17 * 6.0, -0.5, 0.5)
    _sos_hp17 = sig.butter(2, 100 / (sr / 2), btype="high", output="sos")
    _clp17    = sig.sosfilt(_sos_hp17, _clp17).astype(np.float32)

    _diag_fns = [
        ("mel_axis",        mel_axis_weighted),
        ("equal_loudness",  equal_loudness_weighted),
        ("masking",         masking_weighted),
        ("signal_adaptive", signal_adaptive_weighted),
        ("guitar_prior",    guitar_prior_weighted),
        ("harmonic",        harmonic_weighted),
    ]
    _diag_rows = []
    for _nm17, _fn17 in _diag_fns:
        _r_lin  = _fn17(_lin17, _target17, _dry17, sr=sr)
        _r_clip = _fn17(_clp17, _target17, _dry17, sr=sr)
        _diag_rows.append({
            "Metric":        _nm17,
            "linear_approx": f"{_r_lin['score']:.4f}",
            "clipped":       f"{_r_clip['score']:.4f}",
            "linear worse?": "✓" if _r_lin["score"] > _r_clip["score"] else "✗",
        })

    _n_correct17 = sum(1 for r in _diag_rows if r["linear worse?"] == "✓")
    _n_fail17    = len(_diag_fns) - _n_correct17

    _callout17 = (
        mo.callout(
            mo.md(
                f"**{_n_fail17} metric(s) marked ✗** do not correctly distinguish a "
                "linear approximation from clipped audio on this signal. "
                "Treat their scores with caution until the composite weighting is "
                "validated against listening tests."
            ),
            kind="warn",
        )
        if _n_fail17 > 0
        else mo.callout(
            mo.md("All metrics correctly identify the linear approximation as worse than clipped audio."),
            kind="success",
        )
    )

    mo.vstack([
        mo.md(
            f"**{_n_correct17}/{len(_diag_fns)} metrics** correctly score "
            "`linear_approx` worse than `clipped`."
        ),
        mo.ui.table(data=_diag_rows),
        _callout17,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7.0 — Circuit topology and timing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7.1 — KlonDDSP: circuit vs ML correction fraction
    """)
    return


@app.cell
def _(mo, zoo):
    _r_kd = zoo.models.get("klon_ddsp")
    if _r_kd is None:
        mo.md("*klon_ddsp not in results.*")
    elif _r_kd.ddsp_analysis is None:
        print(f"klon_ddsp.ddsp_analysis is None (model found; analysis not populated)")
        mo.md(
            """**DDSP fraction analysis not yet computed.**

    To populate this, re-run with `analyse_ddsp_fraction` after training:

    ```python
    from pedal_model.models.neural.ddsp_klon import KlonDDSP
    # load checkpoint, then:
    analysis = model.analyse_ddsp_fraction(zoo.dry_val, zoo.wet_val)
    zoo.models['klon_ddsp'].ddsp_analysis = analysis
    zoo.save(RESULTS_PATH)
    ```
    """
        )
    else:
        _a_kd = _r_kd.ddsp_analysis
        print(f"klon_ddsp.ddsp_analysis keys: {list(_a_kd.keys())}")
        mo.md(
            f"""
    | Metric | Value |
    |--------|-------|
    | DDSP stage ESR | `{_a_kd['ddsp_esr']:.4f}` |
    | Full model ESR | `{_a_kd['full_esr']:.4f}` |
    | Correction RMS ratio | `{_a_kd['correction_rms_ratio']:.3f}` |
    | **DDSP fraction** | **{_a_kd['ddsp_fraction']:.1%}** |

    A DDSP fraction near 100 % means the Klon circuit topology accounts for almost
    all the pedal response; near 0 % means the GRU is doing all the work.
    """
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7.2 — Timing (neural models)
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, zoo):
    from pedal_model.eval.diagnostics import plot_timing_chart as _ptc72

    _neural_names72 = [n for n in MODEL_NAMES if zoo.models[n].model_type == "neural"]
    if not _neural_names72:
        mo.md("*No neural models in results.*")
    else:
        fig19 = _ptc72(zoo, _neural_names72, color_map=COLOR_MAP)
        mo.mpl.interactive(fig19)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7.3 — Raw per-section metrics table
    """)
    return


@app.cell
def _(MODEL_NAMES, mo):
    raw_model = mo.ui.dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
    raw_model
    return (raw_model,)


@app.cell
def _(mo, np, pd, raw_model, zoo):
    _rows20 = []
    for _sec20, _m20 in zoo.models[raw_model.value].metrics_per_section.items():
        _row20 = {"section": _sec20}
        _row20.update({k: round(v, 4) if np.isfinite(v) else "∞" for k, v in _m20.items()})
        _rows20.append(_row20)
    _df20 = pd.DataFrame(_rows20).set_index("section")
    mo.ui.table(_df20.reset_index())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7.4 — Aggregated summary table
    """)
    return


@app.cell
def _(mo, zoo):
    _df_summary = zoo.to_dataframe()
    mo.ui.table(_df_summary.round(4).reset_index())
    return


if __name__ == "__main__":
    app.run()
