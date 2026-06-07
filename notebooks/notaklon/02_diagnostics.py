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
"""KlonSim model zoo — comprehensive diagnostic notebook.

Run with:
    marimo run  notebooks/notaklon/02_diagnostics.py
    marimo edit notebooks/notaklon/02_diagnostics.py
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import re
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

    PROJECT_ROOT = Path("/mnt/d/Projects/PedalDSP")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from pedal_model.eval.results import ZooResults
    from pedal_model.signals.manifest import Manifest

    RESULTS_PATH = PROJECT_ROOT / "results" / "data" / "klon_sim" / "klon_sim_v1" / "results.pkl"
    zoo = ZooResults.load(RESULTS_PATH)
    manifest = Manifest(PROJECT_ROOT / "data" / "signals" / "val_signal_v1.json")

    sr = zoo.sr
    MODEL_NAMES = list(zoo.models.keys())

    # Consistent colour palette across all plots
    _cmap = plt.cm.tab20
    COLOR_MAP = {name: _cmap(i / max(len(MODEL_NAMES) - 1, 1)) for i, name in enumerate(MODEL_NAMES)}

    # Section bounds from manifest
    SECTION_BOUNDS = {s.label: (s.start_sample, s.end_sample) for s in manifest.sections}
    SECTION_TYPES  = {s.label: s.type for s in manifest.sections}
    SECTION_LABELS = [s.label for s in manifest.sections]

    # Tone sections (have THD/HP metrics)
    TONE_LABELS = [s.label for s in manifest.sections_of_type("stepped_sine_tone")]

    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)
    print(f"Loaded: {len(MODEL_NAMES)} models, {len(SECTION_LABELS)} sections, sr={sr} Hz")
    print(f"Tone sections: {len(TONE_LABELS)}, others: {len(SECTION_LABELS) - len(TONE_LABELS)}")
    return (
        COLOR_MAP,
        MODEL_NAMES,
        SECTION_BOUNDS,
        SECTION_LABELS,
        SECTION_TYPES,
        TONE_LABELS,
        librosa,
        manifest,
        mo,
        np,
        pd,
        plt,
        re,
        sns,
        sr,
        zoo,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Signal overview — RMS envelope
    """)
    return


@app.cell
def _(manifest, mo, sr, zoo):
    from pedal_model.utils.plotting import plot_rms_envelope

    _sections = [(sec.start_sample, sec.end_sample) for sec in manifest.sections]
    fig_ov = plot_rms_envelope(zoo.dry_val, zoo.wet_val, sr, sections=_sections)
    mo.mpl.interactive(fig_ov)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Summary metrics heatmap (mean across all sections)
    """)
    return


@app.cell
def _(MODEL_NAMES, mo, np, pd, zoo):
    from pedal_model.utils.plotting import plot_metrics_heatmap

    _DISPLAY_METRICS = ["ESR", "null_depth_dB", "STFT", "FR_err_dB",
                        "THD_pattern_dist", "HP_sim"]
    _rows = {}
    for _name in MODEL_NAMES:
        _secs = zoo.models[_name].metrics_per_section
        _row = {}
        for _m in _DISPLAY_METRICS:
            _vals = [v[_m] for v in _secs.values() if _m in v and np.isfinite(v[_m])]
            _row[_m] = float(np.mean(_vals)) if _vals else float("nan")
        _rows[_name] = _row

    df_heat = pd.DataFrame(_rows).T
    fig_heat = plot_metrics_heatmap(df_heat, metric_cols=_DISPLAY_METRICS, figsize=(12, 5))
    mo.mpl.interactive(fig_heat)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. ESR per section (grouped by section type)
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, SECTION_TYPES, mo, pd, zoo):
    from pedal_model.utils.plotting import plot_esr_by_section_type

    _rows_esr = []
    for _name in MODEL_NAMES:
        for _sec, _metrics in zoo.models[_name].metrics_per_section.items():
            _rows_esr.append({
                "model": _name,
                "section": _sec,
                "type": SECTION_TYPES.get(_sec, "unknown"),
                "ESR": float(_metrics.get("ESR", float("nan"))),
            })
    df_esr = pd.DataFrame(_rows_esr)
    fig_esr = plot_esr_by_section_type(df_esr, MODEL_NAMES, color_map=COLOR_MAP, figsize=(16, 5))
    mo.mpl.interactive(fig_esr)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Null-test depth per section (dB — higher is better)
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, SECTION_TYPES, mo, np, pd, plt, zoo):
    _rows_nd = []
    for _name in MODEL_NAMES:
        for _sec, _metrics in zoo.models[_name].metrics_per_section.items():
            _nd = float(_metrics.get("null_depth_dB", float("nan")))
            _nd = min(_nd, 80.0) if np.isfinite(_nd) else float("nan")
            _rows_nd.append({
                "model": _name, "section": _sec,
                "type": SECTION_TYPES.get(_sec, "unknown"),
                "null_depth_dB": _nd,
            })
    df_nd = pd.DataFrame(_rows_nd)
    df_nd_grp = df_nd.groupby(["model", "type"])["null_depth_dB"].mean().reset_index()

    _types = sorted(df_nd_grp["type"].unique())
    _x = np.arange(len(_types))
    _w = 0.8 / len(MODEL_NAMES)

    fig_nd, ax_nd = plt.subplots(figsize=(16, 5))
    for _i, _name in enumerate(MODEL_NAMES):
        _sub = df_nd_grp[df_nd_grp["model"] == _name].set_index("type")
        _vals = [float(_sub.loc[t, "null_depth_dB"]) if t in _sub.index else float("nan") for t in _types]
        ax_nd.bar(_x + _i * _w - 0.4, _vals, _w, label=_name,
                  color=COLOR_MAP[_name], alpha=0.85)

    ax_nd.axhline(20, color="green", ls="--", lw=1.2, label="20 dB — inaudible threshold")
    ax_nd.axhline(10, color="orange", ls="--", lw=1.2, label="10 dB — clearly audible")
    ax_nd.set_xticks(_x)
    ax_nd.set_xticklabels(_types, rotation=20, ha="right")
    ax_nd.set_ylabel("null-test depth (dB, mean per type)")
    ax_nd.set_title("Null-test depth per section type — higher is better")
    ax_nd.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    mo.mpl.interactive(fig_nd)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Waveform overlay
    """)
    return


@app.cell
def _(MODEL_NAMES, SECTION_LABELS, mo):
    wf_section = mo.ui.dropdown(
        SECTION_LABELS, value=SECTION_LABELS[3],  # first tone section
        label="Section",
    )
    wf_models = mo.ui.multiselect(
        MODEL_NAMES, value=MODEL_NAMES,
        label="Models to overlay",
    )
    wf_window_ms = mo.ui.slider(10, 200, value=40, step=5, label="Window (ms)")
    wf_offset_ms = mo.ui.slider(0, 500, value=0, step=5, label="Offset into section (ms)")
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
    _sec = wf_section.value
    _start0, _end0 = SECTION_BOUNDS[_sec]
    _off   = int(wf_offset_ms.value * sr / 1000)
    _win   = int(wf_window_ms.value  * sr / 1000)
    _start = min(_start0 + _off, _end0 - _win)
    _end   = min(_start + _win, _end0)

    dry_w  = zoo.dry_val[_start:_end]
    wet_w  = zoo.wet_val[_start:_end]
    t_ms   = 1000.0 * np.arange(len(dry_w)) / sr

    fig_wf, (ax_wf, ax_res) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    ax_wf.plot(t_ms, dry_w, color="gray",  lw=0.7, alpha=0.6, label="dry")
    ax_wf.plot(t_ms, wet_w, color="black", lw=1.0, label="target")
    for _name in wf_models.value:
        _pred = zoo.models[_name].prediction[_start:_end]
        ax_wf.plot(t_ms, _pred, lw=0.8, alpha=0.8, color=COLOR_MAP[_name], label=_name)
    ax_wf.set_ylabel("amplitude")
    ax_wf.set_title(f"Waveform overlay — {_sec}  [{wf_window_ms.value} ms @ offset {wf_offset_ms.value} ms]")
    ax_wf.legend(fontsize=8)

    # Residual (wet - prediction) for each model
    for _name in wf_models.value:
        _pred = zoo.models[_name].prediction[_start:_end]
        ax_res.plot(t_ms, wet_w - _pred, lw=0.7, alpha=0.8, color=COLOR_MAP[_name], label=_name)
    ax_res.axhline(0, color="black", lw=0.5, ls="--")
    ax_res.set_ylabel("residual (target − pred)")
    ax_res.set_xlabel("time within window (ms)")
    ax_res.set_title("Residual (target − prediction)")
    ax_res.legend(fontsize=8)

    plt.tight_layout()
    mo.mpl.interactive(fig_wf)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Null-test residual waveform — |target − prediction|
    """)
    return


@app.cell
def _(SECTION_LABELS, mo):
    nt_section = mo.ui.dropdown(SECTION_LABELS, value=SECTION_LABELS[3], label="Section")
    nt_section
    return (nt_section,)


@app.cell
def _(
    COLOR_MAP,
    MODEL_NAMES,
    SECTION_BOUNDS,
    mo,
    np,
    nt_section,
    plt,
    sr,
    zoo,
):
    _sec = nt_section.value
    _s, _e = SECTION_BOUNDS[_sec]
    wet_nt = zoo.wet_val[_s:_e]
    t_nt   = np.arange(len(wet_nt)) / sr * 1000

    fig_nt, ax_nt = plt.subplots(figsize=(16, 5))
    for _name in MODEL_NAMES:
        _pred = zoo.models[_name].prediction[_s:_e]
        ax_nt.plot(t_nt, np.abs(wet_nt - _pred), lw=0.7, alpha=0.75,
                   color=COLOR_MAP[_name], label=_name)
    ax_nt.set_xlabel("time (ms)")
    ax_nt.set_ylabel("|target − prediction|")
    ax_nt.set_title(f"Null-test residual magnitude — {_sec}")
    ax_nt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    mo.mpl.interactive(fig_nt)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. FFT spectrum comparison
    """)
    return


@app.cell
def _(MODEL_NAMES, SECTION_LABELS, mo):
    fft_section = mo.ui.dropdown(SECTION_LABELS, value=SECTION_LABELS[0], label="Section")
    fft_models  = mo.ui.multiselect(MODEL_NAMES, value=MODEL_NAMES, label="Models")
    mo.hstack([fft_section, fft_models])
    return fft_models, fft_section


@app.cell
def _(COLOR_MAP, SECTION_BOUNDS, fft_models, fft_section, mo, sr, zoo):
    from pedal_model.utils.plotting import compare_spectra_smooth

    _sec = fft_section.value
    _s, _e = SECTION_BOUNDS[_sec]

    # Build signals dict: models first, target last so it renders on top
    _signals: dict = {}
    for _name in fft_models.value:
        _signals[_name] = (zoo.models[_name].prediction[_s:_e], COLOR_MAP[_name])
    _signals["target"] = (zoo.wet_val[_s:_e], "black")

    fig_fft = compare_spectra_smooth(
        _signals, sr, n_bins=512, floor_db=-120.0,
        figsize=(16, 6),
        title=f"Magnitude spectrum — {_sec}",
    )
    mo.mpl.interactive(fig_fft)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Spectrogram grid — target vs top-4 models by ESR
    """)
    return


@app.cell
def _(SECTION_LABELS, mo):
    spec_section = mo.ui.dropdown(SECTION_LABELS, value=SECTION_LABELS[0], label="Section for spectrogram")
    spec_section
    return (spec_section,)


@app.cell
def _(
    MODEL_NAMES,
    SECTION_BOUNDS,
    librosa,
    mo,
    np,
    plt,
    spec_section,
    sr,
    zoo,
):
    _sec = spec_section.value
    _s, _e = SECTION_BOUNDS[_sec]

    # Rank models by mean ESR ascending
    _esr_mean = {}
    for _nm in MODEL_NAMES:
        _vals = [v["ESR"] for v in zoo.models[_nm].metrics_per_section.values()
                 if "ESR" in v and np.isfinite(v["ESR"])]
        _esr_mean[_nm] = float(np.mean(_vals)) if _vals else 999.0
    _top4 = sorted(MODEL_NAMES, key=lambda n: _esr_mean[n])[:4]

    _signals = [("target (wet)", zoo.wet_val[_s:_e])]
    _signals += [(_nm, zoo.models[_nm].prediction[_s:_e]) for _nm in _top4]

    fig_spec, axes_spec = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for _ax, (_label, _x) in zip(axes_spec, _signals):
        _D = librosa.amplitude_to_db(
            np.abs(librosa.stft(_x.astype(np.float32), n_fft=1024, hop_length=256)),
            ref=np.max
        )
        librosa.display.specshow(_D, sr=sr, hop_length=256, x_axis="time", y_axis="hz",
                                 ax=_ax, vmin=-80)
        _ax.set_title(_label, fontsize=9)
        _ax.set_xlabel("")

    axes_spec[0].set_ylabel("frequency (Hz)")
    fig_spec.suptitle(f"Spectrogram comparison — {_sec}", y=1.01)
    plt.tight_layout()
    mo.mpl.interactive(fig_spec)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Frequency response deviation (FR_err_dB) per section
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, SECTION_LABELS, mo, np, plt, zoo):
    _n = len(SECTION_LABELS)
    _x = np.arange(_n)

    fig_fr, ax_fr = plt.subplots(figsize=(18, 5))
    for _name in MODEL_NAMES:
        _vals = [
            zoo.models[_name].metrics_per_section.get(_sec, {}).get("FR_err_dB", float("nan"))
            for _sec in SECTION_LABELS
        ]
        ax_fr.plot(_x, _vals, lw=1.0, marker="o", ms=3, alpha=0.8,
                   color=COLOR_MAP[_name], label=_name)

    ax_fr.set_xticks(_x)
    ax_fr.set_xticklabels(SECTION_LABELS, rotation=55, ha="right", fontsize=6)
    ax_fr.set_ylabel("FR error (dB)")
    ax_fr.set_title("Frequency response deviation per section — lower is better")
    ax_fr.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    mo.mpl.interactive(fig_fr)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. STFT loss per section
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, SECTION_LABELS, mo, np, plt, zoo):
    fig_stft, ax_stft = plt.subplots(figsize=(18, 5))
    _x = np.arange(len(SECTION_LABELS))
    for _name in MODEL_NAMES:
        _vals = [
            zoo.models[_name].metrics_per_section.get(_sec, {}).get("STFT", float("nan"))
            for _sec in SECTION_LABELS
        ]
        ax_stft.plot(_x, _vals, lw=0.9, marker="o", ms=3, alpha=0.8,
                     color=COLOR_MAP[_name], label=_name)

    ax_stft.set_xticks(_x)
    ax_stft.set_xticklabels(SECTION_LABELS, rotation=55, ha="right", fontsize=6)
    ax_stft.set_ylabel("STFT loss")
    ax_stft.set_title("Multi-scale STFT loss per section — lower is better")
    ax_stft.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    mo.mpl.interactive(fig_stft)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. ESR scatter across all sections
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, SECTION_LABELS, mo, np, plt, zoo):
    fig_sc, ax_sc = plt.subplots(figsize=(18, 5))
    _x = np.arange(len(SECTION_LABELS))
    for _name in MODEL_NAMES:
        _vals = [
            min(zoo.models[_name].metrics_per_section.get(_sec, {}).get("ESR", float("nan")), 5.0)
            for _sec in SECTION_LABELS
        ]
        ax_sc.scatter(_x, _vals, s=25, alpha=0.8, color=COLOR_MAP[_name], label=_name, zorder=3)

    ax_sc.axhline(1.0, color="black", ls="--", lw=0.8, label="ESR=1 (silence baseline)")
    ax_sc.axhline(0.01, color="green", ls="--", lw=0.8, label="ESR=0.01 target")
    ax_sc.set_xticks(_x)
    ax_sc.set_xticklabels(SECTION_LABELS, rotation=55, ha="right", fontsize=6)
    ax_sc.set_ylabel("ESR (clipped at 5.0)")
    ax_sc.set_title("ESR scatter — every model × every section")
    ax_sc.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    mo.mpl.interactive(fig_sc)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 12. ESR vs null-test depth scatter (one dot per model × section)
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, np, plt, zoo):
    fig_mv, ax_mv = plt.subplots(figsize=(10, 7))
    for _name in MODEL_NAMES:
        _esrs, _nds = [], []
        for _metrics in zoo.models[_name].metrics_per_section.values():
            _e = _metrics.get("ESR", float("nan"))
            _n = _metrics.get("null_depth_dB", float("nan"))
            if np.isfinite(_e) and np.isfinite(_n):
                _esrs.append(min(_e, 5.0))
                _nds.append(np.clip(_n, -30, 80))
        ax_mv.scatter(_esrs, _nds, s=20, alpha=0.6, color=COLOR_MAP[_name], label=_name)
        if _esrs:
            ax_mv.scatter([float(np.mean(_esrs))], [float(np.mean(_nds))],
                          s=120, marker="D", color=COLOR_MAP[_name], edgecolors="black", lw=1, zorder=5)

    ax_mv.axvline(1.0, color="gray", ls="--", lw=0.8, label="ESR=1")
    ax_mv.axhline(10, color="orange", ls="--", lw=0.8, label="10 dB (audible)")
    ax_mv.axhline(20, color="green", ls="--", lw=0.8, label="20 dB (inaudible)")
    ax_mv.set_xlabel("ESR (clipped at 5)")
    ax_mv.set_ylabel("null-test depth (dB)")
    ax_mv.set_title("ESR vs null-test depth — dots=sections, diamonds=mean")
    ax_mv.legend(fontsize=7)
    plt.tight_layout()
    mo.mpl.interactive(fig_mv)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 13. THD analysis — tone sections only
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, TONE_LABELS, mo, np, plt, zoo):
    if not TONE_LABELS:
        mo.md("*No tone sections found in results.*")
    else:
        # THD_target is the same across models (it's the target signal's THD)
        _first = list(zoo.models.values())[0]
        _thd_target = {
            _sec: _first.metrics_per_section[_sec].get("THD_target", float("nan"))
            for _sec in TONE_LABELS if _sec in _first.metrics_per_section
        }

        _n_tone = len(TONE_LABELS)
        _x = np.arange(_n_tone)
        _w = 0.7 / (len(MODEL_NAMES) + 1)

        fig_thd, (ax_thd, ax_thdp) = plt.subplots(2, 1, figsize=(18, 10))

        # THD_pred per model
        for _i, _name in enumerate(MODEL_NAMES):
            _vals = [
                zoo.models[_name].metrics_per_section.get(_sec, {}).get("THD_pred", float("nan"))
                for _sec in TONE_LABELS
            ]
            ax_thd.bar(_x + _i * _w - 0.35, _vals, _w, label=_name,
                       color=COLOR_MAP[_name], alpha=0.8)

        # THD_target as step overlay
        _t_vals = [_thd_target.get(_sec, float("nan")) for _sec in TONE_LABELS]
        ax_thd.plot(_x, _t_vals, color="black", lw=2,
                    drawstyle="steps-mid", label="THD_target")
        ax_thd.set_xticks(_x)
        ax_thd.set_xticklabels(TONE_LABELS, rotation=40, ha="right", fontsize=7)
        ax_thd.set_ylabel("THD")
        ax_thd.set_title("THD (predicted vs target) per tone section")
        ax_thd.legend(fontsize=7, ncol=4)

        # THD_pattern_dist per model
        for _i, _name in enumerate(MODEL_NAMES):
            _vals = [
                zoo.models[_name].metrics_per_section.get(_sec, {}).get("THD_pattern_dist", float("nan"))
                for _sec in TONE_LABELS
            ]
            ax_thdp.bar(_x + _i * _w - 0.35, _vals, _w, label=_name,
                        color=COLOR_MAP[_name], alpha=0.8)
        ax_thdp.set_xticks(_x)
        ax_thdp.set_xticklabels(TONE_LABELS, rotation=40, ha="right", fontsize=7)
        ax_thdp.set_ylabel("THD pattern distance")
        ax_thdp.set_title("Harmonic pattern distance per tone section — lower is better")
        ax_thdp.legend(fontsize=7, ncol=4)

        plt.tight_layout()
        mo.mpl.interactive(fig_thd)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 14. Harmonic profile similarity (HP_sim) — tone sections
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, TONE_LABELS, mo, np, plt, zoo):
    if not TONE_LABELS:
        mo.md("*No tone sections.*")
    else:
        _n_tone = len(TONE_LABELS)
        _x = np.arange(_n_tone)
        _w = 0.7 / len(MODEL_NAMES)

        fig_hp, ax_hp = plt.subplots(figsize=(18, 5))
        for _i, _name in enumerate(MODEL_NAMES):
            _vals = [
                zoo.models[_name].metrics_per_section.get(_sec, {}).get("HP_sim", float("nan"))
                for _sec in TONE_LABELS
            ]
            ax_hp.bar(_x + _i * _w - 0.35, _vals, _w, label=_name,
                      color=COLOR_MAP[_name], alpha=0.8)

        ax_hp.axhline(1.0, color="green", ls="--", lw=1, label="1.0 = perfect match")
        ax_hp.set_xticks(_x)
        ax_hp.set_xticklabels(TONE_LABELS, rotation=40, ha="right", fontsize=7)
        ax_hp.set_ylabel("HP similarity (cosine)")
        ax_hp.set_title("Harmonic profile similarity per tone section — higher is better")
        ax_hp.legend(fontsize=7, ncol=4)
        plt.tight_layout()
        mo.mpl.interactive(fig_hp)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 15. KlonDDSP: circuit vs ML correction fraction
    """)
    return


@app.cell
def _(mo, zoo):
    _r = zoo.models.get("klon_ddsp")
    if _r is None:
        mo.md("*klon_ddsp not in results.*")
    elif _r.ddsp_analysis is None:
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
        _a = _r.ddsp_analysis
        mo.md(
            f"""
    | Metric | Value |
    |--------|-------|
    | DDSP stage ESR | `{_a['ddsp_esr']:.4f}` |
    | Full model ESR | `{_a['full_esr']:.4f}` |
    | Correction RMS ratio | `{_a['correction_rms_ratio']:.3f}` |
    | **DDSP fraction** | **{_a['ddsp_fraction']:.1%}** |

    A DDSP fraction near 100 % means the Klon circuit topology accounts for almost
    all the pedal response; near 0 % means the GRU is doing all the work.
    """
        )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 16. ESR vs STFT loss — time-domain vs spectral fidelity
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, np, plt, zoo):
    fig_es, ax_es = plt.subplots(figsize=(10, 7))
    for _name in MODEL_NAMES:
        _esrs, _stfts = [], []
        for _m in zoo.models[_name].metrics_per_section.values():
            _e = _m.get("ESR", float("nan"))
            _s = _m.get("STFT", float("nan"))
            if np.isfinite(_e) and np.isfinite(_s):
                _esrs.append(min(_e, 5.0))
                _stfts.append(_s)
        ax_es.scatter(_esrs, _stfts, s=20, alpha=0.6, color=COLOR_MAP[_name], label=_name)
        if _esrs:
            ax_es.scatter([float(np.mean(_esrs))], [float(np.mean(_stfts))],
                          s=120, marker="D", color=COLOR_MAP[_name], edgecolors="black", lw=1, zorder=5)
    ax_es.axvline(1.0, color="gray", ls="--", lw=0.8, label="ESR=1 (worse than silence)")
    ax_es.set_xlabel("ESR (clipped at 5)  ← better")
    ax_es.set_ylabel("STFT loss  ← better")
    ax_es.set_title("ESR vs STFT — time-domain vs spectral fidelity\n"
                    "top-left = good both; right = right spectrum wrong phase; top = wrong spectrum")
    ax_es.legend(fontsize=7)
    plt.tight_layout()
    mo.mpl.interactive(fig_es)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 17. THD_pattern_dist vs HP_sim — harmonic shape vs harmonic similarity
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
                                s=120, marker="D", color=COLOR_MAP[_name], edgecolors="black", lw=1, zorder=5)
        ax_hpsc.set_xlabel("THD pattern distance  ← better")
        ax_hpsc.set_ylabel("HP_sim (harmonic profile cosine similarity)  → better")
        ax_hpsc.set_title("Harmonic shape vs profile similarity (tone sections)\n"
                          "models on the diagonal = metrics agree; off-diagonal = discrepancy")
        ax_hpsc.legend(fontsize=7)
        plt.tight_layout()
        mo.mpl.interactive(fig_hpsc)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 18. ESR vs drive level — does accuracy degrade under harder clipping?

    The Klon's gain stage is drive-dependent: at high drive levels the circuit saturates
    harder, producing a richer harmonic spectrum that is more difficult to model.
    This plot isolates whether model accuracy degrades in that regime.
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, np, plt, re, zoo):
    # Match _lXX or _l-XX suffix where XX is the dB level (positive or negative)
    _drive_re = re.compile(r"_l(-?\d+)")

    def _extract_drive(label: str) -> int | None:
        _m = _drive_re.search(label)
        return int(_m.group(1)) if _m else None

    fig_dr, ax_dr = plt.subplots(figsize=(12, 7))
    for _name in MODEL_NAMES:
        _drives, _esrs_dr = [], []
        for _sec, _metrics in zoo.models[_name].metrics_per_section.items():
            _d = _extract_drive(_sec)
            _e = _metrics.get("ESR", float("nan"))
            if _d is not None and np.isfinite(_e):
                _drives.append(_d)
                _esrs_dr.append(min(_e, 5.0))
        if not _drives:
            continue
        ax_dr.scatter(_drives, _esrs_dr, s=20, alpha=0.5, color=COLOR_MAP[_name])
        _unique = sorted(set(_drives))
        _means = [float(np.mean([_esrs_dr[i] for i, x in enumerate(_drives) if x == u]))
                  for u in _unique]
        ax_dr.plot(_unique, _means, marker="o", ms=7, color=COLOR_MAP[_name], label=_name)

    ax_dr.axhline(1.0, color="gray", ls="--", lw=0.8, label="ESR=1 (worse than silence)")
    ax_dr.axhline(0.01, color="green", ls="--", lw=0.8, label="ESR=0.01 (good fit target)")
    ax_dr.set_xlabel("Drive level (dB)")
    ax_dr.set_ylabel("ESR (clipped at 5.0)  ← better")
    ax_dr.set_title("ESR vs drive level — does the model fail as drive increases?")
    ax_dr.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    mo.mpl.interactive(fig_dr)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 19. Train loss vs validation ESR — undertrained vs wrong architecture
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, np, plt, zoo):
    _neural_models = [(n, zoo.models[n]) for n in MODEL_NAMES
                      if zoo.models[n].model_type == "neural"]
    if not _neural_models or all(r.train_loss is None for _, r in _neural_models):
        mo.md("*Train loss not recorded in this results file — re-run eval to populate.*")
    else:
        fig_tl, ax_tl = plt.subplots(figsize=(8, 6))
        for _name, _r in _neural_models:
            if _r.train_loss is None:
                continue
            _esr_vals = [v["ESR"] for v in _r.metrics_per_section.values()
                         if "ESR" in v and np.isfinite(v["ESR"])]
            _mean_esr = float(np.mean(_esr_vals)) if _esr_vals else float("nan")
            ax_tl.scatter([_r.train_loss], [_mean_esr], s=80, color=COLOR_MAP[_name], zorder=4)
            ax_tl.annotate(_name, (_r.train_loss, _mean_esr),
                           textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax_tl.axhline(1.0, color="gray", ls="--", lw=0.8, label="ESR=1 (silence baseline)")
        ax_tl.set_xlabel("Final training loss  ← better converged")
        ax_tl.set_ylabel("Mean validation ESR  ← better")
        ax_tl.set_title("Train loss vs val ESR\nundertrained = high loss + high ESR; "
                        "overfit = low loss + high ESR")
        ax_tl.legend(fontsize=8)
        plt.tight_layout()
        mo.mpl.interactive(fig_tl)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 20. ESR vs FR_err_dB — overall error vs linear frequency response error
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
                          s=120, marker="D", color=COLOR_MAP[_name], edgecolors="black", lw=1, zorder=5)
    ax_ef.axvline(1.0, color="gray", ls="--", lw=0.8, label="ESR=1")
    ax_ef.set_xlabel("ESR (clipped at 5)  ← better")
    ax_ef.set_ylabel("FR error dB (clipped at 80)  ← better")
    ax_ef.set_title("ESR vs FR error — low ESR + high FR_err = right nonlinearity, wrong EQ")
    ax_ef.legend(fontsize=7)
    plt.tight_layout()
    mo.mpl.interactive(fig_ef)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 21. Timing — fit/train time and predict time
    """)
    return


@app.cell
def _(COLOR_MAP, MODEL_NAMES, mo, np, plt, zoo):
    _fit_s   = [zoo.models[n].fit_time_s     for n in MODEL_NAMES]
    _pred_s  = [zoo.models[n].predict_time_s for n in MODEL_NAMES]
    _y = np.arange(len(MODEL_NAMES))
    _h = 0.35

    fig_tim, ax_tim = plt.subplots(figsize=(12, 6))
    ax_tim.barh(_y + _h / 2, _fit_s,  _h, label="fit / train (s)",
                color=[COLOR_MAP[n] for n in MODEL_NAMES], alpha=0.9)
    ax_tim.barh(_y - _h / 2, _pred_s, _h, label="predict (s)",
                color=[COLOR_MAP[n] for n in MODEL_NAMES], alpha=0.5, hatch="//")
    ax_tim.set_yticks(_y)
    ax_tim.set_yticklabels(MODEL_NAMES)
    ax_tim.set_xscale("log")
    ax_tim.set_xlabel("seconds (log scale)")
    ax_tim.set_title("Fit/train and predict time per model")
    ax_tim.legend()
    plt.tight_layout()
    mo.mpl.interactive(fig_tim)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 22. Raw per-section metrics table
    """)
    return


@app.cell
def _(MODEL_NAMES, mo):
    raw_model = mo.ui.dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
    raw_model
    return (raw_model,)


@app.cell
def _(mo, np, pd, raw_model, zoo):
    _rows = []
    for _sec, _m in zoo.models[raw_model.value].metrics_per_section.items():
        _row = {"section": _sec}
        _row.update({k: round(v, 4) if np.isfinite(v) else "∞" for k, v in _m.items()})
        _rows.append(_row)
    _df = pd.DataFrame(_rows).set_index("section")
    mo.ui.table(_df.reset_index())
    return


@app.cell
def _(mo):
    mo.md("""
    ## 23. Aggregated summary table (mean across sections)
    """)
    return


@app.cell
def _(mo, zoo):
    _df_summary = zoo.to_dataframe()
    mo.ui.table(_df_summary.round(4).reset_index())
    return


if __name__ == "__main__":
    app.run()
