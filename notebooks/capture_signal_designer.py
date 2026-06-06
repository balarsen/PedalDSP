# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy",
#   "scipy",
#   "soundfile",
#   "matplotlib",
#   "pandas",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import io
    import json
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import soundfile as sf
    import marimo as mo

    PROJECT_ROOT = Path("/mnt/d/Projects/PedalDSP")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from generate_capture_signal import (
        CONFIG,
        PRESETS,
        build_signal,
        write_outputs,
    )

    return (
        CONFIG,
        PRESETS,
        Path,
        build_signal,
        json,
        mo,
        np,
        pd,
        plt,
        sf,
        time,
        write_outputs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capture Signal Designer

    Build, preview, and save optimized WAV files for guitar pedal capture.

    1. **Pick a preset** — each preset tunes the parameters for a specific pedal type
    2. **Adjust parameters** in the tabs below — changes update the estimate instantly
    3. **Generate** to build the signal in memory and preview the waveform
    4. **Save** to write the WAV + sidecar JSON files to disk

    > **Master reference:** generate once with the `master` preset and never regenerate it.
    > All capture sessions should use the same master WAV as their dry source.
    """)
    return


@app.cell
def _(CONFIG, PRESETS, mo):
    preset_selector = mo.ui.dropdown(
        options=list(PRESETS.keys()),
        value="fuzz",
        label="Preset",
    )
    output_dir_input = mo.ui.text(
        value=CONFIG["OUTPUT_DIR"],
        label="Output directory",
        full_width=True,
    )
    mo.hstack([
        mo.vstack([mo.md("### Preset"), preset_selector]),
        mo.vstack([mo.md("### Output directory"), output_dir_input]),
    ], gap=2)
    return output_dir_input, preset_selector


@app.cell
def _(CONFIG, PRESETS, preset_selector):
    # Merge preset overrides onto CONFIG defaults — drives control initial values.
    preset_cfg = dict(CONFIG)
    preset_cfg.update(PRESETS[preset_selector.value])
    return (preset_cfg,)


@app.cell
def _(mo, preset_cfg):
    c_global = mo.ui.dictionary({
        "SAMPLE_RATE": mo.ui.dropdown(
            options=[8000, 22050, 44100, 48000, 88200, 96000],
            value=preset_cfg["SAMPLE_RATE"],
            label="Sample rate (Hz)",
        ),
        "BIT_DEPTH": mo.ui.dropdown(
            options=[16, 24, 32],
            value=preset_cfg["BIT_DEPTH"],
            label="Bit depth",
        ),
        "NORMALIZE_HEADROOM_DB": mo.ui.slider(
            -20, 0, value=preset_cfg["NORMALIZE_HEADROOM_DB"], step=0.5,
            label="Normalize headroom (dBFS)  — peak of final output",
        ),
        "SEED": mo.ui.number(
            0, 99999, value=preset_cfg["SEED"], step=1,
            label="RNG seed  — fix for reproducible noise",
        ),
        "WRITE_METADATA_JSON": mo.ui.checkbox(
            value=preset_cfg["WRITE_METADATA_JSON"],
            label="Write _meta.json sidecar",
        ),
        "WRITE_SEGMENT_MAP": mo.ui.checkbox(
            value=preset_cfg["WRITE_SEGMENT_MAP"],
            label="Write _segments.json sidecar",
        ),
    })
    return (c_global,)


@app.cell
def _(mo, preset_cfg):
    c_click = mo.ui.dictionary({
        "CLICK_ENABLED": mo.ui.checkbox(
            value=preset_cfg["CLICK_ENABLED"],
            label="Enable alignment click  (single-sample spike at t=0)",
        ),
        "CLICK_AMPLITUDE": mo.ui.slider(
            0.1, 1.0, value=preset_cfg["CLICK_AMPLITUDE"], step=0.05,
            label="Click amplitude  — high enough for unambiguous cross-correlation",
        ),
        "CLICK_DURATION_MS": mo.ui.slider(
            1.0, 100.0, value=preset_cfg["CLICK_DURATION_MS"], step=1.0,
            label="Click section length (ms)  — spike at sample 0, rest is silence",
        ),
        "CLICK_SILENCE_AFTER_MS": mo.ui.slider(
            100.0, 3000.0, value=preset_cfg["CLICK_SILENCE_AFTER_MS"], step=50.0,
            label="Silence after click (ms)  — lets interface settle before test signal",
        ),
    })
    return (c_click,)


@app.cell
def _(mo, preset_cfg):
    c_sweep = mo.ui.dictionary({
        "SWEEP_ENABLED": mo.ui.checkbox(
            value=preset_cfg["SWEEP_ENABLED"],
            label="Enable log sweep  (Farina exponential sweep)",
        ),
        "SWEEP_FREQ_START": mo.ui.slider(
            10.0, 500.0, value=preset_cfg["SWEEP_FREQ_START"], step=5.0,
            label="Sweep start frequency (Hz)  — lower captures coupling-cap roll-offs",
        ),
        "SWEEP_FREQ_END": mo.ui.slider(
            4000.0, 24000.0, value=preset_cfg["SWEEP_FREQ_END"], step=500.0,
            label="Sweep end frequency (Hz)",
        ),
        "SWEEP_DURATION_SEC": mo.ui.slider(
            2.0, 30.0, value=preset_cfg["SWEEP_DURATION_SEC"], step=1.0,
            label="Sweep duration per repetition (s)",
        ),
        "SWEEP_REPETITIONS": mo.ui.slider(
            1, 5, value=preset_cfg["SWEEP_REPETITIONS"], step=1,
            label="Sweep repetitions  — average for SNR improvement",
        ),
        "SWEEP_SILENCE_BETWEEN_SEC": mo.ui.slider(
            0.1, 5.0, value=preset_cfg["SWEEP_SILENCE_BETWEEN_SEC"], step=0.1,
            label="Silence between sweep repetitions (s)",
        ),
        "SWEEP_GUITAR_RANGE_ENABLED": mo.ui.checkbox(
            value=preset_cfg["SWEEP_GUITAR_RANGE_ENABLED"],
            label="Extra guitar-range sweep  (focused on 80–8000 Hz)",
        ),
        "SWEEP_GUITAR_FREQ_START": mo.ui.slider(
            20.0, 400.0, value=preset_cfg["SWEEP_GUITAR_FREQ_START"], step=5.0,
            label="Guitar sweep start (Hz)  — 82 Hz = low E string",
        ),
        "SWEEP_GUITAR_FREQ_END": mo.ui.slider(
            2000.0, 16000.0, value=preset_cfg["SWEEP_GUITAR_FREQ_END"], step=500.0,
            label="Guitar sweep end (Hz)",
        ),
        "SWEEP_GUITAR_DURATION_SEC": mo.ui.slider(
            2.0, 30.0, value=preset_cfg["SWEEP_GUITAR_DURATION_SEC"], step=1.0,
            label="Guitar sweep duration (s)",
        ),
    })
    return (c_sweep,)


@app.cell
def _(mo, preset_cfg):
    c_amp = mo.ui.dictionary({
        "AMP_SWEEP_ENABLED": mo.ui.checkbox(
            value=preset_cfg["AMP_SWEEP_ENABLED"],
            label="Enable amplitude sweep  — exposes level-dependent clipping (essential for fuzz)",
        ),
        "AMP_SWEEP_FREQ_HZ": mo.ui.slider(
            50.0, 4000.0, value=preset_cfg["AMP_SWEEP_FREQ_HZ"], step=10.0,
            label="Primary carrier frequency (Hz)  — sine tone whose amplitude sweeps",
        ),
        "AMP_SWEEP_ADDITIONAL_FREQS": mo.ui.text(
            value=", ".join(str(f) for f in preset_cfg["AMP_SWEEP_ADDITIONAL_FREQS"]),
            label="Additional carrier frequencies (comma-separated Hz)  — empty = primary only",
            full_width=True,
        ),
        "AMP_SWEEP_DURATION_SEC": mo.ui.slider(
            1.0, 30.0, value=preset_cfg["AMP_SWEEP_DURATION_SEC"], step=0.5,
            label="Duration per carrier frequency (s)",
        ),
        "AMP_SWEEP_MIN_AMPLITUDE": mo.ui.slider(
            0.0001, 0.1, value=preset_cfg["AMP_SWEEP_MIN_AMPLITUDE"], step=0.0005,
            label="Start amplitude  — near-silence to capture linear regime",
        ),
        "AMP_SWEEP_MAX_AMPLITUDE": mo.ui.slider(
            0.3, 1.0, value=preset_cfg["AMP_SWEEP_MAX_AMPLITUDE"], step=0.01,
            label="Peak amplitude  — stay below 1.0 to avoid dry-path clipping",
        ),
        "AMP_SWEEP_SHAPE": mo.ui.dropdown(
            options=["log", "linear", "triangle"],
            value=preset_cfg["AMP_SWEEP_SHAPE"],
            label="Envelope shape  — log: more time at quiet levels (best for fuzz)",
        ),
    })
    return (c_amp,)


@app.cell
def _(json, mo, preset_cfg):
    c_im = mo.ui.dictionary({
        "IM_TONES_ENABLED": mo.ui.checkbox(
            value=preset_cfg["IM_TONES_ENABLED"],
            label="Enable IM tones  — reveals intermodulation products unique to fuzz/saturation",
        ),
        "IM_TONE_PAIRS": mo.ui.text_area(
            value=json.dumps(preset_cfg["IM_TONE_PAIRS"]),
            label="Tone pairs (JSON list of [f1, f2])  — each pair played simultaneously",
            rows=4,
        ),
        "IM_TONE_DURATION_SEC": mo.ui.slider(
            0.5, 10.0, value=preset_cfg["IM_TONE_DURATION_SEC"], step=0.5,
            label="Duration per pair (s)",
        ),
        "IM_TONE_AMPLITUDE": mo.ui.slider(
            0.1, 0.7, value=preset_cfg["IM_TONE_AMPLITUDE"], step=0.05,
            label="Amplitude per tone  — two tones sum, so 0.4 → 0.8 combined",
        ),
    })
    return (c_im,)


@app.cell
def _(mo, preset_cfg):
    c_trans = mo.ui.dictionary({
        "TRANSIENT_ENABLED": mo.ui.checkbox(
            value=preset_cfg["TRANSIENT_ENABLED"],
            label="Enable transient tests  — slow vs. fast pick attack on fuzz capacitor dynamics",
        ),
        "TRANSIENT_FREQ_HZ": mo.ui.slider(
            50.0, 2000.0, value=preset_cfg["TRANSIENT_FREQ_HZ"], step=10.0,
            label="Carrier frequency (Hz)",
        ),
        "TRANSIENT_SLOW_ATTACK_MS": mo.ui.slider(
            1.0, 500.0, value=preset_cfg["TRANSIENT_SLOW_ATTACK_MS"], step=1.0,
            label="Slow attack fade-in (ms)  — simulates volume-knob roll-on",
        ),
        "TRANSIENT_FAST_ATTACK_MS": mo.ui.slider(
            0.1, 20.0, value=preset_cfg["TRANSIENT_FAST_ATTACK_MS"], step=0.1,
            label="Fast attack fade-in (ms)  — simulates hard pick strike",
        ),
        "TRANSIENT_SUSTAIN_MS": mo.ui.slider(
            50.0, 2000.0, value=preset_cfg["TRANSIENT_SUSTAIN_MS"], step=25.0,
            label="Sustain duration (ms)",
        ),
        "TRANSIENT_DECAY_MS": mo.ui.slider(
            10.0, 1000.0, value=preset_cfg["TRANSIENT_DECAY_MS"], step=10.0,
            label="Decay (ms)  — natural note release",
        ),
        "TRANSIENT_REPETITIONS": mo.ui.slider(
            1, 32, value=preset_cfg["TRANSIENT_REPETITIONS"], step=1,
            label="Repetitions per attack type  (slow and fast are separate sequences)",
        ),
        "TRANSIENT_SILENCE_BETWEEN_MS": mo.ui.slider(
            50.0, 5000.0, value=preset_cfg["TRANSIENT_SILENCE_BETWEEN_MS"], step=50.0,
            label="Silence between notes (ms)  — for delay preset: set > longest delay time",
        ),
    })
    return (c_trans,)


@app.cell
def _(mo, preset_cfg):
    c_noise = mo.ui.dictionary({
        "NOISE_ENABLED": mo.ui.checkbox(
            value=preset_cfg["NOISE_ENABLED"],
            label="Enable noise burst  — broadband catch-all after structured sweeps",
        ),
        "NOISE_DURATION_SEC": mo.ui.slider(
            0.5, 30.0, value=preset_cfg["NOISE_DURATION_SEC"], step=0.5,
            label="Noise duration (s)",
        ),
        "NOISE_AMPLITUDE": mo.ui.slider(
            0.05, 1.0, value=preset_cfg["NOISE_AMPLITUDE"], step=0.05,
            label="Noise amplitude  — peak will be ~3× this (random crest factor)",
        ),
        "NOISE_BANDPASS_ENABLED": mo.ui.checkbox(
            value=preset_cfg["NOISE_BANDPASS_ENABLED"],
            label="Bandpass filter noise  — limits energy to guitar operating range",
        ),
        "NOISE_BANDPASS_LOW_HZ": mo.ui.slider(
            10.0, 500.0, value=preset_cfg["NOISE_BANDPASS_LOW_HZ"], step=5.0,
            label="Bandpass low cutoff (Hz)",
        ),
        "NOISE_BANDPASS_HIGH_HZ": mo.ui.slider(
            1000.0, 22000.0, value=preset_cfg["NOISE_BANDPASS_HIGH_HZ"], step=500.0,
            label="Bandpass high cutoff (Hz)",
        ),
    })
    return (c_noise,)


@app.cell
def _(mo, preset_cfg):
    c_silence = mo.ui.dictionary({
        "SILENCE_BETWEEN_SECTIONS_SEC": mo.ui.slider(
            0.1, 5.0, value=preset_cfg["SILENCE_BETWEEN_SECTIONS_SEC"], step=0.1,
            label="Silence between sections (s)  — lets pedal state settle",
        ),
        "SILENCE_AT_END_SEC": mo.ui.slider(
            0.5, 15.0, value=preset_cfg["SILENCE_AT_END_SEC"], step=0.5,
            label="Trailing silence (s)  — must exceed longest delay/reverb tail",
        ),
    })
    return (c_silence,)


@app.cell
def _(
    c_amp,
    c_click,
    c_global,
    c_im,
    c_noise,
    c_silence,
    c_sweep,
    c_trans,
    mo,
):
    param_tabs = mo.ui.tabs({
        "🌐 Global":          c_global,
        "🔔 Click":           c_click,
        "〜 Sweeps":          c_sweep,
        "📈 Amplitude sweep": c_amp,
        "⚡ IM tones":        c_im,
        "🎸 Transients":      c_trans,
        "🌊 Noise":           c_noise,
        "⏱ Silence":          c_silence,
    })
    param_tabs
    return


@app.cell
def _(
    CONFIG,
    c_amp,
    c_click,
    c_global,
    c_im,
    c_noise,
    c_silence,
    c_sweep,
    c_trans,
    json,
):
    resolved_cfg = dict(CONFIG)

    # Merge each section's .value dict
    for _section in [c_global, c_click, c_sweep, c_amp, c_trans, c_noise, c_silence]:
        resolved_cfg.update(_section.value)

    # Parse IM tone pairs from JSON text area
    try:
        _pairs = json.loads(c_im.value["IM_TONE_PAIRS"])
        if not isinstance(_pairs, list):
            raise ValueError
        resolved_cfg["IM_TONE_PAIRS"] = _pairs
    except (json.JSONDecodeError, ValueError):
        resolved_cfg["IM_TONE_PAIRS"] = CONFIG["IM_TONE_PAIRS"]

    # Promote remaining IM scalar controls
    resolved_cfg["IM_TONES_ENABLED"]   = c_im.value["IM_TONES_ENABLED"]
    resolved_cfg["IM_TONE_DURATION_SEC"] = c_im.value["IM_TONE_DURATION_SEC"]
    resolved_cfg["IM_TONE_AMPLITUDE"]   = c_im.value["IM_TONE_AMPLITUDE"]

    # Parse additional amp-sweep frequencies from comma-separated text
    _raw = c_amp.value["AMP_SWEEP_ADDITIONAL_FREQS"].strip()
    if _raw:
        try:
            resolved_cfg["AMP_SWEEP_ADDITIONAL_FREQS"] = [
                float(f.strip()) for f in _raw.split(",") if f.strip()
            ]
        except ValueError:
            resolved_cfg["AMP_SWEEP_ADDITIONAL_FREQS"] = CONFIG["AMP_SWEEP_ADDITIONAL_FREQS"]
    else:
        resolved_cfg["AMP_SWEEP_ADDITIONAL_FREQS"] = []

    # SEED may come back as float from mo.ui.number
    resolved_cfg["SEED"] = int(resolved_cfg["SEED"])
    return (resolved_cfg,)


@app.cell
def _(mo, resolved_cfg):
    _sr  = resolved_cfg["SAMPLE_RATE"]
    _gap = resolved_cfg["SILENCE_BETWEEN_SECTIONS_SEC"]
    _est = 0.0
    _sections = []

    if resolved_cfg["CLICK_ENABLED"]:
        _d = resolved_cfg["CLICK_DURATION_MS"]/1000 + resolved_cfg["CLICK_SILENCE_AFTER_MS"]/1000
        _est += _d + _gap
        _sections.append(("alignment_click", _d))

    if resolved_cfg["SWEEP_ENABLED"]:
        _reps = resolved_cfg["SWEEP_REPETITIONS"]
        _d = resolved_cfg["SWEEP_DURATION_SEC"] * _reps + resolved_cfg["SWEEP_SILENCE_BETWEEN_SEC"] * max(0, _reps - 1)
        _est += _d + _gap
        _sections.append((f"log_sweep ×{_reps}", _d))
        if resolved_cfg["SWEEP_GUITAR_RANGE_ENABLED"]:
            _d2 = resolved_cfg["SWEEP_GUITAR_DURATION_SEC"]
            _est += _d2 + _gap
            _sections.append(("guitar_range_sweep", _d2))

    if resolved_cfg["AMP_SWEEP_ENABLED"]:
        _freqs = [resolved_cfg["AMP_SWEEP_FREQ_HZ"]] + list(resolved_cfg["AMP_SWEEP_ADDITIONAL_FREQS"])
        for _f in _freqs:
            _d = resolved_cfg["AMP_SWEEP_DURATION_SEC"]
            _est += _d + _gap
            _sections.append((f"amp_sweep {int(_f)} Hz", _d))

    if resolved_cfg["IM_TONES_ENABLED"]:
        for _pair in resolved_cfg["IM_TONE_PAIRS"]:
            _d = resolved_cfg["IM_TONE_DURATION_SEC"]
            _est += _d + _gap
            _sections.append((f"IM {_pair[0]}+{_pair[1]} Hz", _d))

    if resolved_cfg["TRANSIENT_ENABLED"]:
        _note_s = (resolved_cfg["TRANSIENT_SLOW_ATTACK_MS"] +
                   resolved_cfg["TRANSIENT_SUSTAIN_MS"] +
                   resolved_cfg["TRANSIENT_DECAY_MS"] +
                   resolved_cfg["TRANSIENT_SILENCE_BETWEEN_MS"]) / 1000
        for _tag, _atk in [("slow attack", resolved_cfg["TRANSIENT_SLOW_ATTACK_MS"]),
                            ("fast attack", resolved_cfg["TRANSIENT_FAST_ATTACK_MS"])]:
            _note_s2 = (_atk + resolved_cfg["TRANSIENT_SUSTAIN_MS"] +
                        resolved_cfg["TRANSIENT_DECAY_MS"] +
                        resolved_cfg["TRANSIENT_SILENCE_BETWEEN_MS"]) / 1000
            _d = _note_s2 * resolved_cfg["TRANSIENT_REPETITIONS"]
            _est += _d + _gap
            _sections.append((f"transients_{_tag} ×{resolved_cfg['TRANSIENT_REPETITIONS']}", _d))

    if resolved_cfg["NOISE_ENABLED"]:
        _d = resolved_cfg["NOISE_DURATION_SEC"]
        _est += _d + _gap
        _sections.append(("noise_burst", _d))

    _tail = resolved_cfg["SILENCE_AT_END_SEC"]
    _est += _tail
    _sections.append(("tail_silence", _tail))

    _rows = "".join(
        f"<tr><td>{lbl}</td><td style='text-align:right'>{d:.1f} s</td></tr>"
        for lbl, d in _sections
    )
    mo.callout(
        mo.md(f"""
    **Estimated duration: {_est:.1f} s &nbsp;({_est/60:.1f} min) &nbsp;·&nbsp; {int(_est * _sr):,} samples &nbsp;·&nbsp; {len(_sections)} sections**

    <table style='font-size:0.85em;width:100%'>
    <tr><th>Section</th><th>Duration</th></tr>
    {_rows}
    </table>
    """),
        kind="info",
    )
    return


@app.cell
def _(mo):
    generate_btn = mo.ui.button(label="⚡  Generate signal", kind="success")
    generate_btn
    return (generate_btn,)


@app.cell
def _(build_signal, generate_btn, mo, resolved_cfg):
    mo.stop(
        generate_btn.value == 0,
        mo.callout(mo.md("Press **Generate signal** to build the audio in memory."), kind="neutral"),
    )
    _signal, _segments = build_signal(resolved_cfg)
    signal     = _signal
    segments   = _segments
    return segments, signal


@app.cell
def _(np, plt, resolved_cfg, segments, signal):
    _sr  = resolved_cfg["SAMPLE_RATE"]
    _t   = np.arange(len(signal)) / _sr

    _fig, _axes = plt.subplots(2, 1, figsize=(14, 6))

    # Full waveform
    _axes[0].plot(_t, signal, linewidth=0.4, color="steelblue", alpha=0.85)
    _colours = plt.cm.tab10.colors
    for _i, _seg in enumerate(segments):
        _x = _seg["start_sample"] / _sr
        _axes[0].axvline(_x, color=_colours[_i % 10], alpha=0.6, linewidth=0.9, linestyle="--")
        _axes[0].text(_x + 0.05, 0.88, _seg["label"],
                      rotation=90, fontsize=6.5, va="top",
                      transform=_axes[0].get_xaxis_transform(),
                      color=_colours[_i % 10])
    _axes[0].set_xlabel("Time (s)")
    _axes[0].set_ylabel("Amplitude")
    _axes[0].set_title("Generated signal — full waveform with section markers")
    _axes[0].set_xlim(0, _t[-1])

    # Spectrogram (first 30s or whole file if shorter)
    _clip_n = min(len(signal), 30 * _sr)
    _axes[1].specgram(signal[:_clip_n], Fs=_sr, NFFT=2048, noverlap=1024,
                      cmap="inferno", vmin=-90)
    _axes[1].set_xlabel("Time (s)")
    _axes[1].set_ylabel("Frequency (Hz)")
    _axes[1].set_title(f"Spectrogram  (first {_clip_n/_sr:.0f} s)")
    _axes[1].set_ylim(0, min(_sr / 2, 20000))

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo, np, pd, resolved_cfg, segments, signal):
    _sr = resolved_cfg["SAMPLE_RATE"]
    _peak_db = 20 * np.log10(max(np.max(np.abs(signal)), 1e-9))

    _df = pd.DataFrame([
        {
            "label":        seg["label"],
            "start (s)":   f"{seg['start_sample']/_sr:.3f}",
            "end (s)":     f"{seg['end_sample']/_sr:.3f}",
            "duration (s)": f"{(seg['end_sample']-seg['start_sample'])/_sr:.3f}",
            "start sample": seg["start_sample"],
            "end sample":   seg["end_sample"],
        }
        for seg in segments
    ])

    mo.vstack([
        mo.callout(mo.md(
            f"**{len(signal)//_sr} s &nbsp;·&nbsp; {len(signal):,} samples &nbsp;·&nbsp;"
            f" {_peak_db:.2f} dBFS peak &nbsp;·&nbsp; {_sr} Hz &nbsp;·&nbsp;"
            f" {resolved_cfg['BIT_DEPTH']}-bit**"
        ), kind="success"),
        mo.md("### Segment map"),
        mo.ui.table(_df),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Audio preview
    """)
    return


@app.cell
def _(mo, np, resolved_cfg, sf, signal):
    import io as _io

    def _to_wav(arr: np.ndarray, sr: int) -> bytes:
        buf = _io.BytesIO()
        sf.write(buf, np.clip(arr, -1.0, 1.0).astype(np.float32), sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    _sr = resolved_cfg["SAMPLE_RATE"]
    _preview_sec = 30
    _preview_n   = min(len(signal), _preview_sec * _sr)

    mo.vstack([
        mo.md(f"*First {_preview_n/_sr:.0f} s of {len(signal)/_sr:.0f} s total*"),
        mo.audio(src=_to_wav(signal[:_preview_n], _sr)),
    ])
    return


@app.cell
def _(mo, preset_selector):
    _is_master = preset_selector.value == "master"
    save_note = mo.callout(
        mo.md(
            "**⚠ Master reference:** this file will be written as `master_reference.wav`. "
            "Once any capture session has used it, never regenerate it."
            if _is_master else
            "Output filename will include the preset name and a timestamp."
        ),
        kind="warn" if _is_master else "info",
    )
    force_overwrite = mo.ui.checkbox(
        value=False,
        label="Allow overwriting master_reference.wav (dangerous — only tick if intentional)",
    ) if _is_master else None

    save_btn = mo.ui.button(label="💾  Save WAV + JSON files", kind="warn")

    mo.vstack([
        save_note,
        force_overwrite if force_overwrite is not None else mo.md(""),
        save_btn,
    ])
    return force_overwrite, save_btn


@app.cell
def _(
    CONFIG,
    Path,
    force_overwrite,
    mo,
    output_dir_input,
    preset_selector,
    resolved_cfg,
    save_btn,
    segments,
    signal,
    time,
    write_outputs,
):
    mo.stop(save_btn.value == 0)

    _preset = preset_selector.value
    _out_dir = Path(output_dir_input.value)
    _cfg = dict(resolved_cfg)
    _cfg["OUTPUT_DIR"] = str(_out_dir)

    if _preset == "master":
        _base_name = CONFIG["MASTER_FILENAME"]
        _master_path = _out_dir / _base_name
        _allow = (force_overwrite is not None and force_overwrite.value)
        if _master_path.exists() and not _allow:
            mo.stop(True, mo.callout(
                mo.md(f"**Blocked:** `{_master_path}` already exists. "
                      "Tick *Allow overwriting* above if this is intentional."),
                kind="danger",
            ))
    else:
        _ts = time.strftime("%Y%m%d_%H%M%S")
        _base_name = CONFIG["SESSION_FILENAME"].format(preset=_preset, timestamp=_ts)

    _paths = write_outputs(signal, segments, _cfg, _preset, _out_dir, _base_name)

    mo.callout(
        mo.vstack([
            mo.md("**Files saved:**"),
            *[mo.md(f"- `{p}`") for p in _paths.values()],
        ]),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
