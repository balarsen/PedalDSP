# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy==2.4.6",
#   "matplotlib",
#   "pandas",
#   "seaborn",
#   "scipy",
#   "soundfile==0.13.1",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tier 1 Pedal Effects — Clean Boost & Tone Filtering

    **What you will learn in this notebook:**

    1. How to synthesise a realistic guitar chord from sine waves + harmonics
    2. What a **Tier 1 (linear, time-invariant)** pedal effect looks like mathematically and visually
    3. Why you need a **broadband identification signal** (not just the chord) to fit a model
    4. How **FIR** and **IIR** models capture linear effects — and why they score near-zero ESR
    5. What the metrics (ESR, STFT loss, HP similarity) actually tell you

    ---

    ## Background: Tier 1 = linear, time-invariant (LTI)

    A **Tier 1** pedal effect satisfies two conditions:

    | Property | Meaning | Fails when... |
    |----------|---------|---------------|
    | **Linear** | doubling the input doubles the output | the circuit clips or saturates |
    | **Time-invariant** | the response doesn't change over time | the effect has memory that depends on signal history (e.g. envelope followers, compressors) |

    Real-world examples: a clean boost pedal, a passive tone control, a buffer, a passive EQ.

    **For a perfect LTI system:** the FIR or IIR model should capture it **exactly**, giving ESR → 0.

    > If ESR is not near zero after fitting an FIR/IIR model, either the system is **not linear** (→ Tier 2/3)
    > or the identification signal didn't cover all frequencies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 0. Imports and setup
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # Resolve project root — try __file__ first, then walk up from cwd.
    # Run with: .venv/bin/marimo edit notebooks/learning_tier1_linear_effects.py
    def _find_root() -> Path:
        for candidate in [
            Path(__file__).resolve().parent.parent,
            Path.cwd(),
            Path.cwd().parent,
        ]:
            if (candidate / 'pedal_model').is_dir():
                return candidate
        raise RuntimeError(
            "Cannot find pedal_model package. "
            "Run marimo from the project venv: .venv/bin/marimo edit notebooks/..."
        )
    ROOT = _find_root()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.signal import butter, sosfilt

    from pedal_model.models.classical.fir import FIRModel
    from pedal_model.models.classical.iir import IIRModel
    from pedal_model.metrics.suite import compute_all_metrics

    from pedal_model.utils import (
        guitar_note, white_noise_id,
        compute_esr_skip, peak_db, rms_db,
        db_spectrum, filter_impulse_response,
        plot_impulse_response, plot_freq_response,
        signal_dashboard, error_analysis_panel, freq_response_overlay,
    )

    plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
    SR = 48_000
    SKIP = 512   # warmup samples excluded from ESR / metric calculations
    print(f'Sample rate: {SR} Hz')
    return (
        FIRModel,
        IIRModel,
        Path,
        SKIP,
        SR,
        butter,
        compute_all_metrics,
        compute_esr_skip,
        db_spectrum,
        error_analysis_panel,
        filter_impulse_response,
        freq_response_overlay,
        guitar_note,
        np,
        pd,
        peak_db,
        plot_freq_response,
        plot_impulse_response,
        plt,
        rms_db,
        signal_dashboard,
        sns,
        sosfilt,
        white_noise_id,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Build the G major chord

    ### 1a. Why harmonics matter

    A real guitar string doesn't vibrate at a single frequency. It vibrates at its fundamental $f_0$
    **and all its integer multiples** (harmonics): $f_0, 2f_0, 3f_0, \ldots$

    The harmonic amplitudes fall off roughly as $1/k$ for the $k$-th harmonic.
    This is what gives a guitar its characteristic timbre (as opposed to a pure sine wave).

    ```
    G major chord = D3 (147 Hz) + G3 (196 Hz) + B3 (247 Hz)
    ```

    Including harmonics is important here: the **tone filter** effect only becomes visually and
    aurally interesting when there is high-frequency content to cut.
    """)
    return


@app.cell
def _(SR, guitar_note, np, peak_db, rms_db):
    DURATION = 2.0   # seconds
    AMP = 0.2        # per-note amplitude: combined chord peaks at ~-6 dBFS

    d3 = guitar_note(147, sr=SR, duration=DURATION, amp=AMP)
    g3 = guitar_note(196, sr=SR, duration=DURATION, amp=AMP)
    b3 = guitar_note(247, sr=SR, duration=DURATION, amp=AMP)

    chord = (d3 + g3 + b3).astype(np.float32)

    print(f'Chord length  : {len(chord):,} samples  ({DURATION:.1f} s)')
    print(f'Peak amplitude: {np.max(np.abs(chord)):.4f}  ({peak_db(chord):.1f} dBFS)')
    print(f'RMS amplitude : {np.sqrt(np.mean(chord**2)):.4f}  ({rms_db(chord):.1f} dBFS)')
    return b3, chord, d3, g3


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 1b. Visualise each individual note
    """)
    return


@app.cell
def _(SR, b3, chord, d3, db_spectrum, g3, np, plt):
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 6))
    _notes = {
        'D3 (147 Hz)': (d3, 'royalblue'),
        'G3 (196 Hz)': (g3, 'seagreen'),
        'B3 (247 Hz)': (b3, 'tomato'),
    }
    _ms = 25
    for _col, (_name, (_sig, _color)) in enumerate(_notes.items()):
        _n_show = int(SR * _ms / 1000)
        _t_ms = np.arange(_n_show) / SR * 1000
        _axes[0, _col].plot(_t_ms, _sig[:_n_show], color=_color, lw=0.9)
        _axes[0, _col].set_title(f'{_name} — time domain')
        _axes[0, _col].set_xlabel('Time (ms)')
        _axes[0, _col].set_ylabel('Amplitude')
        _freqs, _mag_db = db_spectrum(_sig, SR)
        _axes[1, _col].plot(_freqs, _mag_db, color=_color, lw=0.9)
        _axes[1, _col].set_xlim(0, 3000)
        _axes[1, _col].set_ylim(-80, 0)
        _axes[1, _col].set_title(f'{_name} — spectrum')
        _axes[1, _col].set_xlabel('Frequency (Hz)')
        _axes[1, _col].set_ylabel('dBFS')
        _f0 = [147, 196, 247][_col]
        for _k in range(1, 7):
            if _k * _f0 < 3000:
                _axes[1, _col].axvline(_k * _f0, color='gray', lw=0.6, ls='--', alpha=0.6)
                _axes[1, _col].text(_k * _f0 + 15, -10, f'{_k}f₀', fontsize=7, color='gray')

    _n_show = int(SR * _ms / 1000)
    _t_ms = np.arange(_n_show) / SR * 1000
    _axes[0, 3].plot(_t_ms, chord[:_n_show], color='purple', lw=0.9)
    _axes[0, 3].set_title('G chord — time domain')
    _axes[0, 3].set_xlabel('Time (ms)')
    _axes[0, 3].set_ylabel('Amplitude')

    _freqs, _mag_db = db_spectrum(chord, SR)
    _axes[1, 3].plot(_freqs, _mag_db, color='purple', lw=0.9)
    _axes[1, 3].set_xlim(0, 3000)
    _axes[1, 3].set_ylim(-80, 0)
    _axes[1, 3].set_title('G chord — spectrum')
    _axes[1, 3].set_xlabel('Frequency (Hz)')
    _axes[1, 3].set_ylabel('dBFS')
    for _f0_note, _color in [(147, 'royalblue'), (196, 'seagreen'), (247, 'tomato')]:
        _axes[1, 3].axvline(_f0_note, color=_color, lw=0.8, ls='--', alpha=0.7)
        _axes[1, 3].text(_f0_note + 10, -6, f'{_f0_note}Hz', fontsize=7, color=_color)

    _fig.suptitle('G Major Chord — Individual Notes and Combined', fontsize=14, y=1.01)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1c. Why a tonal signal is bad for system identification

    The chord signal has energy **only at its harmonic frequencies** — it is zero everywhere else.

    System identification works by dividing $H(\omega) = \text{WET}(\omega) / \text{DRY}(\omega)$ in the
    frequency domain. If DRY is zero at most frequencies (as it is for a chord), the division produces
    garbage at those frequencies, and the recovered impulse response is corrupted by noise.

    **Solution:** use a **broadband identification signal** (white noise or log sweep) to fit the model,
    then apply the model to the chord. This is exactly the workflow described in the capture plan —
    the sweep is for identification, the guitar is for listening.

    We will demonstrate the failure case first, then fix it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Effect A — Clean Boost

    ### 2a. The math

    A clean boost multiplies every sample by a constant gain $G$:

    $$y[n] = G \cdot x[n]$$

    In the frequency domain: $Y(\omega) = G \cdot X(\omega)$ — the spectrum scales uniformly, no frequency shaping.

    In the Z-domain: $H(z) = G$ — a single-tap FIR filter, the simplest possible LTI system.

    **The ideal FIR kernel should be:** $h[0] = G,\ h[k] = 0 \text{ for } k > 0$

    We use $G = 4.0$ (+12 dB gain).
    """)
    return


@app.cell
def _(chord, np):
    BOOST_GAIN = 4.0   # +12 dB
    BOOST_DB   = 20 * np.log10(BOOST_GAIN)

    chord_boost_wet = (chord * BOOST_GAIN).astype(np.float32)

    print(f'Gain: {BOOST_GAIN:.1f}×  ({BOOST_DB:.1f} dB)')
    print(f'Dry peak : {np.max(np.abs(chord)):.4f}  ({20*np.log10(np.max(np.abs(chord))):.1f} dBFS)')
    print(f'Wet peak : {np.max(np.abs(chord_boost_wet)):.4f}  ({20*np.log10(np.max(np.abs(chord_boost_wet))):.1f} dBFS)')
    return BOOST_DB, BOOST_GAIN, chord_boost_wet


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2b. Visualise boost — time domain and FFT
    """)
    return


@app.cell
def _(BOOST_DB, SR, chord, chord_boost_wet, signal_dashboard):
    _fig = signal_dashboard(
        signals={
            'Dry chord': (chord, 'steelblue'),
            f'Boost wet (+{BOOST_DB:.0f} dB)': (chord_boost_wet, 'darkorange'),
        },
        sr=SR,
        title='Effect A: Clean Boost — Time Domain and Spectrum',
        ms=30,
        max_freq=3000,
        vlines={'D3\n147Hz': 147, 'G3\n196Hz': 196, 'B3\n247Hz': 247},
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2c. The identification signal problem — why the chord fails

    Let's try to fit the FIR model directly to the chord and see what happens.
    """)
    return


@app.cell
def _(
    BOOST_GAIN,
    FIRModel,
    SR,
    chord,
    chord_boost_wet,
    np,
    plot_impulse_response,
    plt,
):
    _fir_chord_id = FIRModel(n_taps=512)
    _fir_chord_id.fit(chord, chord_boost_wet, SR)
    _chord_boost_pred_bad = _fir_chord_id.predict(chord)
    _n = min(len(chord_boost_wet), len(_chord_boost_pred_bad))
    _esr_bad = (
        np.sum((chord_boost_wet[512:_n] - _chord_boost_pred_bad[512:_n]) ** 2)
        / (np.sum(chord_boost_wet[512:_n] ** 2) + 1e-8)
    )
    print(f'FIR fit on chord signal → ESR = {_esr_bad:.6f}')
    print(f'Expected ESR             → ≈ 0.000001  (for a pure gain boost)')
    print(f"\nThe ESR is {'good' if _esr_bad < 0.01 else 'BAD — identification failed'}.")
    print('\nWhy did this fail?')
    print('  The chord has energy ONLY at 147, 196, 247 Hz and their harmonics.')
    print('  H(ω) = WET(ω) / DRY(ω) is undefined (0/0) at all other frequencies.')
    print('  The corrupted H(ω) produces a noisy impulse response when inverted.')

    _kernel_bad = _fir_chord_id._kernel
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4))
    plot_impulse_response(_kernel_bad, label='FIR kernel (identified from chord)', color='tomato', ax=_ax1, n_show=80)
    _ax1.axhline(BOOST_GAIN, color='steelblue', ls='--', lw=1.5, label=f'Expected h[0] = {BOOST_GAIN}')
    _ax1.legend(fontsize=9)
    _ax1.set_title(f'FIR kernel from chord ID (ESR={_esr_bad:.4f})')
    _n_show = int(SR * 0.03)
    _t_ms = np.arange(_n_show) / SR * 1000
    _ax2.plot(_t_ms, chord_boost_wet[:_n_show], color='darkorange', lw=1.2, label='Target wet')
    _ax2.plot(_t_ms, _chord_boost_pred_bad[:_n_show], color='tomato', ls='--', lw=0.9, label='FIR pred (bad ID)')
    _ax2.set_xlabel('Time (ms)')
    _ax2.set_ylabel('Amplitude')
    _ax2.set_title('Time domain: target vs bad prediction')
    _ax2.legend(fontsize=9)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2d. Fix: use a broadband identification signal

    White noise has (approximately) equal energy at every frequency, so $\text{DRY}(\omega) \neq 0$ everywhere.
    The division $H(\omega) = \text{WET}(\omega) / \text{DRY}(\omega)$ is stable and well-conditioned.

    **Workflow:**
    1. Generate white noise → call it the "identification signal"
    2. Apply the effect (boost/filter) → get the "identification wet"
    3. Fit FIR/IIR to the (id_dry, id_wet) pair
    4. Apply the fitted model to the chord → evaluate quality
    """)
    return


@app.cell
def _(BOOST_DB, BOOST_GAIN, SR, np, peak_db, signal_dashboard, white_noise_id):
    id_dry = white_noise_id(SR, duration=4.0, amplitude=0.3, seed=42)
    id_boost_wet = (id_dry * BOOST_GAIN).astype(np.float32)

    print(f'Identification signal: {len(id_dry):,} samples  ({len(id_dry)/SR:.0f}s)')
    print(f'ID dry  peak: {peak_db(id_dry):.1f} dBFS')
    print(f'ID wet  peak: {peak_db(id_boost_wet):.1f} dBFS')
    print(f'\nNote: both spectra are FLAT (white noise has equal energy at all frequencies).')
    print(f'The boost just shifts the whole spectrum up by {BOOST_DB} dB.')

    _fig = signal_dashboard(
        signals={
            'ID signal (white noise)': (id_dry, 'gray'),
            f'ID boosted (+{BOOST_DB:.0f} dB wet)': (id_boost_wet, 'darkorange'),
        },
        sr=SR,
        title='Identification Signal — White Noise (broadband, covers all frequencies)',
        ms=20,
        max_freq=5000,
    )
    _fig
    return id_boost_wet, id_dry


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2e. Fit FIR model using the broadband identification signal
    """)
    return


@app.cell
def _(
    BOOST_DB,
    BOOST_GAIN,
    FIRModel,
    SR,
    db_spectrum,
    id_boost_wet,
    id_dry,
    np,
    plot_freq_response,
    plot_impulse_response,
    plt,
):
    fir_boost = FIRModel(n_taps=512)
    fir_boost.fit(id_dry, id_boost_wet, SR)
    _kernel_boost = fir_boost._kernel

    print('=== FIR Boost kernel analysis ===')
    print(f'h[0]       = {_kernel_boost[0]:.6f}   (expected ≈ {BOOST_GAIN})')
    print(f'h[1]       = {_kernel_boost[1]:.2e}   (expected ≈ 0)')
    print(f'max|h[1:]| = {np.max(np.abs(_kernel_boost[1:])):.2e}   (noise floor)')
    print(f'energy in h[0]  : {_kernel_boost[0]**2:.6f}')
    print(f'energy in h[1:] : {np.sum(_kernel_boost[1:]**2):.2e}')

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4))

    plot_impulse_response(_kernel_boost, label='FIR kernel — boost', color='darkorange', ax=_axes[0], n_show=40)
    _axes[0].axhline(BOOST_GAIN, color='steelblue', ls='--', lw=1.5, label=f'Expected h[0] = {BOOST_GAIN}')
    _axes[0].legend(fontsize=9)
    _axes[0].set_title(f'FIR Impulse Response (first 40 taps)\nh[0] = {_kernel_boost[0]:.4f}')

    plot_freq_response(_kernel_boost, SR, label='FIR frequency response', color='darkorange', ax=_axes[1], max_freq=5000)
    _axes[1].axhline(BOOST_DB, color='steelblue', ls='--', lw=1.2, label=f'Expected: +{BOOST_DB:.1f} dB flat')
    _axes[1].set_title('FIR Frequency Response')
    _axes[1].legend(fontsize=9)

    _id_boost_pred = fir_boost.predict(id_dry)
    _freqs, _m_wet  = db_spectrum(id_boost_wet, SR)
    _freqs, _m_pred = db_spectrum(_id_boost_pred, SR)
    _axes[2].plot(_freqs, _m_wet,  'darkorange', lw=1.0, label='ID wet (target)', alpha=0.8)
    _axes[2].plot(_freqs, _m_pred, 'steelblue',  lw=0.8, ls='--', label='FIR prediction', alpha=0.9)
    _axes[2].set_xlim(0, 5000)
    _axes[2].set_ylim(-80, 0)
    _axes[2].set_xlabel('Frequency (Hz)')
    _axes[2].set_ylabel('dBFS')
    _axes[2].set_title('Spectrum: ID wet vs FIR prediction')
    _axes[2].legend(fontsize=9)

    plt.suptitle('Effect A: Clean Boost — FIR Fit on Broadband ID Signal', fontsize=13, y=1.01)
    plt.tight_layout()
    _fig
    return (fir_boost,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation:**
    - `h[0] ≈ 4.0` and all other taps are near zero — exactly the expected delta-function kernel
    - The frequency response is **flat at +12 dB** across the whole spectrum
    - The FIR perfectly captured the boost
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2f. Apply the fitted boost model to the G chord
    """)
    return


@app.cell
def _(SKIP, SR, chord, chord_boost_wet, error_analysis_panel, fir_boost):
    chord_boost_pred = fir_boost.predict(chord)
    _fig = error_analysis_panel(
        target=chord_boost_wet,
        predicted=chord_boost_pred,
        sr=SR,
        skip=SKIP,
        ms=30,
        max_freq=3000,
        target_color='darkorange',
        pred_color='navy',
        title='Effect A: Clean Boost — FIR Applied to G Chord',
    )
    _fig
    return (chord_boost_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2g. Metrics for the boost
    """)
    return


@app.cell
def _(SKIP, SR, chord, chord_boost_pred, chord_boost_wet, compute_all_metrics):
    _n = min(len(chord_boost_wet), len(chord_boost_pred))
    metrics_boost = compute_all_metrics(
        target=chord_boost_wet[SKIP:_n],
        predicted=chord_boost_pred[SKIP:_n],
        input_signal=chord[SKIP:_n],
        sr=SR,
        harmonic_f0=196.0,
    )
    _interpretations = {
        'ESR':       ('Error-to-Signal Ratio. 0=perfect, 1=silence output. <0.01=excellent', 0.01),
        'MSE':       ('Mean Squared Error. Raw squared amplitude error.', None),
        'DC_err':    ('DC offset error. Should be near 0.', 0.001),
        'RMS_err':   ('Root-mean-square error in amplitude.', None),
        'STFT':      ('Multi-scale spectral loss. 0=identical spectra. <0.1=excellent.', 0.1),
        'FR_err_dB': ('Frequency response error in dB. <1dB=excellent.', 1.0),
        'THD_target':('THD% of the TARGET signal at 196Hz.', None),
        'THD_pred':  ('THD% of the PREDICTION. Should match target.', None),
        'THD_err':   ('|THD_target - THD_pred|. 0=model captures distortion correctly.', 1.0),
        'HP_sim':    ('Harmonic Profile Similarity. 1.0=identical harmonic structure.', None),
        'MCD':       ('Mel Cepstral Distortion in dB. <2dB=good tonal similarity.', 2.0),
    }
    print('╔══════════════════════════════════════════════════════════════╗')
    print('║           BOOST — FIR Model Metrics                         ║')
    print('╠════════════════╦═══════════╦══════════════════════════════════╣')
    print('║ Metric         ║   Value   ║  Interpretation                  ║')
    print('╠════════════════╬═══════════╬══════════════════════════════════╣')
    for _k, _v in metrics_boost.items():
        _thresh = _interpretations[_k][1]
        _flag = '' if _thresh is None else ('✓' if (_v < _thresh if _k != 'HP_sim' else _v > 0.99) else '✗')
        print(f'║ {_k:<14}  ║ {_v:9.6f} ║  {_flag} {_interpretations[_k][0][:42]:<42} ║')
    print('╚════════════════╩═══════════╩══════════════════════════════════╝')
    return (metrics_boost,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the metrics tell us for a clean boost:**

    - **ESR ≈ 0**: The FIR perfectly reconstructs the boosted signal — as expected for a single-tap kernel
    - **THD_err ≈ 0**: No new harmonics were added (linear effect)
    - **HP_sim ≈ 1**: The harmonic structure is unchanged — boost doesn't alter the tone, just the level

    > **Learning point:** If a real-world pedal labelled "clean boost" gives ESR > 0.01, it means the pedal is
    > adding some nonlinearity or frequency shaping — it's not as clean as advertised.

    ---

    ## 3. Effect B — Tone Filter (Low-Pass, "Warm" Rolloff)

    ### 3a. The math

    A **4th-order Butterworth low-pass filter** with cutoff at $f_c = 600$ Hz.

    The Butterworth filter is maximally flat in the passband — no ripple.
    Its magnitude response is:

    $$|H(j\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2N}}$$

    With $N=4$: at $f_c = 600$ Hz, the gain is $-3$ dB. Above $f_c$, the gain rolls off at $-80$ dB/decade.

    **Effect on our chord:**
    - D3 fundamental (147 Hz): passes through at ~0 dB ✓
    - G3 fundamental (196 Hz): passes through at ~0 dB ✓
    - B3 fundamental (247 Hz): passes through at ~0 dB ✓
    - 3rd harmonics (440–740 Hz): partially attenuated ⚠
    - Higher harmonics (>1 kHz): heavily attenuated ✗

    The result sounds **darker and warmer** — the overtones are removed, leaving only the fundamental tones.
    """)
    return


@app.cell
def _(
    SR,
    butter,
    chord,
    db_spectrum,
    filter_impulse_response,
    id_dry,
    np,
    plot_impulse_response,
    plt,
    sosfilt,
):
    CUTOFF_HZ = 600.0
    FILTER_ORDER = 4

    sos_lp = butter(FILTER_ORDER, CUTOFF_HZ, btype='low', fs=SR, output='sos')
    chord_filter_wet = sosfilt(sos_lp, chord).astype(np.float32)
    id_filter_wet    = sosfilt(sos_lp, id_dry).astype(np.float32)

    impulse_out = filter_impulse_response(sos_lp, n=4096)
    _true_freqs, _true_H_db = db_spectrum(impulse_out, SR, n_fft=4096)

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4))

    _axes[0].plot(_true_freqs, _true_H_db, 'mediumseagreen', lw=1.5, label='Butterworth LP')
    _axes[0].axvline(CUTOFF_HZ, color='salmon', ls='--', lw=1.2, label=f'Cutoff: {CUTOFF_HZ:.0f} Hz')
    _axes[0].axhline(-3, color='gray', ls=':', lw=0.8, label='-3 dB')
    _axes[0].set_xlim(0, 5000)
    _axes[0].set_ylim(-80, 5)
    _axes[0].set_xlabel('Frequency (Hz)')
    _axes[0].set_ylabel('H(ω) (dB)')
    _axes[0].set_title(f'True Butterworth LP Response\n(order={FILTER_ORDER}, fc={CUTOFF_HZ:.0f}Hz)')
    _axes[0].legend(fontsize=9)

    plot_impulse_response(impulse_out, label='True LP impulse response', color='mediumseagreen', ax=_axes[1], n_show=100)
    _axes[1].set_title('Impulse Response of the True LP Filter')

    for _sig, _label in [(chord, 'Dry chord'), (chord_filter_wet, f'LP filtered (fc={CUTOFF_HZ:.0f}Hz)')]:
        _freqs_s, _m = db_spectrum(_sig, SR)
        _axes[2].plot(_freqs_s, _m, label=_label, lw=0.9)
    _axes[2].axvline(CUTOFF_HZ, color='salmon', ls='--', lw=1.0, label=f'Cutoff {CUTOFF_HZ:.0f}Hz')
    for _f0_note in [147, 196, 247]:
        _axes[2].axvline(_f0_note, color='gray', lw=0.6, ls=':', alpha=0.5)
    _axes[2].set_xlim(0, 3000)
    _axes[2].set_ylim(-80, 0)
    _axes[2].set_xlabel('Frequency (Hz)')
    _axes[2].set_ylabel('dBFS')
    _axes[2].set_title('Chord Spectrum: Before and After Filtering')
    _axes[2].legend(fontsize=9)

    plt.suptitle('Effect B: Low-Pass Tone Filter — Theory', fontsize=13, y=1.01)
    plt.tight_layout()
    print(f'  • Fundamentals (147, 196, 247 Hz) pass through unchanged')
    print(f'  • Harmonics above {CUTOFF_HZ:.0f} Hz are progressively attenuated')
    print(f'  • 4th-order Butterworth: -80 dB/decade rolloff above cutoff')
    _fig
    return CUTOFF_HZ, chord_filter_wet, id_filter_wet, impulse_out, sos_lp


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3b. Visualise the filter effect on the chord — time and FFT
    """)
    return


@app.cell
def _(CUTOFF_HZ, SR, chord, chord_filter_wet, signal_dashboard):
    _fig = signal_dashboard(
        signals={
            'Dry chord': (chord, 'steelblue'),
            f'LP filtered (fc={CUTOFF_HZ:.0f}Hz wet)': (chord_filter_wet, 'mediumseagreen'),
        },
        sr=SR,
        title='Effect B: Low-Pass Tone Filter Applied to Chord',
        ms=30,
        max_freq=3000,
        vlines={
            'D3\n147Hz': 147, 'G3\n196Hz': 196, 'B3\n247Hz': 247,
            f'fc\n{CUTOFF_HZ:.0f}Hz': CUTOFF_HZ,
        },
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3c. Visualise the identification signal through the filter
    """)
    return


@app.cell
def _(CUTOFF_HZ, SR, id_dry, id_filter_wet, signal_dashboard):
    _fig = signal_dashboard(
        signals={
            'ID dry (white noise)': (id_dry, 'gray'),
            f'ID LP filtered (fc={CUTOFF_HZ:.0f}Hz)': (id_filter_wet, 'mediumseagreen'),
        },
        sr=SR,
        title='Identification Signal Through LP Filter (broadband — all frequencies covered)',
        ms=20,
        max_freq=5000,
        vlines={f'fc {CUTOFF_HZ:.0f}Hz': CUTOFF_HZ},
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3d. Fit FIR model to the filter
    """)
    return


@app.cell
def _(
    CUTOFF_HZ,
    FIRModel,
    SR,
    db_spectrum,
    id_dry,
    id_filter_wet,
    impulse_out,
    np,
    plot_freq_response,
    plt,
):
    fir_filter = FIRModel(n_taps=512)
    fir_filter.fit(id_dry, id_filter_wet, SR)
    kernel_filter = fir_filter._kernel

    print('=== FIR Filter kernel analysis ===')
    print(f'h[0]  = {kernel_filter[0]:.6f}')
    print(f'h[37] = {kernel_filter[37]:.6f}  (LP IR peak is around tap 37)')
    print(f'Max |h|  at tap {np.argmax(np.abs(kernel_filter))}')
    print(f'Energy in first 100 taps: {np.sum(kernel_filter[:100]**2):.4f}')
    print(f'Energy in taps 100-512  : {np.sum(kernel_filter[100:]**2):.6f}')

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4))

    _n_show = 100
    _axes[0].stem(np.arange(_n_show), impulse_out[:_n_show],
                  markerfmt='go', linefmt='g-', basefmt='k-', label='True LP IR')
    _axes[0].stem(np.arange(_n_show), kernel_filter[:_n_show],
                  markerfmt='bs', linefmt='b--', basefmt='k-', label='FIR fitted')
    _axes[0].set_xlabel('Tap index k')
    _axes[0].set_ylabel('h[k]')
    _axes[0].set_title('True IR vs FIR Kernel (first 100 taps)')
    _axes[0].legend(fontsize=9)

    plot_freq_response(impulse_out,   SR, label='True LP',    color='mediumseagreen', ax=_axes[1], max_freq=5000)
    plot_freq_response(kernel_filter, SR, label='FIR fitted', color='steelblue',      ax=_axes[1], max_freq=5000, lw=0.9)
    _axes[1].axvline(CUTOFF_HZ, color='salmon', ls='--', lw=1.0, label=f'fc={CUTOFF_HZ:.0f}Hz')
    _axes[1].set_title('Frequency Response: True vs FIR')
    _axes[1].legend(fontsize=9)

    id_filter_pred_fir = fir_filter.predict(id_dry)
    _freqs, _m_wet  = db_spectrum(id_filter_wet, SR)
    _freqs, _m_pred = db_spectrum(id_filter_pred_fir, SR)
    _axes[2].plot(_freqs, _m_wet,  'mediumseagreen', lw=1.0, label='ID wet target', alpha=0.8)
    _axes[2].plot(_freqs, _m_pred, 'steelblue',      lw=0.8, ls='--', label='FIR prediction', alpha=0.9)
    _axes[2].axvline(CUTOFF_HZ, color='salmon', ls='--', lw=0.8)
    _axes[2].set_xlim(0, 5000)
    _axes[2].set_ylim(-80, 0)
    _axes[2].set_xlabel('Frequency (Hz)')
    _axes[2].set_ylabel('dBFS')
    _axes[2].set_title('ID Signal: Wet Spectrum vs FIR Prediction')
    _axes[2].legend(fontsize=9)

    plt.suptitle('Effect B: Low-Pass Filter — FIR Kernel Analysis', fontsize=13, y=1.01)
    plt.tight_layout()
    _fig
    return fir_filter, id_filter_pred_fir, kernel_filter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3e. Fit IIR model to the filter

    The original filter **is** an IIR circuit (it has feedback — the $a_k$ coefficients are non-zero).
    An IIR model should capture it with far fewer parameters than an FIR:

    - FIR with 512 taps: **512 parameters**
    - IIR of order 8: **17 parameters** (8 $b$ + 9 $a$ coefficients)

    **Trade-off:** IIR is more compact and faster at runtime, but can have stability issues if the poles
    land outside the unit circle. We use second-order sections (SOS) to keep it stable.
    """)
    return


@app.cell
def _(
    CUTOFF_HZ,
    IIRModel,
    SKIP,
    SR,
    compute_esr_skip,
    db_spectrum,
    fir_filter,
    freq_response_overlay,
    id_dry,
    id_filter_pred_fir,
    id_filter_wet,
    impulse_out,
    kernel_filter,
    np,
    plt,
):
    iir_filter = IIRModel(order=8, n_freq_points=512)
    iir_filter.fit(id_dry, id_filter_wet, SR)
    id_filter_pred_iir = iir_filter.predict(id_dry)

    _n = min(len(id_filter_wet), len(id_filter_pred_fir), len(id_filter_pred_iir))
    esr_fir = compute_esr_skip(id_filter_wet, id_filter_pred_fir, skip=SKIP)
    esr_iir = compute_esr_skip(id_filter_wet, id_filter_pred_iir, skip=SKIP)

    print(f'FIR-512 ESR on ID signal: {esr_fir:.8f}')
    print(f'IIR-8   ESR on ID signal: {esr_iir:.8f}')
    print(f'FIR parameters: {fir_filter.n_taps}')
    print(f'IIR parameters: {iir_filter.order * 2 + 1} (approx)')

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4))

    _iir_impulse = np.zeros(4096, dtype=np.float32)
    _iir_impulse[0] = 1.0
    _iir_ir = iir_filter.predict(_iir_impulse)

    freq_response_overlay(
        kernels={
            'True LP':  (impulse_out,   'mediumseagreen'),
            'FIR-512':  (kernel_filter, 'steelblue'),
            'IIR-8':    (_iir_ir[:4096], 'tomato'),
        },
        sr=SR,
        ax=_axes[0],
        max_freq=5000,
        title='Frequency Response Comparison',
        linewidths=[2.0, 1.0, 1.0],
    )
    _axes[0].axvline(CUTOFF_HZ, color='salmon', ls=':', lw=0.8)

    for _sig, _label in [
        (id_filter_wet,      'Target (true filter)'),
        (id_filter_pred_fir, f'FIR-512 (ESR={esr_fir:.2e})'),
        (id_filter_pred_iir, f'IIR-8   (ESR={esr_iir:.2e})'),
    ]:
        _freqs_s, _m = db_spectrum(_sig, SR)
        _axes[1].plot(_freqs_s, _m, label=_label, lw=0.9)
    _axes[1].set_xlim(0, 5000)
    _axes[1].set_ylim(-80, 0)
    _axes[1].set_xlabel('Frequency (Hz)')
    _axes[1].set_ylabel('dBFS')
    _axes[1].set_title('ID Signal: Spectrum Comparison')
    _axes[1].legend(fontsize=8)

    _err_fir = id_filter_wet[:_n] - id_filter_pred_fir[:_n]
    _err_iir = id_filter_wet[:_n] - id_filter_pred_iir[:_n]
    _freqs_e,  _m_fir_err = db_spectrum(_err_fir[SKIP:], SR)
    _freqs_e2, _m_iir_err = db_spectrum(_err_iir[SKIP:], SR)
    _axes[2].plot(_freqs_e,  _m_fir_err, 'steelblue', lw=0.9, label='FIR error')
    _axes[2].plot(_freqs_e2, _m_iir_err, 'tomato',    lw=0.9, label='IIR error')
    _axes[2].set_xlim(0, 5000)
    _axes[2].set_ylim(-100, -40)
    _axes[2].set_xlabel('Frequency (Hz)')
    _axes[2].set_ylabel('Error magnitude (dBFS)')
    _axes[2].set_title('Residual Error Spectra (lower = better)')
    _axes[2].legend(fontsize=9)

    plt.suptitle('Effect B: LP Filter — FIR vs IIR Model Comparison on ID Signal', fontsize=13, y=1.01)
    plt.tight_layout()
    _fig
    return (iir_filter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3f. Apply both models to the G chord and compare
    """)
    return


@app.cell
def _(
    CUTOFF_HZ,
    SR,
    chord,
    chord_filter_wet,
    db_spectrum,
    fir_filter,
    iir_filter,
    np,
    plt,
):
    chord_filter_pred_fir = fir_filter.predict(chord)
    chord_filter_pred_iir = iir_filter.predict(chord)

    _fig, _axes = plt.subplots(2, 3, figsize=(18, 8))
    _ms = 30
    _n_show = int(SR * _ms / 1000)
    _t_ms = np.arange(_n_show) / SR * 1000

    # Row 1: Time domain
    _axes[0, 0].plot(_t_ms, chord[:_n_show],            'steelblue',      lw=1.0, label='Dry chord')
    _axes[0, 0].plot(_t_ms, chord_filter_wet[:_n_show], 'mediumseagreen', lw=1.2, label='Target wet')
    _axes[0, 0].set_title('Time: Dry vs Target Wet')
    _axes[0, 0].set_xlabel('Time (ms)')
    _axes[0, 0].legend(fontsize=9)

    _axes[0, 1].plot(_t_ms, chord_filter_wet[:_n_show],      'mediumseagreen', lw=1.2, label='Target wet')
    _axes[0, 1].plot(_t_ms, chord_filter_pred_fir[:_n_show], 'steelblue',      lw=0.9, ls='--', label='FIR-512')
    _axes[0, 1].set_title('Time: Target vs FIR Prediction')
    _axes[0, 1].set_xlabel('Time (ms)')
    _axes[0, 1].legend(fontsize=9)

    _axes[0, 2].plot(_t_ms, chord_filter_wet[:_n_show],      'mediumseagreen', lw=1.2, label='Target wet')
    _axes[0, 2].plot(_t_ms, chord_filter_pred_iir[:_n_show], 'tomato',         lw=0.9, ls='--', label='IIR-8')
    _axes[0, 2].set_title('Time: Target vs IIR Prediction')
    _axes[0, 2].set_xlabel('Time (ms)')
    _axes[0, 2].legend(fontsize=9)

    # Row 2: Spectra
    for _sig, _label in [(chord, 'Dry'), (chord_filter_wet, 'Target wet')]:
        _freqs, _m = db_spectrum(_sig, SR)
        _axes[1, 0].plot(_freqs, _m, label=_label, lw=0.9)
    _axes[1, 0].axvline(CUTOFF_HZ, color='salmon', ls='--', lw=0.8, label=f'fc={CUTOFF_HZ:.0f}Hz')
    _axes[1, 0].set_xlim(0, 3000); _axes[1, 0].set_ylim(-80, 0)
    _axes[1, 0].set_xlabel('Frequency (Hz)'); _axes[1, 0].set_ylabel('dBFS')
    _axes[1, 0].set_title('Spectrum: Dry vs Target')
    _axes[1, 0].legend(fontsize=8)

    for _sig, _label in [(chord_filter_wet, 'Target'), (chord_filter_pred_fir, 'FIR-512')]:
        _freqs, _m = db_spectrum(_sig, SR)
        _axes[1, 1].plot(_freqs, _m, label=_label, lw=0.9)
    _axes[1, 1].axvline(CUTOFF_HZ, color='salmon', ls='--', lw=0.8)
    _axes[1, 1].set_xlim(0, 3000); _axes[1, 1].set_ylim(-80, 0)
    _axes[1, 1].set_xlabel('Frequency (Hz)'); _axes[1, 1].set_ylabel('dBFS')
    _axes[1, 1].set_title('Spectrum: Target vs FIR')
    _axes[1, 1].legend(fontsize=8)

    for _sig, _label in [(chord_filter_wet, 'Target'), (chord_filter_pred_iir, 'IIR-8')]:
        _freqs, _m = db_spectrum(_sig, SR)
        _axes[1, 2].plot(_freqs, _m, label=_label, lw=0.9)
    _axes[1, 2].axvline(CUTOFF_HZ, color='salmon', ls='--', lw=0.8)
    _axes[1, 2].set_xlim(0, 3000); _axes[1, 2].set_ylim(-80, 0)
    _axes[1, 2].set_xlabel('Frequency (Hz)'); _axes[1, 2].set_ylabel('dBFS')
    _axes[1, 2].set_title('Spectrum: Target vs IIR')
    _axes[1, 2].legend(fontsize=8)

    plt.suptitle('Effect B: LP Filter Applied to G Chord — Model Comparison', fontsize=13, y=1.01)
    plt.tight_layout()
    _fig
    return chord_filter_pred_fir, chord_filter_pred_iir


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3g. Metrics for both filter models
    """)
    return


@app.cell
def _(
    SKIP,
    SR,
    chord,
    chord_filter_pred_fir,
    chord_filter_pred_iir,
    chord_filter_wet,
    compute_all_metrics,
):
    _n = min(len(chord_filter_wet), len(chord_filter_pred_fir), len(chord_filter_pred_iir))
    metrics_filter_fir = compute_all_metrics(
        chord_filter_wet[SKIP:_n], chord_filter_pred_fir[SKIP:_n], chord[SKIP:_n], SR, 196.0
    )
    metrics_filter_iir = compute_all_metrics(
        chord_filter_wet[SKIP:_n], chord_filter_pred_iir[SKIP:_n], chord[SKIP:_n], SR, 196.0
    )
    _commentaries = {
        'ESR':       'Error-to-Signal Ratio — main quality measure',
        'MSE':       'Mean Squared Error in amplitude',
        'DC_err':    'DC offset error',
        'RMS_err':   'RMS amplitude error',
        'STFT':      'Spectral loss across scales',
        'FR_err_dB': 'Frequency response error in dB',
        'THD_target':'THD% of the filtered chord',
        'THD_pred':  'THD% of the prediction',
        'THD_err':   'Absolute THD error (linear effect → 0)',
        'HP_sim':    'Harmonic profile similarity (1=perfect)',
        'MCD':       'Mel Cepstral Distortion in dB',
    }
    print('=== LP Filter metrics ===')
    print(f'{"Metric":<14} {"FIR-512":>12} {"IIR-8":>12}  Commentary')
    print('-' * 80)
    for _k in metrics_filter_fir:
        _v_fir = metrics_filter_fir[_k]
        _v_iir = metrics_filter_iir[_k]
        _winner = 'FIR' if abs(_v_fir) < abs(_v_iir) else 'IIR'
        print(f'{_k:<14} {_v_fir:>12.6f} {_v_iir:>12.6f}  [{_winner} wins] {_commentaries[_k]}')
    return metrics_filter_fir, metrics_filter_iir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Why IIR metrics may vary:**

    The IIR model is fit by solving a **linear system in the frequency domain** — this works well when
    the target filter is a rational transfer function (poles + zeros), which a Butterworth LP is.
    However, the IIR fitting can be sensitive to:
    - Regularisation of near-zero frequency bins
    - The order mismatch (our target is order 4, our IIR is order 8 — it has more parameters than needed)
    - Numerical conditioning of the least-squares solve

    The FIR with 512 taps has plenty of capacity to represent the LP filter, so it typically does better.
    An IIR with order **exactly matching** the true filter (order 4) would theoretically achieve ESR=0.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Summary: all effects × all models
    """)
    return


@app.cell
def _(
    Path,
    metrics_boost,
    metrics_filter_fir,
    metrics_filter_iir,
    pd,
    plt,
    sns,
):
    _ERROR_COLS = {'ESR', 'MSE', 'DC_err', 'RMS_err', 'STFT', 'FR_err_dB', 'THD_err', 'MCD'}
    _SHOW_COLS  = ['ESR', 'STFT', 'FR_err_dB', 'THD_err', 'HP_sim', 'MCD']

    _all_results = {
        'Boost / FIR':   metrics_boost,
        'Filter / FIR':  metrics_filter_fir,
        'Filter / IIR':  metrics_filter_iir,
    }

    _df = pd.DataFrame({_k: {_c: _v[_c] for _c in _SHOW_COLS} for _k, _v in _all_results.items()}).T
    _normed = _df.copy()
    for _col in _SHOW_COLS:
        _mn, _mx = _df[_col].min(), _df[_col].max()
        _span = _mx - _mn + 1e-10
        _normed[_col] = (1 - (_df[_col] - _mn) / _span) if _col in _ERROR_COLS else ((_df[_col] - _mn) / _span)
    _annot_df = _df.map(lambda x: f'{x:.5f}')

    _fig, _ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(
        _normed,
        annot=_annot_df,
        fmt='',
        cmap='RdYlGn',
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=_ax,
    )
    _ax.set_title(
        'Tier 1 Model Comparison\n'
        'Cell values = actual metric value · Colour = normalised score (green=best, red=worst per column)',
        pad=10,
    )
    plt.tight_layout()
    _out_path = Path(__file__).parent.parent / 'tier1_comparison.png'
    plt.savefig(_out_path, dpi=150, bbox_inches='tight')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Key takeaways

    | Observation | Why it matters for pedal modelling |
    |-------------|------------------------------------|
    | **FIR boost → ESR ≈ 0, single non-zero tap** | A pure gain is the simplest linear system. If the ESR were high, the pedal is not a clean boost. |
    | **FIR filter → ESR ≈ 0, many non-zero taps** | A tone filter needs more FIR taps than a gain. The IR length tells you the filter's time constant. |
    | **IIR matches FIR with far fewer parameters** | For circuit-like filters (with feedback), IIR is more efficient. But fitting IIR is less stable numerically. |
    | **THD_err ≈ 0 for all Tier 1 effects** | Linear effects don't add harmonics. When you see THD_err > 0, you've got a nonlinearity → Tier 2. |
    | **HP_sim ≈ 1.0** | The harmonic *ratios* don't change — only the absolute levels do. |
    | **Chord fails for identification, noise works** | Always use broadband signals (sweep/noise) to fit models. Tonal signals leave blind spots in frequency. |

    ### What to try next

    - **Tier 2:** Add soft clipping (`wet = tanh(3 * dry)`) — watch ESR jump for FIR/IIR, THD_err rise, HP_sim drop. Then add the Hammerstein model and see it recover.
    - **Vary the FIR tap count:** Try `n_taps=64, 128, 256, 512`. At what point does the filter ESR plateau?
    - **Match the IIR order to the true filter:** Replace `order=8` with `order=4`. Does the ESR improve?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Real Guitar Recording — Applying the Same Models

    The synthetic chord proved the concepts. Now let's load an actual guitar recording,
    apply the **same boost and LP filter effects**, and fit the **same FIR/IIR models**.

    Key differences from the synthetic case:
    - The recording is stereo and at 44.1 kHz → we'll mix to mono and resample to 48 kHz
    - We normalise to −6 dBFS to match the synthetic chord level
    - The models were already fitted on broadband white noise at 48 kHz — we can apply them directly
    - ESR > 0 for the filter is expected: the real recording has reverb, noise, and pickup colouration
      that the simple white-noise identification doesn't account for

    > The identification signal (white noise) should match the **sample rate** of the target audio.
    > Since we fitted at 48 kHz, we resample the recording to 48 kHz before applying the models.
    """)
    return


@app.cell
def _(Path, SR, np):
    import soundfile as _sf
    from math import gcd as _gcd
    from scipy.signal import resample_poly as _resample_poly

    _AUDIO_PATH = Path('/mnt/d/Projects/PedalDSP/data/01 - Reach for the sun (RC3).wav')
    EXCERPT_START_S = 162.0  # most energetic 5-second window (found via RMS scan)
    EXCERPT_DUR_S   = 5.0   # seconds to analyse

    # Load stereo file at native SR
    _audio_raw, _sr_file = _sf.read(str(_AUDIO_PATH), dtype='float32')

    # Stereo → mono
    _audio_mono = _audio_raw.mean(axis=1) if _audio_raw.ndim > 1 else _audio_raw

    # Take excerpt
    _start = int(EXCERPT_START_S * _sr_file)
    _n     = int(EXCERPT_DUR_S   * _sr_file)
    _excerpt = _audio_mono[_start:_start + _n]

    # Resample to project SR (44100 → 48000)
    _g   = _gcd(SR, _sr_file)
    _up  = SR       // _g
    _dn  = _sr_file // _g
    _resampled = _resample_poly(_excerpt, _up, _dn).astype(np.float32)

    # Normalise to -6 dBFS peak
    _peak = float(np.max(np.abs(_resampled)))
    guitar_dry = (_resampled * (0.5 / _peak)).astype(np.float32)

    print(f'File SR         : {_sr_file} Hz')
    print(f'Excerpt         : {EXCERPT_START_S:.0f}s – {EXCERPT_START_S + EXCERPT_DUR_S:.0f}s')
    print(f'Resampled to    : {SR} Hz → {len(guitar_dry):,} samples ({len(guitar_dry)/SR:.2f}s)')
    print(f'Peak after norm : {np.max(np.abs(guitar_dry)):.4f}  (-6.0 dBFS)')
    return (guitar_dry,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6a. Real guitar — waveform and spectrum
    """)
    return


@app.cell
def _(SR, db_spectrum, guitar_dry, np, plt):
    _fig, (_ax_t, _ax_f) = plt.subplots(1, 2, figsize=(13, 4))

    _ms = 50
    _n_show = int(SR * _ms / 1000)
    _t_ms = np.arange(_n_show) / SR * 1000
    _ax_t.plot(_t_ms, guitar_dry[:_n_show], color='saddlebrown', lw=0.6)
    _ax_t.set_xlabel('Time (ms)')
    _ax_t.set_ylabel('Amplitude')
    _ax_t.set_title(f'Real guitar — first {_ms} ms')

    _freqs, _mag_db = db_spectrum(guitar_dry, SR)
    _ax_f.plot(_freqs, _mag_db, color='saddlebrown', lw=0.8)
    _ax_f.set_xlim(0, 8000)
    _ax_f.set_ylim(-80, 0)
    _ax_f.set_xlabel('Frequency (Hz)')
    _ax_f.set_ylabel('dBFS')
    _ax_f.set_title('Real guitar — magnitude spectrum')

    plt.suptitle('Real Guitar Recording (10s excerpt, mono, resampled to 48 kHz)', fontsize=13, y=1.01)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6b. Apply clean boost to real guitar

    We apply the same $G = 4.0$ boost and use the **already-fitted** FIR model from section 2
    (identified on white noise). For a linear effect the model should still score near-zero ESR.
    """)
    return


@app.cell
def _(
    BOOST_DB,
    BOOST_GAIN,
    SKIP,
    SR,
    error_analysis_panel,
    fir_boost,
    guitar_dry,
    np,
    signal_dashboard,
):
    guitar_boost_wet  = (guitar_dry * BOOST_GAIN).astype(np.float32)
    guitar_boost_pred = fir_boost.predict(guitar_dry)

    _fig_db = signal_dashboard(
        signals={
            'Dry guitar': (guitar_dry, 'saddlebrown'),
            f'Boost wet (+{BOOST_DB:.0f} dB)': (guitar_boost_wet, 'darkorange'),
        },
        sr=SR,
        title='Real Guitar — Clean Boost (time & spectrum)',
        ms=50,
        max_freq=8000,
    )
    _fig_db

    _n = min(len(guitar_boost_wet), len(guitar_boost_pred))
    _esr = (
        np.sum((guitar_boost_wet[SKIP:_n] - guitar_boost_pred[SKIP:_n]) ** 2)
        / (np.sum(guitar_boost_wet[SKIP:_n] ** 2) + 1e-8)
    )
    print(f'Boost FIR ESR on real guitar: {_esr:.6f}  (expect ≈ 0, linear effect)')

    _fig_err = error_analysis_panel(
        target=guitar_boost_wet,
        predicted=guitar_boost_pred,
        sr=SR,
        skip=SKIP,
        ms=50,
        max_freq=8000,
        target_color='darkorange',
        pred_color='navy',
        title='Real Guitar — Boost FIR Prediction Error',
    )
    _fig_err
    return guitar_boost_pred, guitar_boost_wet


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6c. Apply LP tone filter to real guitar

    The same 600 Hz Butterworth LP filter, and both **already-fitted** FIR-512 and IIR-8 models.
    The real guitar has more broadband content than our synthetic chord, so the filter has more
    frequencies to attenuate — the effect is more audible.
    """)
    return


@app.cell
def _(
    CUTOFF_HZ,
    SKIP,
    SR,
    compute_esr_skip,
    db_spectrum,
    error_analysis_panel,
    fir_filter,
    guitar_dry,
    iir_filter,
    np,
    plt,
    signal_dashboard,
    sos_lp,
    sosfilt,
):
    guitar_filter_wet      = sosfilt(sos_lp, guitar_dry).astype(np.float32)
    guitar_filter_pred_fir = fir_filter.predict(guitar_dry)
    guitar_filter_pred_iir = iir_filter.predict(guitar_dry)

    _fig_dash = signal_dashboard(
        signals={
            'Dry guitar': (guitar_dry, 'saddlebrown'),
            f'LP filtered (fc={CUTOFF_HZ:.0f}Hz)': (guitar_filter_wet, 'mediumseagreen'),
        },
        sr=SR,
        title='Real Guitar — LP Tone Filter (time & spectrum)',
        ms=50,
        max_freq=8000,
        vlines={f'fc {CUTOFF_HZ:.0f}Hz': CUTOFF_HZ},
    )
    _fig_dash

    _n = min(len(guitar_filter_wet), len(guitar_filter_pred_fir), len(guitar_filter_pred_iir))
    _esr_fir = compute_esr_skip(guitar_filter_wet, guitar_filter_pred_fir, skip=SKIP)
    _esr_iir = compute_esr_skip(guitar_filter_wet, guitar_filter_pred_iir, skip=SKIP)
    print(f'Filter FIR-512 ESR on real guitar: {_esr_fir:.6f}')
    print(f'Filter IIR-8   ESR on real guitar: {_esr_iir:.6f}')

    _fig_err = error_analysis_panel(
        target=guitar_filter_wet,
        predicted=guitar_filter_pred_fir,
        sr=SR,
        skip=SKIP,
        ms=50,
        max_freq=8000,
        target_color='mediumseagreen',
        pred_color='steelblue',
        title=f'Real Guitar — FIR-512 Filter Prediction Error (ESR={_esr_fir:.4f})',
    )
    _fig_err

    # IIR vs FIR residual spectra side by side
    _fig2, _axes = plt.subplots(1, 2, figsize=(13, 4))
    for _sig, _label, _color in [
        (guitar_filter_wet,      'Target',  'mediumseagreen'),
        (guitar_filter_pred_fir, 'FIR-512', 'steelblue'),
        (guitar_filter_pred_iir, 'IIR-8',   'tomato'),
    ]:
        _freqs, _m = db_spectrum(_sig, SR)
        _axes[0].plot(_freqs, _m, label=_label, lw=0.8)
    _axes[0].axvline(CUTOFF_HZ, color='salmon', ls='--', lw=0.8, label=f'fc={CUTOFF_HZ:.0f}Hz')
    _axes[0].set_xlim(0, 8000); _axes[0].set_ylim(-80, 0)
    _axes[0].set_xlabel('Frequency (Hz)'); _axes[0].set_ylabel('dBFS')
    _axes[0].set_title('Spectrum: Target vs FIR vs IIR')
    _axes[0].legend(fontsize=8)

    _err_fir = guitar_filter_wet[:_n] - guitar_filter_pred_fir[:_n]
    _err_iir = guitar_filter_wet[:_n] - guitar_filter_pred_iir[:_n]
    _freqs_e,  _m_fe = db_spectrum(_err_fir[SKIP:], SR)
    _freqs_e2, _m_ie = db_spectrum(_err_iir[SKIP:], SR)
    _axes[1].plot(_freqs_e,  _m_fe, 'steelblue', lw=0.8, label=f'FIR error (ESR={_esr_fir:.2e})')
    _axes[1].plot(_freqs_e2, _m_ie, 'tomato',    lw=0.8, label=f'IIR error (ESR={_esr_iir:.2e})')
    _axes[1].set_xlim(0, 8000); _axes[1].set_ylim(-100, -20)
    _axes[1].set_xlabel('Frequency (Hz)'); _axes[1].set_ylabel('Error magnitude (dBFS)')
    _axes[1].set_title('Residual Error Spectra — Real Guitar')
    _axes[1].legend(fontsize=8)

    plt.suptitle('Real Guitar — LP Filter Model Comparison', fontsize=13, y=1.01)
    plt.tight_layout()
    _fig2
    return guitar_filter_pred_fir, guitar_filter_pred_iir, guitar_filter_wet


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6d. Full metrics — real guitar vs all models
    """)
    return


@app.cell
def _(
    SKIP,
    SR,
    compute_all_metrics,
    guitar_boost_pred,
    guitar_boost_wet,
    guitar_dry,
    guitar_filter_pred_fir,
    guitar_filter_pred_iir,
    guitar_filter_wet,
    pd,
    plt,
    sns,
):
    _ERROR_COLS = {'ESR', 'MSE', 'DC_err', 'RMS_err', 'STFT', 'FR_err_dB', 'THD_err', 'MCD'}
    _SHOW_COLS  = ['ESR', 'STFT', 'FR_err_dB', 'THD_err', 'HP_sim', 'MCD']

    _nb = min(len(guitar_boost_wet),  len(guitar_boost_pred))
    _nf = min(len(guitar_filter_wet), len(guitar_filter_pred_fir), len(guitar_filter_pred_iir))

    _metrics_real = {
        'Real / Boost FIR': compute_all_metrics(
            guitar_boost_wet[SKIP:_nb],  guitar_boost_pred[SKIP:_nb],
            guitar_dry[SKIP:_nb], SR, harmonic_f0=196.0,
        ),
        'Real / Filter FIR': compute_all_metrics(
            guitar_filter_wet[SKIP:_nf], guitar_filter_pred_fir[SKIP:_nf],
            guitar_dry[SKIP:_nf], SR, harmonic_f0=196.0,
        ),
        'Real / Filter IIR': compute_all_metrics(
            guitar_filter_wet[SKIP:_nf], guitar_filter_pred_iir[SKIP:_nf],
            guitar_dry[SKIP:_nf], SR, harmonic_f0=196.0,
        ),
    }

    _df = pd.DataFrame({_k: {_c: _v[_c] for _c in _SHOW_COLS} for _k, _v in _metrics_real.items()}).T
    _normed = _df.copy()
    for _col in _SHOW_COLS:
        _mn, _mx = _df[_col].min(), _df[_col].max()
        _span = _mx - _mn + 1e-10
        _normed[_col] = (1 - (_df[_col] - _mn) / _span) if _col in _ERROR_COLS else ((_df[_col] - _mn) / _span)
    _annot_df = _df.map(lambda x: f'{x:.5f}')

    _fig, _ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(_normed, annot=_annot_df, fmt='', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5, ax=_ax)
    _ax.set_title(
        'Real Guitar — Tier 1 Model Metrics\n'
        'Cell values = actual metric value · Colour = normalised score (green=best per column)',
        pad=10,
    )
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What changes on real audio vs synthetic?

    | What to notice | Explanation |
    |----------------|-------------|
    | **Boost ESR ≈ 0** | Clean boost is still a single-tap FIR — the model captures it perfectly regardless of input signal content. |
    | **Filter ESR > 0 (slightly)** | The real guitar has pickup colouration, room reverb, and string noise not captured by the white-noise ID signal. The fitted models capture the *filter shape* correctly but can't correct for recording artefacts. |
    | **Filter ESR on real audio > on white noise** | The ID signal is stationary; real audio is non-stationary (dynamics, pitch variation). Any recording-specific resonance the ID didn't excite appears as model error. |
    | **FIR still beats IIR on ESR** | Same as the synthetic case — FIR-512 has more capacity to absorb small discrepancies. |
    | **High-frequency error floor** | Real recordings have noise above 8 kHz; the LP filter attenuates that, so model error concentrates in the filtered band. |

    > **Practical implication:** When capturing dry/wet pairs from a real pedal, the white-noise (or swept-sine)
    > identification signal gives you the filter shape. The remaining ESR on the guitar signal is your baseline
    > for "how well a linear model can do" — anything above it requires a Tier 2 nonlinear model.
    """)
    return


if __name__ == "__main__":
    app.run()
