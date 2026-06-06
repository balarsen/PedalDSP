# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy==2.4.6",
#   "scipy",
#   "torch",
#   "torchaudio",
#   "soundfile==0.13.1",
#   "seaborn",
#   "matplotlib",
#   "pandas",
#   "librosa==0.11.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path("/mnt/d/Projects/PedalDSP")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import soundfile as sf
    import librosa
    import librosa.display
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import marimo as mo

    return (
        DataLoader,
        PROJECT_ROOT,
        TensorDataset,
        mo,
        nn,
        np,
        pd,
        plt,
        sf,
        sns,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Big Muff Pi — End-to-End Pedal Modeling

    This notebook builds a complete modeling pipeline for a **Big Muff Pi fuzz pedal**,
    using real dry/wet captures recorded in the same session.

    We compare:
    - **Classical models**: FIR (linear baseline), Hammerstein (NL + filter), Volterra (2nd-order)
    - **Neural models**: LSTM (recurrent), TCN (dilated convolution)

    The key question: which model best captures the fuzz character?
    A fuzz pedal clips hard — FIR will fail badly. LSTM/TCN should win.

    ---
    """)
    return


@app.cell
def _(PROJECT_ROOT, np, sf):
    STEMS = PROJECT_ROOT / "stems" / "2026-06-04 pi muff"

    dry_raw, sr = sf.read(str(STEMS / "2026-06-04 Brian Larsen - dry.wav"))
    wet_raw, _  = sf.read(str(STEMS / "2026-06-04 Brian Larsen - wet.wav"))

    # Both files are stereo with identical channels — use ch0
    dry = dry_raw[:, 0].astype(np.float32)
    wet = wet_raw[:, 0].astype(np.float32)

    print(f"sr       : {sr} Hz")
    print(f"duration : {len(dry)/sr:.1f} s  ({len(dry):,} samples)")
    print(f"dry rms  : {np.sqrt(np.mean(dry**2)):.6f}  peak: {np.max(np.abs(dry)):.4f}")
    print(f"wet rms  : {np.sqrt(np.mean(wet**2)):.6f}  peak: {np.max(np.abs(wet)):.4f}")
    return STEMS, dry, sr, wet


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Level Matching

    The wet signal is ~3× quieter by RMS — the Big Muff's volume knob was conservative.
    We RMS-normalise the wet to match the dry before training and evaluation.
    This lets us measure *tonal* accuracy independently of volume offset.
    """)
    return


@app.cell
def _(dry, np, wet):
    dry_rms = float(np.sqrt(np.mean(dry**2)))
    wet_rms = float(np.sqrt(np.mean(wet**2)))
    gain    = dry_rms / wet_rms
    wet_norm = (wet * gain).astype(np.float32)

    print(f"dry rms   : {dry_rms:.6f}")
    print(f"wet rms   : {wet_rms:.6f}")
    print(f"gain      : {gain:.3f}×")
    print(f"wet_norm  : rms={np.sqrt(np.mean(wet_norm**2)):.6f}  peak={np.max(np.abs(wet_norm)):.4f}")
    return (wet_norm,)


@app.cell
def _(dry, np, sr, wet_norm):
    split = int(len(dry) * 0.8)

    dry_train = dry[:split].astype(np.float32)
    wet_train = wet_norm[:split].astype(np.float32)
    dry_eval  = dry[split:].astype(np.float32)
    wet_eval  = wet_norm[split:].astype(np.float32)

    print(f"train: {len(dry_train):,} samples  ({len(dry_train)/sr:.1f} s)")
    print(f"eval : {len(dry_eval):,} samples  ({len(dry_eval)/sr:.1f} s)")
    return dry_eval, dry_train, wet_eval, wet_train


@app.cell
def _(dry_train, mo, sr):
    _max_s = int(len(dry_train) / sr)
    train_dur_slider = mo.ui.slider(
        2, _max_s, value=_max_s, step=1,
        label=f"Training data (seconds, max={_max_s}s)",
    )
    train_dur_slider
    return (train_dur_slider,)


@app.cell
def _(dry_train, np, sr, train_dur_slider, wet_train):
    _n = int(train_dur_slider.value * sr)
    dry_train_sub = dry_train[:_n].astype(np.float32)
    wet_train_sub = wet_train[:_n].astype(np.float32)
    print(f"Using {len(dry_train_sub):,} samples  ({len(dry_train_sub)/sr:.1f} s) for training")
    return dry_train_sub, wet_train_sub


@app.cell
def _(mo):
    pregain_slider = mo.ui.slider(
        0, 24, value=0, step=1,
        label="Dry pre-gain (dB) — boosts input into the clipping zone",
    )
    pregain_slider
    return (pregain_slider,)


@app.cell
def _(dry_eval, dry_train_sub, np, pregain_slider):
    _gain = 10 ** (pregain_slider.value / 20.0)
    dry_train_g = np.clip(dry_train_sub * _gain, -1.0, 1.0).astype(np.float32)
    dry_eval_g  = np.clip(dry_eval       * _gain, -1.0, 1.0).astype(np.float32)
    print(f"Pre-gain: +{pregain_slider.value} dB  ({_gain:.2f}×)")
    print(f"dry_train_g  peak: {np.max(np.abs(dry_train_g)):.4f}")
    print(f"dry_eval_g   peak: {np.max(np.abs(dry_eval_g)):.4f}")
    return dry_eval_g, dry_train_g


@app.cell
def _(dry, np, plt, sr, wet_norm):
    n2s = 2 * sr  # 2-second window

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    t2 = np.arange(n2s) / sr
    axes[0].plot(t2, dry[:n2s],      label="dry",      alpha=0.8, linewidth=0.6)
    axes[0].plot(t2, wet_norm[:n2s], label="wet (norm)", alpha=0.8, linewidth=0.6)
    axes[0].set_title("First 2 seconds — dry vs wet (normalised)")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")

    frame = 1024
    n_frames = len(dry) // frame
    env_dry  = np.sqrt(np.mean(dry[:n_frames*frame].reshape(n_frames, frame)**2, axis=1))
    env_wet  = np.sqrt(np.mean(wet_norm[:n_frames*frame].reshape(n_frames, frame)**2, axis=1))
    t_env = np.arange(n_frames) * frame / sr

    axes[1].plot(t_env, env_dry, label="dry envelope",      alpha=0.9)
    axes[1].plot(t_env, env_wet, label="wet envelope (norm)", alpha=0.9)
    axes[1].set_title("Full-length RMS envelope (1024-sample frames)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RMS")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig
    return


@app.cell
def _(dry, plt, sr, wet_norm):
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    clip_s = 10 * sr  # first 10s for spectrograms

    for ax, sig, title in [(ax1, dry[:clip_s], "Dry"), (ax2, wet_norm[:clip_s], "Wet (norm)")]:
        ax.specgram(sig, Fs=sr, NFFT=2048, noverlap=1024, cmap="inferno", vmin=-80)
        ax.set_title(f"{title} — spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 8000)

    plt.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pedal Characterization

    Before fitting any models, we characterise the **Big Muff Pi pedal itself** using only
    the dry/wet capture pair. These plots reveal what the physical pedal actually does:

    | Plot | What to look for |
    |------|-----------------|
    | **Waveform Morphology** | Clipping shape at quiet/median/loud passages + fastest transient |
    | **Gain Reduction** | Whether the pedal compresses (Big Muff is nearly volume-neutral) |
    | **Envelope Comparison** | Attack/release changes introduced by the circuit's capacitors |
    | **Phase Portrait** | Phase-space trajectory — a tanh clipper makes a rounded rectangle |
    | **Odd/Even Harmonic Ratio** | Big Muff is symmetric-clipping → ratio > 0.8 (odd-harmonic fuzz) |
    | **Frequency Smearing** | Harmonic columns above the diagonal confirm strong nonlinearity |
    | **Coherence / Nonlinearity** | Red-shaded regions show which frequencies are nonlinearly generated |
    | **Dynamic Transfer Curve** | Does the clipping shape evolve over time? (capacitor charging) |

    Click **▶ Run Pedal Characterization** — takes ~30–60 s depending on signal length.
    The eval segment (held-out) is used so results are representative of real playing.
    """)
    return


@app.cell
def _(mo):
    pedal_char_btn = mo.ui.button(label="▶  Run Pedal Characterization", kind="success")
    pedal_char_btn
    return (pedal_char_btn,)


@app.cell
def _(dry_eval, mo, pedal_char_btn, sr, wet_eval):
    import tempfile as _tempfile
    from pathlib import Path as _PathPC

    mo.stop(
        pedal_char_btn.value == 0,
        mo.callout(
            mo.md("Click **▶ Run Pedal Characterization** above to generate plots."),
            kind="neutral",
        ),
    )

    from pedal_model.eval.diagnostics import characterize_pedal

    _tmp_pc = _PathPC(_tempfile.mkdtemp(prefix="pedaldsp_char_"))
    _n_pc   = min(len(dry_eval), len(wet_eval))

    pedal_char_figs = characterize_pedal(
        dry_eval[:_n_pc],
        wet_eval[:_n_pc],
        sr,
        output_dir=_tmp_pc,
        pedal_name="pi_muff",
        dry_name="pi_muff_dry",
        wet_name="pi_muff_wet",
    )
    return (pedal_char_figs,)


@app.cell
def _(mo, pedal_char_figs):
    _pc_cats = {
        "Waveform": ["waveform_morphology", "envelope_comparison"],
        "Dynamics": ["gain_reduction", "dynamic_transfer_curve"],
        "Phase / Portrait": ["phase_portrait", "static_transfer_shape"],
        "Harmonics": ["odd_even_harmonic_ratio", "frequency_smearing_matrix"],
        "Spectral": ["coherence_nonlinearity"],
    }

    def _pc_item(name):
        fig = pedal_char_figs.get(name)
        if fig is None:
            return mo.callout(mo.md(f"*{name}* — not generated"), kind="neutral")
        return mo.vstack([mo.md(f"**{name.replace('_', ' ').title()}**"), fig])

    _pc_tab_content = {}
    for _pc_cat, _pc_names in _pc_cats.items():
        _pc_items = [_pc_item(n) for n in _pc_names if n in pedal_char_figs or True]
        _pc_tab_content[_pc_cat] = mo.vstack(
            [_pc_item(n) for n in _pc_names], gap=1
        )

    # Add any plots not covered by the categories above
    _covered = {n for names in _pc_cats.values() for n in names}
    _extras = [n for n in pedal_char_figs if n not in _covered]
    if _extras:
        _pc_tab_content["Other"] = mo.vstack(
            [_pc_item(n) for n in _extras], gap=1
        )

    mo.ui.tabs(_pc_tab_content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classical Models

    Classical models express the pedal as explicit mathematical equations —
    no black box, you can inspect every parameter after fitting.

    ### FIR Filter

    **What it models:** any linear, time-invariant (LTI) system — boosts, EQ, buffers.

    Identify the kernel in the frequency domain:
    $H(\omega) = \text{FFT}(\text{wet}) / \text{FFT}(\text{dry})$, inverse FFT → $h[k]$, window to stabilise.

    **Why it fails on fuzz:** fuzz clips hard — doubling input does not double output.
    FIR can only do spectral shaping, not clipping. Expect ESR ≈ 1.

    ### Hammerstein Model

    **What it models:** static nonlinearity $f(\cdot)$ followed by a linear FIR filter $h[k]$.

    $$v[n] = f(x[n]) \qquad y[n] = \sum_k h[k]\,v[n-k]$$

    $f(\cdot)$ is a degree-7 polynomial fitted by least squares from dry→wet pairs.
    $h[k]$ is identified from the low-amplitude (linear) regime of the signal.
    Captures clipping shape but assumes a *static* nonlinearity — misses capacitor-memory dynamics.

    **Pre-gain note:** if the dry signal never reaches the saturation region, the polynomial fits a
    near-linear curve. Raise the **dry pre-gain** slider until you push samples into clipping.

    ### Volterra Series (2nd order)

    **What it models:** linear + all pairwise lag-product cross-terms.

    $$y[n] = \sum_k h_1[k]\,x[n-k] + \sum_k \sum_{j \geq k} h_2[k,j]\,x[n-k]\,x[n-j]$$

    Memory $M = 20$ gives 230 parameters; identified by least-squares regression.
    Better than Hammerstein because it captures *interactions between past samples*,
    but still cannot model the full infinite-memory dynamics of a capacitor.
    Fitted on every 8th sample to keep the regression matrix tractable.
    """)
    return


@app.cell
def _(PROJECT_ROOT, dry_train_g, sr, time, wet_train_sub):
    import sys as _sys
    if str(PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_ROOT))

    from pedal_model.models.classical.fir import FIRModel
    from pedal_model.models.classical.hammerstein import HammersteinModel
    from pedal_model.models.classical.volterra import VolterraModel

    fir_model = FIRModel(n_taps=1024)
    ham_model = HammersteinModel(poly_order=7, n_taps=512)
    vol_model = VolterraModel(memory=20)

    _t0 = time.time()
    fir_model.fit(dry_train_g, wet_train_sub, sr)
    print(f"FIR fit         : {time.time()-_t0:.2f}s")

    _t0 = time.time()
    ham_model.fit(dry_train_g, wet_train_sub, sr)
    print(f"Hammerstein fit : {time.time()-_t0:.2f}s")

    # Volterra lstsq is expensive on the full signal — subsample every 8th sample
    _t0 = time.time()
    vol_model.fit(dry_train_g[::8], wet_train_sub[::8], sr)
    print(f"Volterra fit    : {time.time()-_t0:.2f}s  (fit on every 8th sample)")
    return fir_model, ham_model, vol_model


@app.cell
def _(dry_eval_g, fir_model, ham_model, time, vol_model):
    _t0 = time.time()
    pred_fir = fir_model.predict(dry_eval_g)
    print(f"FIR predict         : {time.time()-_t0:.2f}s")

    _t0 = time.time()
    pred_ham = ham_model.predict(dry_eval_g)
    print(f"Hammerstein predict : {time.time()-_t0:.2f}s")

    _t0 = time.time()
    pred_vol = vol_model.predict(dry_eval_g)
    print(f"Volterra predict    : {time.time()-_t0:.2f}s")
    return pred_fir, pred_ham, pred_vol


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural Models

    Neural models learn the dry→wet mapping directly from data with no assumption about the
    circuit topology. They are trained end-to-end via backpropagation.

    ---

    ### LSTM (Long Short-Term Memory)

    **Architecture:**
    ```
    x[n]  →  LSTM(input=1, hidden=H, layers=1)  →  Linear(H, 1)  →  ŷ[n]
    ```

    **Why LSTM for fuzz:** the hidden state $(h_t, c_t)$ acts like a set of capacitors —
    it accumulates charge from previous samples and gates how much "memory" flows forward.
    This is exactly the mechanism behind the Big Muff's sustain: capacitor charge keeps the
    transistors biased into the saturation (clipping) region even when the guitar note is decaying.

    **Training procedure (TBPTT — Truncated Backprop Through Time):**
    1. Slice the audio into 2048-sample chunks
    2. Each forward pass processes a full chunk: input shape `(batch, 2048, 1)`
    3. The loss compares model output to wet chunks: `CombinedLoss(α, β)`
    4. Hidden state is *not* carried between batches during training (too complex to batch);
       it *is* carried between chunks during inference via `predict()`
    5. Gradient clipping at norm=1.0 prevents exploding gradients through long sequences

    **Key hyperparameters:**
    - **Hidden size H:** 32 is the standard starting point. Increase to 64–96 for complex effects.
    - **Loss α (time) / β (STFT):** α=0.1 biases toward spectral accuracy; α=0.5–0.9 biases
      toward waveform shape (important for hard clipping — try higher α if output sounds too clean).

    ---

    ### TCN (Temporal Convolutional Network)

    **Architecture:**
    ```
    x  →  Conv1d(1→C)  →  [DilatedBlock(dilation=2ⁱ) × N]  →  Conv1d(C→1)  →  ŷ
    ```
    Each dilated block: `DilatedConv1d → LayerNorm → GELU → residual add`

    **Receptive field** with N=10 blocks, kernel=3, doubling dilations:
    $$\text{RF} = 1 + 2 \sum_{i=0}^{9} (3-1) \cdot 2^i = 4095 \text{ samples} \approx 93\text{ ms}$$

    **Why TCN is competitive:** the large receptive field captures the decay envelope of fuzz
    transients. Training is fully parallel (no sequential dependency), making it 5–10× faster
    per epoch than LSTM. The tradeoff: it has no true "infinite memory" the way LSTM does,
    so very long sustain tails may be modeled less accurately.

    **Training procedure:** identical to LSTM except:
    - Input is reshaped to `(batch, 1, 2048)` — channels-first convention
    - No hidden state to manage; every batch is independent
    - Same `CombinedLoss(α, β)` and AdamW optimizer

    ---

    ### Loss Function

    $$\mathcal{L} = \alpha \cdot \underbrace{\text{MSE}(\hat{y}, y)}_{\text{time-domain}} + \beta \cdot \underbrace{\frac{1}{S}\sum_s \|\log|STFT_s(\hat{y})| - \log|STFT_s(y)|\|_1}_{\text{multi-scale STFT}}$$

    Window sizes $s \in \{32, 128, 512, 2048\}$ samples.

    - **Time-domain term (α):** penalises sample-by-sample amplitude errors directly.
      Higher α → model is forced to match waveform shape, including clipping transitions.
    - **STFT term (β):** penalises spectral shape errors in log-magnitude.
      Higher β → model matches tonal character (harmonic balance) even if timing shifts slightly.

    **For a hard fuzz:** try α=0.5 or higher. The STFT term can "accept" a smooth approximation
    of a square wave that has the right harmonic content but no sharp edges.
    """)
    return


@app.cell
def _(mo):
    neural_controls = mo.ui.array([
        mo.ui.slider(10, 300, value=100, step=10, label="LSTM epochs"),
        mo.ui.slider(10, 200, value=50,  step=10, label="TCN epochs"),
        mo.ui.slider(16, 128, value=32,  step=16, label="LSTM hidden size"),
        mo.ui.slider(0, 100,  value=10,  step=5,  label="Loss α% time-domain  (rest = STFT)"),
    ])
    neural_controls
    return (neural_controls,)


@app.cell
def _(neural_controls):
    lstm_epochs  = neural_controls.value[0]
    tcn_epochs   = neural_controls.value[1]
    lstm_hidden  = neural_controls.value[2]
    loss_alpha   = neural_controls.value[3] / 100.0   # 0.0–1.0
    loss_beta    = 1.0 - loss_alpha
    print(f"LSTM epochs={lstm_epochs}  hidden={lstm_hidden}")
    print(f"TCN  epochs={tcn_epochs}")
    print(f"Loss α={loss_alpha:.2f} (time)  β={loss_beta:.2f} (STFT)")
    return loss_alpha, loss_beta, lstm_epochs, lstm_hidden, tcn_epochs


@app.cell
def _(
    DataLoader,
    TensorDataset,
    dry_train_g,
    loss_alpha,
    loss_beta,
    lstm_epochs,
    lstm_hidden,
    nn,
    time,
    torch,
    wet_train_sub,
):
    from pedal_model.models.neural.lstm import LSTMModel
    from pedal_model.train.losses import CombinedLoss

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {_device}  hidden={lstm_hidden}  α={loss_alpha:.2f}")

    _CHUNK = 2048
    _BS    = 32
    _N     = (len(dry_train_g) // _CHUNK) * _CHUNK

    _dry_t = torch.tensor(dry_train_g[:_N].reshape(-1, _CHUNK, 1), dtype=torch.float32)
    _wet_t = torch.tensor(wet_train_sub[:_N].reshape(-1, _CHUNK, 1), dtype=torch.float32)

    _loader = DataLoader(TensorDataset(_dry_t, _wet_t), batch_size=_BS, shuffle=True, drop_last=True)

    lstm_model = LSTMModel(hidden_size=lstm_hidden, num_layers=1).to(_device)
    _opt  = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
    _sched = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=lstm_epochs)
    _loss_fn = CombinedLoss(alpha=loss_alpha, beta=loss_beta)

    lstm_losses = []
    _t0 = time.time()

    for _epoch in range(1, lstm_epochs + 1):
        lstm_model.train()
        _epoch_loss = 0.0
        for _xb, _yb in _loader:
            _xb, _yb = _xb.to(_device), _yb.to(_device)
            _out, _ = lstm_model(_xb)          # (B, T, 1) — discard hidden
            _l = _loss_fn(_out.squeeze(-1), _yb.squeeze(-1))  # → (B, T)
            _opt.zero_grad()
            _l.backward()
            nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
            _opt.step()
            _epoch_loss += _l.item()
        _sched.step()
        _mean = _epoch_loss / len(_loader)
        lstm_losses.append(_mean)
        if _epoch % 10 == 0 or _epoch == 1:
            print(f"LSTM  epoch {_epoch:3d}/{lstm_epochs}  loss={_mean:.5f}  lr={_sched.get_last_lr()[0]:.2e}")

    print(f"\nLSTM training done in {time.time()-_t0:.1f}s")
    lstm_model.to("cpu")
    return CombinedLoss, lstm_losses, lstm_model


@app.cell
def _(
    CombinedLoss,
    DataLoader,
    TensorDataset,
    dry_train_g,
    loss_alpha,
    loss_beta,
    nn,
    tcn_epochs,
    time,
    torch,
    wet_train_sub,
):
    from pedal_model.models.neural.tcn import TCNModel

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _CHUNK = 2048
    _BS    = 32
    _N     = (len(dry_train_g) // _CHUNK) * _CHUNK

    # TCN expects (B, 1, T)
    _dry_t = torch.tensor(dry_train_g[:_N].reshape(-1, 1, _CHUNK), dtype=torch.float32)
    _wet_t = torch.tensor(wet_train_sub[:_N].reshape(-1, 1, _CHUNK), dtype=torch.float32)

    _loader = DataLoader(TensorDataset(_dry_t, _wet_t), batch_size=_BS, shuffle=True, drop_last=True)

    tcn_model = TCNModel(channels=32, kernel_size=3, n_blocks=10).to(_device)
    _opt   = torch.optim.AdamW(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    _sched = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=tcn_epochs)
    _loss_fn = CombinedLoss(alpha=loss_alpha, beta=loss_beta)

    tcn_losses = []
    _t0 = time.time()

    for _epoch in range(1, tcn_epochs + 1):
        tcn_model.train()
        _epoch_loss = 0.0
        for _xb, _yb in _loader:
            _xb, _yb = _xb.to(_device), _yb.to(_device)
            _out = tcn_model(_xb)               # (B, 1, T)
            _l = _loss_fn(_out, _yb)
            _opt.zero_grad()
            _l.backward()
            nn.utils.clip_grad_norm_(tcn_model.parameters(), 1.0)
            _opt.step()
            _epoch_loss += _l.item()
        _sched.step()
        _mean = _epoch_loss / len(_loader)
        tcn_losses.append(_mean)
        if _epoch % 10 == 0 or _epoch == 1:
            print(f"TCN   epoch {_epoch:3d}/{tcn_epochs}  loss={_mean:.5f}  lr={_sched.get_last_lr()[0]:.2e}")

    print(f"\nTCN training done in {time.time()-_t0:.1f}s")
    tcn_model.to("cpu")
    return tcn_losses, tcn_model


@app.cell
def _(lstm_losses, np, plt, tcn_losses):
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.plot(np.arange(1, len(lstm_losses)+1), lstm_losses, label="LSTM")
    ax_loss.plot(np.arange(1, len(tcn_losses)+1),  tcn_losses,  label="TCN")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Combined loss")
    ax_loss.set_title("Training loss curves")
    ax_loss.legend()
    plt.tight_layout()
    fig_loss
    return


@app.cell
def _(dry_eval_g, lstm_model, tcn_model):
    pred_lstm = lstm_model.predict(dry_eval_g)
    pred_tcn  = tcn_model.predict(dry_eval_g)
    print(f"LSTM prediction shape : {pred_lstm.shape}")
    print(f"TCN  prediction shape : {pred_tcn.shape}")
    return pred_lstm, pred_tcn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics

    Every model is scored on the same set of metrics computed on the held-out eval segment.

    | Metric | Formula (sketch) | Units | Better |
    |--------|-----------------|-------|--------|
    | **ESR** | $\sum(y-\hat y)^2 / \sum y^2$ | — | lower → 0 |
    | **MSE** | $\frac{1}{N}\sum(y-\hat y)^2$ | amplitude² | lower |
    | **DC\_err** | $\|\bar y - \bar{\hat y}\|$ | amplitude | lower |
    | **RMS\_err** | $\sqrt{\text{MSE}}$ | amplitude | lower |
    | **STFT** | mean log-mag L1 across 4 window sizes | — | lower |
    | **FR\_err\_dB** | mean $|H_y(\omega)|$ vs $|H_{\hat y}(\omega)|$ in dB | dB | lower |
    | **THD\_err** | $|\text{THD}_\text{target} - \text{THD}_\text{pred}|$ | % | lower |
    | **HP\_sim** | cosine similarity of normalised harmonic amplitude vectors | — | higher → 1 |
    | **MCD** | Mel Cepstral Distortion (timbral difference) | dB | lower |

    ### Key metrics for fuzz

    - **ESR** is the primary headline number — normalised by signal power so it's fair regardless of level.
    - **THD\_err** and **HP\_sim** are the most musically meaningful for a fuzz pedal.
      THD of a Big Muff Pi can be 60–90%; a model outputting 10% THD will sound clean.
    - **STFT** (multi-scale spectral loss) captures whether the harmonic balance is right
      at different time scales: windows 32→2048 samples.

    ### Reading ESR

    | ESR | Meaning |
    |----|---------|
    | < 0.01 | Near-perfect — barely audible error |
    | 0.01–0.1 | Good — minor artefacts |
    | 0.1–0.5 | Moderate — clearly imperfect but recognisably the same effect |
    | > 0.5 | Poor — missing the core character |
    | ≈ 1.0 | No better than silence |

    ### Reading the heatmap

    Columns are normalised per-metric to [0, 1] (green = best of the five models, red = worst).
    Raw values are printed inside each cell. Green in FIR's row just means FIR is the best
    *among linear models* — not that its absolute score is good.
    """)
    return


@app.cell
def _(
    dry_eval,
    pd,
    pred_fir,
    pred_ham,
    pred_lstm,
    pred_tcn,
    pred_vol,
    sr,
    wet_eval,
):
    from pedal_model.metrics.suite import compute_all_metrics

    _models = {
        "FIR":         pred_fir,
        "Hammerstein": pred_ham,
        "Volterra":    pred_vol,
        "LSTM":        pred_lstm,
        "TCN":         pred_tcn,
    }

    _rows = {}
    for _name, _pred in _models.items():
        # Trim to same length (Volterra prepends zeros, TCN may differ)
        _n = min(len(wet_eval), len(_pred))
        _tgt = wet_eval[:_n]
        _inp = dry_eval[:_n]
        _p   = _pred[:_n]
        try:
            _m = compute_all_metrics(_tgt, _p, _inp, sr)
        except Exception as e:
            print(f"Metrics failed for {_name}: {e}")
            _m = {k: float("nan") for k in ["ESR","MSE","DC_err","RMS_err","STFT","FR_err_dB","THD_err","HP_sim","MCD"]}
        _rows[_name] = _m

    metrics_df = pd.DataFrame(_rows).T
    display_cols = ["ESR", "MSE", "STFT", "FR_err_dB", "THD_err", "HP_sim", "MCD"]
    metrics_df[display_cols]
    return display_cols, metrics_df


@app.cell
def _(display_cols, metrics_df, pd, plt, sns):
    _df = metrics_df[display_cols].copy()

    # Normalize: error cols → invert (lower=better → 1=best), HP_sim → direct
    _error_cols = ["ESR", "MSE", "STFT", "FR_err_dB", "THD_err", "MCD"]
    _sim_cols   = ["HP_sim"]

    _normed = pd.DataFrame(index=_df.index, columns=_df.columns, dtype=float)
    for _col in _error_cols:
        _mn, _mx = _df[_col].min(), _df[_col].max()
        _normed[_col] = 1 - (_df[_col] - _mn) / (_mx - _mn + 1e-10)
    for _col in _sim_cols:
        _mn, _mx = _df[_col].min(), _df[_col].max()
        _normed[_col] = (_df[_col] - _mn) / (_mx - _mn + 1e-10)

    fig_heat, ax_heat = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        _normed.astype(float),
        annot=_df.round(4),
        fmt="",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax_heat,
    )
    ax_heat.set_title("Model Comparison — Big Muff Pi  (green = better)", fontsize=14, pad=14)
    ax_heat.set_xlabel("Metric")
    ax_heat.set_ylabel("Model")
    plt.tight_layout()
    fig_heat
    return


@app.cell
def _(
    dry_eval,
    metrics_df,
    np,
    plt,
    pred_fir,
    pred_ham,
    pred_lstm,
    pred_tcn,
    pred_vol,
    sr,
    wet_eval,
):
    _preds = {
        "FIR":         pred_fir,
        "Hammerstein": pred_ham,
        "Volterra":    pred_vol,
        "LSTM":        pred_lstm,
        "TCN":         pred_tcn,
    }

    _esr = metrics_df["ESR"].dropna()
    _best_name  = _esr.idxmin()
    _worst_name = _esr.idxmax()

    _clip = int(0.5 * sr)  # 0.5 s excerpt

    fig_wave, axes_w = plt.subplots(2, 1, figsize=(13, 7))

    for _ax, _name in zip(axes_w, [_best_name, _worst_name]):
        _n = min(len(wet_eval), len(_preds[_name]))
        _t = np.arange(_clip) / sr
        _ax.plot(_t, dry_eval[:_clip],      label="dry",    alpha=0.5, linewidth=0.7)
        _ax.plot(_t, wet_eval[:_clip],      label="target", linewidth=1.0, color="C1")
        _ax.plot(_t, _preds[_name][:_clip], label=_name,    linewidth=1.0, color="C2", linestyle="--")
        _ax.set_title(f"{_name}  (ESR={metrics_df.loc[_name,'ESR']:.4f})")
        _ax.set_xlabel("Time (s)")
        _ax.legend(loc="upper right")

    plt.suptitle("Best vs Worst model — 0.5 s excerpt", fontsize=13)
    plt.tight_layout()
    fig_wave
    return


@app.cell
def _(
    np,
    plt,
    pred_fir,
    pred_ham,
    pred_lstm,
    pred_tcn,
    pred_vol,
    sr,
    wet_eval,
):
    _preds = {
        "Target (wet)": wet_eval,
        "FIR":          pred_fir,
        "Hammerstein":  pred_ham,
        "Volterra":     pred_vol,
        "LSTM":         pred_lstm,
        "TCN":          pred_tcn,
    }

    _N    = min(len(v) for v in _preds.values())
    _freqs = np.fft.rfftfreq(_N, 1.0/sr)
    _mask  = (_freqs >= 20) & (_freqs <= 20000)

    fig_fr, ax_fr = plt.subplots(figsize=(13, 6))

    for _name, _sig in _preds.items():
        _mag = np.abs(np.fft.rfft(_sig[:_N]))
        _db  = 20 * np.log10(_mag[_mask] + 1e-8)
        ax_fr.plot(_freqs[_mask], _db, label=_name, alpha=0.8, linewidth=0.9)

    ax_fr.set_xscale("log")
    ax_fr.set_xlabel("Frequency (Hz)")
    ax_fr.set_ylabel("Magnitude (dB)")
    ax_fr.set_title("Frequency response — target vs all models")
    ax_fr.legend(fontsize=9)
    plt.tight_layout()
    fig_fr
    return


@app.cell
def _(fir_model, ham_model, lstm_model, np, plt, sr, tcn_model, vol_model):
    # Synthesise 440 Hz sine at low amplitude (3 seconds)
    _f0   = 440.0
    _dur  = 3.0
    _t    = np.arange(int(_dur * sr), dtype=np.float32) / sr
    _sine = (0.5 * np.sin(2 * np.pi * _f0 * _t)).astype(np.float32)

    _models_h = {
        "FIR":         fir_model.predict(_sine),
        "Hammerstein": ham_model.predict(_sine),
        "Volterra":    vol_model.predict(_sine),
        "LSTM":        lstm_model.predict(_sine),
        "TCN":         tcn_model.predict(_sine),
    }

    # Compute harmonic amplitudes 1–10
    _harmonics = np.arange(1, 11)
    _results_h = {}
    for _name, _out in _models_h.items():
        _n = len(_out)
        _spec = np.abs(np.fft.rfft(_out)) / (_n / 2)
        _freqs = np.fft.rfftfreq(_n, 1.0/sr)
        _amps = []
        for _h in _harmonics:
            _target_f = _f0 * _h
            _idx = np.argmin(np.abs(_freqs - _target_f))
            _amps.append(float(_spec[_idx]))
        _a1 = _amps[0] if _amps[0] > 1e-10 else 1e-10
        _results_h[_name] = np.array(_amps) / _a1

    _n_models = len(_results_h)
    fig_harm, axes_h = plt.subplots(1, _n_models, figsize=(14, 5), sharey=True)

    for _ax, (_name, _profile) in zip(axes_h, _results_h.items()):
        _ax.bar(_harmonics, _profile, color="steelblue", edgecolor="navy", alpha=0.8)
        _ax.set_title(_name, fontsize=11)
        _ax.set_xlabel("Harmonic")
        _ax.set_yscale("log")
        _ax.set_xticks(_harmonics)

    axes_h[0].set_ylabel("Relative amplitude (log)")
    plt.suptitle("Harmonic profile at 440 Hz  (normalised to fundamental)", fontsize=13)
    plt.tight_layout()
    fig_harm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Audio Comparison

    Listen to each model against the target. The eval segment is ~11 s of real guitar playing
    the models have never seen. This is the quickest way to hear where each model breaks down.
    """)
    return


@app.cell
def _(
    mo,
    np,
    pred_fir,
    pred_ham,
    pred_lstm,
    pred_tcn,
    pred_vol,
    sf,
    sr,
    wet_eval,
):
    import io as _io

    def _wav_bytes(arr: np.ndarray) -> bytes:
        buf = _io.BytesIO()
        sf.write(buf, np.clip(arr, -1.0, 1.0).astype(np.float32), sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    _n = min(len(wet_eval), len(pred_fir), len(pred_ham), len(pred_vol), len(pred_lstm), len(pred_tcn))

    _rows = [
        ("Target (wet)",  wet_eval[:_n]),
        ("FIR",           pred_fir[:_n]),
        ("Hammerstein",   pred_ham[:_n]),
        ("Volterra",      pred_vol[:_n]),
        ("LSTM",          pred_lstm[:_n]),
        ("TCN",           pred_tcn[:_n]),
    ]

    mo.vstack(
        [
            mo.hstack(
                [mo.md(f"**{label}**&nbsp;&nbsp;"), mo.audio(src=_wav_bytes(sig))],
                justify="start",
                gap=1,
            )
            for label, sig in _rows
        ],
        gap=0.5,
    )
    return


@app.cell
def _(
    STEMS,
    dry_eval,
    np,
    pred_fir,
    pred_ham,
    pred_lstm,
    pred_tcn,
    pred_vol,
    sf,
    sr,
    wet_eval,
):
    _out_dir = STEMS / "predictions"
    _out_dir.mkdir(exist_ok=True)

    _preds = {
        "fir":         pred_fir,
        "hammerstein": pred_ham,
        "volterra":    pred_vol,
        "lstm":        pred_lstm,
        "tcn":         pred_tcn,
    }

    _n = len(dry_eval)
    sf.write(str(_out_dir / "00_dry_eval.wav"),    dry_eval,  sr, subtype="PCM_24")
    sf.write(str(_out_dir / "00_wet_eval.wav"),    wet_eval,  sr, subtype="PCM_24")

    for _name, _pred in _preds.items():
        _p = np.clip(_pred[:_n], -1.0, 1.0).astype(np.float32)
        _path = _out_dir / f"pred_{_name}.wav"
        sf.write(str(_path), _p, sr, subtype="PCM_24")
        print(f"Saved: {_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal Visualization (librosa)

    Full-signal waveform and mel spectrogram of **target** and **best model**,
    using `librosa.display.waveshow` and `librosa.display.specshow`.
    The mosaic layout shares the time axis between the waveform and spectrogram.
    """)
    return


@app.cell
def _(
    metrics_df,
    pred_fir,
    pred_ham,
    pred_lstm,
    pred_tcn,
    pred_vol,
    sr,
    wet_eval,
):
    from pedal_model.utils.plotting import waveform_spectrogram_panel

    _preds_panel = {
        "FIR":         pred_fir,
        "Hammerstein": pred_ham,
        "Volterra":    pred_vol,
        "LSTM":        pred_lstm,
        "TCN":         pred_tcn,
    }
    _esr_p = metrics_df["ESR"].dropna()
    _best_p = _esr_p.idxmin()
    _best_pred_panel = _preds_panel[_best_p]
    _n_p = min(len(wet_eval), len(_best_pred_panel))

    fig_mosaic_target = waveform_spectrogram_panel(
        wet_eval[:_n_p], sr,
        title=f"Target (wet) — full signal",
        waveform_color="#00FF88",
        cmap="magma",
    )
    fig_mosaic_best = waveform_spectrogram_panel(
        _best_pred_panel[:_n_p], sr,
        title=f"{_best_p} prediction — full signal",
        waveform_color="#FF6B6B",
        cmap="magma",
    )
    fig_mosaic_target, fig_mosaic_best
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deep Diagnostics

    Detailed analysis of how well each model captures the Big Muff Pi's nonlinear
    character. Select a model and click **Run Diagnostics** — this runs all 15
    diagnostic plots (static transfer curve, describing function, harmonics,
    spectrogram overlay, etc.) and displays them below organized by category.

    > These plots take ~20–30 s to compute. The eval segment is used as input.
    """)
    return


@app.cell
def _(metrics_df, mo):
    _model_choices = list(metrics_df.index.dropna())
    _esr_d = metrics_df["ESR"].dropna()
    _default_model = _esr_d.idxmin() if len(_esr_d) > 0 else _model_choices[0]

    diag_model_sel = mo.ui.dropdown(
        options=_model_choices,
        value=_default_model,
        label="Model to diagnose",
    )
    diag_run_btn = mo.ui.button(label="▶  Run Diagnostics", kind="success")
    mo.hstack([diag_model_sel, diag_run_btn], gap=2)
    return diag_model_sel, diag_run_btn


@app.cell
def _(
    diag_model_sel,
    diag_run_btn,
    dry_eval,
    mo,
    pred_fir,
    pred_ham,
    pred_lstm,
    pred_tcn,
    pred_vol,
    sr,
    wet_eval,
):
    import tempfile
    from pathlib import Path as _Path

    mo.stop(
        diag_run_btn.value == 0,
        mo.callout(mo.md("Select a model above and click **Run Diagnostics**."),
                   kind="neutral"),
    )

    from pedal_model.eval.diagnostics import run_all_diagnostics

    _preds_d = {
        "FIR":         pred_fir,
        "Hammerstein": pred_ham,
        "Volterra":    pred_vol,
        "LSTM":        pred_lstm,
        "TCN":         pred_tcn,
    }
    _sel  = diag_model_sel.value
    _pred = _preds_d[_sel]
    _n    = min(len(dry_eval), len(wet_eval), len(_pred))

    _tmp_dir = _Path(tempfile.mkdtemp(prefix="pedaldsp_diag_"))

    _diag_figs = run_all_diagnostics(
        dry_eval[:_n],
        wet_eval[:_n],
        _pred[:_n],
        sr,
        output_dir=_tmp_dir,
        model_name=_sel.lower().replace(" ", "_"),
        dry_name="pi_muff_dry",
        wet_name="pi_muff_wet",
    )
    diag_figs = _diag_figs
    return (diag_figs,)


@app.cell
def _(diag_figs, mo):
    _cat = {
        "Static Nonlinearity": ["static_transfer_curve", "describing_function"],
        "Memory / Dynamics":   ["lag_error", "impulse_response", "step_response"],
        "Frequency Domain":    ["transfer_function", "group_delay", "coherence",
                                "spectrogram_overlay"],
        "Nonlinear":           ["harmonic_profile", "imd", "level_dependent_fr",
                                "nonlinear_frequency_map"],
        "Perceptual":          ["error_spectrogram", "aweighted_error"],
    }

    def _fig_cell(name):
        fig = diag_figs.get(name)
        if fig is None:
            return mo.callout(mo.md(f"**{name}** — not generated"), kind="warn")
        return mo.md(f"### {name.replace('_', ' ').title()}") if False else fig

    _tab_content = {}
    for _cat_name, _names in _cat.items():
        _items = []
        for _n in _names:
            _fig = diag_figs.get(_n)
            if _fig is not None:
                _items.append(mo.vstack([mo.md(f"**{_n.replace('_',' ').title()}**"), _fig]))
            else:
                _items.append(mo.callout(mo.md(f"*{_n}* — not available"), kind="neutral"))
        _tab_content[_cat_name] = mo.vstack(_items, gap=1)

    mo.ui.tabs(_tab_content)
    return


@app.cell(hide_code=True)
def _(metrics_df, mo):
    _esr = metrics_df["ESR"].dropna()
    _best  = _esr.idxmin()
    _worst = _esr.idxmax()

    mo.md(f"""
    ## Summary

    | Model | Strength | Expected on fuzz |
    |-------|----------|-----------------|
    | **FIR** | Linear frequency shaping | ❌ No — fuzz is nonlinear |
    | **Hammerstein** | Static clipping + filter | ⚠️  Partial — misses dynamics |
    | **Volterra** | 2nd-order NL interactions | ⚠️  Better — still misses memory effects |
    | **LSTM** | Recurrent state, models capacitor memory | ✅ Best for fuzz character |
    | **TCN** | Large receptive field, fast training | ✅ Strong competitor |

    **By ESR on the eval set:**
    - Best model: **{_best}** (ESR = {_esr[_best]:.4f})
    - Worst model: **{_worst}** (ESR = {_esr[_worst]:.4f})

    **Key takeaway:** FIR/IIR are a baseline. A Big Muff Pi is a hard clipper with
    resonant tone control — the nonlinearity and circuit memory make LSTM/TCN clearly
    superior. The harmonic profile plot shows how many odd harmonics the fuzz generates
    (3rd, 5th, 7th, …) — a good model should reproduce this profile.

    Predictions saved to `data/stems/2026-06-04_pi_muff/predictions/` for listening.
    """)
    return


if __name__ == "__main__":
    app.run()
