# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy==2.4.6",
#   "torch==2.12.0",
#   "soundfile==0.13.1",
#   "matplotlib",
#   "scipy",
#   "pandas",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tier 2 Pedal Modelling — LSTM: Train on First Half, Demo on Second

    **Pipeline in this notebook:**

    1. Load a real audio file → apply a synthetic Tube Screamer to build a (dry, wet) pair
    2. **First 50 %** of the file → train an LSTM
    3. **Second 50 %** → held-out demo: waveforms, spectrograms, metrics, audio playback

    ---

    ### Why LSTM for a guitar pedal?

    A Tube Screamer has **circuit memory** — capacitors charge and discharge with the signal.
    An MLP with a fixed context window can't capture this properly; an LSTM's hidden state can:

    ```
    h[n], c[n] = LSTM_cell( x[n],  h[n-1], c[n-1] )
    ŷ[n]       = Linear( h[n] )
    ```

    The hidden state `h[n]` carries the effect of every past sample forward indefinitely,
    making it ideal for circuits whose character depends on slow-decaying capacitor charge.

    ---

    ### Synthetic Tube Screamer — the topology

    ```
    v[n] = H_pre(z) · x[n]        2nd-order Butterworth bandpass 300 Hz – 5 kHz
    w[n] = tanh( drive · v[n] )    diode-pair soft clipper (symmetric → odd harmonics dominate)
    y[n] = H_post(z) · w[n]       2nd-order Butterworth lowpass 4 kHz
    ```

    The TS circuit is literally this topology in silicon: input filter → diode pair → output filter.
    Increasing `drive` pushes deeper into the tanh knee → more harmonic saturation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 0. Setup
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    def _find_root() -> Path:
        for candidate in [
            Path(__file__).resolve().parent.parent,
            Path.cwd(),
            Path.cwd().parent,
        ]:
            if (candidate / "pedal_model").is_dir():
                return candidate
        raise RuntimeError(
            "pedal_model/ not found. "
            "Run with: .venv/bin/marimo edit notebooks/learning_tier2_lstm_training.py"
        )

    ROOT = _find_root()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (ROOT,)


@app.cell
def _():
    import io
    import time

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import soundfile as sf
    import scipy.signal as ss
    import torch

    from pedal_model.models.neural.lstm import LSTMModel
    from pedal_model.train.losses import CombinedLoss
    from pedal_model.metrics.suite import compute_all_metrics

    plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    return (
        CombinedLoss,
        DEVICE,
        LSTMModel,
        compute_all_metrics,
        io,
        np,
        pd,
        plt,
        sf,
        ss,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Configuration
    """)
    return


@app.cell
def _(mo):
    wav_path_ui = mo.ui.text(
        value="data/01 - Reach for the sun (RC3).wav",
        label="WAV file (relative to project root)",
        full_width=True,
    )
    drive_ui = mo.ui.slider(
        1.0, 20.0, value=8.0, step=0.5,
        label="Drive  (clipping intensity — higher = more harmonic saturation)",
        show_value=True,
    )
    train_secs_ui = mo.ui.slider(
        5, 60, value=20, step=5,
        label="Training seconds (taken from the first half)",
        show_value=True,
    )
    demo_secs_ui = mo.ui.slider(
        5, 30, value=10, step=5,
        label="Demo seconds (taken from the second half)",
        show_value=True,
    )
    hidden_ui = mo.ui.slider(
        8, 64, value=32, step=8,
        label="LSTM hidden size",
        show_value=True,
    )
    epochs_ui = mo.ui.slider(
        5, 100, value=20, step=5,
        label="Training epochs",
        show_value=True,
    )
    lr_ui = mo.ui.dropdown(
        ["1e-4", "3e-4", "1e-3", "3e-3"],
        value="1e-3",
        label="Learning rate",
    )
    chunk_ui = mo.ui.dropdown(
        ["1024", "2048", "4096", "8192"],
        value="4096",
        label="TBPTT chunk size (samples)",
    )
    train_btn = mo.ui.run_button(label="▶  Train LSTM")

    mo.vstack([
        mo.md("### Audio file"),
        wav_path_ui,
        mo.md("### Effect"),
        drive_ui,
        mo.md("### Data splits"),
        mo.hstack([train_secs_ui, demo_secs_ui]),
        mo.md("### Model & optimiser"),
        mo.hstack([hidden_ui, epochs_ui]),
        mo.hstack([lr_ui, chunk_ui]),
        mo.md("---"),
        train_btn,
    ])
    return (
        chunk_ui,
        demo_secs_ui,
        drive_ui,
        epochs_ui,
        hidden_ui,
        lr_ui,
        train_btn,
        train_secs_ui,
        wav_path_ui,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Data — load, synthesise effect, split
    """)
    return


@app.cell
def _(ROOT, sf, wav_path_ui):
    _path = ROOT / wav_path_ui.value
    _audio, sr = sf.read(str(_path), dtype="float32", always_2d=True)
    dry_full = _audio[:, 0]     # left channel → dry signal
    n_mid = len(dry_full) // 2
    _dur = len(dry_full) / sr
    print(f"File     : {_path.name}")
    print(f"Duration : {_dur:.1f}s  ({len(dry_full):,} samples @ {sr} Hz)")
    print(f"Mid-split: sample {n_mid:,}  ({_dur / 2:.1f}s)")
    return dry_full, n_mid, sr


@app.cell
def _(np, ss):
    def simulate_tube_screamer(
        dry: np.ndarray, sr: int, drive: float = 8.0
    ) -> np.ndarray:
        """Tube Screamer approximation: bandpass → tanh soft-clip → lowpass.

        Mirrors the actual TS circuit topology: input tone-shaping filter,
        diode clipping (approximated as tanh), output tone-shaping filter.

        Args:
            dry: Input signal, shape (N,), float32.
            sr: Sample rate in Hz.
            drive: Pre-clip gain. Higher = deeper saturation, more odd harmonics.

        Returns:
            Wet signal, shape (N,), float32, peak-normalised to 0.9.
        """
        sos_pre = ss.butter(2, [300, 5000], btype="bandpass", fs=sr, output="sos")
        v = ss.sosfilt(sos_pre, dry)
        w = np.tanh(drive * v)
        sos_post = ss.butter(2, 4000, btype="lowpass", fs=sr, output="sos")
        out = ss.sosfilt(sos_post, w)
        peak = np.max(np.abs(out))
        return (out * 0.9 / peak if peak > 1e-8 else out).astype(np.float32)

    return (simulate_tube_screamer,)


@app.cell
def _(drive_ui, dry_full, simulate_tube_screamer, sr):
    wet_full = simulate_tube_screamer(dry_full, sr, drive=float(drive_ui.value))
    print(f"Drive={drive_ui.value}  |  Dry RMS={dry_full.std():.4f}  |  Wet RMS={wet_full.std():.4f}")
    return (wet_full,)


@app.cell
def _(demo_secs_ui, dry_full, n_mid, sr, train_secs_ui, wet_full):
    _n_train = min(int(train_secs_ui.value * sr), n_mid)
    _n_demo  = min(int(demo_secs_ui.value  * sr), len(dry_full) - n_mid)

    dry_train = dry_full[:_n_train]
    wet_train = wet_full[:_n_train]
    dry_demo  = dry_full[n_mid : n_mid + _n_demo]
    wet_demo  = wet_full[n_mid : n_mid + _n_demo]

    print(f"Train slice: {len(dry_train):,} samples  ({len(dry_train)/sr:.1f}s)")
    print(f"Demo  slice: {len(dry_demo):,}  samples  ({len(dry_demo)/sr:.1f}s)")
    return dry_demo, dry_train, wet_demo, wet_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Dry vs wet — 30 ms snapshot of training data
    """)
    return


@app.cell
def _(dry_train, np, plt, sr, wet_train):
    _ms = 30
    _n  = int(sr * _ms / 1000)
    _t  = np.arange(_n) / sr * 1000

    _fig, _ax = plt.subplots(figsize=(11, 3))
    _ax.plot(_t, dry_train[:_n], color="steelblue", lw=0.9, label="Dry (input)")
    _ax.plot(_t, wet_train[:_n], color="darkorange", lw=0.9, alpha=0.85, label="Wet (TS sim)")
    _ax.set_xlabel("Time (ms)")
    _ax.set_ylabel("Amplitude")
    _ax.set_title(f"Dry vs Wet — first {_ms} ms of training data")
    _ax.legend(fontsize=9)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Train LSTM

    **Loss function**: `L = α · L2_time + β · STFT_multiscale`  (α = 0.1, β = 0.9)

    The STFT term dominates — it forces tonal accuracy at multiple time-scales
    (window sizes: 32, 128, 512, 2048 samples). The small L2 term anchors absolute amplitude.

    **TBPTT (Truncated Backpropagation Through Time):** gradients are computed over
    fixed-length chunks (default 4096 samples). Hidden state is carried across chunks but
    *detached* from the computation graph between them, preventing gradient explosion over
    very long sequences while still propagating circuit memory.

    > **Tip:** 20 epochs × 20 seconds of audio trains in ~30 s on CPU, ~5 s on GPU.
    > Increase epochs for better accuracy; watch the loss curve for convergence.
    """)
    return


@app.cell
def _(
    CombinedLoss,
    DEVICE,
    LSTMModel,
    chunk_ui,
    dry_train,
    epochs_ui,
    hidden_ui,
    lr_ui,
    mo,
    time,
    torch,
    train_btn,
    wet_train,
):
    mo.stop(
        not train_btn.value,
        mo.callout(mo.md("Press **▶  Train LSTM** above to begin."), kind="info"),
    )

    _hidden = int(hidden_ui.value)
    _epochs = int(epochs_ui.value)
    _lr     = float(lr_ui.value)
    _chunk  = int(chunk_ui.value)

    model       = LSTMModel(hidden_size=_hidden, num_layers=1).to(DEVICE)
    _opt        = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=1e-4)
    _loss_fn    = CombinedLoss(alpha=0.1, beta=0.9)
    _n_chunks   = len(dry_train) // _chunk
    train_losses = []
    _t0          = time.time()

    for _ep in range(_epochs):
        model.train()
        _ep_loss = 0.0
        _h = None

        for _i in range(_n_chunks):
            _cx = dry_train[_i * _chunk : (_i + 1) * _chunk]
            _cy = wet_train[_i * _chunk : (_i + 1) * _chunk]
            _x  = torch.tensor(_cx).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            _y  = torch.tensor(_cy).unsqueeze(0).unsqueeze(-1).to(DEVICE)

            _pred, _h = model(_x, _h)
            _h = (_h[0].detach(), _h[1].detach())

            _loss = _loss_fn(_pred.squeeze(-1), _y.squeeze(-1))
            _opt.zero_grad()
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            _opt.step()
            _ep_loss += _loss.item()

        train_losses.append(_ep_loss / max(_n_chunks, 1))
        if (_ep + 1) % max(1, _epochs // 5) == 0:
            print(f"Epoch {_ep + 1:3d}/{_epochs}  loss = {train_losses[-1]:.5f}")

    model.eval()
    print(f"\nTraining complete in {time.time() - _t0:.1f}s")
    return model, train_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Training loss curve
    """)
    return


@app.cell
def _(np, plt, train_losses):
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.plot(np.arange(1, len(train_losses) + 1), train_losses,
             color="steelblue", lw=1.5, marker="o", markersize=3)
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Combined loss")
    _ax.set_title("Training loss  (0.1 · L2 + 0.9 · STFT)")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Demo — second half (held-out, never seen during training)

    The trained LSTM is run on the second half of the file.
    The **first 4096 samples** are discarded as warmup — the LSTM hidden state starts at zero
    so it takes a few thousand samples to "settle" to the signal's character.
    """)
    return


@app.cell
def _(dry_demo, model, time):
    model.to("cpu")     # predict() creates CPU tensors; move model to match
    _t0 = time.time()
    pred_demo = model.predict(dry_demo)
    WARMUP = 4096
    print(f"Inference: {time.time() - _t0:.2f}s  |  {len(dry_demo):,} samples")
    return WARMUP, pred_demo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Waveform comparison — first 40 ms of demo segment
    """)
    return


@app.cell
def _(np, plt, pred_demo, sr, wet_demo):
    _n_show = int(sr * 0.040)
    _t_ms   = np.arange(_n_show) / sr * 1000
    _err    = wet_demo[:_n_show] - pred_demo[:_n_show]

    _fig, _axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    _axes[0].plot(_t_ms, wet_demo[:_n_show],  color="darkorange", lw=0.9)
    _axes[0].set_title("Target (wet)")
    _axes[0].set_ylabel("Amplitude")

    _axes[1].plot(_t_ms, pred_demo[:_n_show], color="navy", lw=0.9)
    _axes[1].set_title("LSTM Prediction")
    _axes[1].set_ylabel("Amplitude")

    _rms_err = float(np.sqrt(np.mean(_err ** 2)))
    _axes[2].plot(_t_ms, _err, color="tomato", lw=0.7)
    _axes[2].axhline(0, color="black", lw=0.5)
    _axes[2].set_title(f"Residual  (RMS = {_rms_err:.4f})")
    _axes[2].set_xlabel("Time (ms)")
    _axes[2].set_ylabel("Error")

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Spectrogram comparison — 5-second window (post-warmup)
    """)
    return


@app.cell
def _(WARMUP, plt, pred_demo, sr, wet_demo):
    _n_seg = min(sr * 5, len(wet_demo) - WARMUP, len(pred_demo) - WARMUP)
    _tgt   = wet_demo[WARMUP  : WARMUP + _n_seg]
    _prd   = pred_demo[WARMUP : WARMUP + _n_seg]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for _ax, _sig, _title, _cmap in [
        (_ax1, _tgt, "Target spectrogram",     "magma"),
        (_ax2, _prd, "Prediction spectrogram", "viridis"),
    ]:
        _ax.specgram(_sig, NFFT=512, Fs=sr, noverlap=448, cmap=_cmap)
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("Frequency (Hz)")
        _ax.set_ylim(0, 8000)
        _ax.set_title(_title)

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Metrics — evaluated on held-out demo segment
    """)
    return


@app.cell
def _(WARMUP, compute_all_metrics, dry_demo, mo, pd, pred_demo, sr, wet_demo):
    _n   = min(len(wet_demo), len(pred_demo))
    _tgt = wet_demo[WARMUP : _n]
    _prd = pred_demo[WARMUP : _n]
    _dry = dry_demo[WARMUP  : _n]

    _m = compute_all_metrics(_tgt, _prd, _dry, sr)

    _kind = "success" if (_m["ESR"] < 0.05 and _m["STFT"] < 0.10) else "warn"
    mo.vstack([
        mo.callout(
            mo.md(f"""
    **Key metrics** — {(_n - WARMUP) / sr:.1f}s of unseen audio (post-{WARMUP}-sample warmup)

    | Metric | Value | Guideline |
    |--------|-------|-----------|
    | **ESR** | `{_m['ESR']:.4f}` | < 0.01 excellent · < 0.05 good · > 0.1 poor |
    | **STFT loss** | `{_m['STFT']:.4f}` | < 0.05 excellent · < 0.10 good |
    | **MCD** | `{_m['MCD']:.2f} dB` | < 2 dB good |
    | **RMS error** | `{_m['RMS_err']:.4f}` | lower is better |
    """),
            kind=_kind,
        ),
        mo.ui.table(pd.DataFrame([_m]).round(5)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Audio playback — up to 10 s of demo segment (post-warmup)
    """)
    return


@app.cell
def _(WARMUP, dry_demo, io, mo, np, pred_demo, sf, sr, wet_demo):
    def _wav_bytes(sig: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, sig.astype(np.float32), sr, format="WAV")
        buf.seek(0)
        return buf.read()

    _n    = min(len(wet_demo), len(pred_demo))
    _clip = min(sr * 10, _n - WARMUP)
    _s, _e = WARMUP, WARMUP + _clip

    mo.vstack([
        mo.md("**Dry** — original recording, no effect applied:"),
        mo.audio(src=_wav_bytes(dry_demo[_s:_e])),
        mo.md("**Wet (target)** — synthetic Tube Screamer applied to the dry signal:"),
        mo.audio(src=_wav_bytes(wet_demo[_s:_e])),
        mo.md("**LSTM prediction** — what the model learned from the *first* half:"),
        mo.audio(src=_wav_bytes(pred_demo[_s:_e])),
    ])
    return


if __name__ == "__main__":
    app.run()
