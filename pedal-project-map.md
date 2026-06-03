# Pedal Modeling Project Map
## Python · PyTorch · CUDA · WSL2 · VS Code

---

## Project Directory Structure

```
pedal_model/
│
├── capture/
│   ├── generate_sweep.py          # Generate test signals (sweeps, noise, chirps)
│   ├── align.py                   # Load WAV, detect click, align channels
│   └── verify.py                  # Plot and sanity-check capture quality
│
├── data/
│   ├── dataset.py                 # PyTorch Dataset: slice aligned WAV into windows
│   ├── augment.py                 # Optional: gain jitter, noise augmentation
│   └── raw/                       # Your captured WAV files go here
│
├── models/
│   ├── base.py                    # Abstract base class all models inherit
│   │
│   ├── classical/
│   │   ├── fir.py                 # FIR filter fit from frequency response
│   │   ├── iir.py                 # IIR filter (Butterworth/Chebyshev fit)
│   │   ├── hammerstein.py         # Static NL + linear filter
│   │   ├── wiener_hammerstein.py  # Filter + NL + filter
│   │   └── volterra.py            # 2nd/3rd order Volterra series
│   │
│   └── neural/
│       ├── mlp.py                 # Baseline MLP with context window
│       ├── lstm.py                # Single/stacked LSTM
│       ├── gru.py                 # GRU variant (faster than LSTM)
│       ├── tcn.py                 # Temporal Convolutional Network
│       ├── wavenet.py             # Reduced WaveNet
│       ├── ddsp.py                # Differentiable DSP hybrid
│       └── conditioned_lstm.py    # LSTM + knob position conditioning
│
├── metrics/
│   ├── time_domain.py             # ESR, MSE, DC offset, RMS error
│   ├── frequency_domain.py        # STFT loss, frequency response error
│   ├── harmonic.py                # THD, harmonic profile similarity
│   ├── perceptual.py              # Mel cepstral distortion, spectral flatness
│   └── suite.py                   # Run all metrics, return results dict
│
├── train/
│   ├── trainer.py                 # Training loop, checkpointing, CUDA
│   ├── losses.py                  # Combined loss functions for NN training
│   └── scheduler.py               # LR scheduling
│
├── eval/
│   ├── run_eval.py                # Run a model against a capture, compute all metrics
│   ├── compare.py                 # Run ALL models against ALL captures
│   └── report.py                  # Generate the colored comparison heatmap table
│
├── notebooks/
│   ├── 01_capture_verify.ipynb    # Interactive capture inspection
│   ├── 02_classical_models.ipynb  # Step through classical math visually
│   ├── 03_neural_training.ipynb   # Train and inspect neural models
│   └── 04_comparison_table.ipynb  # The final comparison heatmap
│
├── configs/
│   └── experiment.yaml            # Model hyperparameters, paths, settings
│
├── requirements.txt
└── README.md
```

---

## Phase 1 — Capture & Alignment (Start Here)

**Goal**: get clean, aligned (dry, wet) pairs from your WAV file.

### Files to build first
```
capture/generate_sweep.py
capture/align.py
capture/verify.py
```

### Math involved

**Cross-correlation alignment:**
```
lag = argmax( Σ dry[n] · wet[n + k] )   for k in [-N, N]
```
Implemented as `np.correlate` or `scipy.signal.correlate`. O(N log N) via FFT.

**DC offset removal:**
```
x_centered = x - mean(x)
```
Always apply before doing anything else.

**RMS normalization (optional):**
```
x_norm = x / sqrt( mean(x²) )
```

---

## Phase 2 — Classical Models

Build these in order. Each one should reach near-perfect scores on the pedal type it's designed for.

---

### 2A — FIR Filter

**What it models**: Level 1 pedals (boost, buffer, passive EQ)

**The math**:
```
y[n] = Σ h[k] · x[n-k]    for k = 0..N-1
```
h[k] is the filter kernel. Find it by:
1. Take FFT of dry and wet signals
2. Divide: H(ω) = FFT(wet) / FFT(dry)
3. Inverse FFT → h[k]
4. Window it (Hann window) to stabilize

**Implementation**: `scipy.signal.firwin`, `scipy.signal.fftconvolve`

**Expected scores on a clean boost pedal**:
- ESR: < 0.001
- STFT loss: < 0.01
- THD error: ~0 (no harmonics added)

---

### 2B — IIR Filter

**What it models**: Level 1 pedals, tone stacks

**The math**:
```
y[n] = Σ b[k]·x[n-k]  -  Σ a[k]·y[n-k]
```
Rational transfer function. Coefficients found by fitting to measured frequency response.

**Implementation**: `scipy.signal.iirdesign`, `scipy.signal.sosfilt`

Use second-order sections (SOS) for numerical stability — never use direct-form IIR.

---

### 2C — Hammerstein Model

**What it models**: Level 2 pedals (mild overdrive, soft clipping)

**The math**:
```
v[n] = f( x[n] )          → static nonlinearity (waveshaper)
y[n] = Σ h[k] · v[n-k]   → linear filter on output
```

**How to identify f(·)**:
- Feed single sine waves at increasing amplitudes
- Measure output harmonics at each amplitude
- Fit a polynomial: f(x) = a₁x + a₃x³ + a₅x⁵ + ...
  (odd-order terms = symmetric clipping like most diode circuits)

**Identification of h[k]**: measure frequency response at very low amplitude (linear regime) — that's your filter.

**Implementation**: NumPy polynomial fit + scipy convolution

---

### 2D — Wiener-Hammerstein Model

**What it models**: Level 2–3 pedals (Tube Screamer, mild overdrive)

**The math**:
```
v[n] = Σ h₁[k] · x[n-k]   → input filter (tone shaping before clip)
w[n] = f( v[n] )            → static nonlinearity
y[n] = Σ h₂[k] · w[n-k]   → output filter (tone shaping after clip)
```

**Why it works for Tube Screamer**: the TS circuit is literally this topology — an op-amp filter, then diode clipping, then another filter.

**Identification**: best-fit approach using Random Phase Multisine input signals and separating the linear/nonlinear parts. Scipy least-squares optimization.

---

### 2E — Volterra Series (2nd order)

**What it models**: Level 2–3, captures interactions between samples

**The math**:
```
y[n] = Σₖ h₁[k]·x[n-k]                          → linear term
     + Σₖ Σⱼ h₂[k,j]·x[n-k]·x[n-j]             → 2nd order cross-terms
     + Σₖ Σⱼ Σᵢ h₃[k,j,i]·x[n-k]·x[n-j]·x[n-i] → 3rd order (optional)
```

**Practical constraint**: limit memory to M=20–30 samples or parameter count explodes.
- 1st order: M terms
- 2nd order: M² / 2 terms  
- 3rd order: M³ / 6 terms

**Identification**: least-squares regression. Stack all the cross-product terms as columns of a matrix X, solve: h = (XᵀX)⁻¹Xᵀy

**Implementation**: NumPy matrix ops, can GPU-accelerate with PyTorch for large M.

---

## Phase 3 — Neural Models

All neural models share the same training infrastructure. Only the model architecture changes.

### Training Setup (all NN models)

**Input/output**:
```
Input:  x[n-R : n]    → window of R past dry samples (receptive field)
Output: ŷ[n]          → predicted wet sample
```

**Dataset**: slice your aligned WAV into overlapping windows.
Window size = receptive field of the model. Step size = 1 sample (or larger for speed).

**Loss function** (combined):
```
L = α · L_time + β · L_stft

L_time = mean( |y - ŷ|² )                          → L2 time domain
L_stft = Σ_scales || log|STFT(y)| - log|STFT(ŷ)|| → multi-scale spectral
```
α=0.1, β=0.9 is a good starting point. The STFT loss dominates and gives much better tonal accuracy.

**Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4
**Scheduler**: cosine annealing
**Batch size**: 32–64 windows
**Epochs**: 100–500 depending on model size

---

### 3A — MLP Baseline

**Architecture**:
```
Input [R samples] → Linear(R,128) → ReLU → Linear(128,64) → ReLU → Linear(64,1)
```

**Receptive field R**: 512–2048 samples (10–40ms at 48kHz)

**Purpose**: establish a lower bound. Prove that ignoring temporal structure hurts.

**Expected weakness**: poor on anything with envelope-following behavior. The MLP doesn't know that x[n-500] was louder than x[n-100] in any meaningful way — it just sees numbers.

---

### 3B — LSTM

**Architecture**:
```
Input x[n] (scalar, one sample at a time)
  → LSTM(input=1, hidden=32, layers=1)
  → Linear(32, 1)
  → output ŷ[n]
```

Run sample-by-sample. Hidden state h[n] carries circuit memory forward in time.

**Stacked variant**: 2 LSTM layers with hidden=24 each. Slightly better, more expensive.

**Key hyperparameter**: hidden size. 16=fast/weak, 32=sweet spot, 64=more accurate/slower.

**Implementation note**: process in chunks during training (TBPTT — truncated backprop through time, chunk=2048 samples). Reset hidden state between training examples, carry it through during inference.

---

### 3C — GRU

Same as LSTM but replace the cell with a GRU cell:
```
  → GRU(input=1, hidden=32, layers=1)
```

**Why compare**: GRU has fewer parameters than LSTM (no separate cell state). Often trains faster, sometimes matches LSTM quality. Good to know which wins on your data.

---

### 3D — TCN (Temporal Convolutional Network)

**Architecture**:
```
For each block i (i = 0..N_blocks-1):
  DilatedConv1d(channels, kernel=3, dilation=2ⁱ)
  → LayerNorm
  → GELU activation
  → residual connection

Final: Conv1d(channels, 1)  → output
```

**Receptive field**: with 10 blocks, kernel=3, dilation doubling:
```
RF = 1 + 2 · Σᵢ (kernel-1) · 2ⁱ  =  1 + 2 · 2 · (2¹⁰ - 1)  =  4095 samples  (~85ms)
```

**Why it matters**: 85ms of context at 48kHz captures the decay of most clipping transients.

**Speed advantage**: fully parallel during training (unlike LSTM). 5–10x faster to train.

---

### 3E — Reduced WaveNet

**Architecture**:
```
Causal Conv1d (input embedding)
→ N residual blocks, each:
    DilatedCausalConv1d(dilation=2ⁱ)
    → tanh(·) * sigmoid(·)   → gated activation
    → 1x1 Conv (residual + skip)
→ Sum skip connections
→ ReLU → Conv → ReLU → Conv → output
```

**Channels**: 16–32 (much smaller than original WaveNet's 256)
**Blocks**: 3 stacks of 10 (dilations 1,2,4...512)
**Receptive field**: ~6000 samples (125ms)

**Expected**: best accuracy, slowest inference. Use as quality ceiling reference.

---

### 3F — DDSP (Differentiable DSP)

**Architecture**: replace some NN layers with actual DSP operations that have learnable parameters.

```
x[n]
  → Learned FIR filter (coefficients are NN parameters)
  → Learned waveshaper f(·)  (parameterized as a small MLP)
  → Learned IIR filter
  → ŷ[n]
```

Everything is differentiable — train end-to-end with backprop.

**Why it's interesting**: after training, inspect what was learned:
- Plot the waveshaper curve — it should look like a diode clipping curve
- Plot the filter frequency response — it should match the tone stack
- If it doesn't, your data has a problem

**Implementation**: custom PyTorch modules with `torch.nn.Parameter` for DSP coefficients.

---

### 3G — Conditioned LSTM

**Extension of 3B for multi-setting capture**:
```
x[n], h[n-1], c[n-1]
knob_vector [drive, tone, level]  → embedding → injected into LSTM input
  → LSTM → ŷ[n]
```

Requires capturing the pedal at multiple knob positions (e.g. 5x5 grid of drive/tone). One model covers the whole parameter space.

---

## Phase 4 — Metrics Suite

Every model gets scored on every metric. All computed in `metrics/suite.py`.

---

### 4A — Time Domain Metrics

**ESR (Error-to-Signal Ratio)**
```
ESR = Σ(y[n] - ŷ[n])²  /  Σ y[n]²
```
Range: 0 (perfect) → 1 (useless). Target: < 0.01

**MSE**
```
MSE = (1/N) · Σ(y[n] - ŷ[n])²
```

**DC Offset Error**
```
DC_err = |mean(y) - mean(ŷ)|
```
Should be < 1e-4

**RMS Error**
```
RMS_err = sqrt( mean( (y - ŷ)² ) )
```

---

### 4B — Frequency Domain Metrics

**Multi-Scale STFT Loss**
```
For window_sizes in [32, 128, 512, 2048]:
    S_y  = |STFT(y,  window)|
    S_ŷ  = |STFT(ŷ, window)|
    L   += mean( |log(S_y) - log(S_ŷ)| )
```
Implementation: `torchaudio.transforms.Spectrogram`

**Frequency Response Error (dB)**
```
H_y(ω)  = FFT(y)  / FFT(x)    → measured transfer function
H_ŷ(ω) = FFT(ŷ)  / FFT(x)    → modeled transfer function
FR_err  = mean( |20·log10|H_y(ω)| - 20·log10|H_ŷ(ω)|| )   in dB
```

---

### 4C — Harmonic Metrics

**THD (Total Harmonic Distortion)**
Feed a single sine at frequency f₀:
```
THD = sqrt( Σₙ₌₂^N Aₙ² )  /  A₁     × 100%

where Aₙ = amplitude at n·f₀ in the output spectrum
```
Compute at f₀ = 100Hz, 440Hz, 1kHz, 4kHz.

**Harmonic Profile Similarity**
```
profile_y  = [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈] / A₁   → normalized to fundamental
profile_ŷ  = same for model output
HP_sim = cosine_similarity(profile_y, profile_ŷ)         → 1.0 = identical
```

**Even/Odd Harmonic Ratio**
```
even = A₂ + A₄ + A₆ + A₈
odd  = A₃ + A₅ + A₇
EO_ratio = even / odd
```
Tube amp character = high even. Fuzz character = high odd.
Your model should match the target's EO_ratio within 10%.

---

### 4D — Perceptual Metrics

**Mel Cepstral Distortion (MCD)**
```
MCC_y  = DCT( log( MelFilterbank( |STFT(y)|  ) ) )
MCC_ŷ  = DCT( log( MelFilterbank( |STFT(ŷ)|  ) ) )
MCD    = (10/ln10) · sqrt( 2 · Σₖ (MCC_y[k] - MCC_ŷ[k])² )   in dB
```
Lower is better. < 2dB is good. Captures timbral character (tone) independently of phase.

**Log Spectral Distance (LSD)**
```
LSD = sqrt( (1/K) · Σₖ ( 10·log10(|H_y(k)|²) - 10·log10(|H_ŷ(k)|²) )² )
```
In dB. < 1dB is excellent.

---

## Phase 5 — The Comparison Table

`eval/compare.py` runs every model against every capture and builds a results dict:

```python
results = {
    "FIR":              {"ESR": 0.0008, "STFT": 0.012, "THD_err": 0.001, "MCD": 0.8,  "HP_sim": 0.99},
    "IIR":              {"ESR": 0.0006, "STFT": 0.010, "THD_err": 0.001, "MCD": 0.7,  "HP_sim": 0.99},
    "Hammerstein":      {"ESR": 0.012,  "STFT": 0.089, "THD_err": 0.045, "MCD": 2.1,  "HP_sim": 0.87},
    "Wiener-Hammer":    {"ESR": 0.004,  "STFT": 0.031, "THD_err": 0.018, "MCD": 1.4,  "HP_sim": 0.94},
    "Volterra-2nd":     {"ESR": 0.003,  "STFT": 0.025, "THD_err": 0.012, "MCD": 1.2,  "HP_sim": 0.95},
    "MLP":              {"ESR": 0.021,  "STFT": 0.140, "THD_err": 0.092, "MCD": 4.1,  "HP_sim": 0.71},
    "LSTM-32":          {"ESR": 0.002,  "STFT": 0.018, "THD_err": 0.009, "MCD": 0.9,  "HP_sim": 0.97},
    "GRU-32":           {"ESR": 0.0025, "STFT": 0.020, "THD_err": 0.010, "MCD": 1.0,  "HP_sim": 0.97},
    "TCN":              {"ESR": 0.0018, "STFT": 0.015, "THD_err": 0.007, "MCD": 0.85, "HP_sim": 0.98},
    "WaveNet":          {"ESR": 0.0012, "STFT": 0.011, "THD_err": 0.005, "MCD": 0.75, "HP_sim": 0.99},
    "DDSP":             {"ESR": 0.0022, "STFT": 0.017, "THD_err": 0.008, "MCD": 0.88, "HP_sim": 0.98},
}
```

`eval/report.py` renders this as a seaborn heatmap — green=good, red=bad — one row per model, one column per metric. Each metric normalized 0–1 independently so colors are comparable across columns.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render_heatmap(results: dict, pedal_name: str):
    df = pd.DataFrame(results).T

    # Normalize each column: 0=worst, 1=best
    # For error metrics (lower=better): invert
    error_cols = ["ESR", "STFT", "THD_err", "MCD"]
    sim_cols   = ["HP_sim"]

    normed = df.copy()
    for col in error_cols:
        normed[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
    for col in sim_cols:
        normed[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        normed,
        annot=df.round(4),   # show actual values in cells
        fmt='',
        cmap='RdYlGn',
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(f'Model Comparison — {pedal_name}', fontsize=16, pad=20)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Model')
    plt.tight_layout()
    plt.savefig(f'comparison_{pedal_name}.png', dpi=150)
    plt.show()
```

---

## Build Order (Recommended)

```
Week 1 — Foundation
  ✓ WSL2 setup, CUDA verify, pip install
  ✓ capture/generate_sweep.py
  ✓ capture/align.py
  ✓ capture/verify.py
  ✓ Record your first capture WAV

Week 2 — Classical Models
  ✓ metrics/suite.py (all metrics)
  ✓ models/classical/fir.py
  ✓ models/classical/iir.py
  ✓ eval/run_eval.py
  ✓ Prove FIR/IIR = ~perfect on a clean boost pedal

Week 3 — Classical Nonlinear
  ✓ models/classical/hammerstein.py
  ✓ models/classical/wiener_hammerstein.py
  ✓ models/classical/volterra.py
  ✓ Capture a mild overdrive, run comparison

Week 4 — Neural Baseline
  ✓ data/dataset.py
  ✓ train/trainer.py + losses.py
  ✓ models/neural/mlp.py
  ✓ models/neural/lstm.py
  ✓ Prove LSTM beats Volterra on complex pedal

Week 5 — Neural Comparison
  ✓ models/neural/gru.py
  ✓ models/neural/tcn.py
  ✓ models/neural/wavenet.py
  ✓ First full comparison heatmap

Week 6 — Advanced
  ✓ models/neural/ddsp.py
  ✓ models/neural/conditioned_lstm.py
  ✓ eval/compare.py + report.py
  ✓ Full report across all pedals × all models
```

---

## Environment Setup (WSL2 + CUDA)

```bash
# In WSL2 Ubuntu terminal:

# Verify CUDA is visible
nvidia-smi

# Create project environment
python -m venv venv
source venv/bin/activate

# Install everything
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy soundfile librosa matplotlib seaborn einops pedalboard pandas jupyter

# Verify GPU in Python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Open VS Code connected to WSL
code .
```

VS Code extensions to install:
- Remote - WSL
- Python
- Pylance
- Jupyter
- GitLens
