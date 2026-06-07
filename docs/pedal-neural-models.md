# Neural Pedal Models

Neural models learn the dry→wet mapping directly from data with no assumption about the
circuit topology. They are trained end-to-end via backpropagation.

---

## Deployment Targets

Two deployment targets shape every architecture decision:

| Target | Rate | Format | Constraint |
|---|---|---|---|
| **Daisy Seed** | 48 kHz | int16 PCM | Tight CPU/RAM — only the smallest models fit (compact TCN, small WaveNet, distilled DDSP) |
| **Studio One plugin** | 48 kHz (session rate, varies) | float32 | Full model zoo fits; no memory constraint |

**Inference oversampling:** neural models run at 2× target rate and downsample to control
self-aliasing. A 48k deployment model runs internally at 96k and downsamples its output.

**Training rate discipline:** train each model at its deployment rate (downsample 96k→48k first).
Keep a 96k model as a full-bandwidth research artifact.

---

## Provisional Notaklon Results

<!-- TODO: fill in after first full model zoo training pass (Step 11) -->

| Model | ESR | Null depth (dB) | STFT loss | THD pattern | MCD | Passes gate? |
|---|---|---|---|---|---|---|
| FIR | — | — | — | — | — | <!-- TODO --> |
| Hammerstein | — | — | — | — | — | <!-- TODO --> |
| Volterra | — | — | — | — | — | <!-- TODO --> |
| MLP | — | — | — | — | — | <!-- TODO --> |
| TCN (small) | — | — | — | — | — | <!-- TODO --> |
| TCN (larger) | — | — | — | — | — | <!-- TODO --> |
| LSTM-32 | — | — | — | — | — | <!-- TODO --> |
| GRU-32 | — | — | — | — | — | <!-- TODO --> |
| WaveNet | — | — | — | — | — | <!-- TODO --> |
| DDSP | — | — | — | — | — | <!-- TODO --> |

<!-- TODO: add metric-vs-perception correlation plot after blind listening test (Step 12) -->

---

## LSTM (Long Short-Term Memory)

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

**Daisy feasibility:** H=16–24 may fit. Profile first. <!-- TODO: measure FLOPS/latency -->

---

## TCN (Temporal Convolutional Network)

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

**Daisy candidate:** small TCN (C=8, N=6) is the primary Daisy target.
<!-- TODO: measure size/latency on Daisy hardware -->

---

## WaveNet-Style

**Architecture:**
```
Causal Conv1d (input embedding)
→ N residual blocks, each:
    DilatedCausalConv1d(dilation=2ⁱ)
    → tanh(·) * sigmoid(·)   → gated activation
    → 1x1 Conv (residual + skip)
→ Sum skip connections
→ ReLU → Conv → ReLU → Conv → output
```

**Channels:** 16–32 (much smaller than original WaveNet's 256)
**Blocks:** 3 stacks of 10 (dilations 1, 2, 4...512)
**Receptive field:** ~6000 samples (125 ms at 48k)

**Expected:** best accuracy, slowest inference. Quality ceiling reference.
A small variant (channels=8, 2 stacks) is a Daisy candidate.
<!-- TODO: Notaklon null-test depth result here -->

---

## KlonDDSP (Klon-Informed Hybrid DDSP + GRU)

`pedal_model/models/neural/ddsp_klon.py`

**Architecture:** a physics-informed DDSP stage models the Klon Centaur circuit topology
explicitly; a GRU correction stage learns the residual the physics can't capture.

```
x[n]  ──────────────────────────────────────────────────────────┐
        │                                                         │
        ▼                                                         │
   FF1: 1-pole LP (106 Hz, learned α)                            │
        │ ← clean bass pass-through path                         │
        ▼                                                         │
   Clip path: drive → tanh soft-clip                             │
        │                                                         │
        ▼                                                         │
   FF2 blend: sigmoid(w·gain_knob + b) × clip + (1−blend) × FF1 │
        │ ← gain-conditioned clean/dirty mix                     │
        ▼                                                         │
   output_level (sigmoid-scaled amplitude)                        │
        │                                                         │
        └──────────────────────┐ ddsp_out (detached)             │
                                ▼                                 │
                    GRU([dry, ddsp_out, condition])               │
                        ↓                                         │
                    gru_fc (small weights)                        │
                        ↓                                         │
                    correction                                    │
                        │                                         │
                        └──────────────────── + ddsp_out ────────┘
                                                ▼
                                              ŷ[n]
```

**FF1 — clean bass feed-forward:**
The real Klon uses a parallel 106 Hz LPF on the dry signal that bypasses the clipping
stage entirely. This gives the pedal its "transparent" character at lower gain settings —
the bass passes clean while the mids/treble are saturated.

- Pole radius: `α_ff1 = sigmoid(raw_param) × 0.999`  (init ≈ 0.986, matching 106 Hz at 48 kHz)
- Truncated FIR approximation: `h[k] = (1-α) · α^k` for k=0..255, convolved with `F.conv1d`
- Differentiable w.r.t. `_alpha_ff1_raw`

**FF2 — gain-conditioned blend:**
At gain=0 the output is nearly clean (blend → 0 → mostly FF1).
At gain=1 the output is mostly clipped signal (blend → 1 → clip path dominates).

```python
blend = torch.sigmoid(w · gain_knob + b)   # learned affine function of the gain knob
ddsp_out = blend × clip_path + (1 − blend) × ff1_path
```

**GRU context input:**
The GRU receives `[dry, ddsp_out.detach(), condition]` rather than just `[dry, condition]`.
By observing the DDSP stage's output directly, the GRU learns the *residual* — what the
physics got wrong — rather than trying to model everything from scratch.
`.detach()` prevents gradients from flowing back through the DDSP path a second time.

**Conditioned on 3 knobs:** `[gain, tone, volume]` in [0, 1]. Default = 3/11 (nine-o'clock).

**Initialisation:** `gru_fc` is initialised with very small weights (`std=0.001`) so the
model starts near-DDSP-only; the GRU correction grows from near-zero as training proceeds.

---

## DDSP (Differentiable DSP)

**Architecture:** replace some NN layers with actual DSP operations that have learnable parameters.

```
x[n]
  → Learned FIR filter (coefficients are NN parameters)
  → Learned waveshaper f(·)  (parameterized as a small MLP)
  → Learned IIR filter
  → ŷ[n]
```

Everything is differentiable — train end-to-end with backprop.

**Why it's interesting:** after training, inspect what was learned:
- Plot the waveshaper curve — it should look like a diode clipping curve
- Plot the filter frequency response — it should match the Notaklon tone stack
- If it doesn't, the data or level calibration has a problem

**Implementation:** custom PyTorch modules with `torch.nn.Parameter` for DSP coefficients.

**Daisy candidate:** compact DDSP (small FIR taps, 2-layer waveshaper MLP).
<!-- TODO: Notaklon waveshaper curve visualization (Step 11) -->

---

## Conditioned LSTM

**Extension of LSTM for multi-setting capture:**
```
x[n], h[n-1], c[n-1]
knob_vector [drive, tone, level]  → embedding → injected into LSTM input
  → LSTM → ŷ[n]
```

Requires capturing the pedal at multiple knob positions (e.g. 5×5 grid of drive/tone).
One model covers the whole parameter space.

**Planned:** captured at Gain 9/Tone noon and Gain 11/Tone noon initially (see plan §6).
<!-- TODO: add after primary position is validated -->

---

## CircuitDictionary (DARTS-Inspired Classical Model)

`pedal_model/models/classical/circuit_dictionary.py`

A parallel differentiable circuit primitive dictionary with learned architecture weights.
Registered in the eval harness as `"circuit_dict"`.

**Motivation:** rather than committing to a single circuit topology, let the model learn
*which* primitives matter for the pedal. After fitting, the softmax weights reveal the
effective circuit hypothesis — which primitives the data selected.

**Architecture:**
```
x[n]  ──┬── LearnableLowpass(fc)          ──┐
         ├── LearnableHighpass(fc)          ─┤
         ├── LearnableHighShelf(fc, dB)     ─┤
         ├── LearnableBandpass(fc, Q)       ─┤
         ├── LearnableSoftClipper(drive)    ─┤  weighted sum  →  ŷ[n]
         ├── LearnableAsymClipper           ─┤  (softmax α)
         ├── LearnableClipThenFilter        ─┤
         ├── LearnableParallelBlend         ─┤
         ├── SkipConnection (x unchanged)  ─┤
         └── LearnableGain(g)              ─┘
```

**Architecture weights:** `raw_alpha` (10-vector, `nn.Parameter`) → `softmax(raw_alpha)` → weighted sum.
An L1 penalty `λ · mean(|raw_alpha|)` encourages sparsity (few dominant primitives).

**Filter implementation:** all LP/HP filters use a truncated FIR approximation of the IIR
impulse response (`h[k] = (1-a)·a^k`, K=512 taps), convolved via `F.conv1d` with causal
zero-padding. This is fast, differentiable, and avoids sequential sample loops over N.
Biquad (shelf, bandpass) impulse responses are computed via a 512-step recurrence (K << N).

**Training:** Adam, lr=1e-3, 500 epochs, loss = ESR + λ·L1.

**Reading the results:** after `fit()`:
- `model.alpha_` — softmax weights in decreasing order
- `model.primitive_names` — names aligned with `alpha_`
- Topology summary is printed to stdout

**Hyperparameters:**
- `sr`: sample rate (default 48000)
- `n_epochs`: training epochs (default 500)
- `lambda_l1`: L1 sparsity weight (default 0.01)

---

## Loss Function

$$\mathcal{L} = \alpha \cdot \underbrace{\text{MSE}(\hat{y}, y)}_{\text{time-domain}} + \beta \cdot \underbrace{\frac{1}{S}\sum_s \|\log|STFT_s(\hat{y})| - \log|STFT_s(y)|\|_1}_{\text{multi-scale STFT}}$$

Window sizes $s \in \{32, 128, 512, 2048\}$ samples.

- **Time-domain term (α):** penalises sample-by-sample amplitude errors directly.
  Higher α → model is forced to match waveform shape, including clipping transitions.
- **STFT term (β):** penalises spectral shape errors in log-magnitude.
  Higher β → model matches tonal character (harmonic balance) even if timing shifts slightly.

**For a hard fuzz:** try α=0.5 or higher. The STFT term can "accept" a smooth approximation
of a square wave that has the right harmonic content but no sharp edges.

**For the Notaklon** (soft-knee, mostly static NL): α=0.1, β=0.9 is a reasonable start.
<!-- TODO: tune after first training run; log α/β sweep results in configs/ -->
