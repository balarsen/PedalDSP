# Neural Pedal Models

Neural models learn the dry→wet mapping directly from data with no assumption about the
circuit topology. They are trained end-to-end via backpropagation.

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
