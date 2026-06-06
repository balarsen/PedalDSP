# Classical Pedal Models

Classical models express the pedal as explicit mathematical equations.
No training data beyond the fit signal is needed, and you can inspect exactly what was learned.

---

## FIR Filter (Finite Impulse Response)

**What it models:** any *linear, time-invariant* (LTI) system — boosts, EQ, buffers.

**The math:**
$$y[n] = \sum_{k=0}^{N-1} h[k] \cdot x[n-k]$$

$h[k]$ is the filter kernel (impulse response). We identify it in the frequency domain:

1. Take FFT of dry and wet signals
2. Divide: $H(\omega) = \text{FFT}(\text{wet}) / \text{FFT}(\text{dry})$
3. Inverse FFT → $h[k]$, then window to suppress time-aliasing

**Why it fails on fuzz:** fuzz is nonlinear — doubling the input does *not* double the output.
The Big Muff clips hard; FIR cannot reproduce clipping, only spectral shaping.
Expect ESR ≈ 1 (model is no better than silence).

---

## Hammerstein Model

**What it models:** a static nonlinearity followed by a linear filter.

**The math:**
$$v[n] = f(x[n]) \qquad \text{(waveshaper)}$$
$$y[n] = \sum_{k} h[k] \cdot v[n-k] \qquad \text{(output filter)}$$

**How $f(\cdot)$ is identified:** polynomial regression — fit a degree-7 polynomial
$f(x) = a_1 x + a_2 x^2 + \ldots + a_7 x^7$ directly from dry→wet pairs via least squares.
Odd powers model symmetric clipping (diodes); even powers add asymmetry (transistors).

**How $h[k]$ is identified:** measure the frequency response in the low-amplitude (linear) regime
of the signal (lowest 25th percentile of $|x[n]|$), then FIR-divide there.

**Why it partially works on fuzz:** captures the clipping shape, but assumes the nonlinearity
is *static* — it doesn't know that the Big Muff's clipping threshold depends on circuit memory
(capacitor charge). Expect moderate ESR improvement over FIR.

**Pre-gain note:** the polynomial must "see" samples in the saturation region.
If the dry signal is too quiet, $f(\cdot)$ fits a nearly-linear polynomial.
Increase the **dry pre-gain** slider until the polynomial captures the clipping knee.

---

## Volterra Series (2nd order)

**What it models:** linear + all pairwise lag-product interactions.

**The math:**
$$y[n] = \underbrace{\sum_{k} h_1[k]\,x[n-k]}_{\text{linear}} + \underbrace{\sum_{k}\sum_{j \geq k} h_2[k,j]\,x[n-k]\,x[n-j]}_{\text{quadratic cross-terms}}$$

**How identified:** build a regression matrix where each column is either a delayed copy of $x$
(linear) or a product of two delayed copies (quadratic), then solve via least squares.
Memory $M=20$ gives $M + M(M+1)/2 = 230$ parameters.

**Why it's better than Hammerstein:** captures *interactions between past samples* —
e.g. how the previous sample's value shapes the current clipping threshold.
Still limited because it cannot model the infinite-memory dynamics of a capacitor.

**Practical note:** fitting on 54 s of audio at 44100 Hz would require a 2.4M × 230 matrix.
We subsample every 8th sample so the least-squares solve stays tractable.
