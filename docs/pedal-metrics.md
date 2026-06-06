# Metrics Suite

Every model is scored on the same set of metrics, computed on the held-out evaluation segment.
Lower is better for all error metrics; higher is better for similarity metrics.

---

## Time-Domain Metrics

### ESR — Error-to-Signal Ratio *(primary metric)*

$$\text{ESR} = \frac{\sum_n (y[n] - \hat{y}[n])^2}{\sum_n y[n]^2}$$

Range: 0 (perfect) → ∞ (worse than silence). Values > 1 mean the prediction is more wrong
than just outputting silence. **Target: < 0.01.**

ESR normalises by signal power, so it is fair regardless of the target's loudness.
This makes it the go-to metric for audio effect modeling.

### MSE — Mean Squared Error

$$\text{MSE} = \frac{1}{N}\sum_n (y[n] - \hat{y}[n])^2$$

Unnormalised squared error in the time domain. Sensitive to absolute amplitude —
a quiet prediction gets a low MSE even if the shape is wrong. Always read alongside ESR.

### DC Error

$$\text{DC\_err} = |\bar{y} - \bar{\hat{y}}|$$

Difference in DC offset (mean value). Should be < 1e-4. A large DC error means the model
has a constant bias — common when the loss function is not centred.

### RMS Error

$$\text{RMS\_err} = \sqrt{\text{MSE}}$$

Same as MSE but in the same units as the signal (amplitude). Easier to interpret physically.

---

## Frequency-Domain Metrics

### STFT — Multi-Scale Spectral Loss

$$\text{STFT} = \frac{1}{S}\sum_{s} \frac{1}{N_s}\sum_{k} \big|\log|Y_s(k)| - \log|\hat{Y}_s(k)|\big|$$

Computed at window sizes $s \in \{32, 128, 512, 2048\}$ samples, then averaged.

- **Small windows (32):** sensitive to transient timing and onset sharpness
- **Large windows (2048):** sensitive to sustained tonal character and harmonic balance

The log-magnitude formulation matches perceptual loudness (our ears are roughly log-sensitive).
This is the dominant term in the training loss (β).

### FR Error — Frequency Response Error (dB)

$$\text{FR\_err} = \text{mean}_\omega \big| 20\log_{10}|H_y(\omega)| - 20\log_{10}|H_{\hat{y}}(\omega)| \big|$$

where $H_y = \text{FFT}(y)/\text{FFT}(x)$ is the measured transfer function.

Measures how closely the model's overall frequency response (input→output gain vs frequency)
matches the pedal's. A good model should have FR\_err < 1–2 dB across the audio band.

---

## Harmonic Metrics

These require feeding a pure sine tone through each model and measuring its harmonic content.
Evaluated at 440 Hz (concert A).

### THD — Total Harmonic Distortion

$$\text{THD} = \frac{\sqrt{\sum_{n=2}^{N} A_n^2}}{A_1} \times 100\%$$

$A_n$ = amplitude of the $n$-th harmonic in the output spectrum.

The Big Muff Pi is famous for extremely high THD (50–90%). A model that produces
low THD will sound "clean" — the most common failure mode for neural models.

### THD Error

$$\text{THD\_err} = |\text{THD}_\text{target} - \text{THD}_\text{predicted}|$$

How closely the model matches the pedal's actual distortion level.

### HP Similarity — Harmonic Profile Similarity

$$\text{HP\_sim} = \text{cosine\_similarity}(\mathbf{p}_y,\, \mathbf{p}_{\hat{y}})$$

where $\mathbf{p} = [A_1, A_2, \ldots, A_8] / A_1$ is the normalised harmonic amplitude vector.

Measures whether the model reproduces the *shape* of the harmonic spectrum, independent of
overall level. HP\_sim = 1.0 means identical harmonic character; < 0.9 is audibly different.
A fuzz pedal has a characteristic odd-harmonic profile (3rd, 5th, 7th dominate) — the model
should match this.

---

## Perceptual Metrics

### MCD — Mel Cepstral Distortion

$$\text{MCD} = \frac{10}{\ln 10}\sqrt{2\sum_{k=1}^{K}\big(\text{MCC}_y[k] - \text{MCC}_{\hat{y}}[k]\big)^2} \quad \text{dB}$$

where $\text{MCC}$ are Mel Cepstral Coefficients: DCT of the log Mel filterbank energies.

MCD captures *timbral* difference — how different the models sound in terms of tone colour —
independently of pitch or timing. Values < 2 dB are generally indistinguishable by ear;
> 5 dB is clearly different. Developed for speech synthesis evaluation but works well for
guitar tones.

---

## Reading the Heatmap

The comparison heatmap normalises each column independently to [0, 1]:
- **Error metrics** (ESR, MSE, STFT, FR\_err\_dB, THD\_err, MCD): **lower raw value → greener cell**
- **Similarity metrics** (HP\_sim): **higher raw value → greener cell**

Raw values are shown inside each cell. Green = best in class; red = worst in class.
The normalisation is relative to the other models, not to an absolute target —
a green cell in FIR's ESR row just means FIR is the best *of the linear models*, not that
its ESR is good in absolute terms.
