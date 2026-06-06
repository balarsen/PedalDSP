# Signal Design Rationale

**Files:** `data/signals/train_signal_v1.{wav,json}` · `data/signals/val_signal_v1.{wav,json}`
**Rate:** 96 kHz, 32-bit float
**Generator:** `pedal_model/signals/generate.py` · CLI: `python generate_signal.py --signal both`
**Reproduce exactly:** `python generate_signal.py --from-manifest data/signals/train_signal_v1.json`

---

## Why synthetic signals?

An analog pedal is deterministic and near-memoryless: same input → same output every time. We train on synthetic signals because they **tile the input space systematically** — covering amplitude, frequency, and transient corners that any one guitar performance undersamples. Guitar is the held-out generalization test: if the model learned the *transfer function* (the physics of the clipping circuit), it will generalize to guitar. If it only learned the statistics of the training set, it won't.

The synthetic signals fill four coverage gaps that guitar leaves:

| Gap | What fills it |
|---|---|
| Crest factor (sharp transients) | Impulse trains, AM chirps, decaying plucks |
| Spectral envelope dead zones | Pink noise, multitone stacks |
| Continuous amplitude envelope | Amplitude ramps |
| Intermodulation (strings ringing together) | Two-tone sweeps, layered sections |

---

## Amplitude conventions

- **Deterministic signals** (sweeps, tones, ramps, plucks): `amplitude_dbfs` = **peak** dBFS.
- **Stochastic signals** (pink noise, white noise): `amplitude_dbfs` = **RMS** dBFS.
- Klon soft-knee onset is roughly −12 to −6 dBFS at moderate gain settings. The amplitude grid spans −24 dBFS (clean) through −3 dBFS (full drive) to bracket the nonlinear region.

---

## Training Signal — `train_signal_v1` (~13 min, seed 1234)

Total ≈ 640 s · 84 sections · 96 000 Hz · seed 1234

### Design split: Pure / Transient / Layered

| Category | Share | Purpose |
|---|---|---|
| Pure / single-stimulus | ~40% | Clean measurement windows — THD, FR, ESR computed here |
| Transient / dynamic | ~20% | Dynamic range, attack/recovery, envelope |
| Layered / combination | ~40% | Intermodulation richness; maximizes information per second |

Pure segments are retained because summing before a nonlinearity **entangles** responses. Having clean single-stimulus regions protects measurement plots and gives the metrics code clean anchor points.

---

### 1. Log sine sweeps

Three full-band sweeps at increasing amplitude. Each is a Farina exponential sweep: instantaneous phase `φ(t) = 2π·f₁·L·(exp(t/L)−1)`, where `L = T / ln(f₂/f₁)`.

| Label | Duration | Amplitude | Purpose |
|---|---|---|---|
| `sweep_l-24` | 20 s | −24 dBFS peak | Linear frequency response — pedal is barely driven |
| `sweep_l-12` | 20 s | −12 dBFS peak | Onset of soft clipping |
| `sweep_l-3` | 20 s | −3 dBFS peak | Full nonlinearity — harmonic generation at every frequency |

**Why three levels?** The Klon's soft-knee characteristic means the nonlinearity is amplitude-dependent. A single sweep only measures one point on the amplitude dimension. Three sweeps at −24 / −12 / −3 dBFS bracket the clean, onset, and saturated regimes.

**Why 20–20 000 Hz?** Full audio band. Guitar fundamental range is 82–1175 Hz, but the Klon generates harmonics that matter up to ~10 kHz; the 20k upper edge captures ultrasonic content that folds back when downsampled to 48k for deployment.

---

### 2. Stepped sine tones

Each tone is a pure 1 s sine at a fixed frequency and level. 12 frequencies × 5 levels = 60 discrete (freq, level) measurement points.

| Freq (Hz) | Guitar note | Role |
|---|---|---|
| 82.4 | E2 — low E string open | Lowest guitar fundamental |
| 110.0 | A2 — A string open | Second string |
| 165.0 | E3 — approx. D string 2nd fret | Mid-low range |
| 220.0 | A3 — G string / octave A | Core guitar midrange |
| 330.0 | E4 — approx. high E string open | Key guitar note |
| 440.0 | A4 — concert A | THD reference standard |
| 660.0 | E5 — one octave above open high E | First harmonic zone |
| 880.0 | A5 — two octaves above open A | Upper guitar harmonics |
| 1 320.0 | E6 | Fret harmonics / upper partials |
| 2 640.0 | E7 | Harmonic content |
| 5 280.0 | E8 | Near Nyquist at 48k target |
| 10 560.0 | — | Ultrasonic reference |

| Level (dBFS peak) | Meaning |
|---|---|
| −24 | Clean — below soft-knee |
| −18 | Approaching onset |
| −12 | Soft-knee onset |
| −6 | Moderate drive |
| −3 | Maximum drive |

**Why these frequencies?** They are spaced roughly one octave apart, starting at the open low-E (82.4 Hz) and climbing through the entire guitar harmonic range. They are the "anchor grid" for THD measurements — every (freq, level) pair gives one THD figure, building a 12×5 THD surface.

---

### 3. Pink noise

Pink noise (`1/f` spectral density) matches the natural spectral envelope of most music and guitar. Three 20 s blocks at different RMS levels.

| Label | Duration | RMS level | Purpose |
|---|---|---|---|
| `pink_rms-24` | 20 s | −24 dBFS RMS | Clean-level coverage; fills spectral gaps between tone frequencies |
| `pink_rms-12` | 20 s | −12 dBFS RMS | Mid-level drive |
| `pink_rms-3` | 20 s | −3 dBFS RMS | Full-level drive; worst-case spectral input |

Pink noise is stochastic (seed-controlled) — the same seed always produces the same sequence. The `1/f` weighting means lower frequencies carry more energy, matching guitar's emphasis on fundamentals.

---

### 4. White noise

Flat spectrum from DC to Nyquist. Two shorter blocks covering the amplitude extremes.

| Label | Duration | RMS level | Purpose |
|---|---|---|---|
| `white_rms-18` | 15 s | −18 dBFS RMS | Equal energy per Hz — probes high-frequency clipping |
| `white_rms-6` | 15 s | −6 dBFS RMS | High-level flat-spectrum stress test |

White noise is less musically representative than pink but exercises the high-frequency response more evenly, catching any anomalies in the pedal's treble behavior.

---

### 5. Amplitude ramps

A continuous linear-in-dB amplitude sweep on a fixed sine tone. Probes the entire soft-knee curve at each frequency in one shot.

| Freq (Hz) | Guitar note | Duration | Amplitude range |
|---|---|---|---|
| 110.0 | A2 | 10 s | −40 → −2 dBFS |
| 220.0 | A3 | 10 s | −40 → −2 dBFS |
| 440.0 | A4 | 10 s | −40 → −2 dBFS |
| 880.0 | A5 | 10 s | −40 → −2 dBFS |

**Why −40 to −2 dBFS?** −40 dBFS is well below any guitar signal (noise floor); −2 dBFS is near-clipping. The ramp draws a continuous picture of gain vs. input level — the "S-curve" that defines the soft-knee character.

**Why these four frequencies?** The four open A-string octaves (A2, A3, A4, A5) represent the spine of the guitar frequency range. The ramp at each frequency gives a per-octave nonlinearity curve.

---

### 6. Impulse train

Sparse impulses at varying spacings. Tests the pedal's transient response and recovery time.

| Parameter | Value |
|---|---|
| Duration | 30 s |
| Amplitude | −6 dBFS peak |
| Spacings (s) | 0.05, 0.10, 0.20, 0.05, 0.15, 0.30, 0.05, 0.08 (repeating) |

**Why varying spacings?** Closely spaced impulses test whether the pedal has enough memory (or capacitor charge time) to interact with the previous impulse. Wide spacings let it fully recover. The combination builds a picture of transient masking and recovery.

---

### 7. AM chirp

An amplitude-modulated sine carrier with swept modulation frequency. Probes the intersection of frequency content and amplitude dynamics.

| Parameter | Value |
|---|---|
| Duration | 60 s |
| Carrier frequency | 440 Hz (A4) |
| Carrier amplitude | −9 dBFS peak |
| Modulation depth | 80% |
| Modulation frequency sweep | 0.5 → 20 Hz |

The modulation rate sweeps from 0.5 Hz (slow tremolo) to 20 Hz (rapid flutter). At 20 Hz the amplitude changes so fast it becomes a tonal component. This exercises envelope-following behavior and any frequency-dependent gain compression.

---

### 8. Decaying plucks

Exponentially decaying sine bursts — the closest synthetic approximation to a plucked guitar string. Each pluck decays to 1/e in `tau_s = 0.4 s` (i.e., 5 × tau = 2 s total decay).

| Freq (Hz) | Guitar note | Position |
|---|---|---|
| 82.4 | E2 | Low E open |
| 110.0 | A2 | A string open |
| 165.0 | E3 | ~D string 2nd fret |
| 220.0 | A3 | G string / octave A |
| 330.0 | E4 | ~High E open |
| 440.0 | A4 | Concert A |
| 880.0 | A5 | Two octaves above A |

**Why these frequencies?** These are the six open strings of a standard guitar (E2, A2, D3≈165, G3≈196, B3≈247, E4≈330) plus A4 and A5 — the notes a guitarist is most likely to let ring out cleanly. The exponential decay envelope matches real pluck physics (simplified Karplus-Strong).

Pluck total duration = 40 s, with silence gaps between each note at each level.

---

### 9. Two-tone sweeps

A fixed-frequency tone combined with a swept second tone. Generates intermodulation (IM) products — the sum and difference frequencies that appear in the output when a nonlinearity receives two simultaneous inputs.

| Fixed freq | Swept range | Duration | Amplitude |
|---|---|---|---|
| 110.0 Hz (A2) | 20–8000 Hz | 10 s | −9 dBFS peak each |
| 220.0 Hz (A3) | 20–8000 Hz | 10 s | −9 dBFS peak each |
| 440.0 Hz (A4) | 20–8000 Hz | 10 s | −9 dBFS peak each |

**Why three fixed tones?** Guitar playing always involves multiple simultaneous strings. IM products appear at `f₂ ± f₁`, `2f₁ ± f₂`, etc. As the sweep moves through the range, IM products pop in and out — the model must learn to suppress or reproduce them correctly at every combination.

---

### 10. Multitone stacks

All listed frequencies played simultaneously, with Schroeder phases (minimizes crest factor for equal RMS). Three amplitude levels.

| Frequencies (Hz) | Guitar notes | Duration | Levels (dBFS peak) |
|---|---|---|---|
| 82.4, 165, 247, 330, 412, 495, 660, 880 | E2 + harmonic series | 30 s | −18, −9, −3 |

**Why Schroeder phases?** A naive sum of N sines has a crest factor of ~N (each peak adds). Schroeder phases (`φₖ = π·k(k−1)/N`) spread the peaks in time, reducing crest factor to ~3 dB — much closer to real music. This ensures the model sees a realistic amplitude distribution, not artificial spike-and-silence structure.

**Why these 8 frequencies?** They form a rough harmonic series above E2 — matching the spectral structure of a distorted guitar note, where the fundamental and its harmonics are all simultaneously driven into the nonlinearity.

---

### 11. Layered sections

Three multi-layer combos that combine stimulus types for maximum information density per second.

| Label | Content | Duration | Purpose |
|---|---|---|---|
| `layered_sweep_pink` | Log sweep + pink noise | 45 s | Measures FR while spectral fill provides context |
| `layered_multitone_pluck` | Multitone stack + impulses | 45 s | Dense harmonic + transient interaction |
| `layered_three_way` | Sine + pink noise + impulses | 45 s | Worst-case simultaneous stimuli |

**Why layer at all?** A static nonlinearity's output when presented with A+B is not the same as output(A) + output(B). Layered sections force the model to learn this non-superposition property. Keeping some pure-stimulus sections allows measurement plots to remain clean, while layered sections enrich the training distribution.

---

## Validation Signal — `val_signal_v1` (~1 min, seed 42)

Total ≈ 65 s · 38 sections · 96 000 Hz · seed 42

**Design principle: all frequencies are outside the training grid.** A model that learned only the statistics of the training set will fail here. A model that learned the transfer function (the physics) will generalize correctly.

### Frequency separation from training

| Dimension | Training | Validation | Separation |
|---|---|---|---|
| Sweep range | 20–20 000 Hz | 30–16 000 Hz | Different endpoints; model must interpolate |
| Stepped tones | 82.4, 110, 165, 220, 330, 440, 660, 880, 1320, 2640, 5280, 10 560 Hz | **146.8, 196, 246.9, 293.7, 392, 523.3, 784, 987.8 Hz** | Zero overlap — all guitar notes between the training grid |
| Two-tone pairs | Fixed: 110, 220, 440 Hz | **[196+294], [247+370], [392+523] Hz** | Different note pairs; different IM products |
| Pluck frequencies | 82.4, 110, 165, 220, 330, 440, 880 Hz | **146.8, 196, 246.9, 392, 659.3 Hz** | Different strings; different note positions |

---

### Val signal sections

#### Amplitude-stepped sweeps (15 s total)

| Label | Duration | Amplitude | Sweep range |
|---|---|---|---|
| `val_sweep_l-24` | 5 s | −24 dBFS peak | 30–16 000 Hz |
| `val_sweep_l-12` | 5 s | −12 dBFS peak | 30–16 000 Hz |
| `val_sweep_l-3` | 5 s | −3 dBFS peak | 30–16 000 Hz |

Same three amplitude levels as training, different frequency bounds. Provides clean frequency response and clipping-onset measures over a slightly different range.

#### Stepped tones (~12 s total)

All 8 frequencies are real guitar notes NOT present in the training stepped-tone grid.

| Freq (Hz) | Guitar note | In training? |
|---|---|---|
| 146.8 | D3 — D string open | No |
| 196.0 | G3 — G string open | No |
| 246.9 | B3 — B string open | No |
| 293.7 | D4 | No |
| 392.0 | G4 | No |
| 523.3 | C5 | No |
| 784.0 | G5 | No |
| 987.8 | B5 | No |

3 levels each (−18, −9, −3 dBFS) × 0.5 s/tone = 12 s. These are the guitar open strings and common chord tones that fall **between** the training tone grid — a genuinely held-back measurement set.

#### Pink noise (10 s)

| Label | Duration | RMS level |
|---|---|---|
| `val_pink_rms-18` | 5 s | −18 dBFS RMS |
| `val_pink_rms-6` | 5 s | −6 dBFS RMS |

Same RMS levels as two of the three training pink levels. Pink noise is stochastic — different seed state means different actual samples, so this is not a repeat of training even at the same level.

#### Two-tone IM segments (~10 s)

| Label | f₁ (Hz) | f₂ (Hz) | Guitar interval | Duration |
|---|---|---|---|---|
| `val_two_tone_f196_f294` | 196.0 (G3) | 293.7 (D4) | Perfect 5th | 3.33 s |
| `val_two_tone_f247_f370` | 246.9 (B3) | 369.9 (F#4) | Perfect 5th | 3.33 s |
| `val_two_tone_f392_f523` | 392.0 (G4) | 523.3 (C5) | Perfect 4th | 3.33 s |

All pairs are musically natural intervals (5ths and 4ths) between frequencies not in the training two-tone set. Training fixed tones were 110, 220, 440 Hz; none of the val pairs include those.

#### Impulse train (3 s)

Same timing pattern as training (0.05, 0.10, 0.20, 0.05, 0.15 s spacings). The impulse itself is amplitude-agnostic — this section tests time-domain recovery identically to the training version and provides a direct comparison point.

#### Decaying plucks (~10 s)

| Freq (Hz) | Guitar note | In training plucks? |
|---|---|---|
| 146.8 | D3 — D string open | No |
| 196.0 | G3 — G string open | No |
| 246.9 | B3 — B string open | No |
| 392.0 | G4 | No |
| 659.3 | E5 | No |

Same decay constant (τ = 0.4 s) as training, different pitches. Training plucks: E2, A2, E3, A3, E4, A4, A5. Val plucks: D3, G3, B3, G4, E5 — the notes between the training pitches.

---

## Guitar frequency reference

| Note | Freq (Hz) | Training? | Val? |
|---|---|---|---|
| E2 (low E open) | 82.4 | ✓ | — |
| A2 (A string open) | 110.0 | ✓ | — |
| D3 (D string open) | 146.8 | — | ✓ |
| E3 | 164.8 | ✓ (165) | — |
| G3 (G string open) | 196.0 | — | ✓ |
| A3 | 220.0 | ✓ | — |
| B3 (B string open) | 246.9 | — | ✓ |
| D4 | 293.7 | — | ✓ |
| E4 (high E open) | 329.6 | ✓ (330) | — |
| G4 | 392.0 | — | ✓ |
| A4 (concert A) | 440.0 | ✓ | — |
| C5 | 523.3 | — | ✓ |
| E5 | 659.3 | — | ✓ (pluck only) |
| A5 | 880.0 | ✓ | — |
| G5 | 784.0 | — | ✓ |
| B5 | 987.8 | — | ✓ |
| E6 | 1318.5 | ✓ (1320) | — |
| — | 2640.0 | ✓ | — |
| — | 5280.0 | ✓ | — |
| — | 10560.0 | ✓ | — |

Training and validation frequency grids are perfectly interleaved — the guitar note grid is a chromatic comb; training takes every other tooth; val takes the other teeth.

---

## Querying sections by label

The JSON manifest records every section's exact sample range. No hardcoded offsets needed anywhere.

```python
from pedal_model.signals.manifest import Manifest

m = Manifest("data/signals/val_signal_v1.json")

# Get metadata for one section
sec = m.get_section("val_sweep_l-3")
print(sec.start_sample, sec.end_sample, sec.duration_s)

# Load the audio for that section directly from disk
audio = m.load_section("val_sweep_l-3")   # shape (n_samples,), float32

# All swept-sweep sections
sweeps = m.sections_of_type("log_sine_sweep")

# Load full signal once, slice cheaply in memory
full = m.load_all()
plucks = [m.slice_section(s.label, full) for s in m.sections_of_type("decaying_pluck")]
```

---

## Regenerating the signals

The WAV files are gitignored (too large). The JSON manifests are committed and contain every parameter including the seed. To regenerate:

```bash
# Regenerate both from their committed manifests (bit-for-bit identical)
python generate_signal.py --from-manifest data/signals/train_signal_v1.json
python generate_signal.py --from-manifest data/signals/val_signal_v1.json

# Or generate fresh with defaults
python generate_signal.py --signal both
```
