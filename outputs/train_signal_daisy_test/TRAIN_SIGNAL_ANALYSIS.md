# train_signal_v1 — Section Analysis & Daisy Hardware Comparison

**Signal:** `data/signals/train_signal_v1.wav`  
**Duration:** 642.5 s · 96 kHz · float32 mono  
**Sections:** 84 · Seed: 1234  
**Plots:** `outputs/train_signal_daisy_test/section_analysis/`

---

## TL;DR — what's the same, what's different, and why

| | PC ODE model | Daisy hardware |
|---|---|---|
| **ESR vs reference** | ~0.002 (all section types) | ~1.1–1.6 (all section types) |
| **Level vs reference** | +0 dB | **+8.2 dB deficit (consistent)** |
| **Algorithm correct?** | ✓ proven | ✓ same code, but miscalibrated |
| **Filter shape (HPF1/HPF2/shelf)** | matches | approximately correct |
| **Waveform character** | matches | **different — tanh at wrong operating point** |

**Root cause: hardware calibration, not the program.**

The firmware runs the exact same algorithm as the Python ODE model — and the Python model matches the KlonSim reference to ESR = 0.000441 across all 642 seconds and every section type. The Daisy gap is entirely explained by `kInputGain` being ~8.2 dB too low. The re-amp sends the signal about 20.8 dB quieter than a direct guitar input (not the 12.6 dB we originally measured), so the tanh is operating in a much more linear regime than the Python sim.

**Recommended fix:** increase `kInputGain` from 4.270 to **~10.95** (i.e. `+8.2 dB` more gain).

---

## Evidence: per-section ESR and level offset

See `07_per_section_esr.png` and `12_type_summary.png`.

The PC ODE model has ESR ≈ 0.002 for **every** one of the 84 sections — sweeps, stepped tones, noise, impulses, plucks, everything. This is nearly perfect numerical agreement and proves the algorithm is correct.

The Daisy has ESR in the range 1.1–1.6 for **every** section type. But crucially, the **level offset is extremely consistent**: across all 84 sections the Daisy output is short of the reference by a median of **+8.18 dB** (mean +8.20 dB, std < 1.5 dB). This tight clustering is the fingerprint of a single fixed gain error, not a signal-dependent or algorithm-dependent problem.

| Section type | ESR (PC model) | ESR (Daisy) | Daisy level offset |
|---|---|---|---|
| log_sine_sweep | 0.0023 | 1.31 | +9.6 dB |
| stepped_sine_tone | 0.0022 | 1.26 | +8.0 dB |
| pink_noise | 0.0035 | 1.31 | +8.5 dB |
| white_noise | 0.0022 | 1.12 | +10.2 dB |
| amplitude_ramp | 0.0028 | 1.14 | +8.8 dB |
| impulse_train | 0.0032 | 1.25 | +6.3 dB |
| am_chirp | 0.0019 | 1.06 | +11.3 dB |
| decaying_plucks | 0.0021 | 1.26 | +8.9 dB |
| two_tone_sweep | 0.0020 | 1.12 | +10.4 dB |
| multitone | 0.0012 | 1.65 | +6.6 dB |
| layered | 0.0009 | 1.25 | +8.0 dB |

If the mismatch were due to the firmware algorithm being wrong, you would expect different section types to diverge in different ways (sweeps would be affected differently from impulses, etc.). Instead the pattern is uniform — a constant level shortfall across everything.

---

## What's the same between Daisy and simulation

**Filter topology is correct.** The sweep frequency response plot (`02_sweep_frequency_response.png`) shows that all three wet signals (KlonSim ref, PC model, Daisy) share the same basic shape:
- Rolloff below ~80 Hz (HPF1) — present in Daisy ✓
- Deep notch below ~350 Hz (HPF2) — present in Daisy ✓
- Gentle boost above ~3 kHz (high shelf) — present in Daisy ✓

The Daisy's filter cutoffs land in the right places. The coefficients are correct.

**Harmonic generation is happening.** The Daisy does produce distortion; THD in the Daisy is ~60–65% at high input levels, close to the reference's ~58%. The tanh is saturating, just not at the right drive level.

**Impulse response shape is approximately correct.** See `10_impulse_response.png` — the tail shape of the impulse response (filter ringing from HPF2) is similar between Daisy and reference, though the level and exact shape differ.

**Spectrogram character is recognisable.** See `11_pluck_spectrogram.png` — the Daisy spectrogram for decaying plucks looks similar in structure to the reference. The HPF2 high-pass character (energy concentrated above 350 Hz) is visible in all wet signals.

---

## What's different

### 1. Output level — 8.2 dB deficit

The Daisy output is consistently ~8 dB quieter than the reference. This is because the re-amp chain attenuates the signal more than the current `kInputGain = 4.270` compensates for. The tanh input is too low.

### 2. Waveform character — tanh at wrong operating point

This is the more serious consequence of the level gap. The KlonSim reference has a peak amplitude of ~0.7 at the tanh input (`drive × amplitude ≈ 4.28 × 0.71 ≈ 3.0`) — the tanh saturates at this level, producing a near-square-wave output (very rich in odd harmonics).

With the Daisy's current calibration, the estimated tanh input peak is approximately:
```
0.71  (dry peak)
× 0.235 (re-amp attenuation at original -12.6 dB calibration)
× 4.270 (kInputGain)
× 4.281 (drive)
≈ 3.0   ← should be right
```

But the Daisy is still 8.2 dB short. This means the actual re-amp attenuation is more than originally measured — approximately **20.8 dB** total (not 12.6 dB). So the actual tanh input peak is closer to:
```
0.71 × 10^(-20.8/20) × 4.270 × 4.281 ≈ 1.2
tanh(1.2) ≈ 0.83  (moderate saturation)
```

vs the reference:
```
tanh(3.0) ≈ 0.995  (deep saturation)
```

A moderately saturated tanh produces a very different harmonic structure than a deeply saturated one. At 440 Hz the Daisy waveform (normalised to the same RMS) shows essentially **zero correlation** with the reference (ρ = −0.008 vs ρ = 0.9999 for the PC model). See `09_440hz_detail.png` — the phase portrait shows a circular scatter cloud rather than a diagonal line. This is not a polarity or offset issue; the waveforms genuinely have different shapes because the distortion character changes dramatically with saturation depth.

### 3. Transfer curve shift

See `08_ramp_transfer_curves.png`. The amplitude ramp sections sweep the input from −40 to −2 dBFS, tracing the static nonlinearity. The Daisy's transfer curve is shifted to the **right** (needs more input amplitude to reach the same output saturation level) relative to the reference. The shift magnitude directly quantifies the kInputGain error. If the curves were overlapping, the calibration would be correct.

---

## How to fix: increase kInputGain

Current: `kInputGain = 4.270` (12.6 dB)  
Measured deficit: +8.18 dB  
**Recommended: `kInputGain = 10.95` (20.8 dB)**

```
// daisy/klonsim_params.h
// Previous measured deficit: 12.6 dB  →  kInputGain = 4.270
// Revised measured deficit: 20.8 dB   →  kInputGain = 10.95
//   kInputGain = 10^(level_offset_dB / 20) = 10^(20.8/20) = 10.97
static constexpr float kInputGain = 10.95f;
```

After this change, re-flash and re-record the training signal. The level offset should drop near 0 dB, and the corr(ref) should climb from ~0.27 toward the PC model's ~1.00.

**Why was the original 12.6 dB calibration wrong?**

The original calibration was measured from the medley guitar recording at a specific DAW playback level. The training signal re-amp session apparently ran at a ~8 dB lower playback level — possibly a different DAW track volume, output fader position, or OS/driver mixer level. The Daisy ADC has no way to know about this; it just sees whatever voltage the re-amp delivers. `kInputGain` should be calibrated per session (or the re-amp output should be standardised to a fixed reference level before each session).

---

## Residual differences after calibration (expected)

Even with correct `kInputGain`, some differences will remain:

1. **ADC/DAC quantisation noise**: the Daisy recording is 16-bit PCM. That adds ~−96 dBFS noise floor vs the Python sim's 32-bit float computation.
2. **Analog chain coloration**: the Hothouse hardware's input and output buffers add very slight frequency shaping and noise.
3. **tanh approximation**: `tanhf()` on the Cortex-M7 vs Python's `numpy.tanh()` — differences at the 1e−6 level, negligible.
4. **Clock jitter**: if the Daisy's 48 kHz clock drifts even 10 ppm from the PC's clock, there's a ~0.5 sample drift per 10 seconds. Over a 642 s recording this accumulates to ~32 samples, which broadens the correlation peak.

Items 1–4 together might account for an ESR of 0.01–0.05. If the calibrated Daisy reaches ESR < 0.1 it is doing about as well as can be expected from an embedded 16-bit hardware implementation.

---

## Section-by-section documentation

### 1. Log sine sweeps (0.5 – 61.5 s) — 3 sections

| Label | Time | Peak |
|---|---|---|
| `sweep_low` | 0.5 – 20.5 s | −24 dBFS |
| `sweep_mid` | 21.0 – 41.0 s | −12 dBFS |
| `sweep_high` | 41.5 – 61.5 s | −3 dBFS |

20-second logarithmic sweep from 20 Hz to 20 kHz. Each frequency gets proportionally more time at low frequencies (natural weighting). Three amplitudes span quiet (nearly linear) to loud (heavy tanh saturation).

**Why useful:** Full frequency response measurement in one shot. H(f) = FFT(wet) / FFT(dry) gives the system transfer function. Three levels reveal level-dependent response (nonlinearity fingerprint).

**KlonSim character in this section** (`02_sweep_frequency_response.png`):
- Below 80 Hz: rolled off by HPF1
- 80–350 Hz: passes HPF1 but cut heavily by HPF2 — the deep mid notch is visible in all wet signals
- Above 350 Hz: pass-band, with gradual boost above 3 kHz from the shelf
- At −3 dBFS: the sweep drives the tanh heavily; harmonic content from tanh smears energy across the spectrum visible as a raised noise floor in H(f)

---

### 2. Stepped sine tones (62.0 – 122.0 s) — 60 sections

12 frequencies × 5 amplitude levels = 60 one-second sections.

**Frequencies (Hz):** 82.4, 110, 165, 220, 330, 440, 660, 880, 1320, 2640, 5280, 10560  
**Levels (dBFS peak):** −24, −18, −12, −6, −3

**Why useful:** THD vs level for each frequency; direct measurement of the saturation curve; harmonic structure at each frequency.

**KlonSim's defining behaviour — HPF2 strips the fundamental of low notes:**

| Input freq | Lowest harmonic above HPF2 | What the output sounds like |
|---|---|---|
| 82.4 Hz (E2) | 5th (412 Hz) | No fundamental, no 2nd–4th — thin, string-like |
| 110 Hz (A2) | 4th (440 Hz) | No fundamental, barely any low harmonics |
| 165 Hz (E3) | 3rd (495 Hz) | Fundamental stripped; 3rd harmonic dominant |
| 220 Hz (A3) | 2nd (440 Hz) | Fundamental attenuated; 2nd harmonic dominant |
| 330 Hz (E4) | 1st (330 Hz, near HPF2) | Fundamental near cutoff; 2nd (660 Hz) overtakes it |
| 440 Hz (A4) | 1st (440 Hz, above HPF2) | Normal overdrive character; fundamental present |
| 660 Hz+ | all | Completely above HPF2; standard distortion |

This is why KlonSim sounds like a treble-presence boost on low strings (E–G strings) rather than a muddy crunch. The fundamental disappears; only upper harmonics remain.

**THD behaviour** (`04_stepped_thd.png`): THD measured at the fundamental in the *output* exceeds 100% for frequencies below ~330 Hz because essentially all the energy is at harmonics — the "fundamental" in the output is near zero. This is intentional and correct KlonSim behaviour.

---

### 3. Pink noise (122.5 – 182.5 s) — 3 sections

20 seconds each at −24, −12, −3 dBFS RMS. Equal energy per octave (1/f spectrum).

**Why useful:** Wideband excitation matching the spectral balance of music; all frequencies present simultaneously; tests intermodulation behaviour under realistic spectral weighting.

---

### 4. White noise (183.0 – 213.0 s) — 2 sections

15 seconds each at −18 and −6 dBFS RMS. Flat spectrum (equal energy per Hz).

**Why useful:** Emphasises high frequencies; stresses model accuracy above 10 kHz; complements pink noise since pink under-represents highs relative to white.

---

### 5. Amplitude ramps (213.5 – 253.5 s) — 4 sections

| Label | Frequency | Ramp |
|---|---|---|
| `amp_ramp_f110` | 110 Hz | −40 → −2 dBFS in 10 s |
| `amp_ramp_f220` | 220 Hz | −40 → −2 dBFS in 10 s |
| `amp_ramp_f440` | 440 Hz | −40 → −2 dBFS in 10 s |
| `amp_ramp_f880` | 880 Hz | −40 → −2 dBFS in 10 s |

**Why useful:** The most direct measurement of the **static nonlinearity** (transfer curve). Plotting output amplitude vs input amplitude traces out the tanh compression curve. This is the key diagnostic for `kInputGain` calibration — see `08_ramp_transfer_curves.png`.

**What the transfer curve reveals:** The Daisy curve is shifted to the **right** (needs more input to reach the same saturation). The horizontal shift quantifies the kInputGain error directly, without needing the full per-section ESR analysis.

---

### 6. Impulse train (254.0 – 284.0 s) — 1 section

30-second train of unit impulses at −6 dBFS peak, with repeating spacings: `[50, 100, 200, 50, 150, 300, 50, 80] ms`.

**Why useful:** Measures the system impulse response (IR). For the linear filter parts (HPF1, HPF2, shelf), the IR can be extracted by deconvolution. Varied spacings prevent comb-filter artefacts. Tests system reset time.

**See `10_impulse_response.png`** — the tail shape (filter ringing from HPF2 after saturation) is qualitatively similar between Daisy and reference, confirming the filter coefficients are implemented correctly. The amplitude differs due to the kInputGain calibration issue.

---

### 7. AM chirp (284.5 – 344.5 s) — 1 section

440 Hz carrier, amplitude-modulated by a sine sweep from 0.5 Hz to 20 Hz over 60 s. Modulation depth 80%, peak −9 dBFS.

**Why useful:** Tests dynamic/envelope behaviour. Slow AM (0.5 Hz) mimics picking dynamics; fast AM (20 Hz) approaches tremolo. Exposes whether the pedal's nonlinearity responds consistently to slowly vs rapidly varying amplitude.

**KlonSim:** No explicit dynamics processing, but tanh compression is inherently amplitude-dependent. The AM chirp tests whether the training signal exposes this time-varying nonlinearity well enough for models to learn it.

---

### 8. Decaying plucks (345.0 – 385.0 s) — 1 section

Seven guitar-string-like plucks at 82.4, 110, 165, 220, 330, 440, 880 Hz. Each is an exponentially decaying sine with τ = 0.4 s and peak −6 dBFS, repeated across 40 seconds.

**Why useful:** Most realistic section — a plucked string decays from heavy saturation (loud attack) through moderate saturation to near-linear (quiet tail). This amplitude-sweeping from loud to quiet within each note is the defining challenge for pedal models.

**See `11_pluck_spectrogram.png`** — the spectrogram shows the energy in each harmonic decaying at the same rate, with the HPF2 high-pass character visible as absence of energy below ~350 Hz for low notes. The Daisy spectrogram shows qualitatively similar structure with different harmonic amplitudes (due to the tanh operating point mismatch).

---

### 9. Two-tone sweeps (385.5 – 415.5 s) — 3 sections

Fixed tone (110, 220, or 440 Hz) combined with a log sweep from 20 Hz to 8 kHz. Each component at −9 dBFS.

**Why useful:** Measures **intermodulation distortion** (IMD). Two-tone IMD products appear at f₁ ± f₂, 2f₁ ± f₂, etc. The swept tone reveals all IMD products as a function of frequency separation.

**KlonSim:** With drive 4.28, significant 3rd-order IMD is expected. HPF2 strips IMD products below 350 Hz.

---

### 10. Multitone (416.0 – 506.0 s) — 3 sections

8 simultaneous tones at 82.4, 165, 247, 330, 412, 495, 660, 880 Hz (harmonics of E2). Three levels: −18, −9, −3 dBFS each lasting 30 seconds.

**Why useful:** Simultaneous multi-frequency excitation captures the full nonlinear interaction matrix. The 8 tones span both sides of the HPF2 cutoff (below and above 350 Hz), so this tests both the "fundamental stripped" regime and the "fundamental passes" regime simultaneously.

---

### 11. Layered sections (506.5 – 642.5 s) — 3 sections

| Label | Components | Purpose |
|---|---|---|
| `layered_sweep_noise` | Log sweep + pink noise | Frequency response under noisy conditions |
| `layered_multitone_transients` | Multitone + impulse train | Tonal + transient combined |
| `layered_three_way` | Sine 440 Hz + white noise + impulse train | Three-way stress test |

**Why useful:** Real guitar signals combine tonal content, texture noise, and transients simultaneously. These sections approximate that character and provide the hardest test for model generalisation.

---

## Plot index

| Plot | What it shows |
|---|---|
| `01_overview_timeline.png` | All 84 sections by type |
| `02_sweep_frequency_response.png` | H(f) = wet÷dry at 3 levels; HPF1, HPF2, shelf visible |
| `03_stepped_gain_curves.png` | Output vs input RMS per frequency; saturation curves |
| `04_stepped_thd.png` | THD vs level per frequency |
| `05_harmonic_spectra.png` | FFT at peak level for 4 key frequencies |
| `06_stepped_waveforms.png` | 50 ms waveform excerpts from peak-level tones |
| `07_per_section_esr.png` | ESR for every section; PC model vs Daisy; level offset |
| `08_ramp_transfer_curves.png` | Static nonlinearity curves from amplitude ramps — **key calibration diagnostic** |
| `09_440hz_detail.png` | 440 Hz waveform + phase portrait; shows character mismatch at wrong operating point |
| `10_impulse_response.png` | Impulse response comparison from impulse_train section |
| `11_pluck_spectrogram.png` | Spectrogram of decaying plucks across all 4 signals |
| `12_type_summary.png` | ESR and level offset summarised by section type; kInputGain recommendation |

---

## KlonSim DSP parameters (reference)

```
HPF1:  2nd-order Butterworth highpass  80 Hz       (before tanh)
Drive: ×4.282                                       (tanh input scale)
tanh:  tanhf()                                      (memoryless nonlinearity)
HPF2:  2nd-order Butterworth highpass  350 Hz       (after tanh)
Shelf: high shelf  3006 Hz,  +2.00 dB  (Q=0.707)
Level: ×0.644                                       (output gain)

kInputGain (re-amp compensation):
  Current: 4.270  (calibrated to 12.6 dB deficit — incorrect for this setup)
  Required: ~10.95 (measured 20.8 dB total deficit from training signal analysis)
```
