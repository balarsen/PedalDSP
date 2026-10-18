# JHS Notaklon — Hardware Capture Results

Captures evaluated with `scripts/analyze_notaklon_capture.py`.
Output plots: `outputs/notaklon_captures/` (9 plots, 01–09).
Reference signal: `recordings/data/signals/train_signal_v2_48k.npy` (dry, 48 kHz, 642.5 s).

**Pedal:** JHS Notaklon (Klon Centaur clone, with optional "Shamrock" mid-boost + clip mod).  
**Capture date:** 2026-06-18. All captures at 48 kHz, 24-bit.  
**Reamp chain:** DAW → Radial ProRMP → Notaklon input → Focusrite ADC return.  
**Input level to pedal:** approximately −14 dBFS (DAW fader 0 dB through ProRMP).

No KlonSim reference curves are included. KlonSim was used in earlier exploration but is too simple to be a useful comparison for this pedal.

---

## Reamp chain baseline (bypass capture)

The bypass capture has the Notaklon footswitch set to BYPASS — the reamp chain passes through the pedal's true-bypass relay with no processing.

| Property | Value |
|---|---|
| Capture file | `recordings/data/notaklon/captures/bypass/wet_train_signal_v2_48k.wav` |
| Hardware lag | 45 smp (0.938 ms) |
| Peak level | −14.1 dBFS |
| RMS level | −26.6 dBFS |
| Clipped samples | 0 |

**Radial ProRMP character** (visible in plot 02): The bypass FR shows the expected ProRMP transformer behaviour — insertion loss around −5 to −7 dB relative to the direct dry signal, mild bass rolloff below 80 Hz, broad mid plateau, and HF rolloff beginning above 6–8 kHz. This is the reamp tilt that appears in all captures. Plot 03 removes this tilt by subtracting the bypass FR, isolating the pure pedal response.

---

## Alignment and lag

Plot 01 (`01_lag_alignment.png`) shows the cross-correlation on sweep_low (0.5–20 s, −24 dBFS) for all 7 captures. All peaks are sharp and well-defined, confirming the linear-regime xcorr gives reliable lag estimates. Using sweep_high (−3 dBFS, heavily clipped) gives broad, unreliable peaks; the initial G11 measurement (+296 smp) was caused by this error and has been corrected.

| Capture | Lag (smp) | Lag (ms) |
|---|---|---|
| bypass | 45 | 0.938 |
| G9 | 47 | 0.979 |
| G10 | 47 | 0.979 |
| G11-T8 | 48 | 1.000 |
| G11 | 47 | 0.979 |
| G12 | 46 | 0.958 |
| G12-S | 47 | 0.979 |

All lags are consistent within ±1 sample (~21 µs) — the spread is sub-xcorr-peak resolution at this window length. The lag is an audio interface ADC buffer delay, not dependent on the pedal gain or treble setting. All captures use per-capture lag in `params.json`.

---

## Per-capture analysis

### bypass

**File:** `recordings/data/notaklon/captures/bypass/`  
**Status: ✅ Good — use as FR reference for all comparisons**

- Reamp chain only, no pedal processing.
- Zero clipped samples; linear through all amplitude levels.
- Peak −14.1 dBFS is the highest of all captures — the pedal stages in all preset captures reduce output level relative to the reamp (lower peak = more compression/saturation from the gain stage).
- RMS −26.6 dBFS is the lowest of all captures — opposite of what might be expected, but makes sense: the Notaklon adds a mid-band boost and compresses the signal, raising RMS while reducing instantaneous peak headroom.

---

### V12-T12-G9-False (G9)

**File:** `recordings/data/notaklon/captures/V12-T12-G9-False/`  
**Knobs:** volume=12, treble=12, gain=9 o'clock, shamrock=off  
**KlonSim approx:** drive=2.107, clean_blend=0.68, treble_db=0.0, output_level=0.911  
**Status: ✅ Good — lowest gain, most dynamic range**

| Property | Value |
|---|---|
| Lag | 47 smp (0.979 ms) |
| Peak | −15.2 dBFS |
| RMS | −22.6 dBFS |
| Clipped samples | 0 |

G9 is the gentlest preset. The saturation curves (plot 05) show the widest spread between input levels at 220 Hz — meaning G9 retains more dynamic range before clipping than higher gain settings. The harmonic distortion plots (plot 06) show H2 approximately 13–15 dB below H1 at moderate levels — present but relatively mild. In the waveform overlay (plot 07), G9 still shows clearly clipped flat tops at −12 dBFS, indicating the Klon gain structure already drives the clipping stage hard even at minimum gain. The pedal-only FR (plot 03) shows a mid boost peaking around +14 dB at ~500 Hz.

---

### V12-T12-G10-False (G10)

**File:** `recordings/data/notaklon/captures/V12-T12-G10-False/`  
**Knobs:** volume=12, treble=12, gain=10 o'clock, shamrock=off  
**KlonSim approx:** drive=2.702, clean_blend=0.59, treble_db=0.0, output_level=0.911  
**Status: ✅ Good**

| Property | Value |
|---|---|
| Lag | 47 smp (0.979 ms) |
| Peak | −16.9 dBFS |
| RMS | −22.3 dBFS |
| Clipped samples | 0 |

One notch above G9. Peak drops slightly (−16.9 vs −15.2 dBFS) because more gain means earlier saturation — the clipping stage limits harder, reducing the peak relative to the reamp level. Saturation curves (plot 05) show slightly more compression than G9. Pedal FR mid peak ≈ +15 dB at ~450 Hz.

---

### V12-T8-G11-False (G11-T8)

**File:** `recordings/data/notaklon/captures/V12-T8-G11-False/`  
**Knobs:** volume=12, treble=8 o'clock, gain=11 o'clock, shamrock=off  
**KlonSim approx:** drive=3.464, clean_blend=0.50, treble_db=−5.33 dB, output_level=0.911  
**Status: ✅ Good — treble-cut variant of G11**

| Property | Value |
|---|---|
| Lag | 48 smp (1.000 ms) |
| Peak | −17.9 dBFS |
| RMS | −23.5 dBFS |
| Clipped samples | 0 |

Treble at 8 o'clock (two hours left of noon) corresponds to approximately −5.3 dB high-frequency cut in KlonSim terms. Plot 08 (treble comparison) shows the effect clearly: G11-T8 rolls off aggressively from ~400 Hz, reaching −15 dB relative to G11 by 10 kHz. The rolloff looks like a 1st-order shelf starting around 400–500 Hz. This is the lowest peak and lowest RMS of all presets, consistent with the treble cut removing upper-mid energy.

The driven FR (plot 04) shows G11-T8 (purple) sitting well below the cluster of other presets above 1 kHz — the treble cut carries through to the driven regime too.

---

### V12-T12-G11-False (G11)

**File:** `recordings/data/notaklon/captures/V12-T12-G11-False/`  
**Knobs:** volume=12, treble=12, gain=11 o'clock, shamrock=off  
**KlonSim approx:** drive=3.464, clean_blend=0.50, treble_db=0.0, output_level=0.911  
**Status: ✅ Good**

| Property | Value |
|---|---|
| Lag | 47 smp (0.979 ms) |
| Peak | −17.1 dBFS |
| RMS | −22.2 dBFS |
| Clipped samples | 0 |

The reference gain setting at treble noon. Mid boost peak ≈ +16 dB at ~450 Hz in the pedal-only FR (plot 03). Gain vs frequency (plot 09) shows this is the clearest demonstration of the Klon's compression: even at the lowest input level (−24 dBFS, purple), the mid-band output is already heavily compressed. The gap between the −24 dBFS and −3 dBFS curves is small — the pedal is close to saturation across the full input range.

---

### V12-T12-G12-False (G12)

**File:** `recordings/data/notaklon/captures/V12-T12-G12-False/`  
**Knobs:** volume=12, treble=12, gain=12 o'clock, shamrock=off  
**KlonSim approx:** drive=4.441, clean_blend=0.41, treble_db=0.0, output_level=0.911  
**Status: ✅ Good — highest standard gain**

| Property | Value |
|---|---|
| Lag | 46 smp (0.958 ms) |
| Peak | −17.3 dBFS |
| RMS | −22.0 dBFS |
| Clipped samples | 0 |

Maximum gain without the Shamrock mod. The saturation curves (plot 05) show the tightest cluster of any gain setting — nearly flat output across all input levels, indicating the most severe compression. Plot 09 (gain vs freq) confirms: the spread between input levels is smallest here, meaning essentially all dynamic range has been removed by the clipping stage. Harmonic distortion (plot 06) shows the highest H2 levels of the standard gain range. This is the most "all-in" Klon overdrive setting in the capture set.

---

### V12-T12-G12-True (G12-S, Shamrock mod on)

**File:** `recordings/data/notaklon/captures/V12-T12-G12-True/`  
**Knobs:** volume=12, treble=12, gain=12 o'clock, shamrock=on  
**KlonSim approx:** drive=4.441, clean_blend=0.41, treble_db=0.0, output_level=0.911, shamrock=true  
**Status: ✅ Good — Shamrock mod active**

| Property | Value |
|---|---|
| Lag | 47 smp (0.979 ms) |
| Peak | −16.7 dBFS |
| RMS | −21.3 dBFS |
| Clipped samples | 0 |

The JHS Shamrock mod adds a mid boost at ~800 Hz and uses harder/asymmetric clipping diodes. Highest RMS (−21.3 dBFS) of any preset — the mid boost adds more average energy even after saturation. Peak is slightly higher than G12 (−16.7 vs −17.3 dBFS) due to the post-clip gain boost in the Shamrock path.

**Critical finding from plot 08 (right panel):** G12 and G12-S show **nearly identical linear frequency responses** — the curves are indistinguishable from 60 Hz to 8 kHz, with only marginal divergence above 10 kHz. The Shamrock mod does not change the pedal's linear character at all. Its effect is entirely nonlinear: different clipping diode configuration, modified clean blend, and overall output boost. To capture the Shamrock character in a model, it must be captured via the nonlinear signal path (the sweep_high and tone/chord sections of the training signal), not inferred from the linear FR.

---

## Plot commentary

### 01 — Lag / alignment

All 7 xcorr peaks are sharp and unambiguous. The previous G11 error (+296 smp) came from running xcorr on sweep_high (−3 dBFS) where hard clipping produces a broad plateau rather than a clean peak. Using sweep_low (0.5–20 s, −24 dBFS, linear regime) resolves the issue across all captures. Variation within ±1 sample is within the resolution of a 19.5-second xcorr window.

### 02 — Linear frequency response (absolute, with reamp tilt)

All Notaklon presets share the characteristic Klon bandpass EQ: HPF around 80 Hz (input coupling cap), mid boost broadly centred on 400–600 Hz, and HF rolloff above 2–3 kHz. The bypass (dashed) shows the ProRMP baseline — the pedal adds approximately +5 to +8 dB in the 300–800 Hz mid band relative to the reamp. G11-T8 visibly separates from the gain-stack above 500 Hz. G9 is slightly brighter than G10–G12 at high frequencies because it saturates less and more HF content passes through.

### 03 — Pedal-only frequency response (bypass normalised)

The cleanest view of the Notaklon's EQ character. Reamp tilt removed.

- HPF at 80 Hz: below this the signal drops sharply — the Klon's input HPF cap is working as expected.
- Mid peak: all presets peak around 400–600 Hz with +14 to +17 dB above bypass. Peak gain increases modestly with gain setting.
- HF rolloff: treble at noon (all presets except G11-T8) gives a natural HF rolloff above 1–2 kHz, reaching −10 to −15 dB by 10 kHz. This is the Klon's inherent treble character, not a cut.
- G11-T8 treble cut: substantially more HF rolloff than G11 starting at 400 Hz. Down −15 dB at 10 kHz. Confirms the treble knob is an effective tone-shaping control.
- G12-S vs G12: Shamrock mod makes no difference in the linear regime — curves are identical.

### 04 — Driven frequency response (sweep_high, −3 dBFS)

At −3 dBFS all gain settings **converge** to the same output level. This is the Klon's most distinctive characteristic: a fixed saturation ceiling that limits the output regardless of input level or gain setting. All standard presets (G9–G12, G12-S) sit in a tight cluster around −17 to −18 dBFS output across 60 Hz–6 kHz. G11-T8 sits 5–10 dB below the cluster above 1 kHz due to the treble cut. Bypass (dashed) sits above all of them at high frequencies — the reamp chain has better HF extension than the pedal's saturated output.

Above 20 kHz: all traces show an artefact spike — FFT boundary effect, disregard.

### 05 — Saturation curves

The most informative nonlinear diagnostic. At every input level from −25 to −3 dBFS, the Notaklon output stays compressed in a narrow range (~−19 to −18 dBFS). The pedal is essentially a **soft limiter** that adds harmonic colour. Even the lowest gain (G9) compresses the output to within 4 dB regardless of input.

Bypass follows the linear 1:1 line closely — confirming the reamp itself is linear at these levels. The Klon's compression is a property of the pedal, not the reamp chain.

At 220 Hz and low input levels (−25 dBFS), G9 sits 3–4 dB below G10–G12. At 880 Hz the convergence is faster (all 4 gain settings land at the same output even at −25 dBFS input). This suggests the Klon's gain is higher in the midrange than at low frequencies — consistent with the bandpass mid-boost confirmed in plots 02 and 03.

### 06 — Harmonic distortion (440 Hz)

Four rows (G9 → G12) × five input levels. Key observations:

- **H2 dominant**: even-order harmonics dominate over odd-order, confirming asymmetric clipping. H2 is typically 10–15 dB below H1.
- **H3 present, H4/H5 small**: the distortion is soft enough that higher harmonics roll off quickly.
- **Level dependence**: H2/H3 increase relative to H1 as input level rises (more drive → more saturation → more harmonics). This is the expected behaviour of a soft-clipping overdrive.
- **Gain dependence**: G12 shows the highest harmonic levels at any given input — more distortion per unit input, as expected.
- The character is consistent across all gain settings — the ratio between H2 and H3 doesn't change much with gain, only the absolute distortion level does. This suggests the same clipping mechanism operates at all gain settings; only the drive level changes.

### 07 — Waveform overlay (440 Hz tones)

- **−12 dBFS**: All Notaklon presets show visibly flat-topped waveforms while the dry signal and bypass track a clean sine. The Klon is already well into saturation at −12 dBFS input — consistent with the saturation curves.
- **−6 dBFS**: Severe clipping. Output waveform approaches square wave with rounded shoulders — typical of soft-diode clipping.
- **−3 dBFS**: Hard square-wave output from all Notaklon presets. The bypass shows minor rounding at the peaks (ProRMP transformer saturation at maximum input), confirming the reamp is linear for all practical purposes at −12 dBFS and above.
- All four gain settings (G9–G12) produce essentially identical waveform shapes at each level — the saturation ceiling is the same regardless of gain knob position.

### 08 — Treble cut and Shamrock mod

**Treble (G11 vs G11-T8):** The treble control at 8 o'clock produces a smooth 1st-order-style shelf rolloff starting around 400–500 Hz. By 10 kHz the cut is approximately −15 dB relative to G11 (treble at noon). This is a significant tone shaping tool — the treble knob effectively converts the Klon from a bright overdrive to a dark/warm overdrive.

**Shamrock (G12 vs G12-S):** The linear frequency responses are indistinguishable from 60 Hz to 8 kHz. The Shamrock mod operates entirely in the nonlinear domain and cannot be detected from the linear sweep. This has an important implication for modelling: a model trained only on linear data would fail to distinguish G12 from G12-S. The two captures must be differentiated by their nonlinear behaviour (clipping shape, harmonic content, dynamic response).

### 09 — Gain vs frequency at each input level

Each panel (G9, G10, G11, G12) shows gain vs frequency with 5 input levels coloured from dark (−24 dBFS) to bright (−3 dBFS). The gap between the topmost curve (−24 dBFS, most headroom) and the bottommost (−3 dBFS, most compressed) quantifies dynamic range.

- **G9**: widest gap — most dynamic range, least compression.
- **G10 / G11**: progressively tighter — mid-band is compressed more at every input level.
- **G12**: tightest cluster — nearly zero dynamic range. All five input levels converge to essentially the same output gain curve. The pedal is functioning as a hard limiter at this setting.

The mid-hump shape (bandpass peak at 600–900 Hz) is present at all gain levels in the −24 dBFS curves. As input level increases, the hump flattens and the output converges to a frequency-independent saturation ceiling.

---

## Summary table

| Capture | Gain | Treble | Shamrock | Peak (dBFS) | RMS (dBFS) | Lag (smp) | Status |
|---|---|---|---|---|---|---|---|
| bypass | — | — | — | −14.1 | −26.6 | 45 | ✅ Reference |
| G9 | 9 o'clock | noon | off | −15.2 | −22.6 | 47 | ✅ Good |
| G10 | 10 o'clock | noon | off | −16.9 | −22.3 | 47 | ✅ Good |
| G11-T8 | 11 o'clock | 8 o'clock | off | −17.9 | −23.5 | 48 | ✅ Good |
| G11 | 11 o'clock | noon | off | −17.1 | −22.2 | 47 | ✅ Good |
| G12 | noon | noon | off | −17.3 | −22.0 | 46 | ✅ Good |
| G12-S | noon | noon | on | −16.7 | −21.3 | 47 | ✅ Good |

All captures: 48 kHz, 24-bit, 642.5 s, 0 clipped samples. All usable for model training.

**Gain trend:** Peak drops from −15.2 (G9) to −17.3 dBFS (G12) as gain increases, because higher gain drives the clipping stage harder and reduces peak headroom. RMS barely changes (−22.0 to −22.6 dBFS for standard presets) because the compression redistributes energy.

**Shamrock RMS:** G12-S has the highest RMS (−21.3 dBFS) due to the Shamrock mid boost and output gain increase.

**Treble cut RMS:** G11-T8 has the lowest RMS (−23.5 dBFS) — removing treble removes energy.

---

## Known analysis caveats

1. **Reamp tilt in absolute FR plots:** Plot 02 shows absolute levels including the ProRMP transformer tilt. The tilt is real (any model must handle it), but it makes gain comparisons between captures misleading at extreme frequencies. Always use plot 03 (bypass-normalised) to isolate pure pedal EQ character.

2. **Shamrock mod not visible in linear regime:** Plot 08 right panel confirms that G12 and G12-S are spectrally identical in the linear sweep. The Shamrock mod only manifests in nonlinear behaviour. A model that over-relies on spectral features in the linear regime will not distinguish these two presets.

3. **All lags shared (ADC buffer):** The ±1 sample variation in lag (45–48 smp) is not a pedal property — it is an audio interface ADC buffer delay and is consistent across captures. All params.json files record the per-capture lag.

4. **Extra tail in wet files:** All wet files are 76,140 samples (1.586 s) longer than the dry due to DAW export rounding. Alignment is correct for the shared length: `wet[:dry_n_samples]` aligned with `dry`. The tail contains reverberation decay and should be excluded from training windows.

5. **Saturation onset at −14 dBFS input:** The reamp delivers approximately −14 dBFS to the Notaklon input. At this level, even G9 is already near the clipping stage onset. The training signal covers −24 to −3 dBFS sections, so the data spans the full range from mild compression to hard clipping. The linear-regime sections (−24 to −18 dBFS in the signal) will be lightly processed; the high-amplitude sections are the primary distortion data.

6. **xcorr must use sweep_low:** The previous erroneous G11 lag (+296 smp) came from running xcorr on the heavily-clipped sweep_high section. Hard clipping spreads the xcorr peak over hundreds of samples, making it unreliable. Always use the linear-regime section (0.5–20 s, −24 dBFS) for lag estimation.

---

## Pending

- **Model training:** All 7 captures (excluding bypass) are ready for neural model training. The bypass capture can be used as the reamp reference if a model should learn to factor out the reamp tilt.
- **Shamrock nonlinear analysis:** A dedicated distortion/waveform analysis comparing G12 vs G12-S at multiple drive levels would quantify the Shamrock mod's nonlinear signature. The current plots show it's there (slightly higher RMS) but don't isolate the effect cleanly.
- **G12-S harmonic comparison:** Not included in plot 06 (which covers only the gain subset G9–G12). Worth adding a G12 vs G12-S row to show the Shamrock distortion character directly.
- **Additional gain settings:** The current capture set covers gain at 9/10/11/12 o'clock. The lower half of the gain range (7–8 o'clock) is not captured. These settings would produce less clipping and potentially reveal more of the Klon's clean blend character.
- **Treble sweep:** Only two treble settings captured (noon and 8 o'clock). A 3–4 point treble sweep at fixed gain would fully characterise the treble tone stack.
