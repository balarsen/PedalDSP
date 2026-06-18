# GE-7 Hardware Capture Results

Captures evaluated with `scripts/analyze_ge7_capture.py`.
Reference signals: `data/signals/train_signal_v2.wav` (dry, 96 kHz), `train_signal_v2_reamp only.wav` (bypass), `data/ge7/captures/preset_flat/GE7-Flat2 - flat 2.wav` (flat reference).

**Analysis convention:** all gain figures are *preset relative to flat2* (both captured at −10 dB DAW playback), so the playback-level offset cancels and the numbers reflect pure GE-7 EQ change. Comparison target is the Ge7Sim peaking-EQ cascade with Q = 1.4.

---

## Reamp chain baseline

| Property | Value |
|---|---|
| DAW playback fader | −10 dB (applied to all GE-7 captures) |
| Reamp insertion loss @ 82 Hz | −8.8 dB |
| Reamp frequency tilt | +4 dB rise from 82 Hz → 10.5 kHz |
| Reamp saturation onset | ~−12 dBFS input (at 0 dB playback) → −12 dBFS at −10 dB playback is safe |
| Noise floor (reamp silence) | −108.5 dBFS RMS — no hum |
| Flat GE-7 insertion loss | −1.2 dB broadband |

---

## preset_flat (v2 capture — GE7-Flat2)

**File:** `data/ge7/captures/preset_flat/GE7-Flat2 - flat 2.wav`

**Status: ✅ GOOD — use as reference for all comparisons**

- Gain across all 12 frequencies at −18, −12, −6, −3 dBFS: consistent to within 0.25 dB → **perfectly linear**
- Insertion loss: −1.2 dB flat across 82 Hz–10.5 kHz (matches expectation for op-amp buffer losses)
- No saturation at any amplitude level
- This capture is used as the denominator for all preset comparisons; any error here propagates everywhere

---

## preset_smile

**File:** `data/ge7/captures/preset_smile/wet_train_signal_v2.wav`
**Preset:** +8/+4/−2/−6/−6/+4/+8 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ GOOD — usable for model training**

### Measured vs expected EQ response (@ −18 dBFS, relative to flat)

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | +9.24 | +6.36 | +2.88 |
| 110 | +8.18 | +8.11 | **+0.08** |
| 165 | +4.38 | +5.32 | −0.94 |
| 220 | +2.11 | +4.18 | −2.06 |
| 330 | −0.83 | −0.63 | −0.20 |
| 440 | −3.22 | −2.72 | −0.50 |
| 660 | −6.34 | −5.45 | −0.89 |
| 880 | −6.53 | −6.91 | **+0.38** |
| 1320 | −4.49 | −5.92 | +1.44 |
| 2640 | +0.97 | +1.65 | −0.69 |
| 5280 | +5.55 | +6.59 | −1.04 |
| **10560** | **+7.89** | **+2.00** | **+5.88** ⚠️ |

**RMS error: 2.1 dB   Max error: 5.88 dB (10.56 kHz)**

**Linearity spread @ 440 Hz (across −18 to −3 dBFS): 0.01 dB** — essentially perfect. The GE-7 is behaving as a linear EQ at all captured amplitude levels.

### Notes

- The V-shape is clearly present in the data: lows and highs boosted, mids cut.
- 10.56 kHz error (+5.88 dB) is the largest discrepancy. The real GE-7's top band (6.4 kHz, +8 dB) has a wider/brighter skirt than Ge7Sim's Q = 1.4 model — real hardware Q at the top band is probably closer to 0.8–1.0. This is a Ge7Sim model limitation, not a capture problem.
- 220 Hz shows −2.06 dB error (measured less boost than expected). This is within normal hardware tolerance for analog sliders.
- All other bands within ±1.5 dB of Ge7Sim prediction — good agreement.

---

## preset_frown

**File:** `data/ge7/captures/preset_frown/wet_train_signal_v2.wav`
**Preset:** −6/−2/+6/+8/+6/−2/−6 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ GOOD — usable for model training (with caveat on linearity)**

### Measured vs expected EQ response (@ −18 dBFS, relative to flat)

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | −6.24 | −4.55 | −1.69 |
| 110 | −3.95 | −5.58 | +1.63 |
| 165 | −0.82 | −2.57 | +1.75 |
| 220 | +2.47 | −0.85 | +3.32 ⚠️ |
| 330 | +6.12 | +4.77 | +1.35 |
| 440 | +7.51 | +7.37 | **+0.14** |
| 660 | +10.29 | +8.45 | +1.84 |
| 880 | +10.51 | +9.55 | +0.96 |
| 1320 | +6.59 | +7.15 | −0.57 |
| 2640 | +1.92 | +0.30 | +1.62 |
| 5280 | −2.47 | −4.53 | +2.06 |
| **10560** | **−4.98** | **−1.40** | **−3.57** ⚠️ |

**RMS error: 1.95 dB   Max error: 3.57 dB (10.56 kHz)**

**Linearity spread @ 440 Hz (across −18 to −3 dBFS): 3.91 dB** ⚠️

### Notes

- The inverted V-shape is clearly present: mids boosted, lows and highs cut.
- Mid frequencies (440–880 Hz) are within 1 dB of expected — excellent.
- 220 Hz shows +3.32 dB error (measured more boost than expected). This is at the transition between the cut zone and boost zone, where the skirts of adjacent bands interact — small slider position uncertainty has a larger effect here.
- 10.56 kHz error (−3.57 dB) is opposite sign to smile: the frown cuts the top band, and the real GE-7 cuts less at 10.56 kHz than Ge7Sim predicts — same wide-skirt issue from the opposite direction.
- **Linearity spread 3.91 dB**: this is larger than expected for a linear EQ. Investigation needed — possibilities: (a) the mid boost at higher amplitude levels is pushing towards ADC input clip on the return channel; (b) the gain comparison is slightly confounded by the different lag between frown (−7) and flat2 (−60) causing sub-sample timing differences at 440 Hz. The overall RMS error is still good (1.95 dB), suggesting the fundamental response is correct.
- Recommend: re-check the 1824C return channel input gain for headroom before the boosted-mid sections.

---

## Known analysis caveats

1. **Ge7Sim top-band Q mismatch:** The real GE-7's 6.4 kHz band has a wider response than Ge7Sim's Q = 1.4 suggests. Errors at 10.56 kHz are a model artifact, not a capture problem. Consider fitting Q per-band from the capture data for the Ge7Sim update.

2. **Bypass captured at different playback level:** The bypass (`train_signal_v2_reamp only.wav`) was captured at 0 dB playback; all GE-7 presets at −10 dB. The script accounts for this by using flat2 as the comparison reference, not bypass. Do NOT compare preset captures directly to bypass without accounting for the 10.3 dB offset.

3. **−24 dBFS sections excluded:** Peak measurements at −24 dBFS are unreliable due to short 1-second windows and low SNR at that level within the section. All conclusions drawn from −18 to −3 dBFS data only.

4. **Frown linearity spread:** 3.91 dB at 440 Hz requires follow-up. If the next frown capture (with confirmed ADC headroom) shows < 1 dB spread, the current capture still has valid frequency response data for modeling.

---

## preset_bass_boost

**File:** `data/ge7/captures/preset_bass_boost/wet_train_signal_v2.wav`
**Preset:** +12/+8/+4/0/0/0/0 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ Good**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | +15.04 | +9.88 | **+5.16** ⚠️ |
| 110 | +13.57 | +13.04 | +0.53 |
| 165 | +11.02 | +10.59 | +0.43 |
| 220 | +10.50 | +10.44 | +0.07 |
| 330 | +7.29 | +6.53 | +0.76 |
| 440 | +5.50 | +5.43 | +0.07 |
| 660 | +3.42 | +1.97 | +1.45 |
| 880 | +2.16 | +0.95 | +1.21 |
| 1320 | +1.05 | +0.38 | +0.68 |
| 2640 | +0.31 | +0.09 | +0.22 |
| 5280 | +0.11 | +0.02 | +0.09 |
| 10560 | −0.01 | +0.00 | −0.01 |

**RMS error: 1.63 dB   Max error: 5.16 dB (82 Hz)   Linearity spread: 2.27 dB**

Shape is clearly correct — thunderous low-end boost tapering to flat at highs. 82 Hz error (+5.16 dB) is the same bottom-band width issue as Ge7Sim's Q=1.4 being too narrow for the 100 Hz band. Linearity spread of 2.27 dB at 440 Hz may be a minor artefact at a frequency where the bass boost puts a lot of energy into the reamp/ADC chain. Overall the capture is usable.

---

## preset_mid_scoop

**File:** `data/ge7/captures/preset_mid_scoop/wet_train_signal_v2.wav`
**Preset:** 0/0/−8/−12/−10/−4/0 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ Good**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | −1.84 | −0.31 | −1.53 |
| 110 | −2.44 | −0.59 | −1.84 |
| 165 | −3.86 | −1.40 | −2.46 |
| 220 | −5.45 | −2.82 | **−2.63** |
| 330 | −7.90 | −8.07 | +0.16 |
| 440 | −8.67 | −11.17 | +2.50 ⚠️ |
| 660 | −12.33 | −12.92 | +0.59 |
| 880 | −13.48 | −15.01 | +1.53 |
| 1320 | −11.08 | −12.51 | +1.44 |
| 2640 | −8.05 | −7.25 | −0.80 |
| 5280 | −4.81 | −1.99 | **−2.82** ⚠️ |
| 10560 | −2.69 | −0.30 | −2.39 |

**RMS error: 1.92 dB   Max error: 2.82 dB   Linearity spread: 0.03 dB**

Mid-scoop shape clearly present. Linearity is near-perfect (0.03 dB). The low-frequency end (82–220 Hz) shows ~2 dB more cut than expected — the adjacent band skirts interact in a way Ge7Sim underpredicts at mild cut settings. 440 Hz and 5280 Hz errors are at the transition slopes between bands. The capture is usable.

---

## preset_presence

**File:** `data/ge7/captures/preset_presence/wet_train_signal_v2.wav`
**Preset:** 0/0/0/0/+6/+10/+12 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ Good**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | −0.10 | +0.01 | −0.12 |
| 110 | −0.05 | +0.03 | −0.08 |
| 165 | +0.12 | +0.06 | +0.07 |
| 220 | +0.43 | +0.10 | +0.32 |
| 330 | +0.92 | +0.23 | +0.68 |
| 440 | +1.64 | +0.43 | +1.21 |
| 660 | +3.44 | +1.05 | +2.39 |
| 880 | +5.34 | +2.12 | +3.23 ⚠️ |
| 1320 | +8.27 | +6.17 | +2.11 |
| 2640 | +12.14 | +10.81 | +1.33 |
| 5280 | +12.00 | +11.96 | **+0.04** |
| 10560 | +11.79 | +3.55 | **+8.24** ⚠️ |

**RMS error: 2.77 dB   Max error: 8.24 dB (10.56 kHz)   Linearity spread: 0.0 dB**

Shape is correct: flat at lows, big presence boost at highs. Linearity is perfect (0.0 dB). The 10.56 kHz error (+8.24 dB) is the same Ge7Sim Q issue seen in smile — the real GE-7 top band radiates much further above 6.4 kHz than Q=1.4 predicts. 880 Hz shows +3.23 dB error (more boost than expected from mid-band skirt). The capture itself is clean.

---

## preset_high_cut

**File:** `data/ge7/captures/preset_high_cut/wet_train_signal_v2.wav`
**Preset:** 0/0/0/0/−6/−10/−14 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ Good**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | −0.70 | −0.01 | −0.68 |
| 110 | −0.71 | −0.03 | −0.69 |
| 165 | −0.75 | −0.06 | −0.69 |
| 220 | −0.81 | −0.10 | −0.70 |
| 330 | −1.07 | −0.24 | −0.83 |
| 440 | −1.53 | −0.44 | −1.09 |
| 660 | −2.91 | −1.07 | −1.84 |
| 880 | −4.57 | −2.16 | −2.41 |
| 1320 | −7.12 | −6.26 | −0.87 |
| 2640 | −11.80 | −11.16 | −0.64 |
| 5280 | −11.63 | −13.27 | +1.63 |
| 10560 | **−12.36** | **−4.18** | **−8.18** ⚠️ |

**RMS error: 2.64 dB   Max error: 8.18 dB (10.56 kHz)   Linearity spread: 0.01 dB**

Shape is correct: passes lows cleanly, cuts progressively through the highs. Perfect linearity (0.01 dB). Two issues, both Ge7Sim model artefacts rather than capture problems:
1. **Constant ~−0.7 dB low-frequency offset**: all bands below 400 Hz show ~0.7 dB extra cut. This is likely the overall GE-7 circuit applying a slight tilt when the high-frequency bands are cut heavily.
2. **10.56 kHz over-cut (−8.18 dB)**: same Ge7Sim narrow-Q issue — the real GE-7's top band cuts much further above 6.4 kHz than the model predicts.

---

## Summary table

| Preset | Status | RMS error (dB) | Linearity spread (dB) | Notable |
|---|---|---|---|---|
| flat (v2) | ✅ Reference | — | 0.25 | Baseline for all comparisons |
| smile | ✅ Good | 2.10 | 0.01 | 10.56 kHz +5.9 dB (model Q) |
| frown | ✅ Good | 1.95 | 3.91 | Linearity caveat at 440 Hz |
| bass_boost | ✅ Good | 1.63 | 2.27 | 82 Hz +5.2 dB (model Q); best RMS |
| mid_scoop | ✅ Good | 1.92 | 0.03 | Low-freq transition slope errors |
| presence | ✅ Good | 2.77 | 0.00 | 10.56 kHz +8.2 dB (model Q) |
| high_cut | ✅ Good | 2.64 | 0.01 | 10.56 kHz −8.2 dB (model Q); −0.7 dB tilt |

### Cross-preset pattern: Ge7Sim Q mismatch at extreme bands

All captures with large boost/cut at 6.4 kHz (top band) or 100 Hz (bottom band) show 5–8 dB error at the nearest stepped-tone frequency (10.56 kHz or 82 Hz). The real GE-7 band filters are wider than Q=1.4. Estimated real Q:
- Bottom band (100 Hz): Q ≈ 0.7–0.9 (based on bass_boost overshoot at 82 Hz)
- Top band (6.4 kHz): Q ≈ 0.7–0.9 (based on smile/presence/high_cut errors at 10.56 kHz)

This is a Ge7Sim model limitation. It does **not** affect training — models train on real captures, not Ge7Sim output. It does mean Ge7Sim is not a reliable test oracle at extreme frequencies; use the flat capture as baseline instead.

---

## preset_bright

**File:** `data/ge7/captures/preset_bright/wet_train_signal_v2.wav`
**Preset:** 0/0/0/+2/+6/+10/+12 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ Good**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | −0.13 | +0.02 | −0.15 |
| 110 | +0.01 | +0.05 | −0.04 |
| 165 | +0.32 | +0.10 | +0.21 |
| 220 | +0.76 | +0.19 | +0.57 |
| 330 | +1.65 | +0.46 | +1.19 |
| 440 | +2.66 | +0.91 | +1.76 |
| 660 | +4.50 | +2.59 | +1.91 |
| 880 | +6.05 | +3.98 | +2.07 |
| 1320 | +8.33 | +6.80 | +1.53 |
| 2640 | +12.39 | +10.92 | +1.48 |
| 5280 | +11.81 | +11.98 | −0.17 |
| 10560 | +11.36 | +3.56 | **+7.80** ⚠️ |

**RMS error: 2.55 dB   Max error: 7.80 dB (10.56 kHz)   Linearity spread: 0.14 dB**

Shape is very clean — flat lows, progressive rise through the highs. Linearity essentially perfect (0.14 dB). The 10.56 kHz error (+7.80 dB) is the Ge7Sim Q issue: the real GE-7 top band radiates significantly above 6.4 kHz. At 5280 Hz the agreement is nearly perfect (−0.17 dB). Excellent capture.

---

## preset_low_cut

**File:** `data/ge7/captures/preset_low_cut/wet_train_signal_v2.wav`
**Preset:** −14/−10/−6/0/0/0/0 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ Excellent**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | −15.05 | −11.59 | **−3.46** ⚠️ |
| 110 | −14.83 | −15.42 | +0.59 |
| 165 | −11.95 | −13.05 | +1.11 |
| 220 | −11.77 | −13.23 | +1.46 |
| 330 | −8.77 | −8.98 | +0.21 |
| 440 | −7.77 | −7.78 | **+0.02** |
| 660 | −4.53 | −2.84 | −1.69 |
| 880 | −2.63 | −1.39 | −1.24 |
| 1320 | −1.33 | −0.55 | −0.78 |
| 2640 | −0.76 | −0.13 | −0.63 |
| 5280 | −0.68 | −0.03 | −0.65 |
| 10560 | −0.68 | −0.01 | −0.67 |

**RMS error: 1.35 dB   Max error: 3.46 dB (82 Hz)   Linearity spread: 0.02 dB**

Best RMS error of all presets. Near-perfect linearity (0.02 dB). Shape is exactly right: heavy cut at 100–400 Hz, clean flat pass above 800 Hz. The 82 Hz error (−3.46 dB more cut than expected) is the bottom-band wide-skirt issue — same pattern as bass_boost but in the cut direction. 660–880 Hz show ~1.5 dB more cut than expected from the transition slope. No issues with this capture.

---

## preset_warm

**File:** `data/ge7/captures/preset_warm/wet_train_signal_v2.wav`
**Preset:** +6/+4/+2/0/−4/−8/−10 dB @ 100/200/400/800/1.6k/3.2k/6.4k Hz, LEVEL = 0

**Status: ✅ Good**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | +9.22 | +5.02 | **+4.21** ⚠️ |
| 110 | +8.80 | +6.55 | +2.24 |
| 165 | +6.84 | +5.20 | +1.64 |
| 220 | +5.49 | +5.05 | +0.44 |
| 330 | +3.21 | +3.01 | +0.20 |
| 440 | +1.41 | +2.35 | −0.94 |
| 660 | −1.05 | +0.22 | −1.27 |
| 880 | −2.83 | −1.00 | −1.83 |
| 1320 | −5.27 | −4.10 | −1.16 |
| 2640 | −10.35 | −8.37 | −1.97 |
| 5280 | −10.38 | −9.82 | −0.56 |
| 10560 | −10.16 | −2.83 | **−7.33** ⚠️ |

**RMS error: 2.75 dB   Max error: 7.33 dB (10.56 kHz)   Linearity spread: 0.00 dB**

Warm/dark shape clearly present: lows boosted, highs cut. Perfect linearity (0.00 dB). Two known model errors:
1. **82 Hz (+4.21 dB)**: same wide-skirt bottom-band issue as bass_boost — the 100 Hz filter is wider than Q=1.4 predicts
2. **10.56 kHz (−7.33 dB)**: same Ge7Sim Q issue as high_cut — real top band cuts further above 6.4 kHz than model predicts

Mid-range (220–5280 Hz) within ±2 dB. Clean capture.

---

## preset_all_down

**File:** `data/ge7/captures/preset_all_down/wet_train_signal_v2.wav`
**Preset:** −12 dB on all 7 bands, LEVEL = −15 dB (slider at minimum)

**Status: ✅ Good (errors are model-only)**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | −25.86 | −25.91 | **+0.04** |
| 110 | −25.61 | −29.97 | +4.36 |
| 165 | −26.50 | −30.05 | +3.55 |
| 220 | −27.13 | −32.13 | +5.01 |
| 330 | −26.41 | −30.99 | +4.59 |
| 440 | −27.52 | −32.56 | +5.03 |
| 660 | −25.60 | −31.07 | +5.47 |
| 880 | −28.20 | −32.58 | +4.38 |
| 1320 | −26.96 | −30.97 | +4.01 |
| 2640 | −25.43 | −30.37 | +4.95 |
| 5280 | −23.76 | −28.34 | +4.58 |
| 10560 | −23.38 | −18.81 | **−4.57** |

**RMS error: 4.42 dB   Max error: 5.47 dB (660 Hz)   Linearity spread: 0.11 dB**

Linearity is near-perfect (0.11 dB) — no ADC issues despite the extreme setting. The 82 Hz agreement is exceptional (+0.04 dB). Errors at other frequencies are explained by two model limitations:
1. **+4–5 dB across mid-range**: when all 7 bands are cut simultaneously, the overlapping filter skirts interact, producing *less* total cut than Ge7Sim's independent-biquad model predicts. Ge7Sim over-estimates the cut by ~4–5 dB.
2. **10.56 kHz (−4.57 dB)**: same Ge7Sim top-band Q issue — real hardware produces less cut at 10.56 kHz than expected from the narrow Q=1.4 model.

The capture is clean and usable. The errors quantify the Ge7Sim band-interaction model gap, not a recording problem.

*Note: LEVEL was set to −15 dB (slider at minimum), not the −6 dB listed in the initial preset definition. Params corrected.*

---

## preset_all_up

**File:** `data/ge7/captures/preset_all_up/wet_train_signal_v2.wav`  
**Preset:** +12 dB on all 7 bands, LEVEL = +15 dB (slider at maximum)

**Status: ⚠️ DISCARDED — ADC saturation at −10 dB fader**

Total measured gain is ~+35 dB (band overlap + LEVEL pot at max). Even the −18 dBFS sections hit approximately −3 dBFS output. Linearity spread = 13.4 dB. File retained for reference but not used in training.

## preset_all_up_2

**File:** `data/ge7/captures/preset_all_up_2/wet_train_signal_v2.wav`  
**Preset:** +12 dB on all 7 bands, LEVEL = +15 dB (slider at maximum), **DAW fader −30 dB**

**Status: ⚠️ STILL SATURATING — needs recapture at −40 dB fader**

| Freq (Hz) | Measured (dB) | Expected (dB) | Error (dB) |
|---|---|---|---|
| 82 | +34.93 | +25.91 | +9.02 |
| 110 | +32.26 | +29.97 | +2.29 |
| 165 | +34.82 | +30.05 | +4.78 |
| 220 | +34.23 | +32.13 | +2.10 |
| 330 | +34.81 | +30.99 | +3.82 |
| 440 | +36.55 | +32.56 | +3.99 |
| 660 | +34.98 | +31.07 | +3.90 |
| 880 | +37.25 | +32.58 | +4.67 |
| 1320 | +36.58 | +30.97 | +5.61 |
| 2640 | +35.05 | +30.37 | +4.68 |
| 5280 | +33.26 | +28.34 | +4.92 |
| 10560 | +33.26 | +18.81 | **+14.45** ⚠️ |

**RMS error: 6.24 dB   Max error: 14.45 dB (10.56 kHz)   Linearity spread: 12.18 dB**

The fader correction (+20 dB) was applied automatically. The true all-up total gain is **~+35 dB** — all 7 overlapping bands at +12 dB plus the LEVEL pot at maximum (+15 dB labeled, ~+18 dB actual from calibration). This is far higher than Ge7Sim predicts (+32 dB) because independent biquads don't model band overlap.

Headroom calculation at −30 dB fader:
```
−3 dBFS section:  −3 − 30 + 35 = +2 dBFS  → clips
−6 dBFS section:  −6 − 30 + 35 = −1 dBFS  → clips
−12 dBFS section: −12 − 30 + 35 = −7 dBFS → OK
```

**Recapture at −40 dB DAW fader.** At that level all sections clear: −3 dBFS + 35 − 40 = −8 dBFS ✓

---

## Single-slider captures

**Directory:** `data/ge7/captures/single_slider/`  
**Analysis:** `scripts/analyze_single_sliders.py` → `outputs/ge7_captures/single_slider/`

Each band was captured with one slider at +10 dB, all others flat. The LEVEL slider was captured at +10 dB and −10 dB with all band sliders flat.

### Band centre frequency accuracy

| Nominal (Hz) | Actual peak (Hz) | Error (cents) | Measured peak (dB) |
|---|---|---|---|
| 100 | 87 | **−239 ¢** | +9.69 |
| 200 | 190 | −91 ¢ | +8.61 |
| 400 | 383 | −75 ¢ | +7.18 |
| 800 | 792 | −17 ¢ | +7.52 |
| 1600 | 1599 | −1 ¢ | +7.41 |
| 3200 | 3380 | +95 ¢ | +8.42 |
| 6400 | ~13200 (shelf) | N/A | +7.17 |

**The 6400 Hz band is not a peaking filter — it behaves as a high shelf.** The measured frequency response rises continuously from 6 kHz, peaks at ~13 kHz, and never returns to 0 dB within the audible range. This is the root cause of all the large errors at 10.56 kHz seen across every preset that uses the top band. The label "6400 Hz" refers to the filter's half-gain frequency, not its peak.

The lower bands (100–400 Hz) peak 75–239 cents below their labeled frequencies. The 800–1600 Hz range is nearly accurate. The 3200 Hz band peaks ~1 semitone sharp. This is normal for analog graphic EQ designs using gyrators.

### LEVEL slider calibration

| Slider position | Measured gain (200–5000 Hz average) |
|---|---|
| +10 dB | **+12.1 dB** |
| −10 dB | **−14.4 dB** |

The LEVEL pot has a steeper-than-labeled response. This is consistent with a log-taper potentiometer where the midpoint does not correspond to 0 dB. When a preset analysis says "LEVEL = +6 dB", the actual gain is closer to +8 dB.

---

## Summary table

| Preset | Status | RMS error (dB) | Linearity spread (dB) | Notable |
|---|---|---|---|---|
| flat (v2) | ✅ Reference | — | 0.25 | Baseline for all comparisons |
| smile | ✅ Good | 2.10 | 0.01 | 10.56 kHz +5.9 dB (model Q) |
| frown | ✅ Good | 1.95 | 3.91 | Linearity caveat at 440 Hz |
| bass_boost | ✅ Good | 1.63 | 2.27 | 82 Hz +5.2 dB (model Q); best RMS |
| mid_scoop | ✅ Good | 1.92 | 0.03 | Low-freq transition slope errors |
| presence | ✅ Good | 2.77 | 0.00 | 10.56 kHz +8.2 dB (model Q) |
| high_cut | ✅ Good | 2.64 | 0.01 | 10.56 kHz −8.2 dB (model Q); −0.7 dB tilt |
| bright | ✅ Good | 2.55 | 0.14 | 10.56 kHz +7.8 dB (model Q) |
| low_cut | ✅ Excellent | 1.35 | 0.02 | Best RMS; 82 Hz −3.5 dB (model Q) |
| warm | ✅ Good | 2.75 | 0.00 | 82 Hz +4.2 dB and 10.56 kHz −7.3 dB (model Q) |
| all_down | ✅ Good | 4.42 | 0.11 | LEVEL=−15 (max cut); band-overlap + top-band Q errors |
| all_up | ✗ Discarded | — | 13.40 | ADC saturation at −10 dB fader |
| all_up_2 | ⚠️ Recapture | — | 12.18 | ADC still saturating at −30 dB; redo at −40 dB |

### Cross-preset pattern: Ge7Sim Q mismatch at extreme bands

All captures with large boost/cut at 6.4 kHz (top band) or 100 Hz (bottom band) show 5–8 dB error at the nearest stepped-tone frequency (10.56 kHz or 82 Hz). The real GE-7 band filters are wider than Q=1.4. The single-slider analysis confirms:
- **Bottom band (100 Hz)**: actual peak at 87 Hz, filter is wider than Q=1.4. Real Q ≈ 0.7–0.9.
- **Top band (6.4 kHz)**: not a peaking filter at all — behaves as a high shelf peaking at ~13 kHz. Q=1.4 peaking EQ is the wrong model for this band.

This is a Ge7Sim model limitation. It does **not** affect neural model training — models train on real captures, not Ge7Sim output.

### all_down: band-interaction term missing in Ge7Sim

When all 7 bands are boosted/cut equally, the Ge7Sim independent-biquad model underpredicts the total gain change by ~4 dB. The band filter skirts overlap in a way that adds gain (cut direction: more total cut than predicted). A more accurate model would include inter-band coupling terms. For training purposes, the capture is correct; the model comparison is misleading.

---

## Pending

- **`preset_all_up` recapture**: set DAW fader to **−40 dB**, re-record (−30 dB was not enough — real gain is ~+35 dB with LEVEL at max)
- **LEVEL slider investigation**: measured +12.1/−14.4 dB at ±10 dB settings — document in `ge7_plan.md` for calibration reference
- **Ge7Sim top-band fix**: replace the 6400 Hz peaking biquad with a high-shelf filter; fit shelf frequency and Q from single-slider capture

Outputs for completed captures: `outputs/ge7_captures/<preset>/`
