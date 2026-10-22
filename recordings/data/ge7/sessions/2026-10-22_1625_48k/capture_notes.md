# Session 2026-10-22_1625_48k (code 606f705): pre-flight for the record run

Return on In 3, Sweeps only. C1-C4 ok (C2 and C3 run twice; final references band coherence
>= 0.9996). C2: limit on the send side at -9 dBFS sent (1 kHz and 60 Hz agree in sent level, 5 dB
apart in return level). Auto-level sent flat at -12 dB (train) and -15 dB (val), not -23 as at
16:00: the quietest sweep is 57 / 47 dB over the noise. No retakes; every QC item ok.

Repeatability, untouched (flat train, flat val, C4 over ~5 min): within 0.035 dB, 0.3 degrees.
Pre-flight passed: the 0.35 dB val shift of session 1600 did not recur.

Across sessions flat differs by up to 0.5 dB in level and shape (engaged / bypassed at 1 kHz:
-1.65 on 10-21, -1.95 at 16:00, -1.29 at 16:25). The chain cancels in that ratio, so the likely
cause is where the sliders sit in their centre detents each time flat is set: a setting error,
which the record run's re-plugged takes (flat_replug, smile_replug) measure.

## The record run in two sittings (owner, 2026-10-22)

Panel markings (owner): 0, ±15 and two ticks each way, taken to be ±5 and ±10 (evenly spaced:
check). The slider map moved from ±7.5 (between ticks) to ±10, so every position is set on a mark;
A0 pins the parameters as well or slightly better at ±10 (worst lvl_rho_in ±0.87 % vs ±1.00 %).
The random settings already sit on the 5 dB grid (the marks). The GUI draws the same ticks.

1. Sitting 1, this session: Plan page -> Load record run -> Run. Stop at any step boundary (do not
   press Ready). Leave every knob, cable, gain and the pedal's supply as they are.
2. Sitting 2: Desktop shortcut -> Start page -> resume 2026-10-22_1625_48k. Calibrate page ->
   cables as for C3/C4, footswitch OFF -> "Run drift check (bypassed)". It measures the bypassed
   path again against the stored C3 reference without replacing it, saves the check's audio and
   logs the result.
   - ok (< 0.05 dB): Run page, continue where sitting 1 stopped.
   - warn/fail: something moved. Put it back and check again, or recalibrate C2-C4. Takes keep
     the calibration they were recorded with (the dataset divides each by its own), and no
     capture or calibration file is ever overwritten.

## The preset run and the first training (2026-10-22 evening, analysis at code 21f8cdd)

Captured: the 15 folder presets (12 accepted, all_up_2 / all_up_pad15 / presence skipped), each
sweeps + val. **The record run was not loaded**: no slider map, pairs, random settings, repeats
or full-train copies yet. The live training saw 8 fit presets and no validation settings; the
analysis GUI wrote ge7_pedalode_v1.json at 23:30 from those 8.

Capture quality: train vs val (same setting, two signals, ~1 min apart) agree within
0.003-0.075 dB RMS on every preset. Five boosting presets failed linearity on the first attempt
and passed 6 dB quieter (the automatic retake), each failing at the centre of a boosted gyrator
band: 797 Hz (all_up, all_up_3; band 4 has the highest Q, 4.07), 774 Hz (frown), 100 Hz
(bass_boost), 3273 Hz (bright), while smile passed with a -6.1 dBFS return. Hypothesis (inferred,
not proven): the GE-7's gyrator op-amps run out of headroom first (a gyrator swings about Q times
its input at resonance). Pedal sound, for the later clipping capture; avoided for rung A.

Model quality (shape RMS after a level offset, 50 Hz-10 kHz): v1 0.07-0.98 dB on its 8 fit
presets (schematic alone 0.06-3.3, Ge7Sim 0.06-2.8); on the held-out test presets frown 1.23,
smile 1.40, warm 1.26, mid_scoop 2.44 (Ge7Sim 1.70 / 2.23 / 2.90 / 1.81). Better than Ge7Sim on 3
of 4, 5-10x above the 0.26 dB floor; flat's level 0.7 dB off. Cause: presets alone cannot separate
each band's Q, strength, the resistor ratios and the pot law (Qs and strengths only reached
0.2-0.4 of their prior widths); the slider map exists for that. The test presets have been seen
(QC graded smile, mid_scoop and warm against v1), so their final rung A score is partly spent.

QC fixes from this run: "Return present" judged against the take's own noise floor (all_down val
false-failed at -60.4 dBFS RMS, 38 dB above the noise); "Level vs calibration" no longer subtracts
the C4 flat level from a fitted model (it double-counted ~1.3 dB); the analysis GUI refuses to
write a parameter file without validation settings; its table names the bands.

Next: retire v1 (QC grades against it now), drift check, then Load record run in this session
(accepted presets are skipped): map, pairs, random settings, repeats, full-train copies.
