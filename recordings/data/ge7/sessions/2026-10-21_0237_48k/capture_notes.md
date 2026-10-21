# Session 2026-10-21_0237_48k: capture notes

Rig: Main knob max, In 1 gain 4 (XLR from ProDI XLR OUT), In 2 gain 12 (line off), ProDI pad OFF,
loopback sent 30 dB below the pedal send, send level -10 dB. Calibration C1-C4 ok (C2 warns only that the
return is 1.5 dB under the conventional target; it is linear over the take levels).

Checked with `uv run python scripts/session_linearity.py recordings/data/ge7/sessions/2026-10-21_0237_48k`
(sweeps at -24 and -3 dBFS input must agree; THD of the loudest tones).

## Status

| Preset | Max boost | Linearity (dB) | THD (dB) | Status |
|---|---|---|---|---|
| flat | 0 | 0.03 | -66 | **Good** |
| all_down | -12 / LEVEL -15 | 0.02 | -72 | **Good** (QC "fail" is Ge7Sim's LEVEL law, not the take) |
| all_up | +12 / LEVEL +15 | 21.0 | -7.6 | **Clips: retake** (only the pink -24 section is near-linear) |
| bass_boost | +12 at 100 Hz | 10.8 | -8.4 | **Clips: retake** |

Predicted for the rest (not yet captured; to be confirmed by the same check):

| Preset | Max boost | Expectation |
|---|---|---|
| bright, presence | +12 (6.4k), +10 (3.2k) | likely clip: retake |
| frown, smile | +8 | likely clip: retake |
| warm | +6 | may clip mildly: check |
| high_cut, low_cut, mid_scoop | cuts only | should be good |
| all_up_2, all_up_3 | +12 / LEVEL +15, +6 | skipped; same settings family as all_up |

## Before retaking: find where it clips (one diagnostic retake)

Retake **all_up (train)** with the **ProDI -15 dB pad ON**, In 1 gain unchanged, then run the check above.

- **Still clipped** (THD around -8 dB): the GE-7 itself clips. That is real pedal sound. Retake the
  clipping presets at a **lower send level** (profile `send_level_db`, e.g. -25 dB instead of -10)
  with the pad OFF again; no recalibration is needed (the references are ratios). Keep these
  clipped takes as data for a later nonlinear model.
- **Clean** (THD below about -50 dB): the ProDI or the In 1 preamp was clipping. Retake the clipping
  presets **with the pad ON** in a **new session calibrated with the pad on** (C1-C4), since the pad
  changes the chain.

Either way, turn the pad back OFF before capturing anything for this session's calibration.
