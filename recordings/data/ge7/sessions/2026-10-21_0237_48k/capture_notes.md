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

## Result of the pad test (all_up_pad15, 09:09)

- Quiet pink (-24 dBFS) section: -14.5 to -16 dB at every frequency with the pad: the pad works; the
  pedal is linear there. The noise floor dropped 15.4 dB (-82.5 -> -97.9 dBFS): the hiss is the
  GE-7's own, before the pad.
- Loud sections (sweep -3 dBFS, -3 dBFS tones): only -0.1 to -2.7 dB with the pad, still clipped. Both
  takes clip at the same return level, about -14 to -16 dBFS peak.
- So the main limiter is the **rig's return path after the pad** (ProDI transformer or the 1824c In 1),
  with a ceiling near -15 dBFS that does not move with In 1 gain (C2 at max gain and at gain 4 both
  compressed there). Whether the GE-7 also clips at the loudest levels is masked by it.
- C2's "saturation" warnings (0.37 / 0.95 dB at -9 dBFS sent) were this return ceiling, not the
  ProRMP.

bright (09:03) also clips (linearity 7.3 dB, THD -10 dB).

## Proposed (waiting for the owner)

1. Per-preset auto-level: a short probe before each take sets the digital send level so the return
   peaks about 6 dB under the measured ceiling (never above the current -10 dB send). The references
   stay valid; QC already divides by the send level.
2. C2 measures the return ceiling (tones until the return compresses) and sets the take target from
   it, replacing the fixed -18 to -6 dBFS target.
3. Hardware test: return on a rear input (e.g. In 3), C2 only, to see if the ceiling is In 1-specific.
4. Meanwhile the cut-only presets (high_cut, low_cut, mid_scoop) can be captured as they are.

## Owner's answers (2026-10-21) and the plan for tomorrow

- Auto-level per take: yes. Built after this session (C2 measures the ceiling at 1 kHz and 60 Hz;
  each take probes its gain and lowers the digital send level; recorded per take). Lowering In 1 gain
  instead would invalidate C2-C4 and gains nothing (the ceiling is after the gain stage).
- Tomorrow, at the rig:
  1. Hardware test: move the ProDI XLR to a rear input (In 3), set the return channel to 3 on the
     Setup page, run C2 only. Note the ceiling C2 reports; compare with In 1 (about -15 dBFS).
  2. Use whichever input has the higher ceiling, then a **new session**: C1-C4 with the new code.
  3. Recapture the boosting presets (all_up, bass_boost, bright, and the rest of the plan) with
     auto-level on; check each with `scripts/session_linearity.py`.
- This session's good data: flat, all_down (and the cut-only presets if captured).

## frown (captured on code 840e480, before auto-level)

Mildly clipped at the loudest level only: sweep at -3 dBFS reads 0.5 dB (400 Hz) and 2.4 dB (800 Hz)
low, exactly where frown boosts (+6 to +8 dB); loud tones there THD -15.5 dB. The other five sections
(sweeps -24/-12, pink -24/-12/-3) agree within 0.02 dB, coherence 0.9994: usable. Auto-level would
have sent it about 7 dB lower.

## Salvage for rung A

The clipped takes of this session still contain linear sections (frown: all but the -3 dBFS sweep;
the quieter sections of bass_boost and bright; all_up's pink -24). The rung A fitter should take
sections, not whole takes, and drop any section whose response disagrees with the take's quietest
sections by more than the repeatability floor (about 0.15 dB). That keeps today's linear data and
gives a two-session cross-check against tomorrow's auto-levelled recaptures.
