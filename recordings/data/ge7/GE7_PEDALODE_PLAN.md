# GE-7 PedalODE plan: the GE-7's block diagram, with learned block parameters

Status: plan, revised 2026-10-18. **A0 passed and R1 built, 2026-10-18** (see "A0 result" under Rung A
and "R1 as built"); R2 onward needs the rig.
- **Level of description** (Brian, 2026-10-18): the **block diagram**, not individual components. Each
  block is a physical stage with a few physical parameters (gain, corner frequency, Q, pot law).
- **Starting values** (Brian): reasonable guesses. Recovering the true values is the point of the model.
- **Two deployment targets** (Brian, 2026-10-18): the Hothouse/Daisy (primary; pedals chained in
  series to build tones), and a **Studio Pro plugin** (secondary; built in the BrianEQ repo, not here).
  This repo delivers the fully fitted model in a form both can implement and test against.
- **Confidence tags** follow the PedalODE rule: anything not measured says where it came from and how
  sure it is.

## Goal, and what each rung proves

Replace Ge7Sim (a hand-calibrated **series** cascade of 6 peaking biquads + a high shelf + a LEVEL law,
`pedal_model/synthetic/ge7.py`) with a model whose **structure is the GE-7's block diagram**: the
blocks, how they connect, and where the sliders act. Its **block parameters are learned** from hardware
captures, and one model covers every slider setting.

| Rung | Proves | Done when |
|---|---|---|
| A0. Known answer | The fitter recovers this structure's parameters at all | Synthetic data from the model itself, parameters recovered (criteria below) |
| A. Block-diagram fit | This structure explains the GE-7 and generalises to unseen slider settings | Held-out settings within the repeatability floor; every one better than Ge7Sim; parameters consistent across data sets |
| C. Deploy | The same model runs on the Daisy | Firmware nulls against the Python model; CPU budget met |
| C2. Plugin handoff | The same model is complete enough for a Studio Pro plugin | Handoff package (below) passes its own checks at 44.1–192 kHz; BrianEQ builds the plugin |

(Rung B, the component-level comparison with the schematic, is dropped by decision; the name C is kept
so older notes still match.) Rungs in order. Later, beyond this plan: capture and fitting in one closed
loop that decides how much to capture (Brian: "eventually, not yet").

**What "recovered" can and cannot mean here.** With guessed starting values and no schematic values,
there is no external ground truth for the hardware parameters. The evidence that they are right comes
from three places, and the results must not claim more than that:
1. **A0:** the fitter recovers known parameters from synthetic data.
2. **Holdout:** the fitted model predicts slider settings it never saw.
3. **Consistency:** the same parameters come out of different sessions, 48 vs 96 kHz, and different
   subsets of the data.

## The block diagram

**Source and confidence.** Brian's summary (2026-10-18) of a GitHub LTspice teardown of the Hobby-Hour
GE-7 schematic, plus Hobby-Hour, smemmusic and a DIYstompboxes thread. It is **secondary**: nobody has
read the schematic image, and the LEVEL position is "one author's reading". Treat the structure as
likely; every number is a starting guess.

```
IN ─► [1] Input ─┬─► dry path ───────────────────────────────┐
     ×10, zero    │                                           ├─► [5] Switch ─► [6] Output ─► OUT
     ≈ 2.2 kHz    └─► [2] LEVEL ─► [3] EQ summing stage ──────┘    (footswitch;    pole ≈ 2.2 kHz
     (pre-emph.)      pot-split       7 band blocks on               powers up in    (de-emphasis)
                      cut/boost        slider pot-splits             BYPASS)
```

| Block | What it does (teardown) | Confidence | Block parameters |
|---|---|---|---|
| 1. Input | ×10, AC-coupled, zero near 2.2 kHz (pre-emphasis for noise) | Medium | gain, zero f_z, high-pass f_hp1 |
| 2. LEVEL | Pot-split around a unity buffer: cut shunts to ground, boost through a 5:1 divider (≈ +14 dB); wiper RC ≈ 7.2 Hz | Medium; position "likely" | max boost, max cut, pot law, wiper corner |
| 3. EQ summing stage | Two nodes (Eq_Input, Eq_Feedback); each band hangs on its own slider pot between them; toward Eq_Input cuts, toward Eq_Feedback boosts; centre flat | Medium-high (standard graphic-EQ structure) | input/feedback weights, pot law, pot end resistance |
| 3a. Bands 1–6 | Resonant band-pass branches (gyrator-based) at 100 … 3200 Hz | Medium | centre fₖ, Qₖ, strength σₖ |
| 3b. Band 7 | 6.4 kHz is a shelf; construction unknown | Medium that it's a shelf (our own capture agrees) | corner f₇, strength σ₇ (shape per D2) |
| 5. Switch | JFET switch, flip-flop on the footswitch | Medium | none (on/off), but it changes the capture procedure |
| 6. Output | Emitter follower, low-pass ≈ 2.2 kHz cancelling the input zero, AC-coupled | Medium | pole f_p, high-pass f_hp2 |

**Consequences.**
- In **bypass** the signal still goes through blocks 1 and 6, so the bypass response is H_in·H_out:
  flat only as far as the 2.2 kHz zero and pole cancel.
- The **engaged/bypass ratio** is H_level·H_eq: the re-amp chain and the pre/de-emphasis both cancel.
  That makes it the cleanest data for the EQ fit (decision D1).
- **LEVEL order.** For small signals, blocks in series multiply, so LEVEL before or after the EQ gives
  the same response. Order matters only for headroom (clipping in `all_up`), so the model keeps the
  teardown's order.
- **The GE-7 powers up in bypass.** Every capture instruction must say "footswitch LED on".

## The EQ summing stage as blocks

The structure is what the block diagram fixes, and it is not a cascade. Every band sits on a **pot split**
between two shared nodes, and the stage sums them. Written at block level:

    Bₖ(s) = (ωₖ/Qₖ) s / (s² + (ωₖ/Qₖ) s + ωₖ²)          band-pass block (1..6), peak 1 at ωₖ
    B₇(s) = s / (s + ω₇)                                 shelf block (hypothesis H7a, first order)
    yₖ(s) = σₖ · Bₖ(s)                                   band branch, strength σₖ relative to the pot

    slider sₖ ∈ [−15, +15] dB → wiper position xₖ = τ((sₖ + 15)/30) ∈ [0, 1], τ = the pot law
    g1ₖ = 1/(e + xₖ),  g2ₖ = 1/(e + 1 − xₖ)               the two halves of the pot (e: end resistance)

    H_eq(s) = [1 + ρ_b Σₖ g2ₖ yₖ / (g1ₖ + g2ₖ + yₖ)] / [1 + ρ_a Σₖ g1ₖ yₖ / (g1ₖ + g2ₖ + yₖ)]

The pot split sits on the wiper side of every band, which is why the bands interact and why Q changes
with the slider. This is the same formula as the node analysis of the circuit (an ideal summing op-amp,
each band on a pot wiper). A series R–L–C branch is exactly a strength times a normalised band-pass, so
the block parameters (fₖ, Qₖ, σₖ, ρ, e, τ) lose nothing by not naming parts.

*Derived from the block description; which end of the pot boosts is assumed, not checked. Checked
numerically 2026-10-18 with made-up values:*
- *flat at the centre detent (to 1e-15 dB);*
- *boost and cut exact mirror images for ρ_a = ρ_b;*
- *effective Q falling at milder settings;*
- *two-slider settings never equal to the dB sum of the singles, adjacent pairs most.*

Properties to test against data:
- **Centre detent:** flat when ρ_a = ρ_b.
- **Full boost:** 1 + ρ_b·yₖ. **Full cut:** 1/(1 + ρ_a·yₖ), the mirror when ρ_a = ρ_b.
- **Q depends on the slider:** the DIYstompboxes observation, coming from the structure itself.
- **Bands interact:** the error pattern Ge7Sim showed in `docs/ge7_capture_results.md`.

**LEVEL block.** The same pot-split idea around a buffer: cut by shunting, boost through the 5:1 divider,
with its own pot law. The measured law (+10 → +12.1 dB, −10 → −14.4 dB, from `ge7.py`) is asymmetric, and
the block's pot law has to reproduce that.

## Parameters and starting guesses

All positive parameters are fitted as logs. "Teardown" means a value quoted by the secondary source;
"guess" means mine, chosen only to start the fit somewhere sensible.

| Block | Parameter | Start | Source of the start |
|---|---|---|---|
| Input | gain | ×10 (20 dB) | teardown |
| | zero f_z | 2.2 kHz | teardown |
| | high-pass f_hp1 | 10 Hz | guess |
| Output | pole f_p | 2.2 kHz | teardown |
| | high-pass f_hp2 | 5 Hz | guess |
| | overall gain (with input) | measured flat level | measured at C4 |
| LEVEL | max boost | +14 dB (5:1) | teardown |
| | max cut | −15 dB | guess (the panel says ±15) |
| | pot law | linear, plus one shape parameter | guess |
| | wiper corner | 7.2 Hz | teardown |
| EQ | ρ_a, ρ_b | 40, equal | guess, tuned with σ (see note) |
| | end resistance e | 0.01 of the pot | guess |
| | pot law τ | linear, plus one shape parameter | guess |
| Bands 1–6 | fₖ | 100, 200, 400, 800, 1600, 3200 Hz | panel |
| | Qₖ | 2 | guess |
| | σₖ | 1.0 | guess, tuned with ρ (see note) |
| Band 7 | f₇ | 6.4 kHz | panel |
| | σ₇ | 1.0 | guess |

Note on ρ and σ (found while building A0): the branches at centre load both nodes, so one band's boost
is (1+A+B)/(1+A), not 1+B. The first guess (ρ 1, σ 4.7) gave only +9.4 dB. ρ 40 and σ 1.0 were picked
so the middle bands give about ±15 dB at full and ±8.5 dB at ±10, the coarse Studio One-era readings.
The 100 Hz band then gives about +12 dB. These are still guesses, but no longer unreasonable ones.
LEVEL starts at ρ_in 0.914, ρ_fb 1.545, κ 4 (+14/−15 dB at the ends, linear pot), with the overall
ratio gain 0.753 so that all-centre is 0 dB. (Fitting the LEVEL guess to `ge7.py`'s two-point law
drove the pot law into a step, so it was dropped: the fit, not the guess, should find the law.)

About 30 parameters. Measured hints are *not* used as starting values, so the fit cannot just copy
them; they are kept as checks. These are the single-slider peaks near 87, 190, 383, 792, 1599 and
3380 Hz, the 6.4k shelf peaking near 13 kHz, and the flat insertion of −1.2 dB.

## What we have measured (Studio One era, not re-verified)

From `docs/ge7_capture_results.md` and the single-slider calibration in `ge7.py`:
- **Linearity:** within 0.25 dB from −18 to −3 dBFS on flat.
- **Flat:** −1.2 dB insertion loss, broadband.
- **Single sliders at +10 dB:** peaks near 87–3380 Hz, at 72–97 % of the setting.
- **LEVEL:** +10 → +12.1 dB, −10 → −14.4 dB.
- **Ge7Sim:** misses the presets by 1.35–4.42 dB RMS, at band edges and between adjacent bands.
- **Re-amp chain:** a +4 dB tilt from 82 Hz to 10.5 kHz.

These were measured relative to flat, without removing the chain, on the old Studio One rig.

## Step 0: data probes, run before fitting

- **P0.1 Interaction test.**
  - **Capture:** each adjacent pair (6 pairs) at (+15, +15) and (+15, −15), and two distant pairs
    (16 sweep steps, ~20 min).
  - **Compare:** each against the sum, in dB, of the matching single-slider responses from the map.
  - **Expect:** the block model predicts no pair adds up exactly, adjacent pairs least of all. Only the
    ordering and a non-zero deviation are firm predictions; the size depends on the parameters.
  - **Decision D0:** if even adjacent pairs add up within the floor, the shared summing stage doesn't
    matter, a cascade suffices, and we re-plan.
- **P0.2 Reciprocity.** From the slider map's ±10 and ±15 steps, compare dB(+x) with −dB(−x). They are
  equal if ρ_a = ρ_b. An offset or a tilt shows asymmetry.
- **P0.3 Q against the slider.** From the map's band fits, the effective Q at ±10 should be lower than at
  ±15.
- **P0.4 Repeatability floor.** Capture flat and smile three times each, and once more after re-plugging.
  The RMS difference (50 Hz–10 kHz) is the floor every criterion is measured against.

## Rig prerequisites (R1–R2)

**R1. Reference captures, with a guided cable move and a test after each.** The rig's QC currently divides
by a single 1 kHz gain, so the ProRMP's tilt ends up in every "measured" response. Calibration becomes
four steps. The two guitar cables around the pedal get names, printed in the GUI: **cable A** runs ProRMP
OUTPUT → GE-7 IN, **cable B** runs GE-7 OUT → ProDI INPUT. The Setup diagram highlights the cable to move,
and the next step stays locked until the test after each move passes:

| Step | Cable move (exact) | Test that runs next | Must pass |
|---|---|---|---|
| C1 Loopback | None: Out 2 → In 2 already in place | Loopback test (existing) | Loopback found; In 1 silent |
| C2 Chain reference | (1) Pull **cable A** out of GE-7 IN (its other end stays in ProRMP OUTPUT). (2) Pull **cable B** out of ProDI INPUT (leave it hanging from GE-7 OUT). (3) Plug cable A's free end into **ProDI INPUT**. Now ProRMP OUTPUT → cable A → ProDI INPUT, and the pedal is out of the chain. | Chain reference: sweep + tones → H_chain(f), chain gain, linearity | Return present; linear (≤ 0.25 dB); no clipping. Reports the chain tilt. |
| C3 GE-7 bypassed | (1) Pull **cable A** out of ProDI INPUT and plug it back into **GE-7 IN**. (2) Plug **cable B**'s free end back into **ProDI INPUT**. (3) Pedal powered, **footswitch LED OFF** (it powers up bypassed). The chain is back to normal, pedal bypassed. | Bypass reference → H_bypass(f) | Return present; H_bypass/H_chain within a few dB (the pre/de-emphasis mismatch; reported, not judged, until we have a number) |
| C4 GE-7 engaged, flat | No cable moves. **Press the footswitch: LED ON.** All sliders and LEVEL at the centre detent. | Chain test (existing): gain, linearity, predicted peak | Engaged/bypass ≈ flat within ~1.5 dB (the measured flat insertion was −1.2 dB) |

Normalisation after R1, both stored per session:
- QC's "Response vs model" uses H_wet/H_chain, the whole pedal including pre/de-emphasis.
- The rung A EQ fit uses engaged ÷ bypassed (decision D1).

**R1 as built (2026-10-18; `pedal_model/rig/calibrate.py`, the Calibrate page, `tests/rig/test_calibration_steps.py`).**
Each of C2–C4 measures a complex reference H(f) = return ÷ send from a 3 s log sweep, aligned on the
loopback only (so the chain's and the pedal's own delays stay in the phase), stored in the session's
calibration with its raw capture as `calibration/<profile>_ref_<name>_raw_<sr>k_<stamp>.flac`. Re-running a
step clears the steps after it. Differences from the table above, all deliberate:
- **QC divides the GE-7 by the bypass reference, not the chain.** The expected curve (Ge7Sim, later the
  block model's ratio) and the slider-map band fits describe the EQ alone, so engaged ÷ bypassed is the
  like-for-like comparison, and the same quantity as D1. One line in the profile (`qc.reference`:
  `chain`, `bypass` or `engaged_flat`) switches it; other pedals default to `chain`. Old sessions without
  references fall back to the scalar chain gain, with a QC warning.
- **C4 measures a reference sweep** (engaged at the calibration preset), not the chain test, so it can be
  compared with C3 directly.
- **Checks after each move:** C3 fails if bypassed differs from the chain by more than 6 dB (wiring or
  power), and warns if it is identical to the chain within 0.02 dB (cables not moved; a true-bypass pedal
  would also do this). C4 warns if it is identical to bypassed (footswitch still off), and warns above
  3 dB from bypassed (controls not at the preset). The 3 dB is looser than the 1.5 dB above, because the
  measured flat insertion (−1.2 dB) plus any ripple would trip 1.5 dB. Tighten once C4 has been seen on
  the real pedal.
- **Simulator:** the simulated chain now has a +4 dB high-frequency tilt (size from the re-amp-only
  captures, shape illustrative) and cable/bypassed/engaged states, so the demo and the tests exercise the
  normalisation. With only a scalar calibration, the tilt moved the simulated 800 Hz slider-map band fit
  to 2.26 kHz. With the references it is back at 790 Hz.

**First hardware session (2026-10-18, `sessions/2026-10-18_1238_48k`).** Recorded with code that
had a lag bug, so its stored alignments are wrong. `scripts/reanalyse_session.py` recomputes everything
from the raw FLACs. Measured:
- **Interface round trip:** 3718–3725 samples (77.5 ms) at 48 kHz with 1024-sample blocks. It moves a
  few samples between stream opens. The ProRMP/ProDI chain adds +1 sample.
- **Chain response** (re 1 kHz): −2.0 dB at 30 Hz, +1.4 dB at 10 kHz, +1.7 dB at 15 kHz.
- **Bypassed GE-7 vs cable:** −0.34 dB, flat within 0.08 dB from 50 Hz to 10 kHz.
- **Engaged ÷ bypassed at flat:** −1.43 dB (Studio One era: −1.2 dB).
- **Flat preset vs model,** after re-analysis: 0.42 dB RMS (train) and 0.45 dB RMS (val), from 100 Hz
  to 8 kHz.
- **Return level is low:** the training signal peaks near −31 dBFS on In 1. C2 asks for about +19 dB
  more In 1 gain.

**Corrected 2026-10-20: two causes, not one.** (1) Sessions 1238/1307 (stream in the GUI process)
really lost audio: both inputs are equally incoherent with the send. The separate audio process fixed
that; from session 1422 the return (In 1) is coherent to 1.000 in every section. (2) The loopback (In 2)
clips above about -31 dBFS of send, in every session. That alone made the integrity check fail from
21.0 s on, where sweep_mid starts. The 1422 and 1638_2 returns are good data despite their 'fail'
verdicts. Fix: loopback sent 30 dB lower, and C1 tests loop linearity. The note below (now corrected)
first blamed everything on cause (1).

**Resolved, same day (partly wrong, see above): lost audio, not the estimator.** The pink-vs-sweep gap below came from audio
silently missing in the Run-page takes: 28–32 % of 1024-sample blocks in every take of both sessions,
with the driver reporting no dropout. All 13 calibration captures were clean. The Python audio callback
was starved by the GUI's live views. Fixes: the stream now runs in its own process; a QC check compares
the loopback with the send block by block; the takes of both sessions fail it and must be recaptured.
Also found in session 1307: C2 was measured with the pedal still in the chain. C4 now fails when its
curve is identical to C2's, and the chain offset now comes from one capture, because the latency
moves 3–9 samples between stream opens.

**Open issue (superseded by the note above): the pink-noise and sweep responses disagree.** Per-section H1 from the pink-noise sections
reads about 2 dB below the sweep sections, sloping to about 3 dB lower at 30 Hz. It is the same at −24,
−12 and −3 dBFS, and the same in train and val. The sweeps agree with the C4 reference sweep. Because it
doesn't change with level, clipping is not the cause; it looks like a property of how the response is
estimated from noise, or of the noise sections themselves. Not diagnosed yet. QC averages both kinds of
section, so its curve sits between them. **Must be settled before rung A**, because the fit needs the
right H(f). The first check: compare a sweep-only H against a pink-only H on one take, against the
references.

**Gain staging (2026-10-21).** The capture chain must stay linear for engaged ÷ bypassed to cancel
the DI boxes. In session 2026-10-21_0237 the 1824c's return path compressed near -15 dBFS, so every
boosting preset (all_up, bass_boost, bright) captured the DI path's clipping, not the pedal's. Fixes:
C2 now measures that ceiling at 1 kHz and 60 Hz, and auto-level sets a per-take send level so the
return peaks 6 dB under it (recorded per take; the fit uses it). The GE-7's own clipping at
guitar-level input is real sound and is captured separately later, with the rig's ceiling
raised (pad / gain staging) so the pedal clips first.

**For the Klon (nonlinear):** dividing by references only works for a linear pedal. A clipping
pedal needs the input side (send → ProRMP → pedal input) and the output side (pedal output → ProDI
→ In 1) measured separately, since C2 only gives their product. To plan before the Klon work.
Decided 2026-10-22 (owner): build it after the GE-7 record run. It is a send-side step (ProRMP
OUTPUT into a line input whose full scale in dBu comes from the 1824c manual), so each take
records the pedal's input level in dBu. For a nonlinear pedal auto-level is off: the send is
set for a target input level (the pedal's operating point), and the return gets its headroom
from the pad and input gain set before calibration.

**R2. Capture session (the record run; revised 2026-10-22, owner agreed).** 48 kHz, hardware.
The linear fit only uses the log sweeps (the dataset, `pedal_model/rig/dataset.py`, uses sweeps
only: pink noise read ±0.2 dB noisy below 200 Hz and loud pink compressed the rig's low end), so
almost every step plays the 62 s sweep signal instead of the 642 s train signal. Start the session
with the **sweeps** signal kind; the Plan page's **Load record run** button builds the list
(`profiles/ge7.json` "record_run", `Profile.record_run()`):

| Block | Steps | Audio |
|---|---|---|
| C1–C4 | – | 0.5 min (+ cable moves) |
| Repeatability (P0.4): flat and smile (sweeps + val), 2 untouched repeats each, then each once after re-plugging cables A and B | 8 | 11 min |
| Slider map: each slider alone at ±15, ±10 (end stops and panel ticks: set exactly; ±7.5 falls between ticks, and A0 pins the parameters as well or better at ±10) | 32 | 36 min |
| Interaction (P0.1): each adjacent pair at (+15, +15) and (+15, −15), plus 100 Hz + 800 Hz and 800 Hz + 6.4 kHz | 16 | 18 min |
| The other presets, sweeps + val (all_up_2 and all_up_pad15 left out: duplicates) | 11 | 25 min |
| Random settings, 5 dB grid, seeded: 8 for fitting (seed 1), 8 for validation (seed 2) | 16 | 18 min |
| Full train signal on flat, smile and all_up_3 (time-domain checks; keeps the hybrid-ML option) | 3 | 36 min |

About 2.4 h of audio, plus the time to set about 85 settings. Every take is auto-levelled and
checked live for linearity and signal-to-noise. Afterwards: `scripts/ge7_dataset.py <session>`.
Optional at the end: a bypass take to check drift (re-running C3 replaces the reference, so this
is a separate step still to build). A repeat at 96 kHz (slider map plus 4 presets) gives the
sample-rate consistency check.

Random settings are rounded to 5 dB so they can be set repeatably by eye: an unrepeatable slider
position is noise on the model's input. Distant pairs are three octaves apart (choice, 2026-10-22).

## Rung A: fit

**Decisions.**
- **D1. Reference.**
  - **EQ data:** engaged ÷ bypassed, which cancels the chain and the pre/de-emphasis.
  - **Input/output blocks:** fitted from bypass ÷ chain.
  - **Revisit if:** bypass drifts across the session.
- **D2. Band 7 shape.**
  - H7a: a first-order shelf branch, s/(s + ω₇).
  - H7b: a second-order shelf branch.
  - H7c: a low-Q band-pass tuned high.

  Fit H7a first, then the others, and keep the one with the lowest **validation** error (not
  the test set: choosing with it would spend it).
- **D3. Frequency domain first.** Fit H(jω) to the measured complex responses (H1 estimates from the
  sweeps), with no IIR filtering inside the training loop. Afterwards check in the time domain: ESR on the
  val signal through the discretised model.
- **D4. Continuous time.** Fit the s-domain blocks; discretise only to render, with the bilinear transform
  pre-warped at each band.
- **D5. Loss.** Weighted log-magnitude MSE (reuse `losses.log_magnitude_loss`, `make_frequency_weights`)
  plus an excess-phase term weighted by coherence. Phase is valid because takes are aligned on the
  loopback latency.
- **D6. Split (revised 2026-10-22, owner agreed; `ge7.json` record_run.split).**
  - **Fit:** slider map, interaction set, the repeatability takes, the 8 "fit" random settings,
    and the presets not in the test set (flat, all_down, all_up, all_up_3, bass_boost, bright,
    low_cut, high_cut).
  - **Validation:** the 8 "val" random settings. Used to choose between model variants (band-7
    shape, loss weights), never fitted.
  - **Test:** smile, frown, mid_scoop, warm, presence (the musical settings, each mixing boosts
    and cuts). Scored once, at the end.
- **D7. Fitter stages.**
  1. Each band from its own slider-map steps (fₖ, Qₖ, σₖ, with the shared ρ, e, τ).
  2. LEVEL from the LEVEL map steps.
  3. Everything jointly.

  Levenberg–Marquardt (decision A0, `ge7_fitter.py`), several starts per band around the panel
  frequency.

**A0 known answer, before any real data.** Draw parameter values within a plausible range around the
guesses, generate H(jω) for the R2 slider settings, add noise at the measured coherence, and fit from the
guesses.
- **Noise-free:** recover every identifiable parameter within 1 %.
- **With noise:** within its reported uncertainty.

If A0 fails, the problem is the fitter or identifiability, and real data would only hide it.

**A0 result (2026-10-18, `tests/ode/test_ge7_blocks.py`).** Truth drawn around the guesses (fₖ ±15 %,
Q 1.2–3, σ and ρ ±20–30 %, pot_end 0.005–0.03, both pot laws 0.8–1.5, delay ±20 µs); data: all-centre,
slider map, interaction pairs and 24 random settings, 200 log points 20 Hz–19 kHz, complex.
- **Noise-free:** all 31 ratio parameters and all 5 bypass parameters recovered to about 1e-6
  (relative), in 10 Levenberg–Marquardt iterations (3 s on the CPU).
- **0.05 dB noise:** every error within 4.5σ of its reported uncertainty.
- **Identifiability** (Fisher at 0.05 dB white noise; real errors will be larger and partly
  systematic): condition number about 2e4. The weakest directions are the ones expected: the EQ's
  ρ scale against all the σ's, and the LEVEL ratios against the overall gain. Without phase the
  residual delay cannot be identified at all, so the phase term stays.
- **Decision (D7 changed):** Levenberg–Marquardt with a forward-mode Jacobian instead of Adam then
  L-BFGS. It is a small least-squares problem, and JᵀJ is the Fisher report. Revisit if the parameter
  count grows or the fit moves to the time domain.

**A success criteria** (relative to measured floors, not invented numbers):
- every held-out setting within 2× the P0.4 floor (RMS, 50 Hz–10 kHz);
- better than Ge7Sim on every held-out preset;
- P0.1 to P0.3 reproduced with no special-casing;
- **consistency:** parameters from two sessions, from 48 vs 96 kHz, and from the slider map alone vs the
  presets alone, agree within their uncertainties.

**Identifiability.**
- **Expected trade-offs:** ρ against σₖ (both set the boost size), e against the pot law near the ends,
  and overall gains, which are known only as a product.
- **What separates them:** the two slider positions per side, plus the interaction set.
- **New tool:** a Fisher-information/Hessian report at the optimum, saying which parameters (or
  combinations) are pinned, with uncertainties. Results report only those.

## Parked: if the block diagram is too simple (hybrid, physics-informed ML)

Raised by Brian, 2026-10-18, to come back to later. Nothing is built for it yet.

**The concern.** The fit assumes the topology. If the real GE-7 does something the block diagram leaves
out, the model can't sound right however well it is fitted. **The correct sound matters most:** more than
speed, and more than elegant parameters.

**What is and isn't the risk** (my reading, not settled):
- **The optimiser isn't the risk.** Levenberg–Marquardt and Adam minimise the same loss over the same
  structure. Neither one makes the model rely on the topology more or less. LM was chosen only because it
  converges reliably here and gives the uncertainty report.
- **The structure is the risk.** A0 can't detect it, because A0's data come from the model itself. Only real
  data can.

**How we'll know (rung A gives the data).** Model mismatch shows up as residuals that are
- **structured:** they repeat across settings, or follow a slider or a frequency;
- **larger than the repeatability floor (P0.4).**

Noise doesn't do that. The checks:
1. Holdout residual against the floor, per setting and per frequency.
2. The same data fitted by a black-box baseline: the `LinearBiquadModel` cascade, and later a
   knob-conditioned neural model. If a black box clearly beats the block model on holdout, that gap is
   what the structure costs.
3. Time-domain ESR on the val signal, and listening, for what the frequency-domain loss might hide.

**Options if there is a gap, in order:**
1. **Richer blocks where the residual points:**
   - the band-7 alternatives (D2);
   - a second-order term in the pot loading;
   - a finite-gain op-amp;
   - an asymmetric LEVEL law.

   This keeps the model interpretable.
2. **A hybrid model:** the fitted block model, plus a small learned correction trained on what the
   block model can't explain. For example, a few low-order sections whose coefficients come from a small
   network conditioned on the sliders, or a learned correction to each band's (f, Q, σ) as a function of
   all the sliders.
3. **Black-box, conditioned on the controls,** as the fallback if the structure is badly wrong.

**"Physics-informed" constraints for the learned part** (so it can't learn something no real EQ does):
- It stays stable and causal.
- All-centre stays flat.
- Boost and cut stay near mirror images where the circuit makes them so.
- Its output varies smoothly with each slider.
- It's small compared with the block model, through a penalty on its size, so the physics carries the
  sound and the network only fixes the remainder.

**Deployment constraints still apply** (Daisy and the plugin):
- It must be cheap to evaluate when a slider moves.
- It must be expressible in continuous time, or at least at every sample rate the plugin supports.
- It must ship in the handoff package with golden vectors, like the rest.

**Decision:** parked. Revisit after rung A, using the residual and baseline comparison above. Not
before: until real data show a gap, any added flexibility only fits noise.

## Storage and use in the GUI

- **Parameter file** `pedal_model/rig/models/ge7/ge7_pedalode_v<N>.json`:
  - the block diagram's id and version, plus the band-7 hypothesis used;
  - every block parameter with units, its starting guess and its fitted uncertainty;
  - the training sessions and presets, the git commit;
  - the fit metrics, holdout results, repeatability floor and the consistency table;
  - golden test vectors for the Daisy and the plugin (see the plugin section).
  - New file per refit, never overwritten.
- **Profile link:** `profiles/ge7.json` gains `"model": ...`. `qc.expected_response()` evaluates the
  model, falling back to Ge7Sim if no file exists.
- **GUI:** Review shows which model version was used, and the demo backend can simulate the fitted model.

## Rung C: Daisy (decision D8, after rung A)

- **Option 1 (favoured): implement the block diagram directly.**
  - Each band block is one biquad and the shelf block one first- or second-order section, discretised
    once.
  - The summing stage's two node equations are solved in closed form each sample (linear, so no
    iteration).
  - Moving a slider changes only that band's pot-split terms; the band blocks never change.
  - *Medium confidence on numerical behaviour at the extreme settings; prove it in Python first.*
- **Option 2:** at control rate, convert the current H to second-order sections (degree-14 root
  finding on the MCU).

## Secondary target: a Studio Pro plugin (built in BrianEQ, not here)

The plugin is written in the BrianEQ repo. This repo's job is a model complete enough that the plugin
is a port, not a second modelling project. What that requires of the model:

- **Sample-rate independent.** The parameter file holds continuous-time block parameters (D4), never
  digital coefficients. The plugin discretises at the host rate: 44.1, 48, 88.2, 96, 176.4 or 192 kHz.
  The Daisy runs at 48 kHz only, so the plugin is the stricter test.
- **All of the pedal.** All 8 sliders over their full range, the input and output blocks
  (pre/de-emphasis, coupling high-passes), and the bypass path, so "bypassed" can sound like the
  pedal's buffer too. The ratio fit alone (D1) would leave out the outer blocks.
- **Automation-safe.** Plugin sliders move while audio plays. The direct block form (D8 option 1)
  suits this: the band blocks are fixed, and a slider only changes its own pot-split conductances,
  so updates are cheap and the structure stays stable while they change. To prove in Python: a
  full-range slider sweep during audio, no clicks and no instability. *Medium confidence that the
  plain bilinear form is enough; a topology-preserving (zero-delay-feedback) form is the fallback.*
- **One reference, two ports.** A Python reference renderer `render_ge7(params, sliders, sr, x)` is
  the single definition of the sound. Both C++ ports null against it.

**Handoff package** (one folder per fitted model version, next to the parameter file):
1. the parameter file (above);
2. `docs/pedalode_ge7_spec.md`: the block equations, the pot law, the slider-to-wiper mapping, and
   the discretisation recipe, written so it can be implemented without reading Python;
3. golden vectors: H(jω) on a fixed grid for a fixed list of slider settings, plus short
   time-domain input/output pairs at 44.1, 48 and 96 kHz (impulse, a sine sweep, and a slider
   automation sweep), with the tolerances a port must meet;
4. **D9 (decide at rung C):** whether to also write a portable, header-only C++ core here (no
   JUCE and no libDaisy dependency) that both the firmware and the plugin include. It would remove one
   port, but tie the BrianEQ repo to this one.

**Out of scope for now (open question):** the GE-7 runs on 9 V and clips when driven hot. DAW levels
can drive a plugin much harder than a guitar drives the pedal. This plan models only the linear
pedal. A Step 0 level probe can measure where clipping starts, so the plugin can at least document
its valid input range.

## Updates needed in PedalODE (`pedal_model/ode/`, reviewed 2026-10-18)

There is no GE-7 PedalODE code yet. `results/pedalode/GE-7_v1/` holds one flat capture and nothing else.

| Existing | Status for the GE-7 | Update |
|---|---|---|
| `filters.py`: `PeakBiquad`, `HighShelfBiquad`, `SecondOrderHighpass` (digital RBJ parameters, torch) | Digital coefficients, not physical block parameters | New `blocks.py`: continuous-time transfer-function blocks in torch (gain, first-order high- and low-pass, zero, band-pass, shelf branch, pot split, summing stage), evaluated at jω; a bilinear renderer for time-domain checks |
| `wh_model.py`: `LinearBiquadModel` (series cascade) | The wrong structure for the GE-7, but the right **baseline** | Keep. Fit it on the same data as the "cascade" rung, to measure what the block diagram adds. Never delete a simpler rung |
| `fitting.py`: `KlonSimFitter` (one capture, manifest sections, WH stages) | Built for one capture of a nonlinear pedal | New `ge7_fitter.py`: many captures, each with its own slider vector; frequency-domain; stages per D7 |
| `losses.py`: `log_magnitude_loss`, `make_frequency_weights` | Reusable | Add a coherence-weighted excess-phase term |
| `identifiability.py`: section selection, transfer-function helpers | Partly reusable | Add the Fisher/Hessian report |
| `step0.py`: KlonSim pre-fit diagnostics | KlonSim-specific | New `scripts/ge7_step0_probes.py` (P0.1–P0.4) |
| `docs/pedalode_klonsim_spec.md` | Calls the GE-7 "the known-answer linear test" | Write `docs/pedalode_ge7_spec.md` from this plan once A0 passes |
| Rig: `calibrate.py`, `qc.py`, GUI Calibrate page | Scalar chain normalisation | R1: steps C1–C4, H_chain and H_bypass stored, guided cable moves, lock-step tests |

## Order of work

1. R1: guided reference captures, with tests (no hardware needed to build; demo-tested).
2. A0: `blocks.py`, the GE-7 block model, the fitter, and known-answer recovery (no hardware).
3. R2: the capture session (you at the rig).
4. Step 0 probes; settle D0 and D2.
5. Rung A fit, holdout and consistency tables, parameter file v1; the GUI switches to it.
6. Rung C (after D8): the Daisy firmware.
7. Rung C2: the plugin handoff package; the plugin itself is built in BrianEQ.
