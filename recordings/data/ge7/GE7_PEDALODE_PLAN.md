# GE-7 PedalODE plan: the GE-7's block diagram, with learned block parameters

Status: plan, revised 2026-10-18. Nothing here is built yet.
- **Level of description** (Brian, 2026-10-18): the **block diagram**, not individual components. Each
  block is a physical stage with a few physical parameters (gain, corner frequency, Q, pot law).
- **Starting values** (Brian): reasonable guesses. Recovering the true values is the point of the model.
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
| EQ | ρ_a, ρ_b | equal, set so full boost ≈ +15 dB | guess, derived at start-up |
| | end resistance e | 0.01 of the pot | guess |
| | pot law τ | linear, plus one shape parameter | guess |
| Bands 1–6 | fₖ | 100, 200, 400, 800, 1600, 3200 Hz | panel |
| | Qₖ | 2 | guess |
| | σₖ | set so full boost ≈ +15 dB at fₖ | guess, derived at start-up |
| Band 7 | f₇ | 6.4 kHz | panel |
| | σ₇ | set so full boost ≈ +15 dB at 15 kHz | guess |

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
- **P0.2 Reciprocity.** From the slider map's ±7.5 and ±15 steps, compare dB(+x) with −dB(−x). They are
  equal if ρ_a = ρ_b. An offset or a tilt shows asymmetry.
- **P0.3 Q against the slider.** From the map's band fits, the effective Q at ±7.5 should be lower than at
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

**R2. Capture session** (48 kHz, hardware, about 3 hours of audio, mostly unattended; agreed 2026-10-18):
1. C1–C4.
2. Flat ×3 and smile ×3, then one more of each after re-plugging (P0.4).
3. Slider map (32 sweep steps).
4. Interaction set (16 sweep steps).
5. The 14 presets (full signal).
6. 8 random settings (seeded, saved as custom presets) as extra holdout.
7. A bypass capture at the end, to check drift.

A repeat at 96 kHz (slider map plus 4 presets) gives the sample-rate consistency check.

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

  Fit H7a first, then the others, and keep the one with the lowest holdout error.
- **D3. Frequency domain first.** Fit H(jω) to the measured complex responses (H1 estimates from the
  sweeps), with no IIR filtering inside the training loop. Afterwards check in the time domain: ESR on the
  val signal through the discretised model.
- **D4. Continuous time.** Fit the s-domain blocks; discretise only to render, with the bilinear transform
  pre-warped at each band.
- **D5. Loss.** Weighted log-magnitude MSE (reuse `losses.log_magnitude_loss`, `make_frequency_weights`)
  plus an excess-phase term weighted by coherence. Phase is valid because takes are aligned on the
  loopback latency.
- **D6. Split.**
  - **Train:** slider map, interaction set, the old single-slider captures, and half the presets.
  - **Holdout:** smile, frown, mid_scoop, warm and the 8 random settings. Used once, at the end.
- **D7. Fitter stages.**
  1. Each band from its own slider-map steps (fₖ, Qₖ, σₖ, with the shared ρ, e, τ).
  2. LEVEL from the LEVEL map steps.
  3. Everything jointly.

  Adam, then L-BFGS; several starts per band around the panel frequency.

**A0 known answer, before any real data.** Draw parameter values within a plausible range around the
guesses, generate H(jω) for the R2 slider settings, add noise at the measured coherence, and fit from the
guesses.
- **Noise-free:** recover every identifiable parameter within 1 %.
- **With noise:** within its reported uncertainty.

If A0 fails, the problem is the fitter or identifiability, and real data would only hide it.

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

## Storage and use in the GUI

- **Parameter file** `pedal_model/rig/models/ge7/ge7_pedalode_v<N>.json`:
  - the block diagram's id and version, plus the band-7 hypothesis used;
  - every block parameter with units, its starting guess and its fitted uncertainty;
  - the training sessions and presets, the git commit;
  - the fit metrics, holdout results, repeatability floor and the consistency table.
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
6. Rung C (after D8).
