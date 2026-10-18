# GE-7 PedalODE plan: the GE-7's circuit topology, with learned component values

Status: plan, revised 2026-10-18 with the GE-7 signal chain from Brian (summary of an LTspice teardown of
the Hobby-Hour schematic). Nothing here is built yet. Confidence tags follow the PedalODE rule: anything
not measured says where it came from and how sure it is.

## Goal, and what each rung proves

Replace Ge7Sim (a hand-calibrated **series** cascade of 6 peaking biquads + a high shelf + a LEVEL law,
`pedal_model/synthetic/ge7.py`) with a model built from the GE-7's **actual stages and nodes**, whose
**component values are learned** from hardware captures. One model covers every slider setting.

| Rung | Proves | Done when |
|---|---|---|
| A0. Known answer | The fitter recovers this topology's parameters at all | Synthetic data from the model itself, parameters recovered (criteria below) |
| A. Topology fit | This circuit explains the GE-7 and generalises to unseen slider settings | Held-out settings within the repeatability floor; every one better than Ge7Sim |
| B. Component values | The fitted values are the real parts | Identifiable combinations within part tolerances of the schematic values |
| C. Deploy | The same model runs on the Daisy | Firmware nulls against the Python model; CPU budget met |

Rungs in order; a rung starts only when the previous one passes. Later, beyond this plan: capture and
fitting in one closed loop that decides how much to capture (Brian, 2026-10-18: "eventually, not yet").

## The circuit

**Source and confidence.** The stage list below is Brian's summary (2026-10-18) of a GitHub LTspice
teardown of the Hobby-Hour GE-7 schematic, plus Hobby-Hour, smemmusic and a DIYstompboxes thread. It is
**secondary**: nobody has read the schematic image itself, the level position is "one author's reading",
and **no component values are known** except the corner frequencies and ratios quoted. Treat the topology
as likely and every number as a starting point, not a prior.

```
IN ─► [1] Input stage ─┬─► dry path ──────────────────────────────┐
     non-inverting ×10  │                                          ├─► [5] JFET switch ─► [6] Output buffer ─► OUT
     zero ≈ 2.2 kHz     └─► [2] LEVEL ─► [3] EQ bank (summing) ────┘   (flip-flop on      emitter follower
     (pre-emphasis),        unity buffer, pot     2 nodes, 7 branches      the footswitch;   pole ≈ 2.2 kHz
     AC-coupled, 4.5 V      wiper RC ≈ 7.2 Hz,    on slider wipers         powers up in      (de-emphasis),
     bias                   cut: shunt to gnd,                             BYPASS)           AC-coupled
                            boost: 5:1 divider (≈ +14 dB)
```

| Stage | What the teardown says | Confidence | In the model |
|---|---|---|---|
| 1. Input | Non-inverting ×10, AC-coupled, zero near 2.2 kHz (deliberate pre-emphasis for noise) | Medium (secondary source) | Gain G_in, zero f_z, coupling high-pass f_hp1 |
| 2. LEVEL | Unity op-amp buffer; pot wiper with an RC (≈ 7.2 Hz); toward input: shunt (cut); toward inverting node: 5:1 divider (≈ +15 dB) | Medium; position before the EQ is "likely, not certain" | Pot-split network, same trick as the EQ bands |
| 3. EQ bank | One summing op-amp; nodes Eq_Input and Eq_Feedback; each band a branch on its own slider wiper; toward Eq_Input: cut ≈ 15 dB, toward Eq_Feedback: boost ≈ 15 dB | Medium-high (this is the standard graphic-EQ circuit, and the parts list fits) | Exact nodal solution below |
| 3a. Bands 1–6 | Gyrator RLC branches at 100 … 3200 Hz; IC1–IC3 = three dual op-amps = six gyrators | Medium | Series R–L–C admittance per band |
| 3b. Band 7 | 6.4 kHz is a **shelf**, not a resonator; construction unknown | Medium that it's a shelf (agrees with our own measurement); unknown how | Hypotheses H7a–c below |
| 5. Switch | Two JFETs (EQ output, dry signal) driven by a bistable transistor flip-flop on the footswitch | Medium | Not modelled (on/off); matters for the capture procedure |
| 6. Output | AC-coupled emitter follower; R63 + C31 + R62 form a 2.2 kHz low-pass that cancels the input zero | Medium | Pole f_p, coupling high-pass f_hp2 |

**Consequences.**
- In **bypass** the signal still passes through stages 1 and 6, so the bypass response is H_in·H_out,
  which is flat only as far as the 2.2 kHz zero and pole cancel.
- The **engaged/bypass ratio** is H_level·H_eq: the re-amp chain and the pre/de-emphasis both
  cancel. That makes it the cleanest data for the EQ fit (decision D1).
- **Order of the LEVEL stage.** For small signals, linear stages in series multiply, so LEVEL before or
  after the EQ gives the same transfer function. Position matters for headroom (where clipping starts
  in `all_up`), so the model keeps the teardown's order.
- The GE-7 **powers up in bypass.** Every capture instruction must say "footswitch LED on".

## The EQ stage, exactly (ideal op-amp)

Node P = Eq_Input (non-inverting input), node N = Eq_Feedback (inverting input). The source drives P
through R_a; the output feeds N through R_b. Band k: slider pot R_p between P and N, wiper at fraction
xₖ ∈ [0, 1] from P (pot sections R1ₖ = xₖ·R_p to P, R2ₖ = (1 − xₖ)·R_p to N, in practice plus small end
resistors). The branch admittance Yₖ(s) runs from the wiper to ground. With V_P = V_N = v (ideal op-amp),
solving each wiper node gives the current each band draws from P and from N:

    i_Pk = v · G1ₖ Yₖ / (G1ₖ + G2ₖ + Yₖ)        i_Nk = v · G2ₖ Yₖ / (G1ₖ + G2ₖ + Yₖ)     (G = 1/R)

    H_eq(s, x) = V_out / V_in = [1 + R_b Σₖ G2ₖ Yₖ / (G1ₖ + G2ₖ + Yₖ)] / [1 + R_a Σₖ G1ₖ Yₖ / (G1ₖ + G2ₖ + Yₖ)]

*Derived here from the circuit description, for an ideal op-amp; check the node assignment against the
schematic (which end of the pot is boost). The structure, not the labels, is the point. Checked
numerically 2026-10-18 with made-up values: flat at the centre to 1e-15 dB, boost and cut exact
mirrors for R_a = R_b, effective Q falling at milder settings.* Its properties
are each testable against data:
- **Centre detent** (G1 = G2) with R_a = R_b: numerator = denominator, so H_eq = 1 (flat).
- **Full boost** (wiper at N, G2 → ∞): H_eq = 1 + R_b·Yₖ. A series RLC peaks at resonance, giving a peak
  of size set by R_b/Rₖ.
- **Full cut:** H_eq = 1/(1 + R_a·Yₖ): the mirror image when R_a = R_b.
- **Q depends on the slider position:** the pot sections appear in series with Yₖ through
  G1ₖ + G2ₖ + Yₖ, so the effective Q drops at mild settings. This is the DIYstompboxes observation,
  produced by the physics with no extra Q(x) parameter.
- **Band interaction:** all bands share P and N, so adjacent bands interact. A series cascade cannot do
  this, and it is the error pattern recorded in `docs/ge7_capture_results.md`.

**Branch admittances.**
- Bands 1–6, a gyrator RLC: Yₖ(s) = 1 / (Rₖ + s·Lₖ + 1/(s·Cₖ)), with Lₖ the simulated inductance
  (Lₖ = R·R′·C′ inside the gyrator; only Lₖ is visible from outside).
- Band 7, the shelf; construction unknown, so three hypotheses, decided by data (decision D2):
  - **H7a:** an R–C branch (no inductor): Y = s·C/(1 + s·R·C). Boost or cut above 1/(2πRC), which is a
    shelf. *Simplest; my first guess.*
  - **H7b:** a gyrator branch tuned high with low Q, behaving like a shelf within the audio band.
  - **H7c:** something else on the schematic; needs the image.

**LEVEL stage.** The same pot-split trick around a unity buffer: cut by shunting to ground, boost by a
5:1 divider (20·log10 5 ≈ 14 dB; the teardown says "about 15"). The measured law (+10 → +12.1 dB,
−10 → −14.4 dB, from `ge7.py`) is asymmetric. The node model of this stage, with the pot taper as a
parameter, has to reproduce that.

**Parameters** (all positive, fitted as logs):
- EQ stage: R_a, R_b, R_p, and the pot taper (linear assumed; one shape parameter if not).
- Bands 1–6: Rₖ, Lₖ, Cₖ each.
- Band 7: per its hypothesis.
- LEVEL network: its pot, the divider, the wiper RC.
- Input and output: G_in·G_out together (identifiable only as a product), f_z, f_p, f_hp1, f_hp2.

Roughly 35. Only some combinations are identifiable from audio (see Identifiability).

## What we have measured (Studio One era, not re-verified)

From `docs/ge7_capture_results.md` and the single-slider calibration in `ge7.py`:
- **Linearity:** within 0.25 dB from −18 to −3 dBFS on flat.
- **Flat:** −1.2 dB insertion loss, broadband.
- **Single sliders at +10 dB:** peaks near 87, 190, 383, 792, 1599, 3380 Hz, at 72–97 % of the setting.
  The 6.4k band rises like a shelf and peaks near 13 kHz.
- **LEVEL:** +10 → +12.1 dB, −10 → −14.4 dB.
- **Ge7Sim:** misses the presets by 1.35–4.42 dB RMS, at band edges and between adjacent bands.
- **Re-amp chain:** a +4 dB tilt from 82 Hz to 10.5 kHz.

These measurements were made relative to flat, with the chain not removed, on a different rig.
Consistent with the circuit above, but they don't confirm it.

## Step 0: data probes, run before fitting

- **P0.1 Interaction test.**
  - **Capture:** each adjacent pair (6 pairs) at (+15, +15) and (+15, −15), and two non-adjacent pairs as
    controls (16 sweep steps, ~20 min).
  - **Compare:** each against the sum, in dB, of the matching single-slider responses from the slider map.
  - **Expect:** under the nodal model, no pair adds up exactly in dB; adjacent pairs deviate more than
    distant ones. How much depends on the component values: a numerical check with made-up values
    (2026-10-18) gave 12.9 dB for an adjacent pair and 4.6 dB for 100 Hz with 3.2 kHz. So the firm
    prediction is the ordering and that the deviation is non-zero, not a size.
  - **Decision D0:** if even adjacent pairs add up within the floor, the shared-node interaction is
    negligible and a cascade suffices. Then stop and re-plan.
- **P0.2 Reciprocity.** From the slider map's ±7.5 and ±15 dB steps, compare dB(+x) with −dB(−x). They
  are equal if R_a = R_b; a constant offset or a tilt tells us R_a ≠ R_b or the pot is asymmetric.
- **P0.3 Q against the slider.** From the slider map's band fits: the effective Q at ±7.5 should be lower
  than at ±15. This checks the physics.
- **P0.4 Repeatability floor.** Capture flat and smile three times each, and once more after re-plugging.
  The RMS difference (50 Hz–10 kHz) is the floor every criterion below is measured against.

## Rig prerequisites (R1–R2)

**R1. Reference captures, each with a guided cable move and a test right after it.** The rig's QC currently
divides by a single 1 kHz gain, so the ProRMP's tilt ends up in every "measured" response. Calibration
becomes four steps. The two guitar cables around the pedal get names, printed in the GUI:
**cable A** runs ProRMP OUTPUT → GE-7 IN, **cable B** runs GE-7 OUT → ProDI INPUT. The Setup diagram
highlights the cable to move, and the next step stays locked until the test after each move passes:

| Step | Cable move (exact) | Test that runs next | Must pass |
|---|---|---|---|
| C1 Loopback | none: Out 2 → In 2 already in place | Loopback test (existing) | Loopback found; In 1 silent |
| C2 Chain reference | (1) Pull **cable A** out of GE-7 IN (its other end stays in ProRMP OUTPUT). (2) Pull **cable B** out of ProDI INPUT (leave it hanging from GE-7 OUT). (3) Plug cable A's free end into **ProDI INPUT**. Now ProRMP OUTPUT → cable A → ProDI INPUT, and the pedal is out of the chain. | Chain reference: sweep + tones → H_chain(f), chain gain, linearity | Return present; linear (≤ 0.25 dB); no clipping. Reports the chain tilt. |
| C3 GE-7 bypassed | (1) Pull **cable A** out of ProDI INPUT and plug it back into **GE-7 IN**. (2) Plug **cable B**'s free end back into **ProDI INPUT**. (3) Pedal powered, **footswitch LED OFF** (it powers up bypassed). Now the chain is as normal, pedal bypassed. | Bypass reference → H_bypass(f) | Return present; H_bypass/H_chain within a few dB (the pre/de-emphasis mismatch; reported, not judged, until we have a number) |
| C4 GE-7 engaged, flat | No cable moves. **Press the footswitch: LED ON.** All sliders and LEVEL at the centre detent. | Chain test (existing): gain, linearity, predicted peak | Engaged/bypass ≈ flat within ~1.5 dB (the measured flat insertion was −1.2 dB) |

Normalisation after R1:
- QC's "Response vs model" uses H_wet/H_chain, the whole pedal including pre/de-emphasis.
- The rung A fit uses engaged ÷ bypassed (decision D1).

Both are stored per session.

**R2. Capture session** (48 kHz, hardware, about 3 hours of audio, mostly unattended; agreed 2026-10-18):
1. C1–C4.
2. Flat ×3 and smile ×3, then one more of each after re-plugging (P0.4).
3. Slider map (32 sweep steps).
4. Interaction set (16 sweep steps).
5. The 14 presets (full signal).
6. 8 random settings (seeded, saved as custom presets) as extra holdout.
7. A bypass capture at the end, to check drift.

## Rung A: fit

**Decisions.**
- **D1. Reference.**
  - **EQ data:** engaged ÷ bypassed, which cancels the chain and the pre/de-emphasis.
  - **Input/output stages:** fitted separately from bypass ÷ chain.
  - **Revisit if:** bypass drifts across the session (the end-of-session bypass check).
- **D2. Band 7.** Fit H7a first; fit H7b; keep the one with lower holdout error. Use H7c if the schematic
  shows something else.
- **D3. Frequency domain first.** Fit H(jω) to the measured complex responses (H1 estimates from the
  sweeps), with no IIR filtering inside the training loop. Afterwards check in the time domain: ESR on
  the val signal through the discretised model.
- **D4. Continuous time.** Fit the s-domain model; discretise only to render, with the bilinear
  transform pre-warped at each band. The 6.4k band at 48 kHz is where warping matters.
- **D5. Loss.** Weighted log-magnitude MSE (reuse `losses.log_magnitude_loss`, `make_frequency_weights`)
  plus an excess-phase term weighted by coherence. Phase is valid because takes are aligned on the
  loopback latency.
- **D6. Split.**
  - **Train:** slider map, interaction set, the old single-slider captures, and half the presets.
  - **Holdout:** smile, frown, mid_scoop, warm and the 8 random settings. Used once, at the end.
- **D7. Fitter stages.**
  1. Each band from its own slider-map steps: R_k, L_k, C_k, with R_a, R_b and R_p shared.
  2. LEVEL from the LEVEL map steps.
  3. Everything jointly.

  Adam, then L-BFGS; several starts with each band near its panel frequency.

**A0 known answer, before any real data.** Choose parameter values, generate H(jω, x) for the R2 slider
settings, add noise at the measured coherence, and fit.
- **Noise-free:** recover the identifiable combinations within 1 %.
- **With noise:** within their reported uncertainty.

If A0 fails, the problem is the fitter or identifiability, and real data would only hide it.

**A success criteria** (relative to measured floors, not invented numbers):
- every held-out setting within 2× the P0.4 floor (RMS, 50 Hz–10 kHz);
- better than Ge7Sim on every held-out preset;
- P0.1 to P0.3 reproduced with no special-casing.

**Identifiability.**
- **Pinned by the data:** each band's ω = 1/√(LC) and its losses (R) relative to the pot and the summing
  resistors. Overall gains only as a product.
- **Not pinned:** each factor inside a gyrator's L.
- **New tool:** a Fisher-information/Hessian report at the optimum, listing what is pinned, with
  uncertainties.

## Rung B: component values

**Needs the schematic values** (question 1 below). Compare the identifiable combinations with the
schematic, using the parts list's tolerances. If we only know "typical" tolerances (5 % resistors,
5–10 % film capacitors), that's an assumption and has to be labelled as one. Deliverable: recovered values
with uncertainties, the non-identifiable list, and the comparison table.

## Storage and use in the GUI

- **Parameter file** `pedal_model/rig/models/ge7/ge7_pedalode_v<N>.json`:
  - the topology id and version, plus the band-7 hypothesis used;
  - every parameter with units and uncertainty;
  - the training sessions and presets, the git commit;
  - the fit metrics, holdout results and repeatability floor.
  - New file per refit, never overwritten.
- **Profile link:** `profiles/ge7.json` gains `"model": ...`. `qc.expected_response()` evaluates the
  model, falling back to Ge7Sim if no file exists.
- **GUI:** Review shows which model version was used, and the demo backend can simulate the fitted model.

## Rung C: Daisy (decision D8, after rung A)

- **Option 1 (favoured): implement the nodes directly.**
  - Six fixed band branches plus the shelf branch, discretised once.
  - The two summing-node equations solved in closed form each sample (linear, so no iteration).
  - Moving a slider changes only the G1ₖ and G2ₖ conductances. That changes the node equations but
    never the branch filters.
  - *Medium confidence on numerical behaviour at the extreme settings; prove it in Python first.*
- **Option 2:** at control rate, convert the current H to second-order sections (degree-14 root
  finding on the MCU).

## Updates needed in PedalODE (`pedal_model/ode/`, reviewed 2026-10-18)

There is no GE-7 PedalODE code yet. `results/pedalode/GE-7_v1/` holds one flat capture and nothing else.

| Existing | Status for the GE-7 | Update |
|---|---|---|
| `filters.py`: `PeakBiquad`, `HighShelfBiquad`, `SecondOrderHighpass` (digital RBJ parameters, torch) | Wrong level of description: digital biquad coefficients, not circuit elements | New `circuit.py`: continuous-time admittance primitives (R, C, L or gyrator, series and parallel combinations) evaluated at jω in torch; a bilinear renderer for time-domain checks |
| `wh_model.py`: `LinearBiquadModel` (series cascade) | The wrong topology for the GE-7, but the right **baseline** | Keep. Fit it on the same data as the "cascade" rung, to measure how much the topology adds. Never delete a simpler rung |
| `fitting.py`: `KlonSimFitter` (one capture, manifest sections, WH stages) | Built for one capture of a nonlinear pedal | New `ge7_fitter.py`: many captures, each with its own slider vector; frequency-domain; stages per D7 |
| `losses.py`: `log_magnitude_loss`, `make_frequency_weights` | Reusable | Add a coherence-weighted excess-phase term |
| `identifiability.py`: section selection, transfer function helpers | Partly reusable | Add the Fisher/Hessian report |
| `step0.py`: KlonSim pre-fit diagnostics | KlonSim-specific | New `scripts/ge7_step0_probes.py` (P0.1–P0.4) |
| `docs/pedalode_klonsim_spec.md` | Calls the GE-7 "the known-answer linear test" | Write `docs/pedalode_ge7_spec.md` from this plan once A0 passes |
| Rig: `calibrate.py`, `qc.py`, GUI Calibrate page | Scalar chain normalisation | R1: steps C1–C4, H_chain and H_bypass stored, guided cable moves, lock-step tests |

## Order of work

1. R1: the guided reference captures, with tests.
2. A0: `circuit.py`, the GE-7 model, the fitter, and known-answer recovery on synthetic data.
3. R2: the capture session (you at the rig).
4. Step 0 probes; settle D0 and D2.
5. Rung A fit, holdout table, parameter file v1; the GUI switches to it.
6. Rung B (needs the schematic values).
7. Rung C (after D8).

## Open questions

1. **Component values.** The Hobby-Hour schematic image has them (R49, R62, R63, C31, the pots, the
   band parts). Could you read them off and paste them, or attach the image? They're needed for rung B
   and to check which pot end is boost.
2. **Band 7.** If the schematic shows how the 6.4 kHz shelf is built, that settles D2 before any fitting.
