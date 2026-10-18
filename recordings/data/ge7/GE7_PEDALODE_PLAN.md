# GE-7 PedalODE plan: a physical-topology model with learned parameters

Status: plan, 2026-10-18. Nothing here is built yet. Confidence tags follow the PedalODE rule: anything
recalled rather than measured says so, with what is uncertain.

## Goal, and what each rung proves

Replace Ge7Sim (a hand-calibrated **series** cascade of 6 peaking biquads + a high shelf + a LEVEL law,
`pedal_model/synthetic/ge7.py`) with a model whose **structure is the GE-7's circuit topology** and whose
**parameters are learned** from hardware captures. One model must cover every slider setting.

| Rung | Proves | Done when |
|---|---|---|
| A0. Known answer | The fitter can recover this topology's parameters at all | Synthetic data from the model itself, parameters recovered (criteria below) |
| A. Transfer-function topology | Parallel resonators around one gain stage explain the GE-7, and generalise to unseen slider settings | Held-out presets fit to within the capture repeatability floor (below), every one better than Ge7Sim |
| B. Component values | The fitted network maps onto real R, C and pot values | Identifiable component combinations recovered within part tolerances of the schematic |
| C. Deploy | The same model runs on the Daisy | Firmware output nulls against the Python model; CPU budget met |

A must pass before B, B is not needed for C. Do not start a rung before the previous one passes.

## What we know (measured / inferred / assumed)

**Measured** (Studio One-era captures, `docs/ge7_capture_results.md`; single-slider calibration in
`ge7.py`; not re-verified by me):
- Linear: gain variation ≤ 0.25 dB across −18 to −3 dBFS on flat; ≤ 0.14 dB on most presets (frown 3.9 dB
  and bass_boost 2.3 dB are unexplained, probably the return chain, not the pedal).
- Flat insertion loss about −1.2 dB, broadband.
- Single sliders at +10 dB peak near 87, 190, 383, 792, 1599, 3380 Hz (panel: 100…3200); peak gains 72–97 %
  of the slider setting; the 6.4k band rises like a shelf and peaks near 13 kHz.
- LEVEL: +10 → +12.1 dB, −10 → −14.4 dB (asymmetric).
- Ge7Sim misses presets by 1.35–4.42 dB RMS (12 stepped-tone frequencies, relative to flat); the errors
  cluster at band edges (82 Hz, 10.56 kHz) and between adjacent boosted/cut bands. The doc's own words:
  "adjacent band skirts interact in a way Ge7Sim underpredicts".
- Re-amp chain: +4 dB tilt from 82 Hz to 10.5 kHz; noise floor −108.5 dBFS RMS.

**Inferred** from the measurements: the band-interaction errors are what a series cascade cannot produce
and a shared feedback network would; the top band's shelf-like shape suggests its resonator is wide or
coupled to a top-end roll-off. Both are hypotheses to test (Step 0), not conclusions.

**Recalled, not verified** (check against the schematic before relying on any of it):
- The GE-7 is a single op-amp "graphic EQ" stage: each band is a series resonator to ground, hung on a
  slider pot whose ends go to the op-amp's input side and feedback side, so moving the slider moves the
  band from the forward path (boost) to the feedback path (cut). *Medium confidence on the topology class
  (it is the standard graphic-EQ circuit, and the GE-7 is usually described this way); low confidence on
  details such as the exact pot wiring and isolating resistors.*
- The resonators use gyrators (simulated inductors), not real inductors. *Medium confidence.*
- Specific parts (op-amp type, transistor gyrators vs op-amp gyrators, component values): *low
  confidence; do not use any recalled value as a prior.*
- Input/output coupling capacitors give a low-frequency roll-off; the bypass is electronic (buffered).
  *Medium confidence.*

## Step 0: data probes before any model (cheap, decide the topology question with data)

**Decision D0: is it a cascade or a shared feedback network?** Settled by two measurements, not argument:

- **P0.1 Interaction test.** For a cascade, the dB response of two moved sliders is exactly the sum of each
  slider alone. For bands sharing one feedback network it is not, most visibly for adjacent bands. Capture
  the **interaction set**: each adjacent pair (6 pairs) at (+15, +15) and (+15, −15), plus two
  non-adjacent pairs as controls (16 steps, sweep signal, ~20 min). Compare each with the sum of the slider
  map's single-slider responses. Revisit criterion: if the sums match within the repeatability floor, the
  cascade is adequate and rung A changes shape (stop and re-plan).
- **P0.2 Boost/cut reciprocity.** In the standard graphic-EQ topology the cut curve is the reciprocal of
  the boost curve: dB(+x) ≈ −dB(−x) at every frequency (exact for an ideal circuit, approximate in
  practice). The slider map already captures ±7.5 and ±15. Asymmetry tells us the pot wiring is not
  symmetric (the LEVEL law already is not).
- **P0.3 Bandwidth vs slider position.** From the slider-map band fits: does Q change between ±7.5 and
  ±15? ("Proportional-Q" is typical of this topology. *Medium confidence.*)
- **P0.4 Repeatability floor.** Capture flat and smile three times each in one session (and again after
  re-plugging cables). The RMS difference between repeats, 50 Hz–10 kHz, is the best any model can do.
  Every success criterion below is stated relative to it, so we never chase noise.

## Prerequisite rig work (R1–R2)

- **R1. Chain reference.** The rig's QC currently normalises by a single 1 kHz chain gain, so the ProRMP's
  measured +4 dB tilt is baked into every "measured" response. Add a calibration step that records the
  chain with a **cable in place of the pedal** (ProRMP OUT → ProDI IN), and normalise by that full
  H_chain(f), not a scalar. Without R1, rung A would fit the re-amp box. **Decision D1:** the reference is
  the cable bypass, not the GE-7 at flat (flat has its own −1.2 dB and shape; it is data, not the
  reference).
- **R2. Capture set** (one 48 kHz session, hardware):
  calibration + chain reference; flat ×3, smile ×3 (P0.4); the 32-step slider map; the 16-step
  interaction set; the 14 presets (full signal, for time-domain checks later); 8 random multi-slider
  settings (seeded, saved as custom presets) as extra holdout. Roughly 3 hours of capture, mostly
  unattended between slider moves.

## Rung A: transfer-function topology model

**Model (hypothesis; medium confidence that it is the GE-7's exact form).** In continuous time:

    H(s, x) = G · L(level) · Hpf(s) · Lpf(s) · [1 + Σₖ aₖ(xₖ) Bₖ(s)] / [1 + Σₖ bₖ(xₖ) Bₖ(s)]

    Bₖ(s) = (ωₖ/Qₖ) s / (s² + (ωₖ/Qₖ) s + ωₖ²)        band-pass resonator k (k = 1..7)
    aₖ(x) = cₖ · p(+x),  bₖ(x) = cₖ · p(−x)            slider x ∈ [−1, 1] sets boost vs cut
    p(x): monotone pot law, p(0) the centre value     (so at centre aₖ = bₖ and the band cancels)

- At the centre detent each band cancels exactly: H = G·L·Hpf·Lpf, which matches the measured flat.
- Boost/cut reciprocity is built in when the pot law is symmetric; a learned asymmetry parameter allows
  for what P0.2 finds.
- Bands interact through the shared numerator and denominator: P0.1 is what this structure predicts.
- Trainable: ωₖ, Qₖ, cₖ (21), the pot law (1–2 shape parameters shared by the 7 bands), the LEVEL law
  (2), G, the coupling high-pass and the top-end low-pass corners (2–3). About 28 parameters, all physical
  and all positive (parameterised as logs, as in `pedal_model/ode/filters.py`).

**Decisions.**
- **D2. Fit in the frequency domain first.** The pedal is linear, so the measured complex response
  (H1 estimate from the sweeps, `pedal_model/rig/analysis.py`) carries all the information. That avoids
  sequential IIR filtering inside the training loop. Time-domain checks come after the fit (ESR on the val
  signal through the discretised model). Revisit if the time-domain ESR disagrees with the
  frequency-domain fit.
- **D3. Continuous-time parameters, discretise only to render.** Fit H(jω) directly; render with the
  bilinear transform with pre-warping at each ωₖ. The 6.4k band at 48 kHz is where the warping matters most.
- **D4. Loss.** Weighted log-magnitude MSE (reuse `pedal_model/ode/losses.py`: `log_magnitude_loss`,
  `make_frequency_weights`) plus a phase term on the excess phase, weighted by coherence. Phase is
  meaningful because takes are aligned on the loopback latency. Revisit the phase weight if it
  fights the magnitude fit.
- **D5. Data split.** Train: slider map, interaction set, single-slider captures, half the presets.
  Holdout, used only for the final number: smile, frown, mid_scoop, warm, and the 8 random settings.
  Generalising to unseen slider combinations is the whole point of a topology model, so the holdout
  is mostly multi-slider settings.
- **D6. Fitter.** Follow the KlonSim three-stage pattern (`pedal_model/ode/fitting.py`), cut down for a
  linear model: stage 1 each band from its own single-slider data (initialises ωₖ, Qₖ, cₖ); stage 2 the
  pot and LEVEL laws from the slider map; stage 3 everything jointly on all training data. Adam then
  L-BFGS, multi-start ωₖ around the panel frequencies.

**A0 known-answer first** (the GE-7 equivalent of the KlonSim experiment): generate responses from this
same model with chosen parameters, add noise at the measured SNR, and run the fitter.
- Noise-free: recover every parameter within 1 %.
- With noise: within the reported uncertainty.

If A0 fails, the problem is identifiability or the fitter, and real data would only hide that.

**Success criteria for A** (stated against measured floors, not invented targets):
- every held-out setting within 2× the P0.4 repeatability floor (RMS, 50 Hz–10 kHz);
- better than Ge7Sim on every held-out preset;
- the interaction set (P0.1) and the reciprocity test (P0.2) reproduced without special-casing.

**Identifiability.** New work: a Fisher-information/Hessian report at the optimum (which parameter
combinations the data pin down, with uncertainties). Expect cₖ and the pot law to trade off; the slider
map's two positions per side are what separate them. `pedal_model/ode/identifiability.py` has the
section-selection and transfer-function helpers to reuse, but no Fisher tool yet.

## Rung B: component values

Replace (ωₖ, Qₖ, cₖ) by the circuit's components through nodal analysis of the real schematic:
- each gyrator resonator (capacitor + simulated inductor, L = R·R'·C'),
- the slider pots and their taper,
- the op-amp's input and feedback resistors.

Only some combinations are identifiable from audio. For example, a gyrator's L is a product, so its
factors cannot be separated. The deliverable is:
- recovered combinations with uncertainties;
- a statement of which are not identifiable;
- a comparison against the schematic values. Typical tolerances are 5 % resistors and 5–10 % film
  capacitors; that is an assumption, so use the actual parts list.

**Prerequisite: the schematic** (question 1 below). Without it rung B has no ground truth.

## Storage and use in the GUI

- **Parameter file** `pedal_model/rig/models/ge7/ge7_pedalode_v<N>.json`:
  - the topology id and version;
  - every parameter with units and uncertainty;
  - the training data (session folders and preset names), the git commit;
  - the fit metrics, holdout results and repeatability floor.
  - New file for each refit; never overwritten.
- **Profile link:** `profiles/ge7.json` gains `"model": "models/ge7/ge7_pedalode_v1.json"`.
  `qc.expected_response()` evaluates H(jω, x) from it, falling back to Ge7Sim if absent. This makes
  QC's "Response vs model" sharper, and the Plan/Run/Review model curves become the fitted model.
- **Demo:** the demo backend can simulate the fitted model too.
- **Review:** shows the model version used.

## Rung C: Daisy (decide after rung A)

**Decision D7 (deferred):** how to run a 14th-order, slider-dependent rational function on the Seed.
- **Option 1:** implement the topology directly. Seven fixed band-pass biquads feed a linear,
  delay-free feedback sum, which can be solved in closed form each sample because it is linear. The
  sliders only change the branch weights aₖ and bₖ, so moving a slider needs no coefficient
  recomputation. About 7 biquads plus a few multiplies per sample, cheap on a Cortex-M7. *Medium
  confidence on numerical behaviour at extreme settings; prove it in Python first.*
- **Option 2:** at control rate, convert H to second-order sections for the current sliders (root
  finding on a degree-14 polynomial). It's more work on the MCU and needs a robust float root finder.

Lean towards option 1. Decide once rung A shows how strong the interaction is and how many bands matter.

## Files (planned)

| File | What |
|---|---|
| `pedal_model/ode/ge7_topology.py` | The rung A model (torch), H(jω, x) and a discretised renderer |
| `pedal_model/ode/ge7_fitter.py` | The staged fitter (D6) |
| `pedal_model/ode/identifiability.py` | Add the Fisher/Hessian report |
| `scripts/ge7_step0_probes.py` | P0.1–P0.4 from rig sessions |
| `scripts/fit_ge7_pedalode.py` | Fit, write the versioned parameter file, plots, holdout table |
| `pedal_model/rig/models/ge7/` | Versioned parameter files |
| `tests/ode/test_ge7_topology.py` | Centre-detent unity, reciprocity, A0 known-answer recovery |
| `pedal_model/rig/calibrate.py`, `qc.py` | R1: chain reference, H_chain(f) normalisation |

## Order of work

1. R1, the chain reference in the rig, with tests.
2. A0: the model, the fitter and known-answer recovery on synthetic data.
3. R2: capture the hardware session (person at the rig).
4. Step 0 probes P0.1–P0.4; settle D0. If it's a cascade after all, stop and re-plan.
5. Rung A fit and the holdout table; parameter file v1; GUI uses it.
6. Rung B (needs the schematic).
7. Rung C, after decision D7.

## Open questions

1. Do you have the GE-7 schematic (and which revision or serial range)? It's needed for rung B, and to
   confirm or correct the recalled topology above before rung A is built around it.
2. Is it OK to add the "cable instead of pedal" chain-reference step to calibration (R1)? It means one
   re-plug per session.
3. The capture session (R2) is about 3 hours of audio, mostly unattended. Is that acceptable, or should
   the presets use the quick signal (the slider map and interaction set already use the sweep)?
