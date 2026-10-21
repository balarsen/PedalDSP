# Library inventory: what to keep as we move from the GE-7 to the Klon

Written 2026-10-20, after the capture rig and the GE-7 block model worked on hardware. Scope: the older
code in `pedal_model/` (outside `rig/`), `scripts/`, `notebooks/`, the root scripts, and `daisy/`. **Nothing is
removed yet**: this is a proposal to settle with the owner before each removal, and to revisit when the
Klon work shows what it needs.

Order of work: finish the GE-7 (rung A fit, parameter file, Daisy, plugin handoff), then the Klon. "Klon"
means two things here: **KlonSim** (`synthetic/pedal.py`, a synthetic Klon that already runs on the Daisy:
a known-answer target) and the **JHS Notaklon** (the real Klon-style pedal, captured in the Studio One
era; it should be recaptured with the rig).

How "used" was judged: a reverse import index over all scripts, notebooks, tests and `pedal_model/`
(2026-10-20). Of about 18,000 older lines, the rig and the GE-7 model use three modules
(`signals/manifest`, `data/resample`, `synthetic/ge7`).

Labels:
- **Keep:** used now, or clearly needed for the Klon.
- **Rework:** needed, but to be rewritten in the physicist style (CLAUDE.md, Code style) when its turn
  comes.
- **Drop:** superseded or unused. Remove after confirming.
- **Unsure:** decide when the Klon work gets there.

## Signals, data, synthetic pedals

| Module | Lines | Label | Why |
|---|---|---|---|
| `signals/manifest.py` | 230 | **Keep** | The rig reads every session signal through it. |
| `signals/generate.py` + root `generate_signal.py` | 893 | **Keep** | Makes the designed training signals (sweeps, stepped tones, pink noise, plucks). The Klon needs amplitude-stepped coverage of the clipping stage; that is this generator. |
| `data/resample.py` | 109 | **Keep** | The rig uses it (96 → 48 kHz session signals); `prepare_for_daisy` is the Daisy path. |
| `utils/resampling.py` | 117 | **Drop** (merge) | Duplicates `data/resample.py` (plus a 44.1 kHz writer for DAW renders). Fold what the render scripts need into `data/resample.py`. |
| `data/dataset.py` | 277 | **Keep** (if neural) | Windows dry/wet pairs for neural training. Needed only if a neural or hybrid Klon model is trained. |
| `data/augment.py` | 57 | **Drop** | Unused outside its test. |
| `synthetic/ge7.py` (Ge7Sim) | 390 | Keep → **Drop** | QC's expected curve and the demo pedal, until rung A fits the block model; then retire. |
| `synthetic/pedal.py` (KlonSim) + root `make_klon_sim.py` | 455 | **Keep, fix first** | The synthetic Klon is the Klon's known-answer target (an A0 for nonlinear). It carries the in-progress edits behind the **6 failing tests in `tests/synthetic/test_pedal.py`**: resolve those (finish or revert) as the first Klon task. |

## Models

| Module | Lines | Label | Why |
|---|---|---|---|
| `ode/blocks.py`, `ode/ge7_model.py`, `ode/ge7_fitter.py` | 540 | **Keep** | The current GE-7 work, already flattened. The Klon block model should reuse `blocks` and the LM fitter. |
| `ode/wh_model.py` (`WHODEModel`, `LinearBiquadModel`) | 343 | **Keep** | `WHODEModel` (HP → drive → odd nonlinearity → HP → shelf → level) is the KlonSim model that ran on the Daisy. `LinearBiquadModel` is the plan's black-box baseline for the GE-7. |
| `ode/filters.py`, `ode/nonlinearity.py` | 612 | **Keep, rework** | Digital biquads and the learned odd-symmetric MLP nonlinearity: the pieces of the WH model. The nonlinearity is the "physics-informed ML" part of the parked hybrid idea. Rework to continuous-time blocks like `blocks.py`. |
| `ode/fitting.py`, `ode/step0.py`, `ode/losses.py`, `ode/identifiability.py` + `scripts/fit_klonsim_ode.py` | 2,276 | **Rework** | The 3-stage KlonSim fitter and its pre-fit probes. The ideas carry over (stage the fit, probe before fitting, an identifiability report); the code is the deep, comp-sci style we are moving away from. Rewrite for the Klon in the GE-7 fitter's style, keeping what A0-style recovery tests prove. |
| `models/classical/fir.py`, `hammerstein.py`, `wiener_hammerstein.py` | 316 | **Keep** | Small black-box baselines. The Klon plan needs at least one baseline to show what the block structure costs (the hybrid note in the GE-7 plan). |
| `models/classical/iir.py`, `volterra.py` | 241 | **Unsure** | Learning baselines; the zoo results show them weak. Drop unless the Klon comparison wants them. |
| `models/classical/circuit_dictionary.py` | 677 | **Unsure** | A topology-search experiment, unused outside its test. PedalODE's topology-search rung may want the idea; the code probably not. Park. |
| `models/neural/conditioned_lstm.py`, `conditioned_tcn.py` + `scripts/train_ge7_conditioned.py` | 274 | **Keep** | Knob-conditioned black boxes: the realistic Klon baseline and the fallback if the block model falls short (sound matters most). |
| `models/neural/ddsp_klon.py` | 441 | **Keep** | Klon-topology DDSP plus a GRU residual corrector: already the hybrid physics + ML idea parked in the GE-7 plan. The starting point if the Klon needs it. |
| `models/neural/lstm.py`, `gru.py`, `tcn.py` | 292 | **Unsure** | Plain (unconditioned) black boxes used by the learning notebooks. Keep while the notebooks are kept. |
| `models/neural/mlp.py`, `wavenet.py`, `ddsp.py` | 331 | **Drop** | Unused outside their tests; superseded by the conditioned models and `ddsp_klon`. |
| `models/base.py` | 37 | **Keep** | The interface the above share. |

## Training, evaluation, metrics, inference

| Module | Lines | Label | Why |
|---|---|---|---|
| `train/trainer.py`, `losses.py`, `scheduler.py` | 318 | **Keep** | Small; needed for any neural or hybrid Klon model. |
| `metrics/time_domain.py`, `frequency_domain.py`, `harmonic.py`, `suite.py` | 426 | **Keep** | ESR, null depth, STFT error, THD. The Klon needs THD especially: the Big Muff lesson was that ESR hid the missing hard clipping. |
| `metrics/perceptual.py` | 973 | **Unsure** | MCD and friends, used by two notebooks. Keep for listening-related checks if they prove useful; it is large. |
| `eval/harness.py`, `results.py`, `compare.py`, `report.py`, `run_eval.py` + `scripts/fit_models.py`, `retrain_neural.py`, `export_neural_weights.py` + root `run_eval.py` | 1,600 + scripts | **Rework** | The model-zoo machinery, built around the KlonSim `params.json` layout. The Klon needs a way to fit several models on the same takes and compare them; this does that, but heavily. Rebuild a small evaluator on the rig's session layout when the Klon needs one. |
| `eval/diagnostics.py` | 2,804 | **Unsure** (likely mostly drop) | 51 plotting functions for the Notaklon notebooks. Keep the few plots the Klon work actually uses; drop the rest then. |
| `inference/stream_runner.py` + `scripts/render_to_daw.py` | 259 + script | **Keep** | Renders a model over long audio, block by block like the Daisy, for listening. Listening is the final judge. |
| `utils/plotting.py` | 1,821 | **Unsure** | Notebook plotting. Its future follows the notebooks'. |
| `utils/analysis.py`, `utils/synthesis.py` | 228 | **Drop** (check) | Small helpers used only by tests and one notebook; the rig has its own (`rig/analysis.py`). |

## Capture-era code (Studio One), scripts, notebooks

| Item | Label | Why |
|---|---|---|
| `capture/align.py`, `verify.py`, `generate_sweep.py` | **Drop** | Studio One-era alignment and checks; the rig replaced them (loopback alignment, QC). Used only by their tests. |
| `capture/hardware_delay.py` | **Drop** (later) | Superseded by `rig.analysis.find_lag`; still used by two old analysis scripts. Retire with them. |
| `scripts/analyze_ge7_capture.py`, `analyze_single_sliders.py`, `make_ge7_captures.py`, `capture_ge7_checklist.py` | **Drop** (later) | GE-7 Studio One-era analysis; the rig's QC, slider map and Review replace them. Retire when the old GE-7 captures are cleared. |
| `scripts/analyze_notaklon_capture.py`, `analyze_treble_sweep.py`, `analyze_hardware_delay.py`, `analyze_guitar_capture.py`, `analyze_train_signal_sections.py`, `stereo_to_mono.py` | **Unsure** | Notaklon / Studio One-era analysis. Some plots (saturation curves, harmonics vs level) are what the Klon needs; the rig should grow those as QC views, then these retire. |
| `scripts/prepare_notaklon_training.py`, `prepare_klonsim_session.py`, `make_klonsim_captures.py`, `infer_klonsim_on_guitar.py` | **Unsure** | Klon data preparation for the old harness format. Likely replaced by rig sessions; check when the Klon starts. |
| `scripts/export_daisy_coeffs.py`, `compare_ode_daisy.py`, `analyze_daisy_vs_sim.py` | **Keep** | The only working Daisy deployment and verification path (KlonSim on the Hothouse). The template for the GE-7 and Klon firmware. |
| `scripts/compare_sessions.py`, `reanalyse_session.py` | **Keep** | Rig tools. |
| root `validate_pipeline.py` | **Drop** (check) | An old end-to-end smoke test of the WH classical model; superseded by the test suite and the rig. |
| `daisy/` (KlonSim firmware) | **Keep** | Working Hothouse firmware: the template for every deployed pedal. |
| `notebooks/learning_tier*`, `pi_muff_end_to_end.py`, `perceptual_metrics_explorer.py`, `capture_signal_designer.py` | **Unsure** | The learning record; some reference missing files (PROJECT_MAP, Known loose ends). Keep as history; don't maintain. |
| `notebooks/notaklon/*` | **Unsure** | Notaklon zoo results and diagnostics; they depend on `eval/diagnostics.py` and `utils/plotting.py`. Their fate decides those modules'. |

## Data to revisit

- The Notaklon captures (`recordings/data/notaklon/captures/`, gains 9–12 o'clock, Studio One era) were made
  before the rig: scalar normalisation, no loopback alignment. Recapture with the rig before fitting a
  Klon model; keep the old ones for comparison until then.
- The Notaklon results note pending captures: lower gain settings (7–8 o'clock) and a 3–4 point treble sweep.
  Plan them as rig presets (a Notaklon profile: knobs and the Shamrock switch).

## First Klon tasks this implies

1. Resolve the 6 failing `test_pedal.py` tests (KlonSim in-progress edits).
2. A Notaklon rig profile (knob drawing, Shamrock switch), as CLAUDE.md already plans.
3. A Klon block diagram (the plan's ladder: drive stage → clipping → clean-blend parallel path → tone →
   output), fitted the GE-7 way; KlonSim as its known answer; the conditioned TCN as the black-box
   baseline; `ddsp_klon` as the hybrid fallback.
