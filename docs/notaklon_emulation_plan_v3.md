# Notaklon Emulation — Project Plan (v3)

**Target:** JHS Notaklon (DIY Klon clone, Shamrock mod available but **not** in normal use).
**Goal:** Prove the PedalDSP capture → train → validate pipeline end-to-end on a single, well-understood pedal, while training the **full model zoo** (classical + neural + DDSP + hybrid) so the tradeoffs are learned firsthand.
**Deployment targets:** Daisy Seed (48 kHz, tight CPU/RAM) and a Studio One plugin (session rate, typically 48 kHz).
**Workflow:** Marimo notebooks for exploration/visualization; code matures out of notebooks into the `pedal_model` library over time.
**Status:** Plan stage. Code to be written with Claude Code against this spec.

> **v3 changes:** knob positions now driven by the actual tone spreadsheet (primary = Gain 11 / Tone noon; second = Gain 9 / Tone noon; Shamrock boost demoted to optional). Added medley capture protocol with Notaklon isolation. Added directory cleanup (executed as moves). Added generator + reader + test spec and a workflow step to generate signals. Added blocked capture TODO.

---

## 1. Why the Notaklon First

Clean-ish boost / light overdrive with mild, mostly static nonlinearity — the easy end of the complexity ladder:

- Transfer function largely time-invariant (no envelope-following, no modulation).
- Soft-knee clipping, gentle and predictable.
- Owned and known by ear, so model failures are immediately audible.

If we can't nail this, harder targets (compression, fuzz, morphing) aren't worth attempting yet.

---

## 2. Core Methodology: Train Synthetic, Validate on Guitar

An analog pedal is deterministic and near-memoryless: same input → same output. We **train on synthetic signals and validate on real guitar**. A static nonlinearity is characterized by how it maps amplitude → output across frequency; synthetic signals **deterministically tile that input space** where guitar undersamples the high-amplitude / high-frequency corners. Guitar is the held-out generalization test — it tests whether a model learned the **transfer function (physics)** rather than the **statistics of the training set**.

Tailoring to a playing style is **out of scope for now** (possibly permanently).

### Coverage gaps and how synthetic signals fill them
- **Crest factor** (sharp transients) — impulse trains, AM chirps, decaying plucks.
- **Spectral envelope** (formants/dead zones) — pink noise, multitone stacks.
- **Amplitude envelope** (attack→sustain→decay) — decaying-envelope tones, amplitude ramps.
- **Intermodulation** (strings ringing together) — two-tone sweeps, layered sections.

---

## 3. Data Formats (locked)

| Artifact | Rate | Depth | Rationale |
|---|---|---|---|
| **Captured dry/wet pairs** | 96 kHz | 24-bit int | ADC true range ≈ 20–21 bits; 24-bit (~144 dB) far exceeds any pedal's ~90–100 dB SNR. 32-bit buys only float headroom unneeded at capture (levels calibrated). Smaller files. |
| **Generated signals** (train + val) | 96 kHz | 32-bit float | Synthesized and summed (layered blocks); float guarantees no intermediate clipping. |
| **Archive (optional)** | — | — | FLAC lossless (40–60% smaller). Working files stay WAV for fast random-access slicing. |

### Sample-rate strategy
Neither target runs at 96 kHz (Daisy fixed 48k; Studio One typically 48k). We capture at 96k anyway because:
1. **Capture high, train at target** — 96k/24 is archival ground truth; downsample digitally to 48k to train deployment models.
2. **Aliasing is the physics** — Klon clipping makes >20 kHz harmonics that fold back at 48k; capturing at 96k lets us see them before folding and decide deliberately how to model it.

**Discipline:** train each deployment model **at its target rate** (downsample 96k→48k first). Keep a 96k model as a full-bandwidth research artifact. Neural models **oversample at inference** (run at 2×, downsample) to control self-aliasing. Capture **96k only**, downsample digitally.

**Daisy reality:** constraint is CPU/RAM, not VRAM. Only the smallest models fit (compact TCN, small WaveNet, distilled DDSP). The zoo is for learning and the plugin; Daisy gets the one or two that survive the size budget.

---

## 4. Generated Signals + JSON Manifest

The generator is **parameterized** (all freqs, levels, durations, seed exposed) with sensible defaults, and emits **two artifacts from one call**: the WAV and a JSON describing both the **inputs** (every parameter, seed, sample rate, generator version) and the **manifest** (every section: start/end sample, start/end time, type, amplitude, per-section params).

Any notebook computes "THD on the high-amplitude stepped-tone section" by querying the JSON for that section's sample range — no hardcoded offsets, survives regeneration. The schema is a **documented spec in the library**, shared by training and validation signals (this is what makes the validation signal a true cross-project benchmark).

### Manifest schema (sketch)
```json
{
  "schema_version": "1.0",
  "generator_version": "0.1.0",
  "sample_rate": 96000,
  "bit_depth": 32,
  "format": "float",
  "seed": 1234,
  "total_samples": 0,
  "total_duration_s": 0.0,
  "params": {
    "stepped_tones": { "freqs_hz": [], "levels_dbfs": [] },
    "two_tone": { "fixed_freqs_hz": [], "swept_range_hz": [20, 20000], "levels_dbfs": [] }
  },
  "sections": [
    {
      "index": 0, "label": "sweep_low", "type": "log_sine_sweep",
      "start_sample": 0, "end_sample": 1920000,
      "start_s": 0.0, "end_s": 20.0,
      "amplitude_dbfs": -18,
      "params": { "f_start_hz": 20, "f_end_hz": 20000 }
    }
  ]
}
```

### 4.1 Training signal `train_signal_v1` (~13 min)
Labeled sections; ~40% pure / 20% transient / 40% layered. **Defaults below; all are generator parameters.**

**Pure / single-stimulus (measurement + linear characterization)**

| Element | Duration | Purpose |
|---|---|---|
| Log sine sweep, low amplitude | 20 s | Linear FR at clean level |
| Log sine sweep, mid amplitude | 20 s | Onset of soft clipping |
| Log sine sweep, high amplitude | 20 s | Full nonlinearity |
| Stepped sine tones (discrete freqs × levels) | 60 s | THD-vs-level at known points |
| Pink noise, 3 amplitudes (20 s each) | 60 s | Spectral fill, level-dependent |
| White noise, 2 amplitudes | 30 s | Flat-spectrum coverage |
| Amplitude-ramped single tones | 40 s | Continuous level traversal |

**Transient / dynamic**

| Element | Duration | Purpose |
|---|---|---|
| Impulse train (varying spacing) | 30 s | Transient + recovery |
| AM chirps (swept freq × swept envelope) | 60 s | Freq × transient coupling |
| Decaying-envelope tones (pluck-like) | 40 s | Guitar-like attack/decay |

**Layered / combination (intermodulation richness)**

| Element | Duration | Purpose |
|---|---|---|
| Two-tone sweeps (fixed + swept) × several fixed freqs | 90 s | IM products across spectrum |
| Multitone (dense harmonic stacks) × several levels | 90 s | Realistic spectral density |
| Sweep + pink noise layered | 45 s | Coverage + masking |
| Multitone + transients layered | 45 s | Dense + dynamic together |
| Three-way layer (tone + noise + impulses) | 45 s | Worst-case entanglement |

**Total ≈ 13 min.** Pure segments protect measurement plots (clean single-stimulus regions); layered segments enrich training. Summing before a nonlinearity entangles responses — desirable for training density, which is why pure segments are retained for measurement.

### 4.2 Validation superset `val_signal_v1` (~1 min, versioned, reusable across ALL projects)
Spans the union of input regions any pedal type might stress, compactly, everything measurable. Fixed seed, regenerable, versioned.

| Segment | Duration | Purpose |
|---|---|---|
| Amplitude-stepped sweep | 15 s | FR + clipping onset |
| Stepped tones (fixed freqs/levels) | 10 s | THD anchor points, cross-pedal comparable |
| Pink noise, 2 levels | 10 s | Spectral (fuzz-relevant) |
| Two-tone IM segment | 10 s | Intermodulation (fuzz/Muff behavior) |
| Transient/impulse + decaying plucks | 10 s | Compression / envelope effects |
| Silence-gap markers between segments | ~5 s | Auto-segmentation |

Identical across every pedal/model — ESR/THD/null directly comparable in one master benchmark table.

---

## 5. Capture Setup

### Routing
- Guitar DI → interface **input A** (dry reference)
- Guitar DI → Notaklon → interface **input B** (wet)
- Record simultaneously; sample-aligned via shared source.

### Level calibration (BEFORE generating training signals)
1. Record a real guitar DI take; meter RMS and peak at the Notaklon input.
2. Scale synthetic signals to span that measured RMS range (and a little beyond).
3. Mis-scaled synthetic signals teach the wrong region of the nonlinearity curve; calibration exercises the soft-knee where the guitar actually drives it.

### Latency / alignment
- Measure round-trip latency once (loopback impulse); store the sample offset; apply so dry/wet pairs are sample-accurate. Null tests are meaningless without this.

---

## 6. Datasets

| Split | Data | Purpose |
|---|---|---|
| Train | `train_signal_v1` synthetic (dry → real wet) | Tile the input space; fit models |
| Dev/tune | Reserved synthetic slice + `val_signal_v1` | Hyperparameters; keeps guitar pristine |
| Validate | Real guitar DI medley (dry + real wet, **Notaklon-isolated**) | Generalization proof |
| Final gate | Blind A/B/X on the medley | Perceptual sign-off |

### 6.1 Guitar medley (data-selected from the tone sheet)
Six songs spanning low/mid/high Notaklon gain and clean→blues→punk→hard-rock feels (~20 s each, stitched):

| Song | Sheet Notaklon setting | Other pedals in song | Role in medley |
|---|---|---|---|
| Fire on the Mountain | Gain 9 (low) | Comp (low) — **isolate** | Low-gain target |
| Flagpole Sitta | Gain 10–11 (sweet spot) | None — clean Notaklon | Primary-position target |
| Hey Joe | Gain 9 (low) | Wah + Comp — **isolate** | Low-gain, dynamic material |
| I Love Rock 'N Roll | Gain 11–12 (high) | None — clean Notaklon | High-gain target |
| T.N.T. | Gain 11–12 (high) | None — clean Notaklon | High-gain target |
| The Thrill Is Gone | Gain 10 (mid) | Banzai + Comp (high) — **isolate** | Mid-gain, singing-lead material |

**Capture protocol — isolate the Notaklon.** For every segment, the validation capture engages **only the Notaklon** (other pedals bypassed), even on songs that normally stack (Hey Joe, Thrill Is Gone, Fire on the Mountain). The song is the *material*; the isolated Notaklon is the device under test, so the wet signal contains only Notaklon coloration. Optionally also record a **full-chain reference pass** per song for your own listening, clearly labeled and **not** used for model validation.

---

## 7. Model Zoo (train them all)

Built in increasing capability, all sharing one loss/metrics harness. Classical baselines first — they answer: *did the neural nets actually beat the simple thing?* (The repo already contains all of these, plus Volterra and a conditioned-LSTM that seeds the future knob-conditioned model.)

| # | Model | Class | What it teaches |
|---|---|---|---|
| 1 | Static waveshaper + filter (FIR/IIR) | Classical | Memoryless nonlinearity + linear EQ; simplest defensible Klon model |
| 2 | Hammerstein / Wiener–Hammerstein | Classical | Block-oriented system ID; interpretable baseline |
| 3 | Volterra | Classical | Polynomial nonlinearity with memory |
| 4 | MLP (windowed) | Neural | Simplest learned nonlinearity; "is memory needed?" |
| 5 | TCN (small + larger) | Neural | Dilated-conv receptive field; workhorse, Daisy candidate |
| 6 | LSTM / GRU | Neural | Recurrent memory; classic amp/pedal sim |
| 7 | WaveNet-style | Neural | Gated dilated convs; small variant for Daisy |
| 8 | DDSP | Hybrid | Differentiable DSP; interpretable, compact |
| 9 | (later) Conditioned-LSTM | Neural | Seeds the knob-conditioned model |

Each: trained at 96k (research) + 48k (deployment); neural models oversampled at inference; size/CPU recorded for Daisy feasibility.

---

## 8. Metrics & Validation

**Ears are ground truth; metrics are candidates that predict the ears.** A core sub-study finds *which metric best predicts perception*.

### Metric vector (per model, per signal section via manifest)
ESR · multi-scale STFT · frequency-response deviation · THD-pattern distance · null-test depth.

### Metric-vs-perception study
1. Compute the full metric vector for every model on the guitar medley.
2. Blind A/B/X listening test (you) to rank models perceptually.
3. Find which metric best correlates with the perceptual ranking — the proxy to trust going forward.
Deliverable: a **metric-vs-perception correlation plot** justifying the chosen gate.

### Provisional gates (refine after first listening test)
| Metric | Provisional "good" | Notes |
|---|---|---|
| ESR (guitar) | < −30 dB | Waveform fidelity |
| Null depth (guitar) | > 20 dB | Most intuitive single number |
| Multi-scale STFT | lower better, no fixed cut | Perceptual proxy |
| THD-pattern distance | qualitative match first | Harmonic character |

### Training loss (neural)
ESR + multi-scale STFT, weighted 70 / 30.

---

## 9. Visualization / Diagnostic Plots (`pedal_model/utils` + `eval`)

Headline: **coverage / phase-space plots**.
1. Coverage 2D histogram — instantaneous amplitude × spectral centroid of the training signal, **guitar medley overlaid** (the phase-space coverage argument, made visual).
2. Per-section coverage — each block's contribution.
3. Waveform overlay (zoom on a pick transient).
4. Error-over-time residual.
5. Frequency response (model vs real, <1 dB band shaded).
6. THD vs input level.
7. Harmonic bar chart (H2, H3, …).
8. Multi-scale spectrogram diff.
9. Null-test residual (time series + spectrum).
10. Training curves (loss components, train vs dev).
11. Metric-vs-perception correlation plot.
12. Master benchmark table (all models × all metrics on `val_signal_v1`).

---

## 10. Knob Positions — DATA-DRIVEN (separate outputs, not a sweep)

From the tone spreadsheet, the Notaklon is used at a range of gains, Tone almost always noon. Positions chosen by actual usage frequency:

| Priority | Label | Gain | Tone | Output | Shamrock | Representative songs |
|---|---|---|---|---|---|---|
| **1 (primary)** | Sweet spot | 11 | noon | up | off | Flagpole Sitta, T.N.T., I Love Rock 'N Roll, Crossroads, Layla, Pride and Joy |
| **2** | Low gain | 9 | noon | unity | off | Fire on the Mountain, Hey Joe, Thrill Is Gone, Ripple, Estimated Prophet |
| later/optional | Shamrock boost | 11 | noon | up | **on (+4 dB)** | (not seen in current rotation) |

**Sequencing:**
1. Build + fully validate **Sweet spot (Gain 11)** end-to-end (both gates) across the model zoo. This is the most-used setting and the one known best by ear.
2. Then capture/validate **Low gain (Gain 9)** as a second separate output.
3. Reassess; only then consider a knob-conditioned model (the conditioned-LSTM) and a denser sweep.
4. Shamrock boost only if a future song calls for it.

> Note: this replaces the v2 placeholder positions. The Shamrock boost is demoted because it does not appear in the current song rotation. Tone is held at noon throughout — matching documented usage.

---

## 11. Directory Cleanup (Claude Code to execute as moves)

Separate generated/captured **data** from **code**, consolidate planning docs, add per-pedal organization, and eliminate spaces in paths.

### Moves / renames
- `generate_capture_signal.py` (root) → keep a thin root CLI wrapper; library logic lives in `pedal_model/signals/`.
- `tier1_comparison.png` (root) → `results/`.
- `pedal-*.md` (root) + our plan MDs → `docs/`.
- `capture_signals/` → `data/signals/` (generated) — keep captured-run wavs distinct from generated signals; if `capture_signals/` holds prior *captures*, move to `data/captures/…` instead. Claude Code to inspect contents and route accordingly.
- `stems/2026-06-04 pi muff/` → `data/stems/2026-06-04_pi_muff/` (no spaces).

### Target layout
```
PedalDSP/
├─ CLAUDE.md
├─ pyproject.toml / requirements.txt / uv.lock
├─ generate_capture_signal.py        # thin CLI → pedal_model.signals
│
├─ pedal_model/
│  ├─ signals/        # NEW: generator + manifest writer + reader, coverage analysis
│  ├─ capture/        # align, sweep gen, verify, level calibration
│  ├─ data/           # dataset, augmentation, downsample 96k→48k
│  ├─ eval/           # compare, diagnostics, report, run_eval
│  ├─ metrics/        # time_domain, frequency_domain, harmonic, perceptual, suite
│  ├─ models/
│  │  ├─ classical/   # fir, iir, hammerstein, volterra, wiener_hammerstein
│  │  └─ neural/      # lstm, conditioned_lstm, gru, mlp, tcn, wavenet, ddsp
│  ├─ train/          # trainer, losses, scheduler
│  └─ utils/          # analysis, plotting, synthesis
│
├─ tests/             # mirrors pedal_model/, incl. tests/signals/
│
├─ notebooks/
│  └─ notaklon/       # NEW subfolder: notaklon exploration (migrates to library)
│
├─ data/              # NEW: all audio data
│  ├─ signals/        # train_signal_v1.{wav,json}, val_signal_v1.{wav,json}
│  ├─ captures/
│  │  └─ notaklon/
│  │     ├─ gain11_noon/   # primary
│  │     └─ gain09_noon/   # second
│  └─ stems/
│     └─ 2026-06-04_pi_muff/
│
├─ results/           # NEW: PNGs, comparison tables, benchmark exports
├─ docs/              # NEW: all planning + design MDs
└─ configs/experiment.yaml
```

---

## 12. Generator + Reader + Tests (spec for Claude Code)

1. **Standalone CLI** — `generate_capture_signal.py` (root) → `python -m pedal_model.signals.generate`. Creates train + validation WAVs per §4, writing **WAV + JSON manifest** together. All durations/freqs/levels/seed are CLI/config params with defaults.
2. **Library reader** — `pedal_model/signals/manifest.py`: load a WAV + its JSON, expose section slicing **by label** (e.g., `get_section("stepped_tones_high")` → sample range / array) so metrics code grabs named sections without offsets.
3. **End-to-end unit tests** (`tests/signals/`): generate **short** signals (scaled-down durations, ~2–3 s total) and assert
   - sample counts match the manifest,
   - sections tile the file with no gaps/overlaps,
   - reader round-trips (label → expected samples),
   - WAV/JSON sample-rate and length agree.
   Use pytest `tmp_path` so generated WAVs are auto-cleaned after each test (short files, no residue).

---

## 13. Build Order (for Claude Code)
1. **Directory cleanup** (§11) — execute moves/renames; update imports/paths.
2. Manifest schema spec + signal generator (parameterized) emitting WAV + JSON + coverage analysis (§4, §12).
3. Reader function + section-by-label slicing (§12).
4. **End-to-end generator tests** with tmp_path cleanup (§12).
5. **Workflow step — generate signals:** run the generator to produce `train_signal_v1.{wav,json}` and `val_signal_v1.{wav,json}` into `data/signals/`.
6. Capture utilities: latency, alignment, level calibration, dry/wet pairing.
7. Downsample pipeline (96k→48k) + dataset loader/split (reads manifest).
8. Loss (ESR + multi-scale STFT) + metrics module (section-aware).
9. Model zoo on shared harness; 96k + 48k; oversampled neural inference.
10. Viz module (coverage first) + Marimo notebooks in `notebooks/notaklon/`.
11. **Sweet-spot (Gain 11)** position: train all models → metric vectors → blind A/B/X → metric-vs-perception plot → both gates.
12. Master benchmark table on `val_signal_v1`.
13. Add **Low-gain (Gain 9)** as second separate output; repeat.
14. Migrate stabilized notebook code into `pedal_model`.

---

## 14. Open / Blocked Items

- **TODO (BLOCKED — DI boxes in transit):** Capture dry + real-wet for the medley — Fire on the Mountain, Flagpole Sitta, Hey Joe, I Love Rock 'N Roll, T.N.T., The Thrill Is Gone (~20 s each). **Notaklon-isolated** config for model validation; optional full-chain reference pass per song (labeled, not used for validation). Cannot start until DI boxes arrive.
- Level-calibration RMS range — measure once hardware is present (§5).
- Final stepped-tone / two-tone frequency grids — defaults shippable now, tunable via params.
- **Signal-design rationale MD** (`docs/signal_design.md`) — to be written next, documenting why each element/duration/amplitude and the pure/layered split exist, with the coverage argument.
