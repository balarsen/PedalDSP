# Boss GE-7 Equalizer — Modeling Plan

## Overview

The Boss GE-7 is a 7-band graphic equalizer with 8 sliders total (7 bands + output
level). It is a **linear** pedal: each slider position defines a fixed frequency
response with no nonlinear clipping. This makes it an ideal next target after
KlonSim — the model fit should be near-perfect with a classical FIR/IIR approach,
and the multi-preset conditioned model is the scientifically interesting challenge.

**GE-7 sliders and ranges:**

| Slider | Center freq | Range |
|---|---|---|
| Band 1 | 100 Hz | ±15 dB |
| Band 2 | 200 Hz | ±15 dB |
| Band 3 | 400 Hz | ±15 dB |
| Band 4 | 800 Hz | ±15 dB |
| Band 5 | 1.6 kHz | ±15 dB |
| Band 6 | 3.2 kHz | ±15 dB |
| Band 7 | 6.4 kHz | ±15 dB |
| LEVEL | Output | ±15 dB |

Sliders detent at 0 dB. The hardware is op-amp based (active EQ) so even with all
sliders at 0 dB there may be a slight noise floor and input impedance colouration —
worth measuring.

---

## Directory Structure

Mirrors the KlonSim layout exactly. One subdirectory per named preset.

```
recordings/data/ge7/
    presets/                          ← master preset definitions (JSON)
        preset_flat.json
        preset_smile.json
        preset_frown.json
        preset_bass_boost.json
        preset_mid_scoop.json
        preset_presence.json
        preset_high_cut.json
        preset_low_cut.json
        preset_all_up.json
        preset_all_down.json
        preset_bright.json
        preset_warm.json
    captures/                         ← one dir per preset (created at recording time)
        preset_flat/
            wet_train_signal_v1.wav   ← same dry signals as KlonSim
            wet_val_signal_v1.wav
            params.json               ← preset settings + capture metadata
        preset_smile/
            ...

results/ge7/
    per_preset/                       ← one fitted model per preset
        preset_flat/
            staged/
                model_final.pt
                metrics.json
                plots/
        preset_smile/
            ...
    conditioned/                      ← single model conditioned on 8 slider values
        staged/
            model_final.pt
            metrics.json
            plots/
```

---

## Preset Library

12 named presets covering the useful range. Values are dB.

```
             100  200  400  800  1.6k  3.2k  6.4k  LEVEL
flat           0    0    0    0     0     0     0      0    ← baseline, must be first
smile         +8   +4   -2   -6    -6    +4    +8      0    ← classic V-shape
frown         -6   -2   +6   +8    +6    -2    -6      0    ← opposite of smile
bass_boost   +12   +8   +4    0     0     0     0      0    ← thunderous low end
mid_scoop      0    0   -8  -12   -10    -4     0      0    ← classic metal scoop
presence       0    0    0    0    +6   +10   +12      0    ← cut-through-mix hi presence
high_cut       0    0    0    0    -6   -10   -14      0    ← telephone / muffled
low_cut      -14  -10   -6    0     0     0     0      0    ← thins out bass
all_up        +12  +12  +12  +12  +12   +12   +12     +6    ← worst case gain stress
all_down     -12  -12  -12  -12  -12   -12   -12     -6    ← worst case cut stress
bright         0    0    0   +2    +6   +10   +12      0    ← Strat-style sparkle
warm          +6   +4   +2    0    -4    -8   -10      0    ← warm/dark, old amp feel
```

---

## Preset JSON Format

`recordings/data/ge7/presets/preset_smile.json`:

```json
{
    "schema_version": "1.0",
    "pedal": "boss_ge7",
    "preset_name": "smile",
    "description": "Classic V-shape EQ — boost lows + highs, scoop mids",
    "sliders": {
        "hz_100":  8.0,
        "hz_200":  4.0,
        "hz_400": -2.0,
        "hz_800": -6.0,
        "hz_1600": -6.0,
        "hz_3200":  4.0,
        "hz_6400":  8.0,
        "level":    0.0
    },
    "notes": ""
}
```

**Rule:** `level` compensates for perceived loudness. Keep it at 0 dB for most
presets so the EQ comparison is fair. Only `all_up` and `all_down` use non-zero
level to prevent clipping or total silence.

---

## Capture Workflow

### Error-reduction strategy

A `capture_ge7_checklist.py` script drives the session. It reads all preset JSONs,
prints a clear per-slider diagram for each one, waits for the user to confirm the
pedal is set, then prints the DAW instructions. One preset per run. No guessing.

```
╔══════════════════════════════════════════════════════╗
║  PRESET 2 / 12 : smile                              ║
║  Classic V-shape EQ                                  ║
╠══════════════════════════════════════════════════════╣
║  100Hz   200Hz  400Hz  800Hz  1.6k  3.2k  6.4k  LVL ║
║   +8      +4     -2     -6    -6    +4    +8     0   ║
║   ▲▲▲▲   ▲▲    ▽      ▽▽▽▽  ▽▽▽▽  ▲▲    ▲▲▲▲   ─   ║
╚══════════════════════════════════════════════════════╝

  Set the pedal now, then press ENTER to continue...
  
  DAW: Fader at −10 dB → Arm track → Record → play train_signal_v2.wav into pedal
  Save wet recording as:
    recordings/data/ge7/captures/preset_smile/wet_train_signal_v2.wav
  
  Done with TRAIN? Press ENTER, then repeat for VAL.
```

### DAW gain staging

The passive reamp box saturates above approximately −12 dBFS input. The train
signal peaks at −2 dBFS, so the playback track fader **must be set to −10 dB**
before recording any preset. This was verified with the flat capture (flat2):
all amplitude levels −18 through −3 dBFS showed ≤ 0.25 dB gain variation,
confirming full linearity at this playback level.

| DAW setting | Reamp input peak | Result |
|---|---|---|
| 0 dB (unity) | −2 dBFS | Saturates at −6 and −3 dBFS sections |
| **−10 dB** | **−12 dBFS** | **Fully linear — use this** |

The −10 dB fader applies only to the playback (dry send) track. The return
(record) channel gain should be left at whatever gives a healthy capture level
without clipping — the reamp output at −10 dB playback peaks around −19 dBFS,
so adding +6 to +12 dB of return gain is fine if desired.

### Session steps (per preset)

> **Now automated:** the capture rig (`uv run python -m pedal_model.rig --hardware`, profile `ge7`) shows the
> slider positions for each preset, plays the signal at the −10 dB send level from
> `pedal_model/rig/profiles/ge7.json`, records, checks the response against Ge7Sim, and saves the preset folder.
> The manual Studio One steps below are kept for reference.

1. Open `recordings/data/signals/train_signal_v2.wav` in DAW, set playback track fader to **−10 dB**
2. Route playback into GE-7 via reamp box, arm return track
3. Run: `uv run python scripts/capture_ge7_checklist.py --preset preset_smile`
4. Set sliders exactly as shown on screen
5. Press ENTER — record the train signal through the pedal → save wet WAV as
   `recordings/data/ge7/captures/preset_smile/wet_train_signal_v2.wav`
6. Record val signal at same −10 dB fader → save as `wet_val_signal_v2.wav`
7. Press ENTER to confirm — script validates file existence and prints the next
   preset

### After all presets are recorded

Run alignment + quality check across all captures:

```bash
uv run python scripts/verify_ge7_captures.py
```

This calls `pedal_model/capture/align.py` and `verify.py` on every preset and
prints a pass/fail table. Any misaligned or clipped captures are flagged before
any models are trained.

---

## Synthetic Model (for pipeline testing)

`pedal_model/synthetic/ge7.py` — a `Ge7Sim` class that mirrors `KlonSim` exactly.

Signal chain:
```
in → cascade of 7 peak-EQ biquads (Audio EQ Cookbook peaking filter) → ×level → out
```

The Q factor for each band uses the standard graphic EQ approximation:
```
Q = f_c / Δf   where Δf is determined by octave bandwidth
```
The GE-7 bands are spaced one octave apart (100→200→400…) so a Q of ~1.4 gives
good overlap between adjacent bands.

This lets the entire pipeline be tested with synthetic data before any hardware
recording:

```bash
uv run python scripts/make_ge7_captures.py   # generates all 12 synthetic capture dirs
uv run python scripts/fit_ge7_per_preset.py  # trains IIR model on each
```

---

## New Files to Create

| File | What | Priority |
|---|---|---|
| `pedal_model/synthetic/ge7.py` | `Ge7Params`, `Ge7Sliders`, `Ge7Sim` | **1 — needed before anything else** |
| `recordings/data/ge7/presets/*.json` | 12 preset definition files | **1** |
| `scripts/make_ge7_captures.py` | synthetic wet generation (like `make_klon_sim.py`) | **2** |
| `scripts/capture_ge7_checklist.py` | interactive recording guide | **2** |
| `scripts/fit_ge7_per_preset.py` | train IIR model per preset (reuses `fit_models.py` logic) | **3** |
| `scripts/verify_ge7_captures.py` | alignment + level QC across all presets | **3** |
| `tests/synthetic/test_ge7.py` | unit tests for Ge7Sim | **3** |

---

## Files Reused As-Is (no changes needed)

| File | Why it just works |
|---|---|
| `recordings/data/signals/train_signal_v1.wav` | Same input signal — GE-7 is linear so the same sweep/tone coverage works |
| `recordings/data/signals/val_signal_v1.wav` | Same |
| `pedal_model/capture/align.py` | Cross-correlation alignment is pedal-agnostic |
| `pedal_model/capture/verify.py` | Level + DC checks are pedal-agnostic |
| `pedal_model/metrics/` | All metrics (ESR, STFT, FR, THD, MCD) reusable unchanged |
| `pedal_model/train/` | Training loop, losses, scheduler unchanged |
| `pedal_model/models/classical/` | FIR and IIR will be the primary models; WH/Hammerstein too |
| `pedal_model/models/neural/` | TCN / LSTM for conditioned model |
| `pedal_model/utils/plotting.py` | All compare_* functions work on any dry/wet pair |
| `scripts/compare_ode_daisy.py` | Can compare GE-7 captures with minor argument changes |

---

## Model Strategy

### Phase 1 — per-preset classical fit (easy wins)

For each of the 12 presets, fit an FIR filter (via least-squares deconvolution
of the swept sine IR). This should get ESR < 0.001 for a linear EQ. The IIR
fit gives interpretable per-band gains to verify the model matches the preset.

Expected result: near-perfect fit since the GE-7 is linear.

Script: `scripts/fit_ge7_per_preset.py --preset preset_smile`

### Phase 2 — conditioned neural model (the interesting part)

A single TCN or LSTM that takes the 8 slider values as a conditioning vector.
The training set is all 12 presets concatenated. The model learns to produce any
GE-7 response from slider values alone.

This is the Daisy deployment candidate — one model, infinite presets, knobs
mapped to conditioning inputs.

---

## Verification Checklist (before recording)

Run these before the first capture session to confirm the pipeline works
end-to-end with synthetic data:

```bash
# 1. Generate synthetic captures for all presets
uv run python scripts/make_ge7_captures.py

# 2. Train an IIR model on the flat preset
uv run python scripts/fit_ge7_per_preset.py --preset preset_flat

# 3. Verify metrics look sane (ESR should be < 0.001 for synthetic data)
cat results/ge7/per_preset/preset_flat/staged/metrics.json
```

If ESR ≥ 0.01 on synthetic data, the IIR fit or signal chain has a bug —
fix it before recording any hardware.

---

## Unified Training Interface

### Goal

One command trains any pedal. No `--pedal-type` flag. No separate scripts per pedal.

```bash
uv run python scripts/fit_pedal.py --capture-dir recordings/data/ge7/captures/preset_smile
uv run python scripts/fit_pedal.py --capture-dir recordings/data/klon_sim/klon_sim_v1
```

### How it works

`params.json` in the capture directory grows two new fields:

```json
{
  "schema_version": "1.0",
  "pedal": "boss_ge7",
  "pedal_class": "linear_eq",
  "fit_hints": {
    "recommended_models": ["FIR", "IIR"],
    "skip_models": ["WienerHammerstein", "Volterra", "Hammerstein"],
    "fir_length_ms": 100,
    "iir_order": 14
  },
  "sliders": { ... },
  "captures": { ... }
}
```

`fit_pedal.py` reads `params.json`, pulls the hints, and configures the model zoo run:
- If `fit_hints` is present → use recommended models + overrides
- If `fit_hints` is absent → fall back to running all models (safe default)

### Pedal classes and their hints

| `pedal_class` | recommended models | skip | notes |
|---|---|---|---|
| `linear_eq` | FIR, IIR | WH, Volterra, Hammerstein | GE-7; near-perfect FIR fit expected |
| `nonlinear_wh` | WienerHammerstein, TCN, LSTM | FIR | KlonSim; WH is the correct topology |
| `nonlinear_generic` | TCN, LSTM, Hammerstein | — | unknown distortion; try all NL |
| *(absent)* | all | none | fallback: full zoo |

### KlonSim params.json update

Add to `recordings/data/klon_sim/klon_sim_v1/params.json`:
```json
{
  "pedal_class": "nonlinear_wh",
  "fit_hints": {
    "recommended_models": ["WienerHammerstein", "TCN", "LSTM"],
    "skip_models": ["FIR"],
    "neural_epochs": 100
  }
}
```

This is backward-compatible — existing code that ignores `fit_hints` keeps working.

---

## Implementation Status

```
[✓] pedal_model/synthetic/ge7.py       — Ge7Sim, Ge7Sliders, Ge7Params
[✓] recordings/data/ge7/presets/*.json            — 12 preset files
[✓] scripts/make_ge7_captures.py       — synthetic wet gen; all 12 presets generated
[✓] scripts/capture_ge7_checklist.py   — interactive recording session guide
[ ] scripts/verify_ge7_captures.py     — post-capture alignment + level QC
[ ] scripts/fit_pedal.py               — unified training entry point (params.json-driven)
[ ] tests/synthetic/test_ge7.py        — unit tests

Ready to record hardware.  Workflow:
  uv run python scripts/capture_ge7_checklist.py --all
  uv run python scripts/verify_ge7_captures.py
  uv run python scripts/fit_pedal.py --capture-dir recordings/data/ge7/captures/preset_flat
```
