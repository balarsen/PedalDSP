# PedalDSP — Processing Chain: Testing a Real Pedal

End-to-end guide for capturing a guitar pedal, fitting models, and rendering DAW-ready audio.

---

## Overview

```
[Pedal + Loopback Rig]
        ↓
  Step 1: Capture         — record dry/wet WAV pairs
        ↓
  Step 2: Prepare Dataset — create directory structure + params.json
        ↓
  Step 3: Fit Models      — scripts/fit_models.py
        ↓
  Step 4: Inspect Results — notebooks, results.csv
        ↓
  Step 5: Render to DAW   — scripts/render_to_daw.py
        ↓
  [DAW: drag-in and compare]
```

Step 5 (retrain neural — optional) can be inserted between 3 and 4 if you want
more training epochs without re-running slow classical fits.

---

## Step 1: Capture

> **Use the capture rig (since 2026-10-17).** `uv run python -m pedal_model.rig --hardware` plays the signal,
> records the return and a loopback in one ASIO stream, calibrates, checks every take, and writes the
> preset folders with `params.json` already in the format Step 2 describes, so Steps 1 and 2 are done for
> you. Wiring, box settings and troubleshooting: README.md, "Capture rig". The DAW procedure below is the
> older manual method, kept for reference.

### Hardware rig

```
Guitar → DI box (parallel split)
           ├── DRY output → audio interface input A  (record clean)
           └── → Pedal → audio interface input B  (record wet)
```

- Use the same interface for both channels to keep them sample-aligned.
- Set interface input gain identically on both channels; pad if needed.
- **Record at 96 kHz / 24-bit.** The models train and evaluate at 96 kHz;
  capturing at this rate avoids a resample step and gives maximum frequency
  coverage. On a PreSonus Studio 1824c, set the sample rate in Universal Control
  before starting the DAW session.
- 48 kHz is also accepted — the pipeline resamples to 96 kHz automatically — but
  96 kHz is strongly preferred for new captures.
- Target a nominal recording level of **-12 to -18 dBFS** to leave headroom for
  peaks and to match the amplitude range the models are trained on.

### Training signal

Play the `recordings/data/klon_sim/klon_sim_v1/train_signal_v1.wav` file through your DI rig:

```
[DAW or audio player] → DI rig → interface → record stereo (dry=L, wet=R)
```

The training signal contains:
- Sine sweeps at multiple amplitudes (characterises frequency response)
- Stepped-level tones at multiple frequencies (characterises amplitude response)
- Pink noise bursts (wideband coverage)
- Silence gaps (used by the trainer to identify silence blocks)

Record at minimum 60 seconds; the full signal is ~10 minutes and gives better
model coverage.

### Validation signal

Use a separate, shorter signal (15–30 seconds of guitar playing or the
`val_signal_v1.wav`) for validation. **Do not use the same recording as training.**

---

## Step 2: Prepare Dataset

### Directory structure

```
recordings/data/captures/<pedal_name>/
    train_dry.wav          ← dry channel from training recording
    train_wet.wav          ← wet channel from training recording
    val_dry.wav            ← dry channel from validation recording
    val_wet.wav            ← wet channel from validation recording
    train_signal_v1.json   ← signal manifest (copy from recordings/data/klon_sim/)
    params.json            ← config file (create manually — see below)
```

### Creating params.json

```json
{
    "params": {
        "gain": 5,
        "treble": 5,
        "bass": 5,
        "volume": 5
    },
    "captures": {
        "train": {
            "dry_manifest": "train_signal_v1",
            "wet_wav": "train_wet.wav",
            "sample_rate": 96000
        },
        "val": {
            "dry_manifest": "val_signal_v1",
            "wet_wav": "val_wet.wav",
            "sample_rate": 96000
        }
    }
}
```

- `dry_manifest` is the stem of the JSON manifest file (without `.json`).
- `params` is free-form — describes the pedal knob positions used for this capture.
- Set `sample_rate` to match the actual recording rate (96000 or 48000).

### Alignment check

Before fitting, verify that the dry and wet channels are aligned:

```bash
uv run python -c "
import soundfile as sf, numpy as np
dry, sr = sf.read('recordings/data/captures/mypedal/train_dry.wav', dtype='float32')
wet, sr = sf.read('recordings/data/captures/mypedal/train_wet.wav', dtype='float32')
corr = np.correlate(wet[:sr], dry[:sr], mode='full')
lag = np.argmax(corr) - sr + 1
print(f'Lag: {lag} samples ({lag/sr*1000:.2f} ms)')
"
```

A lag of 0–5 samples is expected (interface latency). If the lag is larger,
your recording setup has a latency issue — fix it in your DAW before fitting.
The `pedal_model.capture.align` module can auto-correct small lags.

---

## Step 3: Fit Models

```bash
# Full run — classical + neural, auto-detect CUDA
uv run python scripts/fit_models.py \
    --klon-sim-dir recordings/data/captures/mypedal \
    --name mypedal_v1

# Classical-only first (fast, ~5 min) to check data quality
uv run python scripts/fit_models.py \
    --klon-sim-dir recordings/data/captures/mypedal \
    --name mypedal_v1 \
    --classical-only --skip volterra

# Then add neural with more epochs
uv run python scripts/fit_models.py \
    --klon-sim-dir recordings/data/captures/mypedal \
    --name mypedal_v1 \
    --neural-only --epochs 200 --device cuda
```

**Output:** `results/mypedal_v1/results.pkl` and `results/mypedal_v1/results.csv`

The pkl stores:
- All model predictions on the validation set
- Per-section metrics (ESR, STFT loss, null depth, THD)
- Fitted model objects (classical) and state dicts (neural) — used by render_to_daw.py

### Epoch guidance

| Command | Gradient steps | When to use |
|---------|---------------|-------------|
| `--epochs 50` | ~750k | Quick survey; check all architectures fit |
| `--epochs 100` | ~1.5M | Good convergence for most models |
| `--epochs 200` | ~3.0M | Extended run; diminishing returns check |
| `--epochs 500` | ~7.5M | TCN and DDSP still improving at 200 |

---

## Step 4: Inspect Results

### Quick view

```bash
cat results/mypedal_v1/results.csv
```

Key metrics to check:
- **ESR** (Error-to-Signal Ratio): lower is better; < 0.05 is excellent, > 0.5 means poor fit
- **null_depth_dB**: signal cancellation depth; > 20 dB is good
- **STFT**: spectral loss; lower is better
- **fit_s**: time taken to fit (sanity check — Volterra is slow, FIR is fast)

### Notebooks

```bash
# Launch the zoo results notebook (Marimo):
uv run marimo edit notebooks/notaklon/01_zoo_results.py

# Deep diagnostics:
uv run marimo edit notebooks/notaklon/02_diagnostics.py
```

The notebooks load `results.pkl` and generate:
- Metric heatmap across all models × sections
- Transfer curves, harmonic profiles, spectrogram overlays
- ESR vs STFT scatter to identify model clusters

---

## Step 5 (Optional): Re-train Neural with More Epochs

If neural models need more training without re-running classical fits:

```bash
uv run python scripts/retrain_neural.py \
    --results results/mypedal_v1 \
    --epochs 500 \
    --models tcn lstm \
    --overwrite
```

This saves `{model}_best.pt` and `{model}_weights.pt` into the results directory.
`render_to_daw.py` will prefer `model_state` from `results.pkl`; the .pt files
serve as a fallback if the pkl is absent or you want a standalone checkpoint.

---

## Step 6: Render to DAW

Take a fresh mono dry guitar track and process it through all fitted models:

```bash
uv run python scripts/render_to_daw.py \
    --input path/to/dry_guitar.wav \
    --results results/mypedal_v1

# Select specific models (faster):
uv run python scripts/render_to_daw.py \
    --input dry_guitar.wav \
    --results results/mypedal_v1 \
    --models fir tcn klonsim

# Force re-fitting classical models (if training data has changed):
uv run python scripts/render_to_daw.py \
    --input dry_guitar.wav \
    --results results/mypedal_v1 \
    --refit-classical
```

**Output directory:** `results/mypedal_v1/daw_renders/dry_guitar/`

All rendered WAVs land in the results directory alongside the pkl, so every
artifact for a session is in one place. Override with `--output-dir` if needed.

```
results/mypedal_v1/
    results.pkl
    results.csv
    fitted_params/          ← classical model cache
    daw_renders/
        dry_guitar/
            dry_guitar__dry.wav
            dry_guitar__klonsim.wav   ← reference simulation
            dry_guitar__fir.wav
            dry_guitar__iir.wav
            dry_guitar__hammerstein.wav
            dry_guitar__wiener_hammerstein.wav
            dry_guitar__tcn.wav
            dry_guitar__lstm.wav
            ...
```

Filenames use `{stem}__{model}.wav` (double underscore) so you can split on `__`
unambiguously even when the dry stem itself contains underscores.

All files are 44.1 kHz / 24-bit, ready to import into Studio One or any DAW.

### In-DAW comparison workflow

1. Import all files from `daw_renders/dry_guitar/` onto separate tracks.
2. Set all tracks to the same gain.
3. Phase-flip one track vs another to hear the error (null test).
4. Focus on: transient response, high-frequency extension, saturation character.
5. The `__klonsim` track is the reference — all other tracks are approximations of it.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| ESR > 1.0 for all models | Dry/wet misaligned | Check lag, re-align in DAW |
| Neural ESR much worse than FIR | Not enough epochs | Re-run with `--epochs 200` |
| Volterra fit takes > 30 min | Normal for high-memory setting | Use `--skip volterra` |
| Rendered output clips | RMS mismatch at inference | RMS normalization is applied automatically via `StreamRunner.training_rms`; check StreamRunner INFO logs for the scale factor |
| `model_state` is None in pkl | Old pkl from pre-state-save run | Re-run `scripts/fit_models.py` |
| .pt files not found | Never ran `retrain_neural.py` | Use pkl with model_state (preferred) or run retrain |
| Input is 48 kHz not 96 kHz | Interface sample rate not set correctly | Check Universal Control; set to 96 kHz before recording. 48 kHz is accepted but 96 kHz is preferred |

---

## Script Reference

| Script | Purpose | When to run |
|--------|---------|------------|
| `scripts/fit_models.py` | Fit all models, produce results.pkl | Once per capture; re-run to add neural epochs |
| `scripts/retrain_neural.py` | Re-train neural-only with more epochs | Optional; when neural results need improvement |
| `scripts/render_to_daw.py` | Render new dry audio through all models | Anytime you have a new guitar take to process |
| `run_eval.py` | Backward-compat shim → calls `fit_models.py` | Legacy; prefer `fit_models.py` directly |
