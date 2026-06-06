# Pedal Modeling Project Map
## Python · PyTorch · CUDA · WSL2 · VS Code

---

## Project Directory Structure

```
PedalDSP/
│
├── pedal_model/                       # Library — matures out of notebooks over time
│   │
│   ├── signals/
│   │   ├── generate.py                # Parameterized signal generator + manifest writer
│   │   └── manifest.py                # Manifest reader: Section + Manifest classes
│   │
│   ├── data/
│   │   ├── dataset.py                 # PedalDataset + ChunkDataset; from_manifest() factories
│   │   ├── resample.py                # 96k→48k polyphase resampler + float32↔int16 conversion
│   │   └── augment.py                 # Optional: gain jitter, noise augmentation
│   │
│   ├── capture/
│   │   ├── align.py                   # Latency measurement, cross-correlation alignment
│   │   ├── verify.py                  # Plot and sanity-check capture quality
│   │   └── level_cal.py               # Level calibration (RMS/peak to match real guitar DI)
│   │
│   ├── models/
│   │   ├── base.py                    # Abstract base class (forward(), receptive_field)
│   │   │
│   │   ├── classical/
│   │   │   ├── fir.py                 # FIR filter — linear baseline
│   │   │   ├── iir.py                 # IIR filter (SOS, no instability)
│   │   │   ├── hammerstein.py         # Static NL + linear filter
│   │   │   ├── wiener_hammerstein.py  # Filter + NL + filter
│   │   │   └── volterra.py            # 2nd/3rd order Volterra series
│   │   │
│   │   └── neural/
│   │       ├── mlp.py                 # Windowed MLP baseline
│   │       ├── lstm.py                # LSTM / GRU
│   │       ├── tcn.py                 # Temporal Convolutional Network (Daisy candidate)
│   │       ├── wavenet.py             # Reduced WaveNet
│   │       ├── ddsp.py                # Differentiable DSP hybrid
│   │       └── conditioned_lstm.py    # LSTM + knob-position conditioning
│   │
│   ├── metrics/
│   │   ├── time_domain.py             # ESR, MSE, DC offset, null-test depth
│   │   ├── frequency_domain.py        # Multi-scale STFT loss, frequency response error
│   │   ├── harmonic.py                # THD, harmonic profile similarity, even/odd ratio
│   │   ├── perceptual.py              # Mel cepstral distortion, log spectral distance
│   │   └── suite.py                   # Run all metrics per manifest section → results dict
│   │
│   └── train/
│       ├── trainer.py                 # Training loop, checkpointing, CUDA, config YAML
│       ├── losses.py                  # CombinedLoss (time + multi-scale STFT)
│       └── scheduler.py              # LR scheduling
│
├── data/
│   ├── signals/
│   │   ├── train_signal_v1.wav        # 642.5s generated training signal (96k/float32)
│   │   ├── train_signal_v1.json       # Manifest: 84 sections, seed=1234
│   │   ├── val_signal_v1.wav          # 65s generated validation signal (96k/float32)
│   │   └── val_signal_v1.json         # Manifest: 38 sections, seed=42, held-back freqs
│   │
│   └── captures/
│       └── notaklon/                  # Dry/wet WAV pairs from physical pedal
│           │                          # (DI boxes in transit — Step 6 BLOCKED)
│           ├── <capture_date>/        # e.g. 2026-06-15/
│           │   ├── dry.wav            # 96k/24-bit, input A
│           │   ├── wet_gain11.wav     # 96k/24-bit, Notaklon Gain 11 / Tone noon
│           │   └── wet_gain9.wav      # 96k/24-bit, Notaklon Gain 9 / Tone noon
│           └── medley/                # Guitar medley for validation
│               ├── dry_medley.wav     # Real guitar DI, 96k/24-bit
│               └── wet_medley.wav     # Notaklon-only (no other pedals)
│
├── checkpoints/                       # <model_name>/<timestamp>/ — never overwritten
│
├── results/                           # Metric outputs, comparison heatmaps
│   └── notaklon/
│       └── <!-- TODO: fill after Step 12 -->
│
├── configs/
│   └── experiment.yaml                # Hyperparameters, paths, settings
│
├── notebooks/                         # Marimo — exploration/viz; code matures into library
│   ├── 01_capture_verify.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_neural_training.ipynb
│   └── 04_comparison_table.ipynb
│
├── tests/
│   ├── signals/
│   │   ├── test_generate.py           # 28 tests; bit-for-bit reproducibility regression
│   │   └── test_manifest.py           # 25 tests; Section + Manifest API
│   └── data/
│       ├── test_resample.py           # 23 tests; anti-aliasing, dtype, round-trip
│       └── test_dataset_manifest.py   # 19 tests; from_manifest factories, alignment
│
├── docs/
│   ├── notaklon_emulation_plan_v3.md  # 14-step project plan
│   ├── signal_design.md               # Train + val signal rationale, section tables
│   ├── pedal-metrics.md               # Metric formulas, provisional gates, null-test depth
│   ├── pedal-neural-models.md         # Neural architectures, deployment targets, TODO results
│   ├── pedal-classical-models.md      # Classical model math, TODO Notaklon results
│   └── pedal-project-map.md           # This file
│
├── generate_signal.py                 # CLI wrapper: python generate_signal.py --help
├── pyproject.toml
└── .venv/                             # uv-managed virtualenv (Python 3.12)
```

---

## Deployment Targets

| Target | Rate | Format | Constraint |
|---|---|---|---|
| **Daisy Seed** | 48 kHz | int16 PCM | CPU/RAM budget — only smallest models fit |
| **Studio One plugin** | 48 kHz (session rate) | float32 | Full model zoo; no size constraint |

Both targets are fed by `pedal_model/data/resample.py`:
- `downsample_96k_to_48k()` — polyphase anti-aliased 96k→48k
- `prepare_for_daisy()` — rate + bit-depth in one call
- Studio One only needs rate conversion (float32 kept)

---

## 14-Step Build Order

These map directly to the steps in [notaklon_emulation_plan_v3.md](notaklon_emulation_plan_v3.md).

```
Step 1  ✓  Directory cleanup, audio .gitignore strategy
Step 2  ✓  Parameterized signal generator + JSON manifests
           (generate_signal.py, pedal_model/signals/generate.py)
Step 3  ✓  Manifest reader (pedal_model/signals/manifest.py)
Step 4  ✓  Generator + manifest test suite (tests/signals/)
Step 5  ✓  Generate train_signal_v1 (642.5 s) + val_signal_v1 (65 s)
           (data/signals/*.wav + *.json committed)
Step 6  ✗  Capture utilities — BLOCKED (DI boxes in transit)
           latency measurement, alignment, level calibration, dry/wet pairing
Step 7  ✓  96k→48k resampler + manifest-aware dataset loaders
           (pedal_model/data/resample.py, dataset.py from_manifest factories)
Step 8  ·  Loss + metrics suite (pedal_model/metrics/, pedal_model/train/losses.py)
Step 9  ·  Model zoo (pedal_model/models/classical/ + neural/)
Step 10 ·  Marimo visualization notebooks
Step 11 ·  Training runs — all models on Notaklon capture
Step 12 ·  Benchmark: comparison heatmap + metric-vs-perception study
Step 13 ·  Low-gain capture (Gain 9 / Tone noon) + conditioned model
Step 14 ·  Library migration: clean API, deployment artifacts
```

---

## Environment Setup (WSL2 + uv)

```bash
# In WSL2 Ubuntu terminal:

# Verify CUDA is visible
nvidia-smi

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies from pyproject.toml
uv sync

# Run scripts without activating the venv
uv run python generate_signal.py --help

# Or activate and run directly
source .venv/bin/activate
python generate_signal.py --help

# Add a new dependency (never bare pip install)
uv add <package>

# Verify GPU in Python
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Run tests
uv run pytest tests/ -v
uv run pytest tests/signals/ -v   # fast (no torch import)
uv run pytest tests/data/ -v      # fast (no torch import)
```

VS Code extensions to install:
- Remote - WSL
- Python
- Pylance
- Jupyter
- GitLens
