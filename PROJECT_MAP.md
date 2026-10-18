# PROJECT_MAP.md

Map of the PedalDSP repo, as of 2026-10-17. This was written when `D:\Projects\PedalDSP` (the code) was merged
with the Studio One recording sessions. Rules are in `CLAUDE.md`. The older, more detailed layout notes are in
`docs/pedal_project_map.md`. Entries marked *(guess)* are inferred from file names and need the owner to confirm.

## Top level

```
PedalDSP/
├── README.md                   setup, capture rig wiring + quickstart, troubleshooting
├── CLAUDE.md, PROJECT_MAP.md   working rules, this map
├── pedal_model/                Python library (uv package "pedal-model"), incl. the capture rig (rig/)
├── scripts/                    CLI entry points: analyze_*, fit_*, make_*, export_*, train_*
├── notebooks/                  marimo notebooks (.py); learning tiers, Pi Muff, Notaklon diagnostics
├── tests/                      pytest suite, mirrors pedal_model/
├── daisy/                      Hothouse firmware source: KlonSim ODE model (C++ / libDaisy)
├── external/                   git submodules: libDaisy (v9), DaisySP, HothouseExamples
├── configs/                    experiment YAML; rig.json (capture rig: device, channels, box settings)
├── docs/                       plans, capture results, model/metric references
├── recordings/                 ALL audio (WAVs git-ignored, metadata tracked)
│   ├── <Studio One sessions>/  the five DAW sessions (below)
│   └── data/                   pipeline audio + JSON manifests/params (below)
│       └── <pedal>/sessions/   capture-rig sessions (sessions_demo/ = simulated, git-ignored)
├── outputs/                    analysis plots + summaries per experiment (PNGs/JSON tracked, WAVs ignored)
├── results/                    model-zoo results (CSV tracked, *.pkl ignored), PedalODE fits, firmware build dir
├── checkpoints/                trained weights (ignored)
├── generate_signal.py          CLI: build training/validation signals from parameters or a manifest
├── make_klon_sim.py            render the KlonSim "pedal" over the signals → recordings/data/klon_sim/
├── run_eval.py, validate_pipeline.py   end-to-end eval / pipeline sanity check
└── pyproject.toml, uv.lock, requirements*.txt
```

## `pedal_model/`

| Package | Role |
|---|---|
| `signals/` | `generate.py` (parameterised, seeded signal generator; writes WAV + JSON manifest), `manifest.py` (Section/Manifest reader) |
| `data/` | `dataset.py` (PedalDataset/ChunkDataset, `from_manifest()`), `resample.py` (96→48 kHz polyphase, float32↔int16 for the Daisy), `augment.py` |
| `capture/` | `align.py` (cross-correlation latency), `hardware_delay.py`, `verify.py`, `generate_sweep.py` |
| `rig/` | Closed-loop capture rig: `backends` (ASIO / simulated), `engine`, `signals`, `calibrate`, `qc`, `session`, `compare`, `profiles/` (ge7.json), `gui/` (PySide6). Run `uv run python -m pedal_model.rig` |
| `models/classical/` | FIR, IIR, Hammerstein, Wiener-Hammerstein, Volterra, circuit dictionary |
| `models/neural/` | MLP, LSTM, GRU, TCN, WaveNet, DDSP, DDSP-Klon, conditioned LSTM/TCN (knob inputs) |
| `ode/` | PedalODE models: filters, nonlinearity, Wiener-Hammerstein ODE, fitting, identifiability, step0; GE-7 block model (`blocks`, `ge7_model`, `ge7_fitter`: continuous-time blocks, Levenberg–Marquardt fit, Fisher report) |
| `synthetic/` | `pedal.py` (KlonSim), `ge7.py` (Ge7Sim peaking-EQ cascade, calibrated from the slider captures) |
| `metrics/` | time domain (ESR, null depth), frequency domain (multi-STFT), harmonic (THD), perceptual (MCD); `suite.py` |
| `eval/` | model-zoo harness, comparison, diagnostics, report, results I/O |
| `train/` | trainer, losses, scheduler |
| `inference/` | streaming (block-by-block) runner, the way the Daisy would run it |
| `utils/` | analysis, plotting, resampling, synthesis helpers |

## Capture rig sessions

`recordings/data/<pedal>/sessions/<YYYY-MM-DD_HHMM>_<sr>k/`: `session.json` (settings, rig snapshot,
presets, calibration, per-preset status, event log), dated `calibration_*.json` and `rig_setup_*.json`,
`calibration/` (the C2–C4 reference captures, 24-bit FLAC),
`signals/` (the session's own train/val/sweep signals with manifests), and one folder per preset or map
step with `params.json` (model presets load in `scripts/fit_models.py --klon-sim-dir`; map steps record
their band fit) and self-describing 24-bit FLAC captures,
`<pedal>_<preset>_<role>_{wet,raw}_<sr>k_<date>_<time>.flac`.
Simulated runs go to `sessions_demo/` and are never committed.

**To retire later:** once the rig is proven, the Studio One-era captures (`recordings/data/*/captures/`, the
`recordings/<session>/Media` WAVs) will likely be cleared. Nothing has been removed yet.

## `recordings/data/` (pipeline audio)

Every WAV sits beside the JSON that describes it. Keep the pairs together.

| Path | Contents |
|---|---|
| `signals/` | `train_signal_v1/v2.{wav,json}` (96 kHz float32, 642.5 s), `train_signal_v2_48k.{wav,json,npy}`, `val_signal_v1`, `notaklon_val_take1`, re-amp bypass references (`train_signal_v2_reamp only*.wav`), `legacy/` |
| `ge7/GE7_PEDALODE_PLAN.md` | Plan for the GE-7 topology model with learned parameters (PedalODE rungs A0–C) |
| `ge7/presets/` | 14 preset JSONs (slider dB settings) |
| `ge7/captures/preset_<name>/` | `wet_train_signal_v2.wav` + `params.json` per preset; `preset_flat/GE7-Flat2 - flat 2.wav` is the reference |
| `ge7/captures/single_slider/` | `slider_<f>hz.wav` (one band moved), `slider_level_{p,m}10db.wav` |
| `notaklon/captures/V<v>-T<t>-G<g>-<shamrock>/` | `wet_train_signal_v2_48k.wav` + `params.json`; `bypass/`, `treble_sweep/`, guitar dry/wet pair |
| `notaklon/training/<preset>/` | `wet_train.wav` + harness-format `params.json` (from `scripts/prepare_notaklon_training.py`) |
| `klon_sim/klon_sim_v1/` | KlonSim wet renders of train/val signals + `params.json` |
| `captures/` | the 2026-06-09 medley (stereo + mono); empty Notaklon placeholders |
| `stems/2026-06-04_pi_muff/` | Pi Muff dry/wet stems + `predictions/` from early models |

**Notaklon naming:** `V`/`T`/`G` are the Volume/Treble/Gain knob positions in clock hours. The dials travel from
6 o'clock (minimum) to 4 o'clock (maximum), and 12 is noon. `True`/`False` is the **Shamrock** switch.
`docs/notaklon_capture_plan.md` maps knob positions to normalised parameters.

## `recordings/<session>/` (Studio One sessions)

Each session is a `.song` (a zip of XML, tracked in git) plus `Media/`, `Cache/` and `History/`, which are
ignored. The `.song` files store absolute paths from an earlier location (`D:/Studio One/Projects/PedalDSP/...`).
Studio One finds the WAVs in the session's own `Media/` folder, so a session still opens after its whole folder moves.

1. **`2026-06-09 Dry to DAW/`**: dry guitar DI riffs (*Fire on the Mountain*, *Flagpole Sitta*, *I Love Rock*,
   *TNT*, *Wild Thing*), 48 kHz 16-bit. `Mixdown/2026-06-09 Medley.wav` is the 116.8 s medley used as the dry test
   material everywhere else.
2. **`Testing/`**: the first dry vs wet test. The dry medley is **44.1 kHz** (the only one at that rate).
   `wet.wav` is at 48 kHz *(guess: through the Daisy)*, and `Stems/` holds a dry-minus-wet stem *(guess)*.
3. **`DaisyKlonsim/`** (2026-06-13): KlonSim **running on the Daisy**, re-recorded. `trinDaisy*` is the training
   signal through the Daisy, and `wetDaistKlon 1–5` is the medley through the Daisy (firmware iterations
   *(guess)*). The 32-bit `*_klonsim_wet_48kHz` files are offline references. The analysis is in
   `outputs/train_signal_daisy_test/` and `results/pedalode/klonsim_v1/staged/daisy_comparison/`.
4. **`GE7/GE7/`** (2026-06-14 → 16): the Boss GE-7 recording session. It has single-band takes named after the
   band frequency (`100` … `6400`), level ±10, and about 644 s training-signal runs per preset (flat, smile,
   frown, bass boost, mid scoop, and so on). Its exports became `recordings/data/ge7/captures/`, and the results
   are in `docs/ge7_capture_results.md`.
5. **`notaklon/Notaklon/`** (2026-06-19 → 20): the JHS Notaklon recording session, with the same `V-T-G-Shamrock`
   naming. Its exports became `recordings/data/notaklon/captures/`, and the results are in
   `docs/notaklon_capture_results.md`. The `guitar dry`/`guitar wet` `(2)` suffixes are crossed: lengths pair
   `dry` with `wet(2)`, and the other way round.

Studio One names: `name(2).wav` is an automatic re-take suffix, and `name 2.wav` is a duplicated track. Typos
(`mounatin`, `Daist`, `levl10`, `trin`) are in the original names. Keep them, because the sessions reference them.

## Firmware path

`results/pedalode/klonsim_v1/staged/stage3_params.json` → `scripts/export_daisy_coeffs.py` →
`results/pedalode/klonsim_v1/daisy/` (header + copied `daisy/` sources) → `make && make program-dfu` in WSL.
The libraries are the submodules in `external/` (built once with `make -C external/libDaisy` and
`make -C external/DaisySP` in WSL).

## `docs/`

| Doc | Topic |
|---|---|
| `architecture.md`, `processing_chain.md` | Pipeline design, and how to test a real pedal from start to finish |
| `signal_design.md` | Why the training and validation signals look the way they do |
| `pedal_capture_plan.md`, `ge7_plan.md`, `notaklon_capture_plan.md` | Capture procedures |
| `ge7_capture_results.md`, `notaklon_capture_results.md` | Measured results |
| `notaklon_emulation_plan_v3.md` | The 14-step Notaklon plan |
| `pedal_classical_models.md`, `pedal_neural_models.md`, `pedal_metrics.md` | Model and metric references |
| `klon_sim_explainer.md`, `pedalode_klonsim_spec.md` | KlonSim, and the PedalODE parameter-recovery experiment |
| `daisy_programming.md` | Daisy/Hothouse build, DFU and usbipd setup |

## Known loose ends

- `notebooks/learning_tier1_linear_effects.py` hard-codes `/mnt/d/Projects/PedalDSP/data/01 - Reach for the sun (RC3).wav`.
  That file doesn't exist on either drive.
- The notebooks' default text field `recordings/data/01 - Reach for the sun (RC3).wav` points at the same missing file.
- `results/*.pkl` (up to 1.36 GB each) exist only locally. They're regenerated with `scripts/fit_models.py`.
