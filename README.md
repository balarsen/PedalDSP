# PedalDSP

Learning to build digital guitar pedals for the **Cleveland Music Co. Hothouse** (Electrosmith Daisy Seed).
The loop is **capture → model → compare**: record a real pedal, fit models to it in Python, run the best one
on the Hothouse, and compare the hardware against the original pedal.

- `pedal_model/`: Python library (signals, capture, models, metrics, training, the capture rig)
- `daisy/`: Hothouse firmware (KlonSim); `external/`: libDaisy, DaisySP, HothouseExamples (git submodules)
- `recordings/`: all audio (Studio One sessions and `recordings/data/`); WAVs are not in git
- Working rules: [CLAUDE.md](CLAUDE.md). Repo layout: [PROJECT_MAP.md](PROJECT_MAP.md). Deep docs: [docs/](docs/)

## Setup (Windows)

Python runs on Windows (the audio interface is only reachable there). WSL is used only to build firmware.

```powershell
winget install astral-sh.uv            # once
git submodule update --init --recursive
uv sync --extra rig                    # Python 3.12 + CUDA PyTorch + the capture rig's GUI packages
uv run pytest tests                    # 779 pass; 6 known failures in tests/synthetic/test_pedal.py (work in progress)
```

## Capture rig: closed-loop recording without Studio One

The rig plays a test signal out of the interface, through the pedal, and records the return in the same
ASIO stream. The app tells you how to set the pedal, checks every take, and saves it in the format
`scripts/fit_models.py` reads.

```powershell
uv run python -m pedal_model.rig              # demo mode: a simulated rig, nothing is played
uv run python -m pedal_model.rig --hardware   # the real 1824c over ASIO
```

Steps in the app, each ending in a Continue button to the next: **Start** (sample rate 48/96 kHz, pedal,
quick or full signal) → **Setup** (wiring diagram: every jack named with its connector, every cable with
its level type; Save writes the settings) → **Calibrate** (four guided steps C1–C4: loopback, chain alone,
pedal bypassed, pedal engaged; each says exactly which cable to move, tests the move, and unlocks the next) →
**Plan** (presets with notes and a model-response preview; add your own preset, or the 32-step slider
map) → **Run** (the pedal drawn as it must be set, press Space; live meters, spectrum and response; QC with
a stage-by-stage progress bar; accept or retake) → **Review** (per-step responses, 48 vs 96 kHz comparison,
and the slider-map table). Close Studio One first: ASIO allows one program at a time.

**Slider map (GE-7).** Each slider alone at +15, +7.5, −7.5 and −15 dB, with a ~1 min sweep-only signal.
Every step is fitted with an RBJ peaking filter (centre frequency, Q, bandwidth in octaves, gain); the
6.4k band as a high shelf (corner, slope, gain) and LEVEL as a flat gain. Review shows the measured
numbers next to the same fit applied to the Ge7Sim model.

### Wiring

```
PC (Windows) ──USB── PreSonus Studio 1824c   (ASIO: "Studio USB ASIO Driver")

 SEND      1824c Out 1 (rear, balanced TRS, line level) ──► Radial ProRMP INPUT (XLR)
           ProRMP OUTPUT (1/4" TRS jack, instrument level) ► PEDAL IN (guitar cable)
 RETURN    PEDAL OUT (1/4" TS, instrument level) ─────────► Radial ProDI input
           ProDI XLR output (mic level) ──────────────────► 1824c In 1 (front, XLR; 48 V OFF)
 LOOPBACK  1824c Out 2 (rear, TRS, line level) ───────────► 1824c In 2 (front, TRS; gain at minimum)
```

| Box | Setting | Why |
|---|---|---|
| 1824c Main knob | Mark its position | Out 1/2 are the Main outputs; I believe this knob sets their level (verify on your unit), so it is part of the send level |
| 1824c In 1 | Gain set during calibration; 48 V off | Mic preamp for the ProDI's mic-level output |
| 1824c In 2 | Gain at minimum; 48 V off | Line-level loopback must not clip |
| Radial ProRMP | Input XLR, output 1/4" TRS jack. Level: note the position; ground lift off unless there is hum | Turns line level into a guitar-like instrument signal for the pedal |
| Pedal | Engaged, powered | Device under test |
| Radial ProDI | −15 dB pad (input side): off, on for hot pedals; ground lift (XLR output side): off unless there is hum | Gives the pedal a high-impedance load and sends balanced mic level to In 1 |

**Why the loopback cable?** It records what was sent through the interface only, so every take measures the
interface's own latency and level. Takes are aligned on that latency, so a pedal's own delay (EQ group
delay, a digital pedal's buffer) stays in the data instead of being cut out by cross-correlation.

**Levels in one paragraph.** Line level (about +4 dBu, balanced) is what the interface puts out; instrument
level (roughly 0.1–1 V peak, unbalanced, high impedance) is what a guitar puts out and what pedals expect;
mic level (millivolts, balanced) needs a preamp. The ProRMP converts line → instrument, the ProDI converts
instrument → mic. The app's Setup page has a longer version ("Why these boxes?").

### Troubleshooting

| Symptom (as the app reports it) | Likely cause | Fix |
|---|---|---|
| "Studio USB ASIO Driver" not found | Studio One (or another ASIO app) is open, or the 1824c is off | Close it; power the interface |
| No loopback signal on In 2 | Cable missing, or In 2 gain fully off | Check Out 2 → In 2; In 2 gain just above minimum |
| In 1 hears the loopback | Cables swapped | Out 1 must go to the ProRMP, Out 2 to In 2 |
| No return on In 1 | Pedal off or unpowered, a cable out, ProRMP level at zero, In 1 gain down | Follow the chain on the Setup page |
| Return clipped | In 1 gain too high | Lower In 1 gain or switch on the ProDI pad; recalibrate |
| Gain changes across levels (not linear) | The reamp path is saturating | Lower the Main knob or the ProRMP level, raise In 1 gain |
| Noise floor / hum high | Ground loop or noisy supply | ProDI or ProRMP ground lift; another power supply |
| Dropouts | Block size too small, or other audio software running | Raise the block size (Setup → PC box) |
| Interface latency moved | The ASIO buffer size changed in Universal Control | Recalibrate |

### What a session writes

`recordings/data/<pedal>/sessions/<date>_<time>_<sr>k/` (demo runs go to `sessions_demo/`, never committed):

- `session.json`: settings, rig snapshot, presets, calibration, status and an event log
- `calibration_<date>_<time>.json` and `rig_setup_<date>_<time>.json`: every calibration and every saved
  setup, as dated files (Setup → Save also writes `configs/rig.json` and a dated copy next to it)
- `calibration/<pedal>_ref_<chain|bypass|engaged_flat>_raw_<sr>k_<date>_<time>.flac`: the C2–C4 reference
  captures; their response curves are in the calibration JSON, and QC divides every take by one of them
- `signals/`: the session's own train/val/sweep signals with their manifests
- `<preset>/params.json` plus, per role, `<pedal>_<preset>_<role>_wet_<sr>k_<date>_<time>.flac` (return
  aligned on the loopback latency) and `..._raw_...flac` (the unaligned return + loopback), 24-bit FLAC:
  lossless, about half the size of WAV. File names carry
  the pedal, preset, role, sample rate and time, so a copied-out file is still identifiable; retakes keep
  the earlier files.

Fit a preset with:

```powershell
uv run python scripts/fit_models.py --klon-sim-dir recordings/data/ge7/sessions/<session>/<preset> --classical-only
uv run python scripts/compare_sessions.py <session_48k> <session_96k>     # same presets at two sample rates
```

**Old captures:** the Studio One-era files (`recordings/data/*/captures/`, the `recordings/<session>/Media`
WAVs) stay until the rig is proven; after that they will likely be cleared.

## BrianEQ alongside (to be verified)

The capture app holds the 1824c over ASIO. To watch the return in BrianEQ or BrianLevel at the same time,
open the Standalone's audio settings and pick the Windows Audio device "Mic/Inst/Line In 1/2 (Studio
1824c)". Whether the PreSonus driver allows both at once is still to be tested on the hardware. The rig's
Tools menu launches the installed Standalones.

## Firmware (WSL)

```powershell
wsl.exe -e bash -lc 'cd /mnt/c/Users/Brian/Documents/GitHub/PedalDSP && make -C external/libDaisy && make -C external/DaisySP'   # once
wsl.exe -e bash -lc 'cd /mnt/c/Users/Brian/Documents/GitHub/PedalDSP/daisy && make'
```

Flashing (DFU, `usbipd` to attach the device to WSL): [docs/daisy_programming.md](docs/daisy_programming.md).
