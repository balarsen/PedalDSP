# CLAUDE.md: PedalDSP, learning to build digital guitar pedals on the Hothouse

Read `PROJECT_MAP.md` for the repo layout and what every recording is. This file holds the working rules.
Deeper docs live in `docs/`. Start with `docs/architecture.md`, `docs/processing_chain.md` and
`docs/daisy_programming.md`.

## What this project is

The owner is learning to design digital guitar pedals for the **Cleveland Music Co. Hothouse**, a DIY pedal
platform built around the Electrosmith **Daisy Seed** (STM32H750, ARM Cortex-M7 @ 480 MHz, 24-bit codec).
The Hothouse gives 6 knobs, 3 three-position toggles, 2 footswitches, 2 LEDs, mono in / mono out.

The method is **capture, model, compare**:

1. Play a designed training signal (`recordings/data/signals/`, reproducible from its JSON manifest) or a guitar DI
   through a real pedal, and record dry/wet pairs.
2. Fit models to the pairs in Python (`pedal_model/`): classical (FIR, IIR, Hammerstein, Wiener-Hammerstein,
   Volterra), neural (LSTM, GRU, TCN, WaveNet, DDSP, knob-conditioned variants) and ODE/physics models
   (`pedal_model/ode/`, shared with PedalODE). Score them on the metrics suite (ESR, STFT loss, THD, MCD).
3. Export the winning model to the Daisy (`daisy/`, `scripts/export_daisy_coeffs.py`), flash the Hothouse,
   re-record it, and compare the hardware output with the model and the analog pedal.

Pedals so far: **KlonSim** (a synthetic Klon used to validate the pipeline; it also runs on the Daisy),
**Boss GE-7** (7-band graphic EQ, the linear target), **JHS Notaklon** (Klon-style overdrive, the nonlinear
target), and an early Pi Muff stem pair.

## How to work with the owner

- **Be skeptical.** If a result, number or request does not make sense, say so plainly. Question your own tests too.
- **Say when you are guessing.** Mark assumptions ("I assume ...", "my guess is ..."). Ask when the answer changes
  what you build.
- **Number every question** (1., 2., 3. ...) so the owner can answer by number.
- The owner is a scientist (space physics) and a hobbyist guitarist. They are comfortable with C++ but don't enjoy
  it, and they want to learn. **Explain non-obvious DSP choices** briefly, in comments and in replies.
- Commit only when the owner asks.

## Environment

- **All Python runs on Windows** (since 2026-10-17): capture, analysis, fitting, training, notebooks, tests.
  The audio interface (ASIO) is only reachable from Windows, and the WAVs on C: load fast there. **WSL is used
  only to build firmware.**
- **uv project**, Python 3.12, the `.venv` at the repo root (Windows). `uv sync --extra rig` installs
  everything, including the capture GUI. Run with `uv run <command>`, and add dependencies with `uv add`
  (never bare `pip install`) so `pyproject.toml` and `uv.lock` stay in sync. `requirements.txt` is legacy.
- PyTorch on Windows comes from the CUDA 13.0 index (`[tool.uv.sources]` in pyproject; PyPI's Windows torch is
  CPU-only). GPU: RTX 2070 SUPER, 8 GB. On Linux, PyPI's torch is used.
- Tests: `uv run pytest tests`. The 6 failures in `tests/synthetic/test_pedal.py` come from the
  in-progress `pedal.py` edits brought over from D: (they failed there too); everything else passes.
- Lint: `uv run ruff check` and `uv run ruff format`, line length 100. New code must be clean; much of the
  older code is not yet.
- From WSL, never run `uv sync` in this repo: it would put a Linux `.venv` on top of the Windows one. WSL can
  run the Windows interpreter directly (`./.venv/Scripts/python.exe`), which the firmware Makefile does.
- Use `pathlib.Path` everywhere and never hard-code absolute paths. Scripts use
  `Path(__file__).resolve().parent.parent`, and notebooks use `Path(__file__).resolve().parents[n]`.

## Capture rig (`pedal_model/rig/`)

- Closed-loop capture without Studio One: `uv run python -m pedal_model.rig` (demo; `--hardware` for ASIO).
  Wiring, box settings, level primer and troubleshooting are in README.md and on the GUI's Setup page.
- Core modules (no Qt): `config` (configs/rig.json), `profiles/` (one JSON per pedal: controls, presets
  and their notes, map positions), `signals` (session signals from the 96 kHz masters: full or quick train,
  val, and the sweep-only signal for map steps), `backends` (ASIO via sounddevice, and `SimulatedBackend`),
  `engine` (duplex play/record), `calibrate`, `qc`, `bandfit` (RBJ peak/shelf fits for the slider map),
  `session`, `compare`. GUI in `gui/`: one page per step, each ending in a Continue button.
- **One GUI for every pedal.** A pedal is a profile JSON; the drawing adapts (sliders for the GE-7; knobs
  and switches for the Notaklon/RAT are planned after the GE-7 is proven on hardware, for the Klon + RAT
  combo goal).
- Preset kinds: "model" (train + val, for fitting) and "map" (sweep only, a measurement; its band fit is
  saved in params.json and session.json).
- **Saved files describe themselves** (owner, 2026-10-18): WAVs are named
  `<pedal>_<preset>_<role>_{wet,raw}_<sr>k_<YYYY-MM-DD_HHMM>.flac`; calibrations and setups are also kept as
  dated JSON files; the same fields are inside the JSON. Keep this convention for anything new.
- **Alignment rule:** takes are aligned on the interface latency measured on the loopback channel (Out 2 →
  In 2) plus the chain's fixed offset from calibration, *not* on the cross-correlation peak. The difference
  is the pedal's own delay, which belongs in the training data. QC reports it as "Pedal delay".
- Never play audio on the hardware without the person there: Out 1/2 are the Main outputs and may drive
  monitors. The GUI asks for confirmation before the first sound. Demo mode never opens the sound card.
- Tests (`tests/rig/`) use synthetic masters in tmp_path and the simulated backend: no hardware, no repo audio.
- Not verified yet (needs the person at the rig): the loopback latency and dropout test on the real 1824c,
  whether BrianEQ/BrianLevel can listen on the Windows Audio device while the rig holds ASIO, and whether the
  Main knob really controls Out 1/2.

## Recordings and data

- **All audio lives under `recordings/`:**
  - `recordings/<session>/`: Studio One sessions (`.song` plus `Media/*.wav`). Treat them as read-only.
  - `recordings/data/`: the pipeline's audio and metadata. That includes training signals and manifests
    (`signals/`), GE-7 captures (`ge7/`), Notaklon captures and training sets (`notaklon/`), KlonSim
    (`klon_sim/`) and stems (`stems/`).
- **A WAV sits next to its JSON.** Signal manifests find their audio with `manifest.path.with_suffix(".wav")`, and
  capture folders pair `wet_*.wav` with `params.json`. Move or rename them together, never separately.
- Paths in `params.json` are project-relative (`recordings/data/...`) or relative to the file (`../../signals/...`).
- Generated audio (model renders, comparisons) goes to `outputs/` and `results/`, not `recordings/`.
- **Studio One sessions:** don't edit `.song` files or rename a single `Media/` WAV, because the session finds
  its media by name. Moving a whole session folder is fine. `Cache/` and `History/` are regenerated, so ignore them.
- **Check every file's sample rate and format.** They vary: 44.1/48/96 kHz, and 16-bit, 24-bit or float32.
  The Daisy target is 48 kHz. Resample deliberately with `pedal_model/data/resample.py`, and state it.
- Dry/wet pairs carry hardware round-trip latency. **Align before you compare** (`pedal_model/capture/align.py`,
  `hardware_delay.py`), and report the lag you found.
- Hardware: PreSonus Studio 1824c ("Studio USB ASIO Driver", 20 in / 18 out), Radial ProRMP re-amper, Radial
  ProDI direct box. New captures come from the capture rig (`pedal_model/rig/`); the older ones were recorded in
  Studio One. Once the rig is proven, the Studio One-era capture files will likely be cleared (not before).

## Git

- Audio and big binaries are **never** committed: `*.wav`, `*.pkl`, `*.npy`, `*.pt`, `checkpoints/`, the Studio
  One `Cache/` and `History/` folders, and firmware build output are ignored. GitHub rejects files over 100 MB.
  On 2026-10-17, multi-GB `results.pkl` files were stripped from unpushed history.
- Plots (`*.png`) and small JSON/CSV results in `outputs/` and `results/` are tracked.
- Capture audio is 24-bit FLAC (lossless; 49–64 % of the WAV size on real captures). Session signals
  and model renders stay float32 WAV (FLAC stores integers only).
- Line endings are LF (`.gitattributes`: `* text=auto eol=lf`). Keep it that way: the firmware build runs in
  WSL, and the history was written with LF.
- Capture sessions: `session.json`, `params.json` and signal manifests are tracked (small, and they record how
  each capture was made); their WAVs are not. Demo runs (`recordings/data/*/sessions_demo/`) are ignored.
- Submodules in `external/` are pinned commits. Updating one is a deliberate commit that records the new
  pin; rebuild and re-test the firmware when you do.

## Firmware (Daisy / Hothouse)

- The source is in `daisy/` (`klonsim_pedal.cpp`, `klonsim_params.h`, `Makefile`). `scripts/export_daisy_coeffs.py`
  writes the params header and copies the sources into `results/pedalode/klonsim_v1/daisy/`, a complete
  build folder.
- **Libraries are git submodules** in `external/` (since 2026-10-17): `libDaisy` (v9.0.0+), `DaisySP`, and
  `HothouseExamples` (for `src/hothouse.h/.cpp` and the example effects). `git submodule update --init
  --recursive` fetches everything, including HothouseExamples' own older nested libDaisy/DaisySP (pinned 2024);
  those nested copies are **not** used: our firmware and the examples build against `external/libDaisy` and
  `external/DaisySP` (examples: `make LIBDAISY_DIR=… DAISYSP_DIR=…`). The Makefile finds the repo root with
  `git rev-parse`. The old clones in `D:\Projects` are no longer used.
- Build check 2026-10-17 (libDaisy v9.0.0-5): KlonSim 89,440 B flash (68.2%); HelloWorld, BasicTremolo,
  AmnesiaDelay, EchoKingMkII all build.
- Toolchain (WSL): `arm-none-eabi-gcc` 13.2, `make`, `dfu-util`. From Windows, wrap the command:
  `wsl.exe -e bash -lc 'cd /mnt/c/Users/Brian/Documents/GitHub/PedalDSP/daisy && make'` (from Git Bash, set
  `MSYS_NO_PATHCONV=1` so `/mnt/c/...` paths are not rewritten). Build libraries once with
  `make -C external/libDaisy` and `make -C external/DaisySP`. `usbipd` 5.3 is installed on Windows for
  attaching the DFU device. Never flash without the person asking.
- **DFU:** hold **both** footswitches for 2 s (the firmware must already be running), or hold BOOT on the Seed
  and tap RESET. USB reaches WSL through `usbipd attach --wsl --busid <id>`. Full steps are in
  `docs/daisy_programming.md`.
- Embedded rules: the audio callback is hard real-time, so **no allocation, no locks, no `printf`** in it. Use
  `float`, not `double` (the FPU is single-precision). Smooth knob values to avoid zipper noise. Watch CPU load,
  and keep large buffers in SDRAM (`DSY_SDRAM_BSS`). Test a model offline in Python before porting it.

## Code style

- PEP 8, `ruff check .` (line length 100). Type-annotate every function signature, with no `Any` unless unavoidable.
- Prefer `np.ndarray` and `torch.Tensor` in signatures, and give shapes in docstrings. Constants are
  `UPPER_SNAKE_CASE` at module level.
- **Functions:** every public function gets a one-line summary plus `Args` / `Returns`. Document units
  (`sr: int  # sample rate in Hz`), valid ranges and non-obvious defaults. Skip obvious parameters.
- **Inline comments** only for a non-obvious why: a hidden constraint, a numerical workaround, a DSP identity.
  No comments that restate the code.
- **Modules** get a one-line docstring naming their responsibility.

## Testing

- Every module in `capture/`, `models/`, `metrics/` and `train/` has a matching `tests/test_<module>.py`. Use
  `pytest`, not `unittest.TestCase`, and run `uv run pytest tests/`.
- Test pure functions on synthetic inputs with known answers, plus edge cases (empty input, one sample, all zeros,
  perfect prediction). Compare with `np.testing.assert_allclose(rtol=1e-5)`, not `==`.
- Model tests check the output shape, that the forward pass runs on CPU (GPU variants use `pytest.mark.skipif`),
  and that the loss is finite and > 0.
- Data pipeline tests use short synthetic WAVs made in the test. Windows must stay inside the audio, and dry/wet
  alignment must survive slicing.
- No file I/O beyond `tmp_path`, and no mocking of NumPy, SciPy or PyTorch internals. Seed `np.random` and
  `torch` in every test that uses randomness. Tests must pass on CPU only.

## Models, training, metrics

- All models inherit from `pedal_model/models/base.py` and implement `forward()` and `receptive_field`.
- Classical models run with NumPy and SciPy only, no PyTorch. Neural models support `torch.no_grad()` inference.
- Never hard-code a sample rate; pass `sr: int`. No global mutable state.
- Checkpoints go in `checkpoints/<model_name>/<timestamp>/` and are never overwritten. Log the epoch, train
  loss, val loss and val ESR. Training scripts take a config YAML (`configs/`).
- Metrics return a plain `float`, detached from the graph. `metrics/suite.py` is the only place that calls them
  all together, and metric modules don't import each other.

## Related projects

- **GE-7 PedalODE plan**: `recordings/data/ge7/GE7_PEDALODE_PLAN.md` (block-diagram model: summing EQ stage,
  learned parameters; replaces Ge7Sim as the GUI's model once rung A passes).
- **PedalODE**: physics-grounded ODE circuit-topology discovery and parameter fitting. It uses these GE-7 and
  Notaklon captures as its rungs (`docs/pedalode_klonsim_spec.md`, `pedal_model/ode/`).
- **BrianAudioLab** (local folder `../BrianEQ`): JUCE analyzer plug-ins for Studio One. Its planned `brianpedal`
  analyzer overlaps with this work.
