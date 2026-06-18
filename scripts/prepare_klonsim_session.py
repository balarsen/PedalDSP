#!/usr/bin/env python3
"""Prepare 48 kHz source files for the KlonSim three-way comparison.

Reads the existing 44.1 kHz dry recording, resamples it to 48 kHz, then
generates two wet references at 48 kHz from the same source:

  dry_48kHz.wav         — resampled dry (play this into the Daisy)
  klonsim_wet_48kHz.wav — Python KlonSim simulation (ground-truth reference)
  model_wet_48kHz.wav   — fitted ODE model (the PC reproduction)

All three files share exactly the same input, so the comparison is clean.
The Daisy wet is NOT generated here — you record that from the hardware.

Usage:
    uv run python scripts/prepare_klonsim_session.py
    uv run python scripts/prepare_klonsim_session.py --dry <path> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from pedal_model.utils.resampling import resample_audio
from pedal_model.synthetic.pedal import KlonSim, KlonSimParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DRY = (
    PROJECT_ROOT
    / "outputs"
    / "2026-06-09_medley_mono"
    / "2026-06-09_medley_mono__dry.wav"
)
DEFAULT_PARAMS = (
    PROJECT_ROOT
    / "results"
    / "pedalode"
    / "klonsim_v1"
    / "staged"
    / "stage3_params.json"
)
DEFAULT_SIM_PARAMS = (
    PROJECT_ROOT / "data" / "klon_sim" / "klon_sim_v1" / "params.json"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "outputs" / "2026-06-09_medley_mono"
)
TARGET_SR = 48_000


# ── KlonSim ODE model (mirrors infer_klonsim_on_guitar.py exactly) ────────────

def _highshelf_biquad(fc: float, gain_db: float, sr: int) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    sqrt_A = math.sqrt(A)
    w0 = 2.0 * math.pi * fc / sr
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2.0 * 0.707)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2.0 * sqrt_A * alpha)
    b1 = -2.0 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2.0 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2.0 * sqrt_A * alpha
    a1 = 2.0 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2.0 * sqrt_A * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _apply_ode_model(
    dry: np.ndarray,
    fc1: float,
    fc2: float,
    shelf_freq: float,
    shelf_gain_db: float,
    drive: float,
    output_level: float,
    sr: int,
) -> np.ndarray:
    sos1 = scipy.signal.butter(2, fc1 / (sr / 2.0), btype="high", output="sos")
    sos2 = scipy.signal.butter(2, fc2 / (sr / 2.0), btype="high", output="sos")
    sos_shelf = _highshelf_biquad(shelf_freq, shelf_gain_db, sr)
    x = scipy.signal.sosfilt(sos1, dry.astype(np.float64))
    x = np.tanh(x * drive)
    x = scipy.signal.sosfilt(sos2, x)
    x = scipy.signal.sosfilt(sos_shelf, x)
    return (x * output_level).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry", type=Path, default=DEFAULT_DRY,
                   help="44.1 kHz dry guitar WAV.")
    p.add_argument("--params", type=Path, default=DEFAULT_PARAMS,
                   help="stage3_params.json from the ODE fit.")
    p.add_argument("--sim-params", type=Path, default=DEFAULT_SIM_PARAMS,
                   help="params.json used to generate the original KlonSim wet signals.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    for path, label in [
        (args.dry, "dry WAV"),
        (args.params, "ODE params"),
        (args.sim_params, "sim params"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Strip __dry suffix for cleaner output names
    base = args.dry.stem
    for sfx in ("__dry", "_dry"):
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break

    # ── Step 1: resample dry to 48 kHz ───────────────────────────────────────
    dry_raw, src_sr = sf.read(str(args.dry), dtype="float32", always_2d=False)
    if dry_raw.ndim > 1:
        dry_raw = dry_raw[:, 0]

    print(f"[prepare] Dry:  {args.dry.name}  ({src_sr} Hz, {len(dry_raw)/src_sr:.1f} s)")

    dry_48kHz = resample_audio(dry_raw, src_sr, TARGET_SR, quality="HQ")

    dry_out = args.out_dir / f"{base}__dry_48kHz.wav"
    sf.write(str(dry_out), dry_48kHz, TARGET_SR, subtype="FLOAT")
    print(f"[prepare] Saved dry_48kHz:        {dry_out}")

    # ── Step 2: KlonSim Python simulation (ground-truth reference) ───────────
    sim_doc = json.loads(args.sim_params.read_text())
    sim_p = sim_doc.get("params", {})
    klonsim_params = KlonSimParams(
        drive=float(sim_p.get("drive", 4.0)),
        input_hpf_hz=float(sim_p.get("input_hpf_hz", 80.0)),
        clip_hpf_hz=float(sim_p.get("clip_hpf_hz", 350.0)),
        presence_hz=float(sim_p.get("presence_hz", 3000.0)),
        presence_db=float(sim_p.get("presence_db", 2.0)),
        output_level=float(sim_p.get("output_level", 0.7)),
        noise_floor_dbfs=float(sim_p.get("noise_floor_dbfs", -70.0)),
        noise_seed=int(sim_p.get("noise_seed", 42)),
    )
    pedal = KlonSim(klonsim_params)
    klonsim_wet = pedal.process(dry_48kHz, TARGET_SR)

    ref_out = args.out_dir / f"{base}__klonsim_wet_48kHz.wav"
    sf.write(str(ref_out), klonsim_wet, TARGET_SR, subtype="FLOAT")
    print(f"[prepare] Saved klonsim_wet_48kHz: {ref_out}")
    print(f"          KlonSim params: drive={klonsim_params.drive:.3f}  "
          f"hpf1={klonsim_params.input_hpf_hz:.0f}Hz  "
          f"hpf2={klonsim_params.clip_hpf_hz:.0f}Hz  "
          f"shelf={klonsim_params.presence_hz:.0f}Hz {klonsim_params.presence_db:+.2f}dB  "
          f"level={klonsim_params.output_level:.3f}")

    # ── Step 3: ODE fitted model ──────────────────────────────────────────────
    ode = json.loads(args.params.read_text())
    fc1          = float(ode["fc1_hz"])
    fc2          = float(ode["fc2_hz"])
    shelf_freq   = float(ode["shelf_freq_hz"])
    shelf_gain   = float(ode["shelf_gain_db"])
    drive        = float(ode["drive"])
    output_level = float(ode["output_level"])

    model_wet = _apply_ode_model(
        dry_48kHz, fc1, fc2, shelf_freq, shelf_gain, drive, output_level, TARGET_SR
    )

    model_out = args.out_dir / f"{base}__model_wet_48kHz.wav"
    sf.write(str(model_out), model_wet, TARGET_SR, subtype="FLOAT")
    print(f"[prepare] Saved model_wet_48kHz:  {model_out}")
    print(f"          ODE params:    drive={drive:.3f}  "
          f"hpf1={fc1:.0f}Hz  hpf2={fc2:.0f}Hz  "
          f"shelf={shelf_freq:.0f}Hz {shelf_gain:+.2f}dB  "
          f"level={output_level:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    daisy_out = args.out_dir / f"{base}__daisy_wet.wav"
    print(f"""
[prepare] Done. Three files ready at 48 kHz:
  dry_48kHz:        {dry_out}
  klonsim_wet_48kHz:{ref_out}
  model_wet_48kHz:  {model_out}

Next steps:
  1. Flash Daisy with KlonSim firmware (see docs/daisy_programming.md)
  2. Set DAW to 48 kHz
  3. Play into Daisy: {dry_out}
  4. Record Daisy output to: {daisy_out}
  5. Run comparison:

uv run python scripts/compare_ode_daisy.py \\
  --dry   {dry_out} \\
  --ref   {ref_out} \\
  --pc    {model_out} \\
  --daisy {daisy_out}
""")


if __name__ == "__main__":
    main()
