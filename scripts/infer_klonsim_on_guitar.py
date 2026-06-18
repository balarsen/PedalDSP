#!/usr/bin/env python3
"""Apply fitted KlonSim model parameters to any dry guitar recording.

Runs the KlonSim topology (HPF1 → drive → tanh → HPF2 → shelf → level)
using scipy biquads with the recovered parameters from stage3_params.json.
The biquad coefficients match HighShelfBiquad and SecondOrderHighpass exactly
(Audio EQ Cookbook bilinear-transform formulas at the training sample rate).

Usage:
    uv run python scripts/infer_klonsim_on_guitar.py <dry.wav> [--params ...] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf


def _highshelf_biquad(fc: float, gain_db: float, sr: int) -> np.ndarray:
    """Audio EQ Cookbook high-shelf biquad SOS (matches HighShelfBiquad.forward exactly).

    Args:
        fc: Shelf frequency in Hz.
        gain_db: Shelf gain in dB (positive = boost).
        sr: Sample rate in Hz.

    Returns:
        SOS array, shape (1, 6), for use with scipy.signal.sosfilt.
    """
    A = 10.0 ** (gain_db / 40.0)
    sqrt_A = math.sqrt(A)
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    Q = 0.707  # maximally flat shelf slope, matches HighShelfBiquad default
    alpha = sin_w0 / (2.0 * Q)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2.0 * sqrt_A * alpha)
    b1 = -2.0 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2.0 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2.0 * sqrt_A * alpha
    a1 = 2.0 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2.0 * sqrt_A * alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def apply_klonsim_model(
    dry: np.ndarray,
    fc1: float,
    fc2: float,
    shelf_freq: float,
    shelf_gain_db: float,
    drive: float,
    output_level: float,
    sr: int,
) -> np.ndarray:
    """Apply KlonSim topology with given parameters via scipy sosfilt.

    Signal chain:
        dry → HPF1(fc1) → ×drive → tanh → HPF2(fc2) → Shelf(shelf_freq, shelf_gain_db) → ×level → wet

    HPF1 and HPF2 are 2nd-order Butterworth highpass filters (matches SecondOrderHighpass).
    The nonlinearity is exact tanh (NL R²=0.9988 in the trained model, so this is a faithful
    approximation — the ESR difference vs the full trained MLP is < 0.01).

    Args:
        dry: Dry input, shape (N,), float32 or float64.
        fc1: Pre-filter cutoff in Hz.
        fc2: Post-filter cutoff in Hz.
        shelf_freq: High shelf frequency in Hz.
        shelf_gain_db: Shelf gain in dB.
        drive: Drive multiplier applied before the nonlinearity.
        output_level: Output level scalar.
        sr: Sample rate in Hz.

    Returns:
        Wet output, shape (N,), float32.
    """
    sos_hpf1 = scipy.signal.butter(2, fc1 / (sr / 2.0), btype="high", output="sos")
    sos_hpf2 = scipy.signal.butter(2, fc2 / (sr / 2.0), btype="high", output="sos")
    sos_shelf = _highshelf_biquad(shelf_freq, shelf_gain_db, sr)

    x = scipy.signal.sosfilt(sos_hpf1, dry.astype(np.float64))
    x = np.tanh(x * drive)
    x = scipy.signal.sosfilt(sos_hpf2, x)
    x = scipy.signal.sosfilt(sos_shelf, x)
    return (x * output_level).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dry", type=Path, help="Dry guitar WAV file.")
    p.add_argument(
        "--params",
        type=Path,
        default=Path("results/pedalode/klonsim_v1/staged/stage3_params.json"),
        help="stage3_params.json from fit_klonsim_ode.py.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same directory as the dry file.",
    )
    args = p.parse_args()

    if not args.dry.is_file():
        raise FileNotFoundError(f"Dry WAV not found: {args.dry}")
    if not args.params.is_file():
        raise FileNotFoundError(f"Params JSON not found: {args.params}")

    dry, sr = sf.read(str(args.dry), dtype="float32")
    if dry.ndim > 1:
        dry = dry[:, 0]  # take first channel if stereo/multichannel

    with open(args.params) as f:
        params = json.load(f)

    fc1 = float(params["fc1_hz"])
    fc2 = float(params["fc2_hz"])
    shelf_freq = float(params["shelf_freq_hz"])
    shelf_gain_db = float(params["shelf_gain_db"])
    drive = float(params["drive"])
    output_level = float(params["output_level"])

    print(f"[infer] Input:  {args.dry}  ({len(dry) / sr:.2f} s at {sr} Hz)")
    print(
        f"[infer] Params: fc1={fc1:.1f} Hz  fc2={fc2:.1f} Hz  "
        f"shelf={shelf_freq:.0f} Hz {shelf_gain_db:+.2f} dB  "
        f"drive={drive:.3f}  level={output_level:.3f}"
    )

    wet = apply_klonsim_model(dry, fc1, fc2, shelf_freq, shelf_gain_db, drive, output_level, sr)

    out_dir = args.out_dir or args.dry.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip trailing _dry / __dry suffix so outputs don't read as "medley__dry__wet"
    base = args.dry.stem
    for _sfx in ("__dry", "_dry"):
        if base.endswith(_sfx):
            base = base[: -len(_sfx)]
            break

    model_path = out_dir / f"{base}__model_wet.wav"
    sf.write(str(model_path), wet, sr, subtype="FLOAT")
    print(f"[infer] Saved:  {model_path}")

    # Copy the KlonSim reference wet (sibling file: __dry → __klonsim)
    klonsim_src = args.dry.parent / f"{base}__klonsim.wav"
    if klonsim_src.is_file():
        klonsim_dst = out_dir / f"{base}__klonsim_wet.wav"
        shutil.copy2(str(klonsim_src), str(klonsim_dst))
        print(f"[infer] Copied: {klonsim_dst}  (KlonSim reference)")
    else:
        print(f"[infer] Note:   KlonSim reference not found at {klonsim_src} — skipping copy")


if __name__ == "__main__":
    main()
