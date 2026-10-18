#!/usr/bin/env python3
"""Generate KlonSim v2 wet captures at a specific knob setting.

Processes the train and val signals through the improved KlonSim (parallel
clean-blend topology, asymmetric clipping) and saves wet WAVs alongside a
versioned params.json.

Usage:
    # Gain 11 o'clock, treble noon, volume noon (sweet spot)
    uv run python scripts/make_klonsim_captures.py --gain 11 --treble 12 --volume 12

    # Gain 9 o'clock (low-gain setting)
    uv run python scripts/make_klonsim_captures.py --gain 9 --treble 12 --volume 12 --name gain9_noon

    # Custom params (bypass knob mapping)
    uv run python scripts/make_klonsim_captures.py --drive 5.0 --blend 0.4 --asymmetry 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pedal_model.synthetic.pedal import KlonKnobs, KlonSim, KlonSimParams

SIGNALS_DIR = PROJECT_ROOT / "recordings" / "data" / "signals"
CAPTURES_BASE = PROJECT_ROOT / "recordings" / "data" / "klon_sim"

# Use the same 48 kHz training signal as the GE-7 captures so KlonSim and
# GE-7 wet signals are on identical dry source — required for chaining them.
TRAIN_WAV = SIGNALS_DIR / "train_signal_v2_48k.wav"
TRAIN_MANIFEST = SIGNALS_DIR / "train_signal_v2_48k.json"

# Val fraction: last VAL_FRAC of the signal (same split as train_ge7_conditioned.py)
VAL_FRAC: float = 0.1


def make_captures(params: KlonSimParams, out_dir: Path) -> None:
    """Generate wet WAVs and write params.json.

    Uses train_signal_v2_48k.wav (same dry source as GE-7 captures).  Splits
    into train (first 90%) and val (last 10%) to match the GE-7 training split.

    Args:
        params: KlonSim parameters to use for processing.
        out_dir: Output directory (created if absent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pedal = KlonSim(params)

    if not TRAIN_WAV.exists():
        raise FileNotFoundError(f"Dry signal not found: {TRAIN_WAV}")

    dry, sr = sf.read(str(TRAIN_WAV), dtype="float32", always_2d=False)
    total_n = len(dry)
    val_n = int(total_n * VAL_FRAC)
    train_n = total_n - val_n
    print(f"  Dry: {TRAIN_WAV.name}  {total_n/sr:.1f}s @ {sr} Hz", flush=True)
    print(f"  Split: train={train_n/sr:.1f}s  val={val_n/sr:.1f}s")

    for label, dry_slice in [("train", dry[:train_n]), ("val", dry[train_n:])]:
        print(f"  Processing {label} …", flush=True)
        wet = pedal.process(dry_slice, sr)
        out_wav = out_dir / f"wet_{label}_signal_v2_48k.wav"
        sf.write(str(out_wav), wet, sr, subtype="FLOAT")
        print(f"  Saved {out_wav.relative_to(PROJECT_ROOT)}")
        esr = float(np.mean((wet - dry_slice) ** 2) / (np.mean(dry_slice ** 2) + 1e-10))
        print(f"    ESR vs dry: {esr:.3f}")

    # Write params.json
    meta = {
        "schema_version": "2.0",
        "date_created": datetime.now(timezone.utc).isoformat(),
        "pedal_type": "klon_sim",
        "pedal_version": KlonSim.VERSION,
        "params": asdict(params),
        "captures": {
            "train": {
                "dry_signal": "train_signal_v2_48k",
                "dry_manifest": str(TRAIN_MANIFEST.relative_to(PROJECT_ROOT)),
                "wet_wav": "wet_train_signal_v2_48k.wav",
                "sample_rate": sr,
                "n_samples": train_n,
                "duration_s": round(train_n / sr, 2),
            },
            "val": {
                "dry_signal": "train_signal_v2_48k",
                "dry_manifest": str(TRAIN_MANIFEST.relative_to(PROJECT_ROOT)),
                "wet_wav": "wet_val_signal_v2_48k.wav",
                "sample_rate": sr,
                "n_samples": val_n,
                "duration_s": round(val_n / sr, 2),
            },
        },
    }
    params_path = out_dir / "params.json"
    params_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved {params_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate KlonSim v2 wet captures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Knob-position interface (preferred)
    knob_group = parser.add_argument_group("Knob positions (o'clock, 6=min, 17=max)")
    knob_group.add_argument("--gain", type=float, default=11.0,
                            help="Gain knob in o'clock hours (e.g. 9, 11, 12).")
    knob_group.add_argument("--treble", type=float, default=12.0,
                            help="Treble knob in o'clock hours.")
    knob_group.add_argument("--volume", type=float, default=12.0,
                            help="Volume knob in o'clock hours.")

    # Direct-param overrides
    p_group = parser.add_argument_group("Direct parameter overrides (bypass knob mapping)")
    p_group.add_argument("--drive", type=float, default=None,
                         help="Drive multiplier override.")
    p_group.add_argument("--blend", type=float, default=None,
                         help="clean_blend override [0,1].")
    p_group.add_argument("--asymmetry", type=float, default=None,
                         help="clip_asymmetry override.")

    parser.add_argument("--name", type=str, default=None,
                        help="Output subdirectory name (default: gain<N>_t<N>_v<N>).")
    args = parser.parse_args()

    # Build params from knob positions
    knobs = KlonKnobs.at_oclock(
        gain=args.gain,
        treble=args.treble,
        volume=args.volume,
    )
    params = knobs.to_params()

    # Apply direct overrides
    if args.drive is not None:
        params.drive = args.drive
    if args.blend is not None:
        params.clean_blend = args.blend
    if args.asymmetry is not None:
        params.clip_asymmetry = args.asymmetry

    if args.name is None:
        g = int(round(args.gain))
        t = int(round(args.treble))
        v = int(round(args.volume))
        args.name = f"klon_sim_v2_gain{g}_t{t}_v{v}"

    out_dir = CAPTURES_BASE / args.name
    print(f"KlonSim v{KlonSim.VERSION} — {args.name}")
    print(f"  drive={params.drive:.3f}  blend={params.clean_blend:.3f}  "
          f"treble={params.treble_db:+.1f}dB  asymmetry={params.clip_asymmetry:.3f}")
    print(f"  output_level={params.output_level:.3f}  shamrock={params.shamrock}")
    print(f"  → {out_dir.relative_to(PROJECT_ROOT)}\n")

    make_captures(params, out_dir)
    print("\nDone. Update fit_klonsim_ode.py --wet / --dry to point at the new files.")


if __name__ == "__main__":
    main()
