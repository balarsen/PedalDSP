#!/usr/bin/env python3
"""make_klon_sim.py — generate KlonSim synthetic dry/wet pairs.

Reads dry signals from recordings/data/signals/, applies KlonSim, writes wet WAVs and a
params.json to recordings/data/klon_sim/klon_sim_v1/ (or --output-dir).

Usage:
    python make_klon_sim.py                    # process both train and val
    python make_klon_sim.py --signal val       # val only
    python make_klon_sim.py --drive 6 --noise-seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from pedal_model.synthetic.pedal import KlonSim, KlonSimParams

PROJECT_ROOT = Path(__file__).resolve().parent
SIGNALS_DIR = PROJECT_ROOT / "recordings" / "data" / "signals"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "recordings" / "data" / "klon_sim" / "klon_sim_v1"

_SIGNAL_NAMES = {
    "train": "train_signal_v1",
    "val": "val_signal_v1",
}


def _process_one(
    split: str,
    pedal: KlonSim,
    output_dir: Path,
) -> dict:
    signal_name = _SIGNAL_NAMES[split]
    dry_path = SIGNALS_DIR / f"{signal_name}.wav"

    if not dry_path.exists():
        print(
            f"  ERROR: {dry_path} not found — run generate_signal.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    dry, sr = sf.read(str(dry_path), dtype="float32", always_2d=False)
    duration = len(dry) / sr
    print(f"  {split}: {signal_name}  ({duration:.1f} s, sr={sr} Hz)")

    wet = pedal.process(dry, sr)

    wet_name = f"wet_{signal_name}.wav"
    wet_path = output_dir / wet_name
    sf.write(str(wet_path), wet, sr, subtype="FLOAT")

    # Store dry manifest path relative to params.json location
    dry_manifest_rel = f"../../signals/{signal_name}.json"

    return {
        "dry_signal": signal_name,
        "dry_manifest": dry_manifest_rel,
        "wet_wav": wet_name,
        "sample_rate": sr,
        "n_samples": len(wet),
        "duration_s": round(duration, 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate KlonSim synthetic wet captures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--signal",
        choices=["train", "val", "both"],
        default="both",
        help="Which signal(s) to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
    )
    parser.add_argument("--drive", type=float, default=4.0, metavar="X")
    parser.add_argument("--input-hpf-hz", type=float, default=80.0, metavar="HZ")
    parser.add_argument("--clip-hpf-hz", type=float, default=350.0, metavar="HZ")
    parser.add_argument("--presence-hz", type=float, default=3000.0, metavar="HZ")
    parser.add_argument("--presence-db", type=float, default=2.0, metavar="DB")
    parser.add_argument("--output-level", type=float, default=0.7, metavar="X")
    parser.add_argument("--noise-floor-dbfs", type=float, default=-70.0, metavar="DB")
    parser.add_argument("--noise-seed", type=int, default=42, metavar="N")
    args = parser.parse_args(argv)

    params = KlonSimParams(
        drive=args.drive,
        input_hpf_hz=args.input_hpf_hz,
        clip_hpf_hz=args.clip_hpf_hz,
        presence_hz=args.presence_hz,
        presence_db=args.presence_db,
        output_level=args.output_level,
        noise_floor_dbfs=args.noise_floor_dbfs,
        noise_seed=args.noise_seed,
    )
    pedal = KlonSim(params)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"] if args.signal == "both" else [args.signal]

    print(f"output → {output_dir}")
    captures: dict[str, dict] = {}
    for split in splits:
        captures[split] = _process_one(split, pedal, output_dir)

    params_doc = {
        "schema_version": "1.0",
        "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **pedal.to_dict(),
        "captures": captures,
    }
    params_path = output_dir / "params.json"
    params_path.write_text(json.dumps(params_doc, indent=2))

    print("\ndone")
    for split, info in captures.items():
        wet_abs = output_dir / info["wet_wav"]
        print(f"  wet ({split}):  {wet_abs}")
    print(f"  params:       {params_path}")


if __name__ == "__main__":
    main()
