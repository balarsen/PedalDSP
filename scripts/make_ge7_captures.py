#!/usr/bin/env python3
"""Generate synthetic GE-7 wet captures for all preset JSON files.

Reads each preset from recordings/data/ge7/presets/, applies Ge7Sim to the shared
train/val dry signals, and writes wet WAVs + params.json to
recordings/data/ge7/captures/<preset_name>/.

Use this to test the full pipeline before any hardware is recorded.

Usage:
    uv run python scripts/make_ge7_captures.py
    uv run python scripts/make_ge7_captures.py --preset preset_smile
    uv run python scripts/make_ge7_captures.py --presets-dir recordings/data/ge7/presets
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from pedal_model.synthetic.ge7 import Ge7Sim, Ge7Sliders

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_DIR = PROJECT_ROOT / "recordings" / "data" / "signals"
PRESETS_DIR = PROJECT_ROOT / "recordings" / "data" / "ge7" / "presets"
CAPTURES_DIR = PROJECT_ROOT / "recordings" / "data" / "ge7" / "captures"

_SIGNAL_NAMES = {
    "train": "train_signal_v1",
    "val":   "val_signal_v1",
}


def _process_preset(preset_path: Path, captures_dir: Path) -> None:
    preset = json.loads(preset_path.read_text())
    name = preset["preset_name"]
    sliders = Ge7Sliders.from_dict(preset["sliders"])
    pedal = Ge7Sim.from_sliders(sliders)

    out_dir = captures_dir / f"preset_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    captures: dict[str, dict] = {}
    for split, signal_name in _SIGNAL_NAMES.items():
        dry_path = SIGNALS_DIR / f"{signal_name}.wav"
        if not dry_path.exists():
            print(f"  ERROR: {dry_path} not found — run generate_signal.py first.",
                  file=sys.stderr)
            sys.exit(1)

        dry, sr = sf.read(str(dry_path), dtype="float32", always_2d=False)
        wet = pedal.process(dry, sr)
        wet_name = f"wet_{signal_name}.wav"
        sf.write(str(out_dir / wet_name), wet, sr, subtype="FLOAT")

        captures[split] = {
            "dry_signal": signal_name,
            "dry_manifest": f"../../../signals/{signal_name}.json",
            "wet_wav": wet_name,
            "sample_rate": sr,
            "n_samples": len(wet),
            "duration_s": round(len(wet) / sr, 3),
        }
        print(f"    {split}: {out_dir / wet_name}")

    params_doc = {
        "schema_version": "1.0",
        "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "synthetic",
        **pedal.to_dict(),
        "preset_description": preset.get("description", ""),
        "captures": captures,
    }
    (out_dir / "params.json").write_text(json.dumps(params_doc, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", metavar="NAME",
                   help="Single preset name, e.g. preset_smile. Default: all presets.")
    p.add_argument("--presets-dir", type=Path, default=PRESETS_DIR)
    p.add_argument("--captures-dir", type=Path, default=CAPTURES_DIR)
    args = p.parse_args()

    if args.preset:
        preset_files = [args.presets_dir / f"{args.preset}.json"]
        if not preset_files[0].exists():
            # Try with prefix already included
            preset_files = [args.presets_dir / f"{args.preset}"]
        if not preset_files[0].exists():
            print(f"ERROR: preset not found: {preset_files[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        preset_files = sorted(args.presets_dir.glob("preset_*.json"))
        if not preset_files:
            print(f"ERROR: no preset_*.json files found in {args.presets_dir}",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Processing {len(preset_files)} preset(s) → {args.captures_dir}\n")
    for pf in preset_files:
        preset = json.loads(pf.read_text())
        print(f"  [{preset['preset_name']}]  {preset.get('description', '')}")
        _process_preset(pf, args.captures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
