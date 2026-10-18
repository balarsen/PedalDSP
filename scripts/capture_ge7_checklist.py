#!/usr/bin/env python3
"""Interactive recording guide for Boss GE-7 capture sessions.

Prints a clear slider diagram for one preset and walks you through
recording the train and val signals. Validates that the output files
actually exist before moving on.

Usage:
    uv run python scripts/capture_ge7_checklist.py --preset preset_smile
    uv run python scripts/capture_ge7_checklist.py --all      # walk all presets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = PROJECT_ROOT / "recordings" / "data" / "ge7" / "presets"
CAPTURES_DIR = PROJECT_ROOT / "recordings" / "data" / "ge7" / "captures"
SIGNALS_DIR = PROJECT_ROOT / "recordings" / "data" / "signals"

_BANDS = [
    ("100Hz",  "hz_100"),
    ("200Hz",  "hz_200"),
    ("400Hz",  "hz_400"),
    ("800Hz",  "hz_800"),
    ("1.6k",   "hz_1600"),
    ("3.2k",   "hz_3200"),
    ("6.4k",   "hz_6400"),
    ("LEVEL",  "level"),
]

_SLIDER_RANGE = 15  # ±15 dB


def _bar(db: float, width: int = 4) -> str:
    """One-line ASCII slider indicator. Centre = 0 dB, arrows show direction."""
    if abs(db) < 0.5:
        return "  0  "
    n = int(round(abs(db) / _SLIDER_RANGE * width))
    n = max(1, n)
    if db > 0:
        return f"+{db:>4.0f} " + "▲" * n
    else:
        return f"{db:>4.0f} " + "▽" * n


def _print_preset(preset: dict, index: int, total: int) -> None:
    name = preset["preset_name"]
    desc = preset.get("description", "")
    sliders = preset["sliders"]

    width = 58
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║  PRESET {index} / {total} : {name:<{width - 16}}║")
    print(f"║  {desc:<{width - 2}}║")
    print(f"╠{border}╣")

    # header
    labels = "  ".join(f"{lbl:>6}" for lbl, _ in _BANDS)
    print(f"║  {labels}  ║")

    # values row
    values = "  ".join(f"{sliders.get(key, 0.0):>+6.1f}" for _, key in _BANDS)
    print(f"║  {values}  ║")

    # bar row
    bars = "  ".join(f"{_bar(sliders.get(key, 0.0)):>6}" for _, key in _BANDS)
    print(f"║  {bars}  ║")
    print(f"╚{border}╝")


def _wait(prompt: str) -> None:
    try:
        input(f"\n  {prompt}")
    except (KeyboardInterrupt, EOFError):
        print("\n  Aborted.")
        sys.exit(0)


def _run_preset(preset: dict, index: int, total: int) -> None:
    name = preset["preset_name"]
    out_dir = CAPTURES_DIR / f"preset_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dry = SIGNALS_DIR / "train_signal_v1.wav"
    val_dry = SIGNALS_DIR / "val_signal_v1.wav"
    train_wet = out_dir / "wet_train_signal_v1.wav"
    val_wet = out_dir / "wet_val_signal_v1.wav"

    _print_preset(preset, index, total)

    if not train_dry.exists() or not val_dry.exists():
        print(f"\n  WARNING: dry signals not found in {SIGNALS_DIR}")
        print("  Run:  uv run python generate_signal.py --signal both")

    _wait("Set the GE-7 sliders as shown above, then press ENTER...")

    print(f"""
  ─── TRAIN signal ───────────────────────────────────────────
  DAW: arm return track, play into GE-7:
    {train_dry}

  Save the recorded wet output to:
    {train_wet}
""")
    _wait("Press ENTER when the TRAIN recording is saved...")
    if not train_wet.exists():
        print(f"  WARNING: file not found: {train_wet}")
        print("  Save the file there and re-run, or press ENTER to continue anyway.")
        _wait("Press ENTER to continue...")

    print(f"""
  ─── VAL signal ─────────────────────────────────────────────
  DAW: play into GE-7 (same slider positions!):
    {val_dry}

  Save the recorded wet output to:
    {val_wet}
""")
    _wait("Press ENTER when the VAL recording is saved...")
    if not val_wet.exists():
        print(f"  WARNING: file not found: {val_wet}")
        _wait("Press ENTER to continue anyway...")

    # Write a stub params.json so the capture dir is recognisable
    params_path = out_dir / "params.json"
    if not params_path.exists():
        doc = {
            "schema_version": "1.0",
            "source": "hardware",
            "pedal": "boss_ge7",
            "pedal_class": "linear_eq",
            "fit_hints": {
                "recommended_models": ["FIR", "IIR"],
                "skip_models": ["WienerHammerstein", "Volterra", "Hammerstein"],
                "fir_length_ms": 100,
                "iir_order": 14,
            },
            "preset_name": name,
            "preset_description": preset.get("description", ""),
            "sliders": preset["sliders"],
            "captures": {
                "train": {
                    "dry_signal": "train_signal_v1",
                    "dry_manifest": "../../../signals/train_signal_v1.json",
                    "wet_wav": "wet_train_signal_v1.wav",
                },
                "val": {
                    "dry_signal": "val_signal_v1",
                    "dry_manifest": "../../../signals/val_signal_v1.json",
                    "wet_wav": "wet_val_signal_v1.wav",
                },
            },
        }
        params_path.write_text(json.dumps(doc, indent=2))
        print(f"  Wrote: {params_path}")

    ok = train_wet.exists() and val_wet.exists()
    status = "✓ COMPLETE" if ok else "⚠ INCOMPLETE"
    print(f"\n  {status}: preset_{name}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", metavar="NAME",
                       help="Single preset, e.g. preset_smile")
    group.add_argument("--all", action="store_true",
                       help="Walk through all presets in order")
    p.add_argument("--presets-dir", type=Path, default=PRESETS_DIR)
    args = p.parse_args()

    if args.all:
        preset_files = sorted(args.presets_dir.glob("preset_*.json"))
        # Ensure flat is first (it's the baseline)
        flat = args.presets_dir / "preset_flat.json"
        if flat in preset_files:
            preset_files = [flat] + [f for f in preset_files if f != flat]
    else:
        name = args.preset if args.preset.startswith("preset_") else f"preset_{args.preset}"
        preset_files = [args.presets_dir / f"{name}.json"]
        if not preset_files[0].exists():
            print(f"ERROR: {preset_files[0]} not found", file=sys.stderr)
            sys.exit(1)

    total = len(preset_files)
    print(f"\nBoss GE-7 Capture Session — {total} preset(s)")
    print("─" * 60)
    print("BEFORE YOU START:")
    print("  • Pedal in BYPASS OFF (engaged)")
    print("  • Guitar → DI → interface IN → GE-7 IN")
    print("  • GE-7 OUT → interface IN (return) — record the wet signal only")
    print("  • Set DAW to 48 kHz, 24-bit")
    print("  • Verify levels are not clipping before each preset")
    print("─" * 60)

    for i, pf in enumerate(preset_files, start=1):
        preset = json.loads(pf.read_text())
        _run_preset(preset, i, total)

    print("\n" + "─" * 60)
    print(f"Session complete — {total} preset(s) captured.")
    print(f"Run quality check:  uv run python scripts/verify_ge7_captures.py")


if __name__ == "__main__":
    main()
