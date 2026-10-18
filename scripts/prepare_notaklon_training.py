"""Prepare JHS Notaklon hardware captures for model-zoo training.

Converts the notaklon capture format (recordings/data/notaklon/captures/<preset>/params.json)
to the klon_sim harness format (recordings/data/notaklon/training/<preset>/params.json),
applying lag correction to the wet WAV in the process.

The training and validation signals use the same audio file (train = val).
Val ESR therefore reflects training convergence, not held-out generalisation —
but it is still useful for comparing model architectures on the same data.

Usage:
    # Prepare all standard captures:
    uv run python scripts/prepare_notaklon_training.py

    # Prepare a subset:
    uv run python scripts/prepare_notaklon_training.py --captures V12-T12-G11-False V12-T12-G12-False

    # Re-process already-prepared captures:
    uv run python scripts/prepare_notaklon_training.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CAPTURES_DIR = ROOT / "recordings/data/notaklon/captures"
TRAINING_DIR = ROOT / "recordings/data/notaklon/training"
DRY_NPY      = ROOT / "recordings/data/signals/train_signal_v2_48k.npy"

# Relative path from any training/<preset>/ directory to the 48 kHz manifest.
# Goes up 3 levels: <preset>/ → training/ → notaklon/ → recordings/data/, then into signals/.
MANIFEST_REL = "../../../signals/train_signal_v2_48k.json"

# Captures to prepare by default (exclude bypass — no pedal processing).
DEFAULT_CAPTURES: list[str] = [
    "V12-T12-G9-False",
    "V12-T12-G10-False",
    "V12-T8-G11-False",
    "V12-T12-G11-False",
    "V12-T12-G12-False",
    "V12-T12-G12-True",
    "V10-T10-G1-False",
]


def prepare_capture(capture_dir: Path, out_dir: Path, *, force: bool = False) -> None:
    """Lag-correct wet WAV and write klon_sim-format params.json."""
    params_path = capture_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"No params.json in {capture_dir}")

    src = json.loads(params_path.read_text())
    lag = int(src["wet_signal"]["lag_samples"])
    wet_src = ROOT / src["wet_signal"]["file"]

    out_dir.mkdir(parents=True, exist_ok=True)
    wet_out = out_dir / "wet_train.wav"

    if wet_out.exists() and not force:
        print(f"  {capture_dir.name}: already prepared — skip (use --force to redo)")
        return

    print(f"  {capture_dir.name}  lag={lag} smp → lag-correcting wet WAV …")
    wet, sr = sf.read(str(wet_src), dtype="float32", always_2d=False)
    if sr != 48_000:
        raise ValueError(f"Expected 48 kHz, got {sr} in {wet_src}")

    dry = np.load(str(DRY_NPY))
    n = min(len(dry), len(wet) - lag)
    wet_aligned = wet[lag: lag + n]

    peak = float(20 * np.log10(np.abs(wet_aligned).max() + 1e-12))
    rms  = float(20 * np.log10(np.sqrt(np.mean(wet_aligned**2)) + 1e-12))
    sf.write(str(wet_out), wet_aligned, sr, subtype="PCM_24")
    print(f"    wet_train.wav  {n/sr:.2f}s  peak={peak:.1f} dBFS  rms={rms:.1f} dBFS")

    training_params = {
        "schema_version": "1.0",
        "date_created": datetime.now(timezone.utc).isoformat(),
        "pedal_type": "jhs_notaklon",
        "source_capture": str(capture_dir.relative_to(ROOT)),
        "source_params": src,
        "captures": {
            "train": {
                "dry_manifest": MANIFEST_REL,
                "wet_wav": "wet_train.wav",
                "sample_rate": 48_000,
                "n_samples": n,
                "duration_s": round(n / 48_000, 3),
            },
            # Val = train (same file) — gives training-set ESR for convergence monitoring.
            "val": {
                "dry_manifest": MANIFEST_REL,
                "wet_wav": "wet_train.wav",
                "sample_rate": 48_000,
                "n_samples": n,
                "duration_s": round(n / 48_000, 3),
            },
        },
    }
    (out_dir / "params.json").write_text(json.dumps(training_params, indent=2))
    print(f"    params.json written → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Notaklon hardware captures for model-zoo training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--captures", nargs="*", default=None,
        help="Capture directory names to prepare. Default: all standard captures.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process captures even if output already exists.",
    )
    args = parser.parse_args()

    names = args.captures or DEFAULT_CAPTURES
    print(f"Preparing {len(names)} capture(s) into {TRAINING_DIR}\n")

    for name in names:
        cap_dir = CAPTURES_DIR / name
        if not cap_dir.exists():
            print(f"  SKIP {name}: directory not found")
            continue
        try:
            prepare_capture(cap_dir, TRAINING_DIR / name, force=args.force)
        except Exception as exc:
            print(f"  ERROR {name}: {exc}")

    print("\nTo run training (one capture at a time, or in parallel):")
    for name in names:
        label = name.replace("-", "_").replace("False", "F").replace("True", "T")
        print(
            f"  uv run python scripts/fit_models.py"
            f" --klon-sim-dir recordings/data/notaklon/training/{name}"
            f" --name notaklon_{label}"
            f" --device cuda --epochs 100"
        )


if __name__ == "__main__":
    main()
