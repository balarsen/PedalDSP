#!/usr/bin/env python3
"""stereo_to_mono.py — extract mono from a dual-mono stereo WAV.

Verifies L and R channels are identical (or near-identical) before extracting.
Writes {stem}_mono.wav alongside the input file.

Usage:
    uv run python scripts/stereo_to_mono.py path/to/file.wav
    uv run python scripts/stereo_to_mono.py path/to/file.wav --force   # skip corr check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf

CORR_THRESHOLD = 0.999  # below this we warn; below 0.99 we abort unless --force


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract mono from a dual-mono stereo WAV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, metavar="WAV", help="Input stereo WAV file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write output even if L/R channels differ significantly.",
    )
    args = parser.parse_args(argv)

    src = args.input.resolve()
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    info = sf.info(str(src))
    print(f"File     : {src.name}")
    print(f"SR       : {info.samplerate} Hz")
    print(f"Channels : {info.channels}")
    print(f"Duration : {info.duration:.1f}s  ({int(info.duration//60)}m {info.duration%60:.1f}s)")
    print(f"Format   : {info.format} / {info.subtype}")

    if info.channels == 1:
        print("Already mono — nothing to do.")
        sys.exit(0)

    if info.channels != 2:
        print(f"ERROR: expected 1 or 2 channels, got {info.channels}.", file=sys.stderr)
        sys.exit(1)

    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    L, R = audio[:, 0], audio[:, 1]

    peak_L = 20.0 * np.log10(np.max(np.abs(L)) + 1e-9)
    peak_R = 20.0 * np.log10(np.max(np.abs(R)) + 1e-9)
    rms_L  = 20.0 * np.log10(np.sqrt(np.mean(L**2)) + 1e-9)
    rms_R  = 20.0 * np.log10(np.sqrt(np.mean(R**2)) + 1e-9)
    corr   = float(np.corrcoef(L, R)[0, 1])

    print(f"Peak L/R : {peak_L:.1f} / {peak_R:.1f} dBFS")
    print(f"RMS  L/R : {rms_L:.1f} / {rms_R:.1f} dBFS")
    print(f"L/R corr : {corr:.6f}", end="  ")

    if corr >= CORR_THRESHOLD:
        print("✓ dual-mono confirmed")
    elif corr >= 0.99:
        print("⚠ channels are similar but not identical — using L channel")
        if not args.force:
            print("  Pass --force to write anyway.")
            sys.exit(1)
    else:
        print("✗ channels differ significantly — this is true stereo")
        if not args.force:
            print("  Pass --force to extract L channel anyway.")
            sys.exit(1)

    out = src.parent / (src.stem + "_mono.wav")
    sf.write(str(out), L, sr, subtype=info.subtype)
    print(f"Written  : {out}")


if __name__ == "__main__":
    main()
