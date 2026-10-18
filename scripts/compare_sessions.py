#!/usr/bin/env python3
"""Compare the same presets captured in two rig sessions (for example 48 kHz vs 96 kHz).

Prints the RMS difference of the measured responses below and above 10 kHz for every preset both
sessions captured, and saves an overlay plot per preset.

Usage:
    uv run python scripts/compare_sessions.py <session_dir_a> <session_dir_b> [--preset smile]
    [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pedal_model.rig.compare import agreement, common_presets, load, preset_response  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--preset", action="append", help="limit to these presets (repeatable)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "session_compare")
    args = p.parse_args()

    sa, sb = load(args.a), load(args.b)
    presets = args.preset or common_presets(sa, sb)
    if not presets:
        raise SystemExit("No preset was captured in both sessions.")
    out = args.out / f"{sa.root.name}__vs__{sb.root.name}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'preset':14s} {'<10k RMS dB':>12s} {'>10k RMS dB':>12s} {'offset dB':>10s}")
    for name in presets:
        ra, rb = preset_response(sa, name), preset_response(sb, name)
        if ra is None or rb is None:
            print(f"{name:14s}   (missing in one session)")
            continue
        ag = agreement(ra, rb)
        print(
            f"{name:14s} {ag['rms_below_10k_db']:12.2f} {ag['rms_above_10k_db']:12.2f} "
            f"{ag['offset_db']:10.2f}"
        )
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.semilogx(
            ra.freqs_hz, ra.measured_db, label=f"{ra.session} ({ra.sample_rate // 1000} kHz)"
        )
        ax.semilogx(
            rb.freqs_hz, rb.measured_db, label=f"{rb.session} ({rb.sample_rate // 1000} kHz)"
        )
        if ra.expected_db is not None:
            ax.semilogx(ra.freqs_hz, ra.expected_db, "--", color="0.4", label="model")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title(f"{name}: below 10 kHz RMS diff {ag['rms_below_10k_db']:.2f} dB")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=120)
        plt.close(fig)
    print(f"Plots: {out}")


if __name__ == "__main__":
    main()
