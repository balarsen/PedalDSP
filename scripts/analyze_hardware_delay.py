"""CLI: measure hardware round-trip delay between dry and wet captures.

Usage examples:

    # Single wet file:
    uv run scripts/analyze_hardware_delay.py \\
        --dry recordings/data/notaklon/dry_ref.wav \\
        --wet recordings/data/notaklon/captures/V12-T12-G9-False/wet.wav \\
        --out outputs/delay_G9

    # Multiple wet files (labelled):
    uv run scripts/analyze_hardware_delay.py \\
        --dry recordings/data/notaklon/dry_ref.wav \\
        --wet G9:recordings/data/notaklon/captures/V12-T12-G9-False/wet.wav \\
              G10:recordings/data/notaklon/captures/V12-T12-G10-False/wet.wav \\
              G11:recordings/data/notaklon/captures/V12-T12-G11-False/wet.wav \\
        --out outputs/notaklon_delay \\
        --xcorr-start 0.5 --xcorr-end 20.0 \\
        --zoom-t 254.0 --zoom-half-ms 30
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pedal_model.capture.hardware_delay import analyze_delay


def _parse_wet(tokens: list[str]) -> dict[str, Path] | Path:
    """Parse --wet tokens into labelled dict or single path."""
    if len(tokens) == 1 and ":" not in tokens[0]:
        return Path(tokens[0])
    result: dict[str, Path] = {}
    for tok in tokens:
        if ":" in tok:
            label, path = tok.split(":", 1)
            result[label] = Path(path)
        else:
            # bare path — use stem as label
            p = Path(tok)
            result[p.stem] = p
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure hardware round-trip delay between dry and wet captures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry", required=True, type=Path,
        help="Dry reference WAV (mono or first channel used).",
    )
    parser.add_argument(
        "--wet", required=True, nargs="+",
        help=(
            "Wet WAV(s). Plain path for a single file; "
            "LABEL:path ... for multiple labelled captures."
        ),
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for plots and results.json.",
    )
    parser.add_argument(
        "--sr", type=int, default=48000,
        help="Expected sample rate in Hz (default: 48000).",
    )
    parser.add_argument(
        "--xcorr-start", type=float, default=0.5,
        help="Start of xcorr analysis window in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--xcorr-end", type=float, default=20.0,
        help="End of xcorr analysis window in seconds (default: 20.0). "
             "Use a low-amplitude / linear-regime section.",
    )
    parser.add_argument(
        "--max-lag", type=float, default=0.5,
        help="Xcorr search radius in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--zoom-t", type=float, default=None,
        help="Centre time (seconds) for the waveform zoom plot. "
             "Omit to skip the zoom plot.",
    )
    parser.add_argument(
        "--zoom-half-ms", type=float, default=30.0,
        help="Half-width of the zoom window in ms (default: 30).",
    )
    parser.add_argument(
        "--zoom-title", type=str, default="Waveform zoom",
        help="Title for the waveform zoom plot.",
    )
    args = parser.parse_args()

    wet = _parse_wet(args.wet)

    print(f"Dry : {args.dry}")
    if isinstance(wet, dict):
        for label, p in wet.items():
            print(f"Wet : {label} → {p}")
    else:
        print(f"Wet : {wet}")
    print(f"Out : {args.out}")
    print(f"Xcorr window: {args.xcorr_start:.1f}–{args.xcorr_end:.1f} s "
          f"(max lag ±{args.max_lag:.2f} s)")
    print()

    lags = analyze_delay(
        dry=args.dry,
        wet=wet,
        sr=args.sr,
        out_dir=args.out,
        xcorr_start_s=args.xcorr_start,
        xcorr_end_s=args.xcorr_end,
        max_lag_s=args.max_lag,
        t_zoom_s=args.zoom_t,
        zoom_half_ms=args.zoom_half_ms,
        zoom_title=args.zoom_title,
    )

    print()
    print("Results:")
    for label, lag in lags.items():
        print(f"  {label:20s}  {lag:+5d} smp  ({lag / args.sr * 1000:+.2f} ms)")


if __name__ == "__main__":
    main()
