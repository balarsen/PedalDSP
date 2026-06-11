#!/usr/bin/env python3
"""fit_models.py — fit the full model zoo on a pedal capture directory.

This is the primary entry point for evaluating a captured pedal effect.
It fits classical models (FIR, IIR, Hammerstein, WienerHammerstein, Volterra,
CircuitDictionary) and trains neural models (LSTM, GRU, TCN, MLP, DDSP,
KlonDDSP) against a dry/wet capture pair, then saves results.pkl + results.csv.

Processing chain:
    1.  Capture dry/wet audio → data/captures/<pedal>/
    2.  Create params.json → data/captures/<pedal>/params.json
    3.  **Run this script** → results/<name>/results.pkl + results.csv
    4.  Inspect results → notebooks/notaklon/01_zoo_results.py
    5.  Render to DAW → scripts/render_to_daw.py
    6.  (Optional) Re-train neural with more epochs → scripts/retrain_neural.py

Usage:
    uv run python scripts/fit_models.py \\
        --klon-sim-dir data/klon_sim/klon_sim_v1 \\
        --name klon_v1

    # Classical-only quick run:
    uv run python scripts/fit_models.py --classical-only --skip volterra

    # Neural-only with more epochs:
    uv run python scripts/fit_models.py --neural-only --epochs 200 --device cuda

    # Override training WAVs with custom files:
    uv run python scripts/fit_models.py \\
        --klon-sim-dir data/klon_sim/klon_sim_v1 \\
        --dry-wav data/captures/Medley_mono.wav \\
        --wet-wav data/captures/Medley_wet.wav \\
        --name medley_v1

Epoch guidance (full 61.68 M-sample signal at 96 kHz):
    --epochs 50   → ~750k gradient steps, fast survey run
    --epochs 100  → ~1.5M gradient steps, good convergence
    --epochs 200  → ~3.0M gradient steps, extended run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit the model zoo on a pedal capture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--klon-sim-dir",
        type=Path,
        default=Path("data/klon_sim/klon_sim_v1"),
        metavar="DIR",
        help="Directory with params.json and WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Where to write results.pkl / results.csv. "
        "Default: results/<name or klon-sim-dir relative path>.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="NAME",
        help="Experiment name used as the output directory leaf. "
        "When set, output goes to results/<name>/. "
        "When --dry-wav is also set and --name is omitted, the dry WAV stem is used.",
    )
    parser.add_argument(
        "--dry-wav",
        type=Path,
        default=None,
        metavar="WAV",
        help="Override the training dry WAV from params.json with this file.",
    )
    parser.add_argument(
        "--wet-wav",
        type=Path,
        default=None,
        metavar="WAV",
        help="Override the training wet WAV from params.json with this file.",
    )
    parser.add_argument(
        "--dry-val-wav",
        type=Path,
        default=None,
        metavar="WAV",
        help="Override the validation dry WAV from params.json with this file.",
    )
    parser.add_argument(
        "--wet-val-wav",
        type=Path,
        default=None,
        metavar="WAV",
        help="Override the validation wet WAV from params.json with this file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="Training epochs for LSTM, GRU, TCN, DDSP, KlonDDSP.",
    )
    parser.add_argument(
        "--circuit-dict-epochs",
        type=int,
        default=200,
        metavar="N",
        help="Training epochs for CircuitDictionary. "
        "200 epochs = 1.55 full signal passes at window=480k.",
    )
    parser.add_argument(
        "--tcn-epochs",
        type=int,
        default=0,
        metavar="N",
        help="Override epoch count for TCN only (0 = use --epochs).",
    )
    parser.add_argument(
        "--circuit-dict-window",
        type=int,
        default=480_000,
        metavar="N",
        help="Window size (samples) per epoch for CircuitDictionary.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device for neural training.",
    )
    parser.add_argument(
        "--mlp-train",
        type=int,
        default=5_000_000,
        metavar="N",
        help="Random subsample size for MLP (0 = full signal).",
    )
    parser.add_argument(
        "--max-hammerstein",
        type=int,
        default=5_000_000,
        metavar="N",
        help="Subsample cap for Hammerstein polynomial lstsq stage.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help="Parallel workers for classical fitting. 0 = cpu_count.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="MODEL",
        help="Model keys to skip, e.g. --skip volterra tcn.",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Run classical models only.",
    )
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Run neural models only.",
    )
    args = parser.parse_args(argv)

    from pedal_model.eval.harness import EvalConfig, run_zoo

    klon_dir = args.klon_sim_dir.resolve()

    # Output directory priority: --output-dir > --name > --dry-wav stem > klon_sim_dir
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    elif args.name:
        out_dir = PROJECT_ROOT / "results" / args.name
    elif args.dry_wav is not None:
        out_dir = PROJECT_ROOT / "results" / args.dry_wav.resolve().stem
    else:
        rel = (
            klon_dir.relative_to(PROJECT_ROOT)
            if klon_dir.is_relative_to(PROJECT_ROOT)
            else klon_dir.name
        )
        out_dir = PROJECT_ROOT / "results" / rel

    cfg = EvalConfig(
        klon_sim_dir=klon_dir,
        output_dir=out_dir,
        max_hammerstein_samples=args.max_hammerstein or 5_000_000,
        max_mlp_samples=args.mlp_train or 0,
        neural_epochs=args.epochs,
        tcn_epochs=args.tcn_epochs,
        circuit_dict_epochs=args.circuit_dict_epochs,
        circuit_dict_window_size=args.circuit_dict_window,
        device=args.device,
        run_classical=not args.neural_only,
        run_neural=not args.classical_only,
        classical_workers=args.workers,
        skip_models=args.skip or [],
        dry_wav=args.dry_wav.resolve() if args.dry_wav else None,
        wet_wav=args.wet_wav.resolve() if args.wet_wav else None,
        dry_val_wav=args.dry_val_wav.resolve() if args.dry_val_wav else None,
        wet_val_wav=args.wet_val_wav.resolve() if args.wet_val_wav else None,
        experiment_name=args.name,
    )

    print(f"klon_sim dir : {cfg.klon_sim_dir}")
    print(f"output dir   : {cfg.output_dir}")
    print(f"classical    : {cfg.run_classical}  neural: {cfg.run_neural}")
    ep_line = (
        f"epochs       : neural={cfg.neural_epochs}  "
        f"circuit_dict={cfg.circuit_dict_epochs}  device={cfg.device}"
    )
    if cfg.tcn_epochs > 0:
        ep_line += f"  tcn={cfg.tcn_epochs}"
    print(ep_line)
    if cfg.skip_models:
        print(f"skipping     : {cfg.skip_models}")
    print()

    results = run_zoo(cfg)

    print("\nAggregated metrics (mean across sections):")
    df = results.to_dataframe()
    if not df.empty:
        print(df[["ESR", "null_depth_dB", "STFT", "fit_s"]].to_string())


if __name__ == "__main__":
    main()
