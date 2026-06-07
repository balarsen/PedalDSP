#!/usr/bin/env python3
"""run_eval.py — run the model zoo on a klon_sim capture directory.

Usage:
    python run_eval.py
    python run_eval.py --klon-sim-dir data/klon_sim/klon_sim_v1 --epochs 50
    python run_eval.py --classical-only --skip volterra
    python run_eval.py --neural-only --epochs 100 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run model zoo on a klon_sim capture.",
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
        "Default: results/<klon-sim-dir relative path>.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        metavar="N",
        help="Training epochs for each neural model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device for neural training.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=500_000,
        metavar="N",
        help="Max samples fed to neural training (0 = unlimited).",
    )
    parser.add_argument(
        "--max-classical",
        type=int,
        default=200_000,
        metavar="N",
        help="Max training samples for Hammerstein/WH models.",
    )
    parser.add_argument(
        "--max-volterra",
        type=int,
        default=100_000,
        metavar="N",
        help="Max training samples for Volterra (large matrices).",
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
    if args.output_dir is None:
        rel = klon_dir.relative_to(PROJECT_ROOT) if klon_dir.is_relative_to(PROJECT_ROOT) else klon_dir.name
        out_dir = PROJECT_ROOT / "results" / rel
    else:
        out_dir = args.output_dir.resolve()

    cfg = EvalConfig(
        klon_sim_dir=klon_dir,
        output_dir=out_dir,
        max_classical_samples=args.max_classical or None,
        max_volterra_samples=args.max_volterra,
        max_train_samples=args.max_train or None,
        neural_epochs=args.epochs,
        device=args.device,
        run_classical=not args.neural_only,
        run_neural=not args.classical_only,
        classical_workers=args.workers,
        skip_models=args.skip or [],
    )

    print(f"klon_sim dir : {cfg.klon_sim_dir}")
    print(f"output dir   : {cfg.output_dir}")
    print(f"classical    : {cfg.run_classical}  neural: {cfg.run_neural}")
    print(f"epochs       : {cfg.neural_epochs}  device: {cfg.device}")
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
