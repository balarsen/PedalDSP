#!/usr/bin/env python3
"""run_eval.py — run the model zoo on a klon_sim capture directory.

Usage:
    python run_eval.py
    python run_eval.py --klon-sim-dir data/klon_sim/klon_sim_v1 --epochs 50
    python run_eval.py --classical-only --skip volterra
    python run_eval.py --neural-only --epochs 50 --device cuda

Epoch guidance (full 61.68M-sample signal at 96 kHz):
    --epochs 50   → ~750k gradient steps (15k chunks × 50), full signal coverage
    --epochs 100  → ~1.5M gradient steps, good convergence for most architectures
    --epochs 200  → ~3.0M gradient steps, extended run for diminishing returns check
    --circuit-dict-epochs 200  → 1.55 full signal passes at 480k window (default)
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
        default=50,
        metavar="N",
        # 50 epochs × 15,058 chunks = ~753k gradient steps on the full signal;
        # 6× more total steps than the old 1000 epochs × 122 fixed chunks,
        # with full signal diversity instead of the same 0.81% repeated.
        help="Training epochs for LSTM, GRU, TCN, DDSP, KlonDDSP.",
    )
    parser.add_argument(
        "--circuit-dict-epochs",
        type=int,
        default=200,
        metavar="N",
        help="Training epochs for CircuitDictionary (independent of --epochs). "
        "200 epochs = 1.55 full signal passes at 480k window.",
    )
    parser.add_argument(
        "--tcn-epochs",
        type=int,
        default=0,
        metavar="N",
        help="Override epoch count for TCN only (0 = use --epochs). "
        "TCN val_esr=0.0088 at 200ep and still descending; 500ep targets <0.005.",
    )
    parser.add_argument(
        "--circuit-dict-window",
        type=int,
        default=480_000,
        metavar="N",
        help="Window size (samples) per epoch for CircuitDictionary. "
        "480000 = 5 s at 96 kHz; one full pass every 129 epochs.",
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
        help="Random subsample size for MLP (0 = full signal). "
        "MLP is pointwise so random scatter is correct.",
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
    if args.output_dir is None:
        rel = klon_dir.relative_to(PROJECT_ROOT) if klon_dir.is_relative_to(PROJECT_ROOT) else klon_dir.name
        out_dir = PROJECT_ROOT / "results" / rel
    else:
        out_dir = args.output_dir.resolve()

    cfg = EvalConfig(
        klon_sim_dir=klon_dir,
        output_dir=out_dir,
        max_hammerstein_samples=args.max_hammerstein or 5_000_000,
        max_mlp_samples=args.mlp_train or 0,  # 0 → use full signal
        neural_epochs=args.epochs,
        tcn_epochs=args.tcn_epochs,
        circuit_dict_epochs=args.circuit_dict_epochs,
        circuit_dict_window_size=args.circuit_dict_window,
        device=args.device,
        run_classical=not args.neural_only,
        run_neural=not args.classical_only,
        classical_workers=args.workers,
        skip_models=args.skip or [],
    )

    print(f"klon_sim dir : {cfg.klon_sim_dir}")
    print(f"output dir   : {cfg.output_dir}")
    print(f"classical    : {cfg.run_classical}  neural: {cfg.run_neural}")
    ep_line = f"epochs       : neural={cfg.neural_epochs}  circuit_dict={cfg.circuit_dict_epochs}  device={cfg.device}"
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
