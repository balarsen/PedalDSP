#!/usr/bin/env python3
"""Fit KlonSim parameters using the PedalODE three-stage approach.

Usage:
    python scripts/fit_klonsim_ode.py [options]

Options:
    --stage {1,2,3,all}   Run only the specified stage (default: all).
    --random-inits N       Also run Stage 3 from N random initialisations and
                           compare against the staged init (default: 0).
    --dry PATH             Override dry signal WAV path.
    --wet PATH             Override wet (KlonSim-processed) WAV path.
    --manifest PATH        Override manifest JSON path.
    --results-dir PATH     Output directory (default: results/pedalode/klonsim_v1).
    --device {cpu,cuda}    Torch device (default: cuda if available else cpu).
    --sr N                 Override sample rate (default: read from WAV file).
    --steps1 N             Stage 1 optimisation steps (default: 2000).
    --steps2 N             Stage 2 optimisation steps (default: 3000).
    --steps3 N             Stage 3 optimisation steps (default: 5000).
    --window N             Stage 3 window size in samples (default: 4096).
    --seed N               Global random seed (default: 42).
    --quiet                Suppress per-step logging.
    --verbose              Extra diagnostic info (tabbed in) at each stage.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["0", "1", "2", "3", "all"], default="all")
    p.add_argument("--random-inits", type=int, default=0, metavar="N")
    p.add_argument("--dry", type=Path, default=Path("data/signals/val_signal_v1.wav"))
    p.add_argument("--wet", type=Path, default=Path("data/klon_sim/klon_sim_v1/wet_val_signal_v1.wav"))
    p.add_argument("--manifest", type=Path, default=Path("data/signals/val_signal_v1.json"))
    p.add_argument("--results-dir", type=Path, default=Path("results/pedalode/klonsim_v1"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--sr", type=int, default=None)
    p.add_argument("--steps1", type=int, default=2000)
    p.add_argument("--steps2", type=int, default=3000)
    p.add_argument("--steps3", type=int, default=5000)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stall-patience", type=int, default=500, metavar="N",
                   help="Early-stop a stage after N steps with no improvement (default: 500).")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="Print extra diagnostic info (tabbed in) at each stage.")
    return p.parse_args()


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file, return (audio_float32, sr)."""
    audio, sr = sf.read(str(path), dtype="float32")
    return audio, int(sr)


def _try_load_step0_params(results_dir: Path) -> dict:
    """Load step0_params.json written by Stage 0, or return empty dict if absent."""
    p = results_dir / "staged" / "step0_params.json"
    if p.is_file():
        with open(p) as f:
            data = json.load(f)
        print(f"[fit_klonsim_ode] Loaded Step 0 estimates from {p}")
        return data
    return {}


def _model_from_step0(s0: dict) -> "WHODEModel":
    """Build WHODEModel initialised from Step 0 estimates (falls back to GT-approximate defaults)."""
    from pedal_model.ode.wh_model import WHODEModel

    fc1  = float(s0.get("bode_fc1_hz")      or 80.0)
    fc2  = float(s0.get("bode_fc2_hz")      or 350.0)
    drv  = float(s0.get("gc_drive")         or 4.0)
    lvl  = float(s0.get("gc_output_level")  or 0.7)

    print(
        f"[fit_klonsim_ode] WHODEModel init: fc1={fc1:.1f} Hz  fc2={fc2:.1f} Hz  "
        f"drive={drv:.3f}  output_level={lvl:.3f}"
    )
    return WHODEModel(
        fc1_init=fc1,
        fc2_init=fc2,
        shelf_freq_init=3000.0,
        shelf_gain_db_init=2.0,
        drive_init=drv,
        output_level_init=lvl,
        use_odd_symmetric_nl=True,
    )


def _run_one(
    model: "WHODEModel",
    dry: np.ndarray,
    wet: np.ndarray,
    sections: list[dict],
    sr: int,
    args: argparse.Namespace,
    suffix: str = "",
) -> dict:
    from pedal_model.ode.fitting import KlonSimFitter

    results_dir = args.results_dir
    if suffix:
        results_dir = results_dir / suffix

    fitter = KlonSimFitter(
        model=model,
        dry=dry,
        wet=wet,
        manifest_sections=sections,
        sr=sr,
        results_dir=results_dir,
        device=args.device,
        verbose=not args.quiet,
        debug=args.verbose,
    )

    sp = args.stall_patience
    stage = args.stage
    if stage == "all":
        return fitter.run(
            stage1_steps=args.steps1,
            stage2_steps=args.steps2,
            stage3_steps=args.steps3,
            stage3_window=args.window,
            stall_patience=sp,
        )
    elif stage == "1":
        fitter._run_stage1(n_steps=args.steps1, stall_patience=sp)
        return fitter._build_report()
    elif stage == "2":
        fitter._run_stage2(n_steps=args.steps2, stall_patience=sp)
        return fitter._build_report()
    elif stage == "3":
        fitter._run_stage3(n_steps=args.steps3, window=args.window, stall_patience=sp)
        return fitter._build_report()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    args = _parse_args()

    # Resolve device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fit_klonsim_ode] Device: {args.device}")

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load audio
    print(f"[fit_klonsim_ode] Loading dry:  {args.dry}")
    dry, sr_dry = _load_wav(args.dry)
    print(f"[fit_klonsim_ode] Loading wet:  {args.wet}")
    wet, sr_wet = _load_wav(args.wet)

    if sr_dry != sr_wet:
        print(f"WARNING: dry sr={sr_dry} != wet sr={sr_wet}", file=sys.stderr)

    sr = args.sr if args.sr is not None else sr_dry
    if len(dry) != len(wet):
        n = min(len(dry), len(wet))
        print(f"WARNING: length mismatch, trimming both to {n} samples", file=sys.stderr)
        dry, wet = dry[:n], wet[:n]

    # Load manifest
    print(f"[fit_klonsim_ode] Manifest:     {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)
    sections = manifest["sections"]
    print(f"[fit_klonsim_ode] {len(sections)} sections, {len(dry)/sr:.1f} s at {sr} Hz")
    if args.verbose:
        for sec in sections:
            print(f"    {sec.get('label','?'):40s}  {sec.get('type','?'):22s}  "
                  f"{sec.get('amplitude_dbfs', 0):+6.1f} dBFS  "
                  f"{(sec['end_sample']-sec['start_sample'])/sr:.2f} s")

    # Stage 0: pre-fitting analysis (no model, no ODE)
    s0_params: dict = {}
    if args.stage in ("0", "all"):
        from pedal_model.ode.step0 import run_step0
        print("\n" + "=" * 72)
        print("  STAGE 0 — Pre-fitting analysis (SciPy only)")
        print("=" * 72)
        s0_params = run_step0(
            dry=dry,
            wet=wet,
            sections=sections,
            sr=sr,
            results_dir=args.results_dir / "staged",
            verbose=not args.quiet,
        )
        if args.stage == "0":
            return
    elif args.stage in ("1", "2", "3"):
        # Running a single stage without Stage 0 — try to load prior Step 0 estimates
        s0_params = _try_load_step0_params(args.results_dir)

    # Primary run: staged initialisation from Step 0 estimates
    print("\n" + "=" * 72)
    print("  PRIMARY RUN — three-stage staged initialisation")
    print("=" * 72)
    model = _model_from_step0(s0_params)
    result_staged = _run_one(model, dry, wet, sections, sr, args, suffix="staged")

    # Optional random initialisations for Stage 3 comparison
    if args.random_inits > 0 and (args.stage in ("3", "all")):
        from pedal_model.ode.wh_model import WHODEModel
        print(f"\n[fit_klonsim_ode] Running {args.random_inits} random Stage 3 initialisations…")
        random_results = []
        for i in range(args.random_inits):
            torch.manual_seed(args.seed + 1 + i)
            np.random.seed(args.seed + 1 + i)
            # Random initialisation: fc1 ∈ [40, 200], fc2 ∈ [200, 1000], drive ∈ [1, 8]
            rng = np.random.default_rng(args.seed + 1 + i)
            m_rand = WHODEModel(
                fc1_init=float(rng.uniform(40.0, 200.0)),
                fc2_init=float(rng.uniform(200.0, 1000.0)),
                shelf_freq_init=float(rng.uniform(1000.0, 5000.0)),
                shelf_gain_db_init=float(rng.uniform(-3.0, 6.0)),
                drive_init=float(rng.uniform(1.0, 8.0)),
                output_level_init=float(rng.uniform(0.2, 1.0)),
                use_odd_symmetric_nl=True,
            )
            # Only run Stage 3 from this random init (skip Stage 1+2)
            from pedal_model.ode.fitting import KlonSimFitter
            fitter_r = KlonSimFitter(
                model=m_rand,
                dry=dry,
                wet=wet,
                manifest_sections=sections,
                sr=sr,
                results_dir=args.results_dir / f"random_init_{i}",
                device=args.device,
                verbose=not args.quiet,
            )
            fitter_r._run_stage3(n_steps=args.steps3, window=args.window, stall_patience=args.stall_patience)
            r = fitter_r._build_report()
            random_results.append({"init": i, "esr": r["esr"], "params": r["final_params"]})
            print(f"  Random init {i}: ESR={r['esr']:.5f}")

        print("\n  ESR comparison:")
        print(f"    Staged init: {result_staged['esr']:.5f}")
        for rr in random_results:
            print(f"    Random init {rr['init']}: {rr['esr']:.5f}")

    # Final summary
    esr = result_staged["esr"]
    nl_r2 = result_staged["nl_r2"]
    print("\n" + "=" * 72)
    print(f"  COMPLETION REPORT")
    print("=" * 72)
    print(f"  ESR (full val signal): {esr:.5f}  {'PASS ✓' if esr < 0.01 else 'FAIL ✗'}")
    print(f"  NL R² vs tanh:         {nl_r2:.6f}  {'PASS ✓' if nl_r2 > 0.999 else 'FAIL ✗'}")
    gen = result_staged.get("generalization_esr", {})
    if gen:
        print("  Generalization ESR (held-out sections):")
        for label, g_esr in gen.items():
            print(f"    {label}: {g_esr:.5f}")
    print(f"  Results saved to: {args.results_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
