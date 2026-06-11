#!/usr/bin/env python3
"""retrain_neural.py — re-train neural models and save state dicts.

Use this when you want to extend neural training (more epochs, different
device) without re-running the slower classical model fits from fit_models.py.

Reads training data from the klon_sim capture referenced in results.pkl,
trains each neural model using the same harness infrastructure as fit_models.py,
and saves state dicts as {results_dir}/{model_key}_weights.pt and
{model_key}_best.pt (best checkpoint, with epoch/val_esr metadata).

render_to_daw.py will prefer model_state from results.pkl over .pt files;
these .pt files serve as a standalone fallback if results.pkl is absent or stale.

Usage:
    uv run python scripts/retrain_neural.py \\
        --results results/data/klon_sim/klon_sim_v1

    # Shorter run for quick testing:
    uv run python scripts/retrain_neural.py \\
        --results results/data/klon_sim/klon_sim_v1 --epochs 10

    # Specific models only:
    uv run python scripts/retrain_neural.py \\
        --results results/data/klon_sim/klon_sim_v1 --models lstm tcn
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import soundfile as sf

from pedal_model.eval.harness import EvalConfig, _NEURAL_REGISTRY, _bad_block_set
from pedal_model.eval.results import ZooResults
from pedal_model.signals.manifest import Manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-train neural models and save state dicts for render_to_daw.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/data/klon_sim/klon_sim_v1"),
        metavar="DIR",
        help="Directory containing results.pkl from run_eval.py.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="Training epochs for each neural model.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="MODEL",
        help=f"Models to export. Default: all. Choices: {list(_NEURAL_REGISTRY)}",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-train and overwrite existing .pt files.",
    )
    args = parser.parse_args(argv)

    results_path = args.results.resolve()
    pkl_path = results_path / "results.pkl"
    if not pkl_path.exists():
        _LOG.error("results.pkl not found at %s", pkl_path)
        sys.exit(1)

    zoo = ZooResults.load(pkl_path)
    params_path = Path(zoo.config.get("params_path", ""))
    if not params_path.exists():
        _LOG.error("params.json not found at %s", params_path)
        sys.exit(1)

    with open(params_path) as fh:
        params = json.load(fh)

    klon_sim_dir = params_path.parent
    sr = params["captures"]["train"]["sample_rate"]

    # Resolve data paths
    dry_manifest_rel = params["captures"]["train"]["dry_manifest"]
    dry_train_wav = (klon_sim_dir / dry_manifest_rel).resolve().with_suffix(".wav")
    wet_train_wav = klon_sim_dir / params["captures"]["train"]["wet_wav"]
    dry_manifest_path = (klon_sim_dir / dry_manifest_rel).resolve()

    _LOG.info("Loading training data…")
    dry_train, _ = sf.read(str(dry_train_wav), dtype="float32", always_2d=False)
    wet_train, _ = sf.read(str(wet_train_wav), dtype="float32", always_2d=False)

    dry_val_rel = params["captures"]["val"]["dry_manifest"]
    dry_val_wav = (klon_sim_dir / dry_val_rel).resolve().with_suffix(".wav")
    wet_val_wav = klon_sim_dir / params["captures"]["val"]["wet_wav"]
    dry_val, _ = sf.read(str(dry_val_wav), dtype="float32", always_2d=False)
    wet_val, _ = sf.read(str(wet_val_wav), dtype="float32", always_2d=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Build EvalConfig matching the original run
    cfg = EvalConfig(
        klon_sim_dir=klon_sim_dir,
        output_dir=results_path,
        neural_epochs=args.epochs,
        device=str(device),
        run_classical=False,
        run_neural=True,
    )

    train_manifest = Manifest(dry_manifest_path)
    bad_blocks = _bad_block_set(train_manifest, cfg.neural_chunk_size, len(dry_train))

    requested = set(args.models or list(_NEURAL_REGISTRY))
    unknown = requested - set(_NEURAL_REGISTRY)
    if unknown:
        _LOG.error("Unknown model(s): %s", unknown)
        sys.exit(1)

    keys = [k for k in _NEURAL_REGISTRY if k in requested]
    _LOG.info("Will export: %s  (device=%s, epochs=%d)", keys, device, args.epochs)

    from pedal_model.eval.harness import (
        _train_recurrent,
        _train_tcn,
        _train_mlp,
    )

    # klon_ddsp and ddsp use the same TCN-style trainer
    _TCN_LIKE = {"tcn", "ddsp", "klon_ddsp", "circuit_dict"}
    _RECURRENT = {"lstm", "gru", "conditioned_lstm"}
    _MLP_LIKE = {"mlp"}

    for key in keys:
        out_path = results_path / f"{key}_weights.pt"
        if out_path.exists() and not args.overwrite:
            _LOG.info("[%s] checkpoint already exists at %s — skipping (use --overwrite)", key, out_path)
            continue

        module_path, class_name, kwargs = _NEURAL_REGISTRY[key]
        mod = importlib.import_module(module_path)
        model = getattr(mod, class_name)(**kwargs)
        model.to(device)

        _LOG.info("[%s] training %s epochs…", key, args.epochs)
        t0 = time.perf_counter()

        try:
            if key in _RECURRENT:
                train_loss, train_meta = _train_recurrent(
                    model, dry_train, wet_train, cfg, device, bad_blocks=bad_blocks
                )
            elif key in _TCN_LIKE:
                train_loss, train_meta = _train_tcn(
                    model, dry_train, wet_train, cfg, device,
                    dry_val=dry_val, wet_val=wet_val,
                )
            elif key in _MLP_LIKE:
                train_loss, train_meta = _train_mlp(model, dry_train, wet_train, cfg, device)
            else:
                _LOG.warning("[%s] no known trainer — using _train_tcn as fallback", key)
                train_loss, train_meta = _train_tcn(model, dry_train, wet_train, cfg, device)

            elapsed = time.perf_counter() - t0

            # Save final weights (model already holds best state after training)
            torch.save(model.state_dict(), str(out_path))
            _LOG.info("[%s] saved weights → %s  (%.0fs)", key, out_path, elapsed)

            # Save best checkpoint with metadata for render_to_daw.py
            best_path = results_path / f"{key}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": train_meta.get("epoch", args.epochs),
                    "val_esr": float(train_loss),
                    "grad_norm_max": train_meta.get("grad_norm_max", 0.0),
                },
                str(best_path),
            )
            _LOG.info(
                "[%s] saved best checkpoint → %s  (epoch=%s, val_esr/loss=%.4f, grad_norm_max=%.1f)",
                key,
                best_path,
                train_meta.get("epoch", args.epochs),
                train_loss,
                train_meta.get("grad_norm_max", 0.0),
            )

            if train_meta.get("grad_norm_max", 0.0) > 100.0:
                _LOG.warning(
                    "[%s] grad_norm_max=%.1f at best checkpoint. "
                    "Consider reducing learning rate or increasing gradient clip threshold. "
                    "Training may be unstable.",
                    key,
                    train_meta["grad_norm_max"],
                )

        except Exception as exc:
            _LOG.error("[%s] training failed: %s", key, exc)
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
