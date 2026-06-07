#!/usr/bin/env python3
"""Validate the WienerHammerstein streaming implementation.

Two tests are run:

1. Synthetic WH system — a known h1/f/h2 ground truth.
   Fit with streaming and verify ESR < 0.05 (model has the correct topology
   so a good fit is guaranteed if the streaming math is correct).
   Also compares streaming Stage-2 coefficients against non-streaming numpy
   lstsq on the same data (should agree to <1 % relative error).

2. KlonSim real signal — fits WH on the actual 61.68M-sample training signal
   and reports ESR on the val set.  ESR > 1 here is expected (Klon topology
   does not match the WH structure) but the fit must be finite.

Usage:
    python validate_pipeline.py
    python validate_pipeline.py --skip-klon   # synthetic test only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────── #

def _butter_lp(fc: float, sr: float, order: int = 4) -> np.ndarray:
    sos = butter(order, fc / (sr / 2), btype="low", output="sos")
    return sos


def _butter_hp(fc: float, sr: float, order: int = 4) -> np.ndarray:
    sos = butter(order, fc / (sr / 2), btype="high", output="sos")
    return sos


def _apply_sos(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sosfilt(sos, x).astype(np.float32)


def _apply_known_wh(
    dry: np.ndarray, h1_sos: np.ndarray, h2_sos: np.ndarray
) -> np.ndarray:
    """Apply known WH: LP filter → tanh soft-clip → HP filter."""
    v = _apply_sos(h1_sos, dry)
    w = np.tanh(2.0 * v).astype(np.float32)
    return _apply_sos(h2_sos, w)


def _esr(target: np.ndarray, pred: np.ndarray) -> float:
    err = float(np.sum((pred - target) ** 2))
    sig = float(np.sum(target ** 2)) + 1e-8
    return err / sig


# ── Test 1: Synthetic WH ──────────────────────────────────────────────────── #

def test_synthetic_wh(n_samples: int = 2_000_000, sr: float = 48_000.0) -> bool:
    """Fit WH streaming on a known synthetic system; compare against lstsq.

    Signal design mirrors the klon_sim training signal structure:
    - First 96k samples at low amplitude (-34 dBFS ≈ 0.02) for linear-regime
      Stage-1 identification (the block-selector picks this region).
    - Remaining samples at normal amplitude (0.3) for NL coverage.

    System: LP(800 Hz) → tanh(2x) → LP(4000 Hz).  Both filters are LP so the
    Stage-1 estimate H1*H2 ≈ H1 in H1's passband — the implicit assumption
    that makes WH identification tractable.
    """
    print("=" * 60)
    print("Test 1: Synthetic WH — streaming vs lstsq on known system")
    print("=" * 60)

    rng = np.random.default_rng(42)
    # Low-amplitude section (linear regime for Stage-1 h1 estimation)
    n_linear = 96_000
    dry_lin = (rng.standard_normal(n_linear) * 0.02).astype(np.float32)
    # Full-amplitude section (nonlinear coverage for Stage-2 polynomial fit)
    dry_full = (rng.standard_normal(n_samples - n_linear) * 0.30).astype(np.float32)
    dry = np.concatenate([dry_lin, dry_full])

    # LP+LP system: H2 ≈ 1 in H1's passband, so Stage-1 can identify H1 well.
    h1_sos = _butter_lp(800.0, sr, order=4)
    h2_sos = _butter_lp(4000.0, sr, order=4)

    wet = _apply_known_wh(dry, h1_sos, h2_sos)

    print(f"  dry:  {n_samples/1e6:.1f}M samples  "
          f"linear={n_linear//1000}k@0.02 + rest@0.3  peak={np.abs(dry).max():.3f}")
    print(f"  wet:  peak={np.abs(wet).max():.3f}")

    # — Fit streaming WH —
    from pedal_model.models.classical.wiener_hammerstein import WienerHammersteinModel

    model = WienerHammersteinModel(poly_order=5, n_taps=256)
    t0 = time.perf_counter()
    model.fit(dry, wet, int(sr))
    fit_s = time.perf_counter() - t0

    # Predict on the last 100k samples (out of training distribution window)
    test_slice = slice(-100_000, None)
    pred = model.predict(dry[test_slice])
    tgt = wet[test_slice]
    esr_streaming = _esr(tgt, pred[:len(tgt)])
    print(f"  streaming fit: {fit_s:.1f}s  ESR={esr_streaming:.6f}")
    if esr_streaming > 0.5:
        print("  note: ESR > 0.5 is expected — Stage-1 can only estimate H1*H2,")
        print("        not H1 alone; ESR tests topology fit, not streaming correctness.")

    # — Non-streaming reference: direct XTX/XTy on full signal using model's h1 —
    # This isolates whether the streaming accumulation is mathematically correct.
    from scipy.signal import fftconvolve

    h1 = model._h1.astype(np.float64)
    v_full = fftconvolve(dry.astype(np.float64), h1, mode="full")[:n_samples]
    powers = np.arange(1, model.poly_order + 1)
    _BLOCK = 65_536
    XTX_ref = np.zeros((len(powers), len(powers)), dtype=np.float64)
    XTy_ref = np.zeros(len(powers), dtype=np.float64)
    for bstart in range(0, n_samples, _BLOCK):
        bend = min(bstart + _BLOCK, n_samples)
        v_b = v_full[bstart:bend]
        w_b = wet[bstart:bend].astype(np.float64)
        X_b = np.column_stack([v_b ** p for p in powers])
        XTX_ref += X_b.T @ X_b
        XTy_ref += X_b.T @ w_b
    ridge = 1e-6 * float(np.trace(XTX_ref)) / len(powers)
    coeffs_ref = np.linalg.solve(XTX_ref + ridge * np.eye(len(powers)), XTy_ref)

    # Streaming coefficients: _poly_coeffs is in polyval order (highest power first)
    # Convert to same order as coeffs_ref (power 1, 2, ..., poly_order)
    coeffs_stream = model._poly_coeffs[:-1][::-1]  # drop constant, reverse to asc order
    rel_err = np.abs(coeffs_stream - coeffs_ref) / (np.abs(coeffs_ref) + 1e-8)
    max_rel_err = float(rel_err.max())
    print(f"  Stage-2 coeff max rel. error vs reference XTX: {max_rel_err:.6f}")

    # PASS conditions:
    # 1. Stage-2 streaming math matches reference XTX to < 1% (proves streaming is correct).
    # 2. Full model ESR < 0.05 (proves Stage-1 h1 estimation + Stage-2 + Stage-3 work end-to-end).
    streaming_math_ok = max_rel_err < 0.01
    esr_ok = esr_streaming < 0.05
    passed = streaming_math_ok and esr_ok
    status = "PASS" if passed else "FAIL"
    print(f"  → {status}  (streaming_math<1%: {streaming_math_ok},  ESR<0.05: {esr_ok})\n")
    return passed


# ── Test 2: KlonSim real signal ───────────────────────────────────────────── #

def test_klon_wh() -> bool:
    """Fit WH streaming on the real klon_sim training signal."""
    import json
    import soundfile as sf

    from pedal_model.models.classical.wiener_hammerstein import WienerHammersteinModel

    print("=" * 60)
    print("Test 2: WH streaming on klon_sim real capture")
    print("=" * 60)

    klon_dir = PROJECT_ROOT / "data" / "klon_sim" / "klon_sim_v1"
    params_path = klon_dir / "params.json"
    if not params_path.exists():
        print(f"  SKIP — {params_path} not found")
        return True

    params = json.loads(params_path.read_text())
    sr = params["captures"]["train"]["sample_rate"]
    dry_manifest_rel = params["captures"]["train"]["dry_manifest"]
    dry_train_wav = (klon_dir / dry_manifest_rel).resolve().with_suffix(".wav")
    wet_train_wav = klon_dir / params["captures"]["train"]["wet_wav"]
    dry_val_wav = (klon_dir / params["captures"]["val"]["dry_manifest"]).resolve().with_suffix(".wav")
    wet_val_wav = klon_dir / params["captures"]["val"]["wet_wav"]

    print(f"  loading train signal ({params['captures']['train']['duration_s']:.0f}s)…")
    dry_train, _ = sf.read(str(dry_train_wav), dtype="float32", always_2d=False)
    wet_train, _ = sf.read(str(wet_train_wav), dtype="float32", always_2d=False)
    dry_val, _ = sf.read(str(dry_val_wav), dtype="float32", always_2d=False)
    wet_val, _ = sf.read(str(wet_val_wav), dtype="float32", always_2d=False)

    print(f"  train: {len(dry_train)/1e6:.2f}M samples  val: {len(dry_val)/1e6:.2f}M samples")

    model = WienerHammersteinModel(poly_order=5, n_taps=512)
    t0 = time.perf_counter()
    model.fit(dry_train, wet_train, sr)
    fit_s = time.perf_counter() - t0

    pred_val = model.predict(dry_val)
    n = min(len(wet_val), len(pred_val))
    esr_val = _esr(wet_val[:n], pred_val[:n])

    finite_ok = np.isfinite(pred_val).all()
    print(f"  fit: {fit_s:.1f}s  val_ESR={esr_val:.4f}  finite_pred={finite_ok}")

    # ESR > 1 is expected for Klon (topology mismatch is fine).
    # We only fail if the prediction is non-finite or the fit crashed.
    passed = finite_ok and np.isfinite(esr_val)
    status = "PASS" if passed else "FAIL"
    note = "  (ESR > 1 expected — Klon topology ≠ WH)" if esr_val > 1.0 else ""
    print(f"  → {status}{note}\n")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────── #

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate WH streaming pipeline.")
    parser.add_argument("--skip-klon", action="store_true",
                        help="Skip Test 2 (klon_sim real signal fit).")
    args = parser.parse_args(argv)

    results: list[bool] = []
    results.append(test_synthetic_wh())
    if not args.skip_klon:
        results.append(test_klon_wh())

    print("=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
