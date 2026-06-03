"""Evaluate a single model against a capture WAV and print the metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..capture.align import load_and_align
from ..metrics.suite import compute_all_metrics


def evaluate_model(
    model,
    capture_path: Path | str,
    harmonic_f0: float = 440.0,
) -> dict[str, float]:
    """Run a fitted/loaded model on a capture and compute all metrics.

    Args:
        model: Any object with a .predict(dry) -> wet method.
        capture_path: Path to the stereo capture WAV.
        harmonic_f0: Fundamental frequency for THD/HP metrics in Hz.

    Returns:
        Metrics dict (see metrics/suite.py for keys).
    """
    dry, wet_target, sr = load_and_align(capture_path)
    wet_pred = model.predict(dry)

    # Trim to equal length (Volterra pads with zeros)
    n = min(len(wet_target), len(wet_pred))
    return compute_all_metrics(wet_target[:n], wet_pred[:n], dry[:n], sr, harmonic_f0)


def print_metrics(metrics: dict[str, float], model_name: str = "") -> None:
    header = f"=== {model_name} ===" if model_name else "=== Results ==="
    print(header)
    for k, v in metrics.items():
        print(f"  {k:<14} {v:.6f}")
