"""Run all fitted models against all captures and build a results table."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .run_eval import evaluate_model


def compare_models(
    models: list,
    capture_paths: list[Path | str],
    harmonic_f0: float = 440.0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate every model on every capture.

    Args:
        models: List of fitted model objects with .name and .predict() attributes.
        capture_paths: List of stereo WAV paths to evaluate against.
        harmonic_f0: Fundamental frequency for harmonic metrics in Hz.

    Returns:
        Nested dict: results[capture_name][model_name] → metrics dict.
    """
    results: dict[str, dict[str, dict[str, float]]] = {}
    for path in capture_paths:
        cap_name = Path(path).stem
        results[cap_name] = {}
        print(f"\n--- Capture: {cap_name} ---")
        for model in models:
            print(f"  Running {model.name}…", end=" ", flush=True)
            try:
                metrics = evaluate_model(model, path, harmonic_f0)
                results[cap_name][model.name] = metrics
                print(f"ESR={metrics['ESR']:.4f}")
            except Exception as exc:
                print(f"FAILED: {exc}")
                results[cap_name][model.name] = {}
    return results
