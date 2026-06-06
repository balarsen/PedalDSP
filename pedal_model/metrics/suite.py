"""Runs all metrics and returns a unified results dict."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .frequency_domain import compute_fr_error, compute_multiscale_stft_loss
from .harmonic import (
    compute_hp_similarity,
    compute_thd,
    compute_thd_pattern_distance,
)
from .perceptual import compute_mcd
from .time_domain import (
    compute_dc_error,
    compute_esr,
    compute_mse,
    compute_null_depth,
    compute_rms_error,
)

if TYPE_CHECKING:
    from pedal_model.signals.manifest import Manifest

# Frequency at which harmonic metrics are evaluated.
_HARMONIC_F0 = 440.0


def compute_all_metrics(
    target: np.ndarray,
    predicted: np.ndarray,
    input_signal: np.ndarray,
    sr: int,
    harmonic_f0: float = _HARMONIC_F0,
) -> dict[str, float]:
    """Compute the full metrics suite for one model output.

    Args:
        target: Reference wet audio, shape (N,), float32, range [-1, 1].
        predicted: Model wet output, same shape.
        input_signal: Dry input used to produce both, same shape.
        sr: Sample rate in Hz.
        harmonic_f0: Fundamental frequency used for THD/HP metrics in Hz.

    Returns:
        Dict mapping metric name → float value. Keys:
        ESR, null_depth_dB, MSE, DC_err, RMS_err, STFT, FR_err_dB,
        THD_target, THD_pred, THD_err, THD_pattern_dist, HP_sim, MCD.
    """
    thd_target = compute_thd(target, harmonic_f0, sr)
    thd_pred = compute_thd(predicted, harmonic_f0, sr)

    return {
        "ESR": compute_esr(target, predicted),
        "null_depth_dB": compute_null_depth(target, predicted),
        "MSE": compute_mse(target, predicted),
        "DC_err": compute_dc_error(target, predicted),
        "RMS_err": compute_rms_error(target, predicted),
        "STFT": compute_multiscale_stft_loss(target, predicted),
        "FR_err_dB": compute_fr_error(target, predicted, input_signal, sr),
        "THD_target": thd_target,
        "THD_pred": thd_pred,
        "THD_err": abs(thd_target - thd_pred),
        "THD_pattern_dist": compute_thd_pattern_distance(target, predicted, harmonic_f0, sr),
        "HP_sim": compute_hp_similarity(target, predicted, harmonic_f0, sr),
        "MCD": compute_mcd(target, predicted, sr),
    }


def compute_per_section(
    manifest: "Manifest",
    dry: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    section_labels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute metrics per manifest section.

    Slices *dry*, *target*, and *predicted* by section sample range and runs
    the metric suite on each slice.  Harmonic metrics are only added for
    sections of type ``"stepped_sine_tone"`` whose ``params`` dict contains
    a ``"freq_hz"`` key.

    Args:
        manifest: Signal manifest describing section boundaries.
        dry: Full dry audio, shape (total_samples,), float32.
        target: Full target (wet) audio, same shape.
        predicted: Full predicted audio, same shape.
        sr: Sample rate in Hz.
        section_labels: If given, compute only these section labels.
            Defaults to all sections.

    Returns:
        ``{section_label: {metric_name: float}}``.  Every section contains:
        ``ESR``, ``null_depth_dB``, ``STFT``, ``FR_err_dB``.
        Sine-tone sections additionally contain: ``THD_target``,
        ``THD_pred``, ``THD_pattern_dist``, ``HP_sim``.
    """
    sections = manifest.sections
    if section_labels is not None:
        label_set = set(section_labels)
        sections = [s for s in sections if s.label in label_set]

    results: dict[str, dict[str, float]] = {}
    for sec in sections:
        a, b = sec.start_sample, sec.end_sample
        dry_s = dry[a:b]
        tgt_s = target[a:b]
        pred_s = predicted[a:b]

        row: dict[str, float] = {
            "ESR": compute_esr(tgt_s, pred_s),
            "null_depth_dB": compute_null_depth(tgt_s, pred_s),
            "STFT": compute_multiscale_stft_loss(tgt_s, pred_s),
            "FR_err_dB": compute_fr_error(tgt_s, pred_s, dry_s, sr),
        }

        if sec.type == "stepped_sine_tone" and "freq_hz" in sec.params:
            f0 = float(sec.params["freq_hz"])
            row["THD_target"] = compute_thd(tgt_s, f0, sr)
            row["THD_pred"] = compute_thd(pred_s, f0, sr)
            row["THD_pattern_dist"] = compute_thd_pattern_distance(tgt_s, pred_s, f0, sr)
            row["HP_sim"] = compute_hp_similarity(tgt_s, pred_s, f0, sr)

        results[sec.label] = row

    return results
