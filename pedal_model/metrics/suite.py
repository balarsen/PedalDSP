"""Runs all metrics and returns a unified results dict."""
from __future__ import annotations

import numpy as np

from .frequency_domain import compute_fr_error, compute_multiscale_stft_loss
from .harmonic import compute_hp_similarity, compute_thd
from .perceptual import compute_mcd
from .time_domain import compute_dc_error, compute_esr, compute_mse, compute_rms_error

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
        ESR, MSE, DC_err, RMS_err, STFT, FR_err_dB, THD_target,
        THD_pred, THD_err, HP_sim, MCD, LSD_dB.
    """
    thd_target = compute_thd(target, harmonic_f0, sr)
    thd_pred = compute_thd(predicted, harmonic_f0, sr)

    return {
        "ESR": compute_esr(target, predicted),
        "MSE": compute_mse(target, predicted),
        "DC_err": compute_dc_error(target, predicted),
        "RMS_err": compute_rms_error(target, predicted),
        "STFT": compute_multiscale_stft_loss(target, predicted),
        "FR_err_dB": compute_fr_error(target, predicted, input_signal, sr),
        "THD_target": thd_target,
        "THD_pred": thd_pred,
        "THD_err": abs(thd_target - thd_pred),
        "HP_sim": compute_hp_similarity(target, predicted, harmonic_f0, sr),
        "MCD": compute_mcd(target, predicted, sr),
    }
