"""Time-domain error metrics."""
import numpy as np


def compute_esr(target: np.ndarray, predicted: np.ndarray) -> float:
    """Error-to-Signal Ratio: lower is better, 0 = perfect.

    Args:
        target: Reference audio, shape (N,), float32.
        predicted: Model output, same shape as target.

    Returns:
        ESR in [0, ∞). Values > 1 mean the model is worse than silence.
    """
    return float(np.sum((target - predicted) ** 2) / (np.sum(target ** 2) + 1e-8))


def compute_mse(target: np.ndarray, predicted: np.ndarray) -> float:
    """Mean squared error.

    Args:
        target: Reference audio, shape (N,).
        predicted: Model output, same shape.

    Returns:
        MSE as a float.
    """
    return float(np.mean((target - predicted) ** 2))


def compute_dc_error(target: np.ndarray, predicted: np.ndarray) -> float:
    """Absolute DC offset difference between target and predicted.

    Args:
        target: Reference audio, shape (N,).
        predicted: Model output, same shape.

    Returns:
        |mean(target) - mean(predicted)|. Should be < 1e-4 for good models.
    """
    return float(abs(np.mean(target) - np.mean(predicted)))


def compute_rms_error(target: np.ndarray, predicted: np.ndarray) -> float:
    """Root-mean-square error.

    Args:
        target: Reference audio, shape (N,).
        predicted: Model output, same shape.

    Returns:
        RMS error as a float.
    """
    return float(np.sqrt(np.mean((target - predicted) ** 2)))


def compute_null_depth(target: np.ndarray, predicted: np.ndarray) -> float:
    """Null-test depth in dB: how much quieter the error is than the target.

    Higher is better.  > 20 dB means the error is inaudible in most contexts;
    < 10 dB is clearly audible.  Returns ``inf`` for identical signals.
    Returns ``0.0`` when the target is silence (undefined ratio).

    Args:
        target: Reference audio, shape (N,), float32, range [-1, 1].
        predicted: Model output, same shape.

    Returns:
        Null depth in dB ∈ (−∞, +∞].
    """
    error_power = float(np.mean((target - predicted) ** 2))
    if error_power == 0.0:
        return float("inf")
    signal_power = float(np.mean(target ** 2))
    if signal_power < 1e-12:
        return 0.0
    return float(-20.0 * np.log10(np.sqrt(error_power / signal_power)))
