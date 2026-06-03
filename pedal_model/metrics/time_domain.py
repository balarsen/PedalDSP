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
