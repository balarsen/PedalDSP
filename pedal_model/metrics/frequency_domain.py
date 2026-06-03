"""Frequency-domain error metrics."""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft


def compute_multiscale_stft_loss(
    target: np.ndarray,
    predicted: np.ndarray,
    window_sizes: list[int] | None = None,
) -> float:
    """Multi-scale log-magnitude STFT loss.

    Args:
        target: Reference audio, shape (N,), float32.
        predicted: Model output, same shape.
        window_sizes: FFT window sizes. Defaults to [32, 128, 512, 2048].

    Returns:
        Average log-magnitude L1 error across all scales.
    """
    if window_sizes is None:
        window_sizes = [32, 128, 512, 2048]

    total = 0.0
    for win in window_sizes:
        hop = win // 4
        n_frames = max(1, (len(target) - win) // hop + 1)

        s_target = np.zeros((win // 2 + 1, n_frames))
        s_pred = np.zeros_like(s_target)
        window = np.hanning(win)

        for i in range(n_frames):
            start = i * hop
            frame_t = target[start : start + win]
            frame_p = predicted[start : start + win]
            if len(frame_t) < win:
                frame_t = np.pad(frame_t, (0, win - len(frame_t)))
                frame_p = np.pad(frame_p, (0, win - len(frame_p)))
            s_target[:, i] = np.abs(rfft(frame_t * window))
            s_pred[:, i] = np.abs(rfft(frame_p * window))

        log_target = np.log(s_target + 1e-8)
        log_pred = np.log(s_pred + 1e-8)
        total += float(np.mean(np.abs(log_target - log_pred)))

    return total / len(window_sizes)


def compute_fr_error(
    target: np.ndarray,
    predicted: np.ndarray,
    input_signal: np.ndarray,
    sr: int,
) -> float:
    """Mean frequency-response error in dB between target and predicted transfer functions.

    Args:
        target: Reference wet audio, shape (N,).
        predicted: Model wet output, same shape.
        input_signal: The dry input used to produce both, same shape.
        sr: Sample rate in Hz (unused in current implementation, reserved for future windowing).

    Returns:
        Mean absolute error in dB across positive frequencies.
    """
    n = len(input_signal)
    X = rfft(input_signal, n=n)
    denom = np.abs(X) + 1e-8

    H_target = np.abs(rfft(target, n=n)) / denom
    H_pred = np.abs(rfft(predicted, n=n)) / denom

    db_target = 20.0 * np.log10(H_target + 1e-8)
    db_pred = 20.0 * np.log10(H_pred + 1e-8)

    return float(np.mean(np.abs(db_target - db_pred)))
