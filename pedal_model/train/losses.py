"""Combined time-domain + multi-scale STFT loss for neural model training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFTLoss(nn.Module):
    """Log-magnitude L1 error across multiple STFT window sizes."""

    def __init__(self, window_sizes: list[int] | None = None) -> None:
        """
        Args:
            window_sizes: FFT window lengths. Defaults to [32, 128, 512, 2048].
        """
        super().__init__()
        self.window_sizes = window_sizes or [32, 128, 512, 2048]

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss.

        Args:
            predicted: Model output, shape (batch, time) or (batch, 1, time).
            target: Reference audio, same shape.

        Returns:
            Scalar loss tensor.
        """
        if predicted.dim() == 3:
            predicted = predicted.squeeze(1)
            target = target.squeeze(1)

        loss = torch.zeros((), device=predicted.device)
        for win in self.window_sizes:
            hop = win // 4
            window = torch.hann_window(win, device=predicted.device)
            s_pred = torch.stft(predicted, win, hop, win, window, return_complex=True)
            s_targ = torch.stft(target, win, hop, win, window, return_complex=True)
            log_pred = torch.log(s_pred.abs() + 1e-8)
            log_targ = torch.log(s_targ.abs() + 1e-8)
            loss = loss + F.l1_loss(log_pred, log_targ)
        return loss / len(self.window_sizes)


class ESRLoss(nn.Module):
    """Differentiable Error-to-Signal Ratio loss.

    Normalises the squared error by the signal power so the loss is
    invariant to the overall output level of the pedal — a 0 dBFS and a
    −12 dBFS version of the same distortion shape give the same ESR.
    """

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ESR.

        Args:
            predicted: Model output, any shape.
            target: Reference audio, same shape.

        Returns:
            Scalar ESR in [0, ∞).  0 = perfect.
        """
        error_power = torch.sum((predicted - target) ** 2)
        signal_power = torch.sum(target ** 2) + 1e-8
        return error_power / signal_power


class CombinedLoss(nn.Module):
    """α · ESR + β · STFT_loss.

    Default 70/30 split: ESR anchors waveform fidelity while the STFT term
    penalises spectral shape errors.  The 70% ESR weight suits soft-clipping
    pedals (Notaklon-style) where waveform alignment matters.  For harder fuzz
    increase β to 0.5–0.7 so the STFT term tolerates phase-shifted approximations
    of square waves.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3) -> None:
        """
        Args:
            alpha: Weight for the ESR term. Default 0.7.
            beta: Weight for the multi-scale STFT term. Default 0.3.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.esr_loss = ESRLoss()
        self.stft_loss = MultiScaleSTFTLoss()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predicted: Model output, shape (batch, time) or (batch, 1, time).
            target: Reference audio, same shape.

        Returns:
            Scalar loss tensor.
        """
        esr = self.esr_loss(predicted, target)
        stft = self.stft_loss(predicted, target)
        return self.alpha * esr + self.beta * stft
