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


class CombinedLoss(nn.Module):
    """α · L2_time + β · STFT_loss.

    α=0.1, β=0.9 by default: STFT loss dominates for better tonal accuracy,
    while the small L2 term anchors absolute amplitude.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.9) -> None:
        """
        Args:
            alpha: Weight for the L2 time-domain term.
            beta: Weight for the multi-scale STFT term.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.stft_loss = MultiScaleSTFTLoss()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predicted: Model output, shape (batch, time) or (batch, 1, time).
            target: Reference audio, same shape.

        Returns:
            Scalar loss tensor.
        """
        l2 = F.mse_loss(predicted, target)
        stft = self.stft_loss(predicted, target)
        return self.alpha * l2 + self.beta * stft
