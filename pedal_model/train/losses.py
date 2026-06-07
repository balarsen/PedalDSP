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
            # Accept (batch, 1, time) channels-first OR (batch, time, 1) channels-last
            if predicted.shape[1] == 1:
                predicted = predicted.squeeze(1)
                target = target.squeeze(1)
            elif predicted.shape[-1] == 1:
                predicted = predicted.squeeze(-1)
                target = target.squeeze(-1)

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


def _hp_preemphasis(x: torch.Tensor, coef: float = 0.85) -> torch.Tensor:
    """Apply a 1-pole high-pass pre-emphasis filter in-place (no gradient).

    y[n] = x[n] - coef * x[n-1]

    This is the Wright et al. (2019) pre-emphasis filter. With coef=0.85 the
    -3 dB point is roughly 2 kHz at 48 kHz sample rate, re-weighting gradient
    energy toward the treble frequencies that define overdrive character.

    Args:
        x: Audio tensor, shape (..., time). Applied along the last dimension.
        coef: Feedback coefficient. Higher = more high-pass emphasis.

    Returns:
        Pre-emphasised tensor, same shape as input.
    """
    # First-difference with leaky integration: y[n] = x[n] - coef * x[n-1]
    return x[..., 1:] - coef * x[..., :-1]


class PreEmphasisESRLoss(nn.Module):
    """ESR computed on high-pass pre-emphasised signals (Wright et al. 2019).

    Applying a 1-pole HP filter before ESR re-weights the loss toward
    mid/high frequencies. Without pre-emphasis the bulk of the audio power
    lives at low frequencies so the gradient signal is dominated by bass
    accuracy; with pre-emphasis the network is forced to fit the treble
    detail that defines the Klon's character.

    Reference: Wright et al., "Real-Time Guitar Amplifier Emulation with
    Deep Learning", Applied Sciences, 2020.
    """

    def __init__(self, coef: float = 0.85) -> None:
        """
        Args:
            coef: HP pre-emphasis coefficient. 0.85 ≈ 2 kHz at 48 kHz.
                  Range (0, 1): higher = more treble emphasis.
        """
        super().__init__()
        self.coef = coef
        self._esr = ESRLoss()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute pre-emphasised ESR.

        Args:
            predicted: Model output, any shape with time on last axis.
            target: Reference audio, same shape.

        Returns:
            Scalar ESR of the pre-emphasised signals.
        """
        pred_hp = _hp_preemphasis(predicted, self.coef)
        targ_hp = _hp_preemphasis(target, self.coef)
        return self._esr(pred_hp, targ_hp)


class CombinedLoss(nn.Module):
    """α · PreEmphasisESR + β · STFT_loss  (Wright et al. 2019 formulation).

    Default 75/25 split. The pre-emphasis ESR term pushes the network to fit
    treble frequencies where the Klon's character lives; the STFT term keeps
    the overall spectral shape honest. For harder fuzz devices increase β to
    0.5–0.7 so the STFT term tolerates phase-shifted square-wave approximations.

    Set ``preemphasis_coef=0.0`` to revert to plain ESR (coef=0 → identity).
    """

    def __init__(
        self,
        alpha: float = 0.75,
        beta: float = 0.25,
        preemphasis_coef: float = 0.85,
    ) -> None:
        """
        Args:
            alpha: Weight for the pre-emphasis ESR term. Default 0.75.
            beta: Weight for the multi-scale STFT term. Default 0.25.
            preemphasis_coef: HP coefficient for pre-emphasis. Default 0.85
                (~2 kHz at 48 kHz). Set to 0.0 to disable pre-emphasis.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if preemphasis_coef > 0.0:
            self.esr_loss: nn.Module = PreEmphasisESRLoss(coef=preemphasis_coef)
        else:
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
