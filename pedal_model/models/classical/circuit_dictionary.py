"""Parallel differentiable circuit primitive dictionary with learned architecture weights."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import PedalModel

# Impulse response / FIR kernel length used for all IIR approximations.
# K=512 captures 99.9% of IIR energy at pole radii up to ~0.986.
_K: int = 512


def _convolve(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Causal linear convolution of a 1-D signal with a 1-D kernel.

    Prepends (K-1) zeros to enforce causality, then applies F.conv1d.

    Args:
        x: Input signal, shape (N,), float32.
        kernel: FIR kernel, shape (K,), float32.

    Returns:
        Output signal, shape (N,), float32.
    """
    K = kernel.shape[0]
    # Reshape for conv1d: (batch=1, channels=1, N) and (out_ch=1, in_ch=1, K)
    x_2d = x.unsqueeze(0).unsqueeze(0)          # (1, 1, N)
    k_2d = kernel.unsqueeze(0).unsqueeze(0)      # (1, 1, K)
    # Causal zero-padding: pad (K-1) zeros on the left
    x_padded = F.pad(x_2d, (K - 1, 0))
    out = F.conv1d(x_padded, k_2d)              # (1, 1, N)
    return out.squeeze(0).squeeze(0)             # (N,)


def _lp_kernel(a: torch.Tensor, K: int = _K) -> torch.Tensor:
    """Build a 1-pole lowpass impulse response kernel of length K.

    h[k] = (1 - a) * a^k, which is the impulse response of H(z) = (1-a)/(1-a*z^-1).

    Args:
        a: Pole radius, scalar tensor, range (0, 1). Differentiable.
        K: Number of taps.

    Returns:
        Kernel tensor, shape (K,), float32.
    """
    k = torch.arange(K, dtype=torch.float32, device=a.device)
    return (1.0 - a) * torch.pow(a, k)


def _hp_kernel(a: torch.Tensor, K: int = _K) -> torch.Tensor:
    """Build a 1-pole highpass impulse response kernel of length K.

    h_hp = delta - h_lp  =>  h_hp[0] = a, h_hp[k>=1] = -(1-a)*a^k

    Args:
        a: Pole radius (derived from LP cutoff), scalar tensor.
        K: Number of taps.

    Returns:
        Kernel tensor, shape (K,), float32.
    """
    lp = _lp_kernel(a, K)
    delta = torch.zeros(K, dtype=torch.float32, device=a.device)
    delta[0] = 1.0
    return delta - lp


def _biquad_kernel(
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    K: int = _K,
) -> torch.Tensor:
    """Compute the impulse response of a biquad filter via a K-step recurrence.

    This K-step loop is fast (K=512 << N) and is used ONLY to compute the
    impulse-response kernel, not to iterate over the full N-sample signal.
    Gradients flow through b0..a2 via the loop.

    Args:
        b0, b1, b2: Normalised feed-forward coefficients.
        a1, a2: Normalised feedback coefficients (sign convention: y[n] = ... - a1*y[n-1] - a2*y[n-2]).
        K: Kernel length (taps).

    Returns:
        Impulse-response kernel, shape (K,), float32.
    """
    h = []
    y_prev1 = torch.zeros((), dtype=torch.float32, device=b0.device)
    y_prev2 = torch.zeros((), dtype=torch.float32, device=b0.device)
    for n in range(K):
        x_n = 1.0 if n == 0 else 0.0
        x_p1 = 1.0 if n == 1 else 0.0
        x_p2 = 1.0 if n == 2 else 0.0
        y_n = b0 * x_n + b1 * x_p1 + b2 * x_p2 - a1 * y_prev1 - a2 * y_prev2
        h.append(y_n)
        y_prev2 = y_prev1
        y_prev1 = y_n
    return torch.stack(h)


# ---------------------------------------------------------------------------
# Circuit primitives
# ---------------------------------------------------------------------------


def _prim_params(prim: nn.Module) -> str:
    """Return a short human-readable string of a primitive's learned parameter values."""
    parts: list[str] = []
    if hasattr(prim, "log_fc"):
        sr = getattr(prim, "sr", 48_000)
        fc = math.exp(prim.log_fc.item())  # type: ignore[attr-defined]
        fc = max(100.0, min(fc, sr / 2.0 - 100.0))
        parts.append(f"fc={fc:.0f}Hz")
    if hasattr(prim, "log_Q"):
        Q = math.exp(prim.log_Q.item())  # type: ignore[attr-defined]
        Q = max(0.3, min(Q, 8.0))
        parts.append(f"Q={Q:.2f}")
    if hasattr(prim, "gain_dB"):
        parts.append(f"gain={prim.gain_dB.item():.1f}dB")  # type: ignore[attr-defined]
    if hasattr(prim, "log_drive_pos"):
        dp = math.exp(prim.log_drive_pos.item())  # type: ignore[attr-defined]
        dn = math.exp(prim.log_drive_neg.item())  # type: ignore[attr-defined]
        parts.append(f"drive+={dp:.2f} drive-={dn:.2f}")
    elif hasattr(prim, "log_drive"):
        parts.append(f"drive={math.exp(prim.log_drive.item()):.2f}")  # type: ignore[attr-defined]
    if hasattr(prim, "log_gain") and not hasattr(prim, "log_drive_pos"):
        parts.append(f"gain={math.exp(prim.log_gain.item()):.2f}x")  # type: ignore[attr-defined]
    return "  ".join(parts)


class LearnableLowpass(nn.Module):
    """1-pole IIR lowpass filter with a learned cutoff frequency."""

    def __init__(self, sr: int) -> None:
        """
        Args:
            sr: Sample rate in Hz.
        """
        super().__init__()
        self.sr = sr
        # log_fc parameterises fc = exp(log_fc) for unconstrained optimisation
        self.log_fc = nn.Parameter(torch.tensor(math.log(106.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lowpass filter to signal.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Filtered signal, shape (N,), float32.
        """
        # 100 Hz minimum: at fc=100, a^512 ≈ 0.001 (0.1% truncation error).
        # At fc=20 Hz (old minimum), a^512 ≈ 0.26 — catastrophic truncation that
        # causes loss spikes whenever the optimizer pushes fc toward very low values.
        fc = torch.exp(self.log_fc).clamp(100.0, self.sr / 2.0 - 100.0)
        a = torch.exp(torch.tensor(-2.0 * math.pi, dtype=torch.float32) * fc / self.sr)
        kernel = _lp_kernel(a)
        return _convolve(x, kernel)


class LearnableHighpass(nn.Module):
    """1-pole IIR highpass filter with a learned cutoff frequency."""

    def __init__(self, sr: int) -> None:
        """
        Args:
            sr: Sample rate in Hz.
        """
        super().__init__()
        self.sr = sr
        self.log_fc = nn.Parameter(torch.tensor(math.log(300.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply highpass filter to signal.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Filtered signal, shape (N,), float32.
        """
        fc = torch.exp(self.log_fc).clamp(100.0, self.sr / 2.0 - 100.0)
        a = torch.exp(torch.tensor(-2.0 * math.pi, dtype=torch.float32) * fc / self.sr)
        kernel = _hp_kernel(a)
        return _convolve(x, kernel)


class LearnableHighShelf(nn.Module):
    """High-shelf filter: out = x + (g - 1) * HP(x).

    Implements a shelf as a blend of dry signal and a high-passed copy,
    giving a boost/cut above fc. Simple and fully differentiable.
    """

    def __init__(self, sr: int) -> None:
        """
        Args:
            sr: Sample rate in Hz.
        """
        super().__init__()
        self.sr = sr
        self.log_fc = nn.Parameter(torch.tensor(math.log(1000.0)))
        # shelf_gain_dB: learned gain for the high-shelf boost/cut, clamped to [-12, +12] dB
        self.gain_dB = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply high-shelf filter to signal.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Filtered signal, shape (N,), float32.
        """
        fc = torch.exp(self.log_fc).clamp(100.0, self.sr / 2.0 - 100.0)
        a = torch.exp(torch.tensor(-2.0 * math.pi, dtype=torch.float32) * fc / self.sr)
        kernel = _hp_kernel(a)
        hp_x = _convolve(x, kernel)
        gain_dB_clamped = self.gain_dB.clamp(-12.0, 12.0)
        g = torch.pow(torch.tensor(10.0), gain_dB_clamped / 20.0)
        return x + (g - 1.0) * hp_x


class LearnableBandpass(nn.Module):
    """Biquad bandpass filter using Audio EQ Cookbook formulas."""

    def __init__(self, sr: int) -> None:
        """
        Args:
            sr: Sample rate in Hz.
        """
        super().__init__()
        self.sr = sr
        self.log_fc = nn.Parameter(torch.tensor(math.log(800.0)))
        self.log_Q = nn.Parameter(torch.tensor(math.log(1.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filter to signal.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Filtered signal, shape (N,), float32.
        """
        fc = torch.exp(self.log_fc).clamp(100.0, self.sr / 2.0 - 100.0)
        Q = torch.exp(self.log_Q).clamp(0.3, 8.0)

        w0 = 2.0 * math.pi * fc / self.sr
        sin_w0 = torch.sin(w0)
        cos_w0 = torch.cos(w0)
        alpha = sin_w0 / (2.0 * Q)

        b0_raw = sin_w0 / 2.0
        b1_raw = torch.zeros((), dtype=torch.float32, device=x.device)
        b2_raw = -sin_w0 / 2.0
        a0 = 1.0 + alpha
        a1_raw = -2.0 * cos_w0
        a2_raw = 1.0 - alpha

        b0n = b0_raw / a0
        b1n = b1_raw / a0
        b2n = b2_raw / a0
        a1n = a1_raw / a0
        a2n = a2_raw / a0

        kernel = _biquad_kernel(b0n, b1n, b2n, a1n, a2n)
        return _convolve(x, kernel)


class LearnableSoftClipper(nn.Module):
    """Symmetric soft-clipper: out = gain * tanh(drive * x)."""

    def __init__(self) -> None:
        super().__init__()
        self.log_drive = nn.Parameter(torch.tensor(0.0))
        self.log_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply symmetric soft-clip nonlinearity.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Clipped signal, shape (N,), float32.
        """
        drive = torch.exp(self.log_drive).clamp(0.1, 50.0)
        gain = torch.exp(self.log_gain).clamp(0.1, 10.0)
        return gain * torch.tanh(drive * x)


class LearnableAsymClipper(nn.Module):
    """Asymmetric soft-clipper with separate positive/negative drive.

    out = gain * (tanh(drive_pos * x+) - tanh(drive_neg * x-))
    where x+ = clamp(x, min=0), x- = clamp(x, max=0).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_drive_pos = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_drive_neg = nn.Parameter(torch.tensor(math.log(1.5)))
        self.log_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply asymmetric soft-clip nonlinearity.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Clipped signal, shape (N,), float32.
        """
        drive_pos = torch.exp(self.log_drive_pos).clamp(0.1, 50.0)
        drive_neg = torch.exp(self.log_drive_neg).clamp(0.1, 50.0)
        gain = torch.exp(self.log_gain).clamp(0.1, 10.0)
        pos_part = torch.tanh(drive_pos * x.clamp(min=0.0))
        neg_part = torch.tanh(drive_neg * x.clamp(max=0.0))
        return gain * (pos_part - neg_part)


class LearnableClipThenFilter(nn.Module):
    """Soft-clip stage followed by a high-shelf filter."""

    def __init__(self, sr: int) -> None:
        """
        Args:
            sr: Sample rate in Hz.
        """
        super().__init__()
        self.sr = sr
        self.log_drive = nn.Parameter(torch.tensor(0.0))
        self.log_clip_gain = nn.Parameter(torch.tensor(0.0))
        self.log_fc = nn.Parameter(torch.tensor(math.log(2000.0)))
        self.shelf_gain_dB = nn.Parameter(torch.tensor(0.0))
        # Reuse the shelf primitive for the filter stage
        self._shelf = LearnableHighShelf(sr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-clip then apply high-shelf.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Processed signal, shape (N,), float32.
        """
        drive = torch.exp(self.log_drive).clamp(0.1, 50.0)
        clip_gain = torch.exp(self.log_clip_gain).clamp(0.1, 10.0)
        clipped = clip_gain * torch.tanh(drive * x)

        # Override shelf parameters with this module's own params
        fc = torch.exp(self.log_fc).clamp(100.0, self.sr / 2.0 - 100.0)
        a = torch.exp(torch.tensor(-2.0 * math.pi, dtype=torch.float32) * fc / self.sr)
        kernel = _hp_kernel(a)
        hp = _convolve(clipped, kernel)
        gain_dB_clamped = self.shelf_gain_dB.clamp(-12.0, 12.0)
        g = torch.pow(torch.tensor(10.0), gain_dB_clamped / 20.0)
        return clipped + (g - 1.0) * hp


class LearnableParallelBlend(nn.Module):
    """Parallel blend of clean path and soft-clipped path.

    out = w_clean * x + w_dirty * tanh(drive * x)
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_clean = nn.Parameter(torch.tensor(0.5))
        self.w_dirty = nn.Parameter(torch.tensor(0.5))
        self.log_drive = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blend clean and distorted signal paths.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Blended signal, shape (N,), float32.
        """
        w_clean = self.w_clean.clamp(0.0, 2.0)
        w_dirty = self.w_dirty.clamp(0.0, 2.0)
        drive = torch.exp(self.log_drive).clamp(0.1, 50.0)
        return w_clean * x + w_dirty * torch.tanh(drive * x)


class SkipConnection(nn.Module):
    """Identity primitive — passes the input unchanged. No learnable parameters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input unchanged.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            x unchanged.
        """
        return x


class LearnableGain(nn.Module):
    """Simple gain stage: out = gain * x."""

    def __init__(self) -> None:
        super().__init__()
        self.log_gain = nn.Parameter(torch.tensor(0.0))  # gain=1 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scalar gain to signal.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Scaled signal, shape (N,), float32.
        """
        gain = torch.exp(self.log_gain).clamp(0.01, 10.0)
        return gain * x


# ---------------------------------------------------------------------------
# ESR loss (module-level helper)
# ---------------------------------------------------------------------------


def _esr(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Error-to-Signal Ratio loss. Lower is better; 0 = perfect prediction.

    Args:
        predicted: Model output, shape (N,), float32.
        target: Reference signal, shape (N,), float32.

    Returns:
        Scalar tensor ESR in [0, inf).
    """
    error = predicted - target
    return (error ** 2).sum() / (target ** 2).sum().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# CircuitDictionary model
# ---------------------------------------------------------------------------


class CircuitDictionary(nn.Module, PedalModel):
    """Parallel differentiable circuit primitive dictionary with learned architecture weights.

    Inspired by DARTS (Liu et al. 2019). Each primitive processes the full
    input signal independently. A learned softmax-weighted sum combines the
    outputs. An L1 penalty on the architecture weights encourages sparsity.

    After fitting, ``self.alpha_`` (softmax of raw weights) gives the learned
    mixture and ``self.primitive_names`` gives the corresponding labels.
    """

    def __init__(
        self,
        sr: int = 48000,
        n_epochs: int = 500,
        lambda_l1: float = 0.1,
        window_size: int = 65536,
    ) -> None:
        """
        Args:
            sr: Sample rate in Hz (used to initialise frequency-based primitives).
            n_epochs: Number of Adam optimisation epochs during fit().
            lambda_l1: L1 regularisation coefficient on the architecture weights.
                Default 0.05 — stronger than the original 0.01 to break the
                uniform-softmax plateau more aggressively.
            window_size: Samples per epoch during training. A new random contiguous
                window is drawn from the full training signal each epoch so that all
                parts of the signal are visited over the course of training.
                At 96 kHz, 65536 ≈ 0.68 s; at 48 kHz ≈ 1.37 s.
        """
        nn.Module.__init__(self)
        self.sr = sr
        self.n_epochs = n_epochs
        self.lambda_l1 = lambda_l1
        self.window_size = window_size

        self._primitives: nn.ModuleList | None = None
        self.raw_alpha: nn.Parameter | None = None  # type: ignore[assignment]
        self.alpha_: np.ndarray | None = None
        self.primitive_names: list[str] | None = None

    # ------------------------------------------------------------------
    # PedalModel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable model identifier."""
        return "CircuitDict"

    @property
    def receptive_field(self) -> int:
        """Number of past samples the model depends on (matches FIR kernel length)."""
        return _K

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        """Optimise architecture weights and primitive parameters via Adam.

        Each epoch draws a fresh random contiguous window of ``window_size``
        samples from the full training signal so that all parts of the signal
        are visited over the course of training. With n_epochs=500 and
        window_size=65536 at 96 kHz, ~33M samples of coverage are accumulated
        across the 64-second capture (~53% coverage per run).

        Args:
            dry: Full dry training signal, shape (N,), float32, range [-1, 1].
            wet: Full target wet signal, same shape as dry.
            sr: Sample rate in Hz.
        """
        self.sr = sr
        self._build(sr)

        N = len(dry)
        W = min(self.window_size, N)

        # Sequential epoch cycling: step through signal deterministically so every
        # sample is visited exactly once per ceil(N/W) epochs.  Guarantees 100%
        # coverage when n_epochs ≥ ceil(N/W); dramatically better than random
        # windows which give ~65% coverage at 1000 epochs via the coupon-collector.
        _max_start = max(1, N - W + 1)
        _full_pass_epochs = int(np.ceil(N / W))
        print(f"      circuit_dict: window={W} samples ({W/1e3:.0f}k), "
              f"one full pass = {_full_pass_epochs} epochs, "
              f"running {self.n_epochs} epochs "
              f"({min(100.0, self.n_epochs / _full_pass_epochs * 100):.1f}% signal coverage)")

        # Separate param groups:
        #   raw_alpha at high lr so topology differentiates faster than filters change.
        #   Filter primitives (log_fc, log_Q, gain_dB) at low lr — their gradients
        #   are amplified by exp() in the fc→a→kernel chain, so small steps are safer.
        #   Nonlinear / gain primitives at base lr.
        _FILTER_TYPES = (
            LearnableLowpass, LearnableHighpass, LearnableHighShelf,
            LearnableBandpass, LearnableClipThenFilter,
        )
        filter_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for prim in self._primitives:  # type: ignore[union-attr]
            target = filter_params if isinstance(prim, _FILTER_TYPES) else other_params
            target.extend(prim.parameters())

        opt = torch.optim.Adam([
            {"params": [self.raw_alpha], "lr": 1e-2},   # fast: topology weights
            {"params": filter_params,    "lr": 1e-4},   # slow: filter cutoffs
            {"params": other_params,     "lr": 1e-3},   # base: drive/gain scalars
        ])

        for epoch in range(1, self.n_epochs + 1):
            # Sequential cycling: position advances by W each epoch, wrapping at end.
            start = ((epoch - 1) * W) % _max_start
            dry_t = torch.tensor(dry[start : start + W], dtype=torch.float32)
            wet_t = torch.tensor(wet[start : start + W], dtype=torch.float32)

            opt.zero_grad()
            out = self._forward(dry_t)
            esr = _esr(out, wet_t)
            l1 = self.raw_alpha.abs().mean()  # type: ignore[union-attr]
            loss = esr + self.lambda_l1 * l1
            loss.backward()
            # Per-group clipping: filter fc→a→kernel chain amplifies gradients through
            # exp(), so filter params get a tighter clip than alpha/nonlinear params.
            torch.nn.utils.clip_grad_norm_(filter_params, max_norm=0.1)
            torch.nn.utils.clip_grad_norm_([self.raw_alpha] + other_params, max_norm=1.0)  # type: ignore[list-item]
            opt.step()

            loss_val = loss.item()

            if loss_val > 2.0:
                with torch.no_grad():
                    _a = F.softmax(self.raw_alpha, dim=0)  # type: ignore[arg-type]
                    _names = [p.__class__.__name__.replace("Learnable", "") for p in self._primitives]  # type: ignore[union-attr]
                    _top = sorted(zip(_a.tolist(), _names), reverse=True)
                print(f"    spike ep={epoch}  loss={loss_val:.3f}  esr={esr.item():.3f}  "
                      f"top3: {', '.join(f'{n}:{w:.2f}' for w, n in _top[:3])}")

            if epoch % 50 == 0 or epoch == 1:
                coverage_pct = min(100.0, epoch * W / N * 100.0)
                with torch.no_grad():
                    alpha = F.softmax(self.raw_alpha, dim=0)  # type: ignore[arg-type]
                    entropy = -(alpha * (alpha + 1e-8).log()).sum().item()
                    top_idx = int(alpha.argmax().item())
                    top_name = self._primitives[top_idx].__class__.__name__.replace("Learnable", "")  # type: ignore[index]
                    top_w = alpha[top_idx].item()
                print(f"      epoch {epoch}/{self.n_epochs}  loss={loss_val:.4f}  "
                      f"esr={esr.item():.4f}  l1={l1.item():.4f}  "
                      f"H={entropy:.3f}  top={top_name}({top_w:.2f})  "
                      f"coverage={coverage_pct:.1f}%")

        with torch.no_grad():
            self.alpha_ = F.softmax(self.raw_alpha, dim=0).numpy()  # type: ignore[arg-type]
            self.primitive_names = [p.__class__.__name__ for p in self._primitives]  # type: ignore[union-attr]

        final_coverage = min(100.0, self.n_epochs * W / N * 100.0)
        print(f"  circuit_dict data coverage: {final_coverage:.1f}% of training signal")
        print("  circuit_dict learned topology:")
        order = sorted(
            zip(self.alpha_, self._primitives, self.primitive_names),  # type: ignore[union-attr]
            reverse=True,
        )
        for w, prim, pname in order:
            params = _prim_params(prim)
            param_str = f"  {params}" if params else ""
            print(f"    {w:.2f}  {pname:<26}{param_str}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run the fitted model on a dry input signal.

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        if self._primitives is None:
            raise RuntimeError("Call fit() before predict().")
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32)
            out = self._forward(x_t)
        return out.numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, sr: int) -> None:
        """Instantiate all primitives and initialise raw architecture weights.

        Args:
            sr: Sample rate in Hz.
        """
        primitives = [
            LearnableLowpass(sr),
            LearnableHighpass(sr),
            LearnableHighShelf(sr),
            LearnableBandpass(sr),
            LearnableSoftClipper(),
            LearnableAsymClipper(),
            LearnableClipThenFilter(sr),
            LearnableParallelBlend(),
            SkipConnection(),
            LearnableGain(),
        ]
        self._primitives = nn.ModuleList(primitives)
        # Small random init breaks the all-zeros symmetry so softmax is not
        # perfectly uniform at epoch 0 and the L1 gradient is non-zero from
        # the first update (|∂/∂x| is 0 at x=0 due to subgradient at the kink).
        self.raw_alpha = nn.Parameter(0.1 * torch.randn(len(primitives)))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the softmax-weighted sum of all primitive outputs.

        Args:
            x: Input signal, shape (N,), float32.

        Returns:
            Blended output, shape (N,), float32.
        """
        weights = F.softmax(self.raw_alpha, dim=0)  # type: ignore[arg-type]
        outputs = [p(x) for p in self._primitives]  # type: ignore[union-attr]
        return sum(w * o for w, o in zip(weights, outputs))  # type: ignore[return-value]
