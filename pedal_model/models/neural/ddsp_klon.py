"""KlonDDSP: Klon-topology DDSP circuit cascaded with a GRU residual corrector.

Stage 1 (DDSP): differentiable Klon-style signal chain with learnable parameters.
  - Input HPF (feedforward FIR: y[n] = x[n] - α_in · x[n-1])
  - Drive + tanh clipping
  - Post-clip HPF
  - High-frequency shelf (1st-difference boost with learnable gain)
  - Output level scalar

Stage 2 (ML): small GRU that sees the dry signal and adds a correction.
  correction = f_GRU(dry ‖ condition)
  output = ddsp_out + correction

Conditioning: the three physical Klon knobs [gain, treble, volume] ∈ [0, 1]³
are concatenated to every GRU time step as side-chain information, allowing the
same model to represent different knob positions at inference time.

After training, call analyse_ddsp_fraction(dry, wet) to decompose how much of
the pedal response is captured by the physics-informed DDSP stage vs. the
data-driven GRU correction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..base import PedalModel

# Default knob positions — 9 o'clock on a 6-to-5-o'clock travel (≈ 27% of range)
_NINE_OCLOCK = float(3.0 / 11.0)
DEFAULT_CONDITION = np.array([_NINE_OCLOCK, _NINE_OCLOCK, _NINE_OCLOCK], dtype=np.float32)
N_KNOBS = 3  # gain, treble, volume


class KlonDDSP(nn.Module, PedalModel):
    """Klon-topology differentiable circuit + GRU residual corrector.

    Conditioning: each forward pass accepts an optional `condition` tensor
    of shape (n_knobs,) or (batch, n_knobs) with normalised knob positions
    [gain, treble, volume] ∈ [0, 1]. At 9 o'clock each value is ≈ 0.273.
    When condition=None, DEFAULT_CONDITION (9 o'clock) is used.

    Training: use train/trainer.py with ChunkDataset. Input/output shape is
    (batch, time, 1). CombinedLoss (ESR 70 / STFT 30) is recommended.

    Args:
        gru_hidden: Hidden units in the GRU corrector.
        n_fir_pre: Optional learnable FIR pre-filter taps (0 = skip).
    """

    def __init__(self, gru_hidden: int = 16, n_fir_pre: int = 0) -> None:
        nn.Module.__init__(self)
        # DDSP parameters — unconstrained; sigmoid/exp applied in properties
        self._alpha_in_raw = nn.Parameter(torch.zeros(1))
        self._alpha_post_raw = nn.Parameter(torch.zeros(1))
        self._log_drive = nn.Parameter(torch.tensor(1.4))  # exp → ~4× drive
        self._level_raw = nn.Parameter(torch.zeros(1))     # sigmoid → 0.5
        self._shelf_gain = nn.Parameter(torch.zeros(1))    # 0 = no shelf

        # Optional FIR pre-filter
        self._use_pre_fir = n_fir_pre > 0
        if self._use_pre_fir:
            kernel = torch.zeros(1, 1, n_fir_pre)
            nn.init.dirac_(kernel)
            self.pre_fir_kernel = nn.Parameter(kernel)

        # GRU corrector: (dry + condition) → correction to add onto DDSP output
        # Input dim = 1 (dry sample) + N_KNOBS (condition) = 4
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(1 + N_KNOBS, gru_hidden, batch_first=True)
        self.gru_fc = nn.Linear(gru_hidden, 1)

    # ── Derived parameters ────────────────────────────────────────────────── #

    @property
    def alpha_in(self) -> torch.Tensor:
        """HPF coefficient ∈ (0, 0.999). Higher → lower cutoff frequency."""
        return torch.sigmoid(self._alpha_in_raw) * 0.999

    @property
    def alpha_post(self) -> torch.Tensor:
        """Post-clip HPF coefficient ∈ (0, 0.999)."""
        return torch.sigmoid(self._alpha_post_raw) * 0.999

    @property
    def drive(self) -> torch.Tensor:
        """Pre-gain into the tanh stage. Always positive; clamped to [1, 20]."""
        return torch.exp(self._log_drive).clamp(1.0, 20.0)

    @property
    def output_level(self) -> torch.Tensor:
        """Output gain scalar ∈ (0, 1)."""
        return torch.sigmoid(self._level_raw)

    # ── PedalModel interface ──────────────────────────────────────────────── #

    @property
    def name(self) -> str:
        return f"KlonDDSP-GRU{self.gru_hidden}"

    @property
    def receptive_field(self) -> int:
        return 4096

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        raise NotImplementedError("Use train/trainer.py to train KlonDDSP.")

    # ── Forward pass ─────────────────────────────────────────────────────── #

    def _hpf(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Feedforward FIR high-pass: y[n] = x[n] - alpha · x[n-1].

        Causal, fully differentiable w.r.t. alpha.

        Args:
            x: (batch, time)
            alpha: scalar tensor ∈ (0, 1)

        Returns:
            Filtered signal, same shape as x.
        """
        return torch.cat([x[..., :1], x[..., 1:] - alpha * x[..., :-1]], dim=-1)

    def _ddsp_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Klon-like differentiable signal chain.

        Args:
            x: (batch, time) dry signal

        Returns:
            (batch, time) processed signal
        """
        y = x

        if self._use_pre_fir:
            import torch.nn.functional as F

            n = self.pre_fir_kernel.shape[-1]
            y3d = y.unsqueeze(1)
            y3d = F.conv1d(F.pad(y3d, (n - 1, 0)), self.pre_fir_kernel)
            y = y3d.squeeze(1)

        # 1. Input HPF
        y = self._hpf(y, self.alpha_in)
        # 2. Drive + tanh
        y = torch.tanh(y * self.drive)
        # 3. Post-clip HPF
        y = self._hpf(y, self.alpha_post)
        # 4. High shelf: y + gain * 1st-difference (boosts high frequencies)
        shelf = torch.cat([torch.zeros_like(y[..., :1]), y[..., 1:] - y[..., :-1]], dim=-1)
        y = y + self._shelf_gain * shelf
        # 5. Output level
        y = y * self.output_level
        return y

    def _expand_condition(
        self,
        condition: torch.Tensor | None,
        batch: int,
        time: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast condition to (batch, time, N_KNOBS) for GRU concatenation.

        Args:
            condition: (N_KNOBS,), (batch, N_KNOBS), or None.
            batch: Batch dimension.
            time: Time dimension.
            device: Target device.

        Returns:
            (batch, time, N_KNOBS) float32 tensor.
        """
        if condition is None:
            cond = torch.tensor(DEFAULT_CONDITION, device=device)  # (3,)
        else:
            cond = condition.to(device=device, dtype=torch.float32)

        if cond.dim() == 1:
            cond = cond.unsqueeze(0)  # (1, 3)
        # (batch, 3) → (batch, 1, 3) → (batch, time, 3)
        return cond.unsqueeze(1).expand(batch, time, N_KNOBS)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDSP circuit + GRU correction, optionally conditioned on knob positions.

        Args:
            x: Dry input, shape (batch, time) or (batch, time, 1).
            hidden: GRU hidden state from the previous chunk,
                shape (1, batch, gru_hidden). None for the first chunk.
            condition: Knob positions [gain, treble, volume] ∈ [0, 1],
                shape (N_KNOBS,) or (batch, N_KNOBS). None → 9 o'clock defaults.

        Returns:
            (output, hidden):
                output — shape (batch, time, 1).
                hidden — GRU state to carry across chunks.
        """
        if x.dim() == 3:
            x_2d = x.squeeze(-1)
        else:
            x_2d = x  # (batch, time)

        batch, time = x_2d.shape

        # Stage 1: physics-inspired DDSP circuit
        ddsp_out = self._ddsp_stage(x_2d)  # (batch, time)

        # Stage 2: GRU on dry + condition
        cond_exp = self._expand_condition(condition, batch, time, x_2d.device)
        gru_in = torch.cat([x_2d.unsqueeze(-1), cond_exp], dim=-1)  # (batch, time, 4)
        gru_out, new_hidden = self.gru(gru_in, hidden)
        correction = self.gru_fc(gru_out)  # (batch, time, 1)

        output = ddsp_out.unsqueeze(-1) + correction  # (batch, time, 1)
        return output, new_hidden

    # ── Inference ────────────────────────────────────────────────────────── #

    def predict(
        self,
        x: np.ndarray,
        chunk_size: int = 65536,
        condition: np.ndarray | None = None,
    ) -> np.ndarray:
        """Chunked inference with GRU hidden state carried across chunks.

        Args:
            x: Dry input, shape (N,), float32.
            chunk_size: Samples per chunk.
            condition: Knob positions [gain, treble, volume] ∈ [0, 1], shape (3,).
                None → 9 o'clock defaults (all ≈ 0.273).

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        self.eval()
        N = len(x)
        out = np.zeros(N, dtype=np.float32)
        hidden = None
        cond_t = (
            torch.tensor(condition, dtype=torch.float32)
            if condition is not None
            else None
        )
        with torch.no_grad():
            for start in range(0, N, chunk_size):
                chunk = x[start : start + chunk_size]
                t = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)  # (1, T)
                y, hidden = self(t, hidden, cond_t)
                hidden = hidden.detach()
                out[start : start + len(chunk)] = y.squeeze().cpu().numpy()
        return out

    def predict_ddsp(self, x: np.ndarray) -> np.ndarray:
        """Run the DDSP circuit stage only — no GRU correction.

        Useful for measuring how much of the pedal response the physics model
        captures before the data-driven stage adds its correction.

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            DDSP-stage output, shape (N,), float32.
        """
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, N)
            ddsp_out = self._ddsp_stage(t)
        return ddsp_out.squeeze(0).cpu().numpy().astype(np.float32)

    # ── Analysis ─────────────────────────────────────────────────────────── #

    def analyse_ddsp_fraction(
        self,
        dry: np.ndarray,
        wet: np.ndarray,
        condition: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Decompose DDSP vs. ML contribution after training.

        Runs both the DDSP-only stage and the full model on *dry*, then computes
        ESR ratios to quantify what fraction of the pedal response is captured
        by the physics-informed circuit model vs. the GRU correction.

        Args:
            dry: Dry input, shape (N,), float32.
            wet: Reference wet output, shape (N,), float32.
            condition: Knob positions [gain, treble, volume]. None → 9 o'clock.

        Returns:
            dict with:
                ddsp_esr: ESR of the DDSP stage alone vs. wet.
                full_esr: ESR of the full model (DDSP + GRU) vs. wet.
                correction_rms_ratio: RMS(correction) / RMS(full_output).
                ddsp_fraction: Fraction of total explained variance from DDSP.
                    1.0 = DDSP explains everything; 0.0 = GRU does all the work.
        """
        from pedal_model.metrics.time_domain import compute_esr

        ddsp_out = self.predict_ddsp(dry)
        full_out = self.predict(dry, condition=condition)
        correction = full_out - ddsp_out

        ddsp_esr = compute_esr(wet, ddsp_out)
        full_esr = compute_esr(wet, full_out)

        full_rms = float(np.sqrt(np.mean(full_out ** 2))) + 1e-12
        correction_rms_ratio = float(np.sqrt(np.mean(correction ** 2))) / full_rms

        ddsp_explained = max(0.0, 1.0 - ddsp_esr)
        full_explained = max(1e-8, 1.0 - full_esr)
        ddsp_fraction = float(np.clip(ddsp_explained / full_explained, 0.0, 1.0))

        return {
            "ddsp_esr": float(ddsp_esr),
            "full_esr": float(full_esr),
            "correction_rms_ratio": float(correction_rms_ratio),
            "ddsp_fraction": ddsp_fraction,
        }
