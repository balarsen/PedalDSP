"""KlonDDSP: Klon-topology DDSP circuit cascaded with a GRU residual corrector.

Stage 1 (DDSP): differentiable Klon-style signal chain with learnable parameters.
  - Input HPF (feedforward FIR: y[n] = x[n] - α_in · x[n-1])
  - Drive + tanh clipping (the nonlinear core)
  - Post-clip HPF
  - High-frequency shelf (1st-difference boost with learnable gain)
  - FF1: single-pole 106 Hz LPF applied to the dry signal in parallel
  - FF2: gain-conditioned blend of the clipped path and FF1 clean path
  - Output level scalar

Stage 2 (ML): small GRU that sees [dry, ddsp_out.detach(), condition] and adds
a correction to the physics-stage output.
  output = ddsp_out + f_GRU(dry ‖ ddsp_out.detach() ‖ condition)

The FF1 + FF2 pair is the key missing ingredient in vanilla clipping models of the
Klon.  The gain knob simultaneously sets the pre-clip gain AND the blend ratio
between the clean-bass (FF1) and clipped-treble paths, which is what gives the
Klon its "transparent overdrive" character at lower drive settings.

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
import torch.nn.functional as F

from ..base import PedalModel

# Default knob positions — 9 o'clock on a 6-to-5-o'clock travel (≈ 27% of range)
_NINE_OCLOCK = float(3.0 / 11.0)
DEFAULT_CONDITION = np.array([_NINE_OCLOCK, _NINE_OCLOCK, _NINE_OCLOCK], dtype=np.float32)
N_KNOBS = 3  # gain, treble, volume

# GRU receives: dry (1) + ddsp_out context (1) + knobs (N_KNOBS)
_GRU_INPUT_SIZE = 2 + N_KNOBS

# FF1 LPF taps — 256 taps captures 99.9 % of the 106 Hz IIR energy at 48 kHz
_FF1_TAPS = 256


class KlonDDSP(nn.Module, PedalModel):
    """Klon-topology differentiable circuit + GRU residual corrector.

    The DDSP stage now implements the full Klon parallel architecture:
      - Clip path: HPF → gain/tanh → post-HPF → shelf
      - FF1 clean path: 106 Hz LPF applied to the dry input
      - FF2 blend: gain-conditioned mix of clip and FF1 paths

    The GRU corrector receives [dry, ddsp_out.detach(), condition] so it can
    directly observe the physics-stage output and learn its residual error
    rather than reconstructing the correction from the dry signal alone.

    Conditioning: each forward pass accepts an optional `condition` tensor
    of shape (n_knobs,) or (batch, n_knobs) with normalised knob positions
    [gain, treble, volume] ∈ [0, 1]. At 9 o'clock each value is ≈ 0.273.
    When condition=None, DEFAULT_CONDITION (9 o'clock) is used.

    Args:
        gru_hidden: Hidden units in the GRU corrector.
        n_fir_pre: Optional learnable FIR pre-filter taps (0 = skip).
    """

    def __init__(self, gru_hidden: int = 16, n_fir_pre: int = 0) -> None:
        nn.Module.__init__(self)

        # ── Clip-path DDSP parameters ──────────────────────────────────── #
        # Unconstrained raw values; sigmoid/exp applied in properties below.
        self._alpha_in_raw = nn.Parameter(torch.zeros(1))
        self._alpha_post_raw = nn.Parameter(torch.zeros(1))
        self._log_drive = nn.Parameter(torch.tensor(1.4))   # exp → ~4× drive
        self._level_raw = nn.Parameter(torch.zeros(1))      # sigmoid → 0.5
        self._shelf_gain = nn.Parameter(torch.zeros(1))     # 0 = no shelf

        # ── FF1: learnable 1-pole LPF (initialised to 106 Hz at 48 kHz) ── #
        # a ≈ 0.986 → sigmoid(4.34) * 0.999 ≈ 0.986
        # fc ≈ (1 - a) * sr / (2π) = 0.014 * 48000 / 6.28 ≈ 107 Hz
        self._alpha_ff1_raw = nn.Parameter(torch.tensor(4.34))

        # ── FF2: gain-conditioned blend (clip path vs. FF1 clean path) ─── #
        # blend = sigmoid(blend_w * gain_knob + blend_b)
        # Init: blend_w=1, blend_b=0 → sigmoid(gain_knob) ≈ 0.5 at 9-o'clock
        self._ff2_blend_w = nn.Parameter(torch.tensor(1.0))
        self._ff2_blend_b = nn.Parameter(torch.tensor(0.0))

        # ── Optional FIR pre-filter ────────────────────────────────────── #
        self._use_pre_fir = n_fir_pre > 0
        if self._use_pre_fir:
            kernel = torch.zeros(1, 1, n_fir_pre)
            nn.init.dirac_(kernel)
            self.pre_fir_kernel = nn.Parameter(kernel)

        # ── GRU corrector ─────────────────────────────────────────────── #
        # Input: [dry, ddsp_out.detach(), gain, treble, volume]
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(_GRU_INPUT_SIZE, gru_hidden, batch_first=True)
        self.gru_fc = nn.Linear(gru_hidden, 1)
        # Small init: model starts near-DDSP-only; Kaiming-uniform default
        # produces ~0.1 RMS correction noise that immediately pushes ESR > 1.
        nn.init.normal_(self.gru_fc.weight, std=1e-3)
        nn.init.zeros_(self.gru_fc.bias)

    # ── Derived parameters ────────────────────────────────────────────────── #

    @property
    def alpha_in(self) -> torch.Tensor:
        """HPF coefficient ∈ (0, 0.999). Higher → sharper high-pass."""
        return torch.sigmoid(self._alpha_in_raw) * 0.999

    @property
    def alpha_post(self) -> torch.Tensor:
        """Post-clip HPF coefficient ∈ (0, 0.999)."""
        return torch.sigmoid(self._alpha_post_raw) * 0.999

    @property
    def alpha_ff1(self) -> torch.Tensor:
        """FF1 LPF coefficient ∈ (0, 0.999). Initialised to ≈ 106 Hz at 48 kHz."""
        return torch.sigmoid(self._alpha_ff1_raw) * 0.999

    @property
    def drive(self) -> torch.Tensor:
        """Pre-clip gain, clamped to [1, 20]."""
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

    # ── Filter primitives ─────────────────────────────────────────────────── #

    def _hpf(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Causal first-difference HPF: y[n] = x[n] - alpha · x[n-1].

        Args:
            x: (batch, time)
            alpha: scalar ∈ (0, 1)

        Returns:
            Filtered signal, same shape.
        """
        return torch.cat([x[..., :1], x[..., 1:] - alpha * x[..., :-1]], dim=-1)

    def _lpf(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Causal 1-pole LPF as a truncated FIR: h[k] = (1-a) · a^k.

        The truncated impulse response captures 99.9 % of the IIR energy at
        K=256 taps for a ≈ 0.986 (106 Hz at 48 kHz). Fully differentiable
        w.r.t. `a` through the pow() and multiplication.

        Args:
            x: (batch, time)
            a: scalar LPF coefficient ∈ (0, 1)

        Returns:
            Filtered signal, same shape.
        """
        K = _FF1_TAPS
        k = torch.arange(K, device=x.device, dtype=x.dtype)
        h = (1 - a) * torch.pow(a, k)               # (K,) geometric series
        kernel = h.flip(0).view(1, 1, K)             # (1, 1, K) causal
        x3d = x.unsqueeze(1)                         # (batch, 1, time)
        y3d = F.conv1d(F.pad(x3d, (K - 1, 0)), kernel)
        return y3d.squeeze(1)                        # (batch, time)

    # ── DDSP circuit ──────────────────────────────────────────────────────── #

    def _ddsp_stage(
        self,
        x: torch.Tensor,
        gain_knob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Klon-like differentiable signal chain with FF1 + FF2.

        Args:
            x: (batch, time) dry signal.
            gain_knob: Gain knob position ∈ [0, 1], scalar or (batch,).
                None → 9 o'clock default (≈ 0.273).

        Returns:
            (batch, time) processed signal.
        """
        # Resolve gain_knob to a tensor on the same device
        if gain_knob is None:
            gk = torch.tensor(_NINE_OCLOCK, device=x.device, dtype=x.dtype)
        else:
            gk = gain_knob.to(device=x.device, dtype=x.dtype)

        y = x

        # Optional FIR pre-filter
        if self._use_pre_fir:
            n = self.pre_fir_kernel.shape[-1]
            y3d = y.unsqueeze(1)
            y3d = F.conv1d(F.pad(y3d, (n - 1, 0)), self.pre_fir_kernel)
            y = y3d.squeeze(1)

        # ── Clip path ──────────────────────────────────────────────────── #
        clip = self._hpf(y, self.alpha_in)          # input HPF
        clip = torch.tanh(clip * self.drive)         # drive + clip
        clip = self._hpf(clip, self.alpha_post)      # post-clip HPF
        shelf = torch.cat(                           # 1st-diff high shelf
            [torch.zeros_like(clip[..., :1]), clip[..., 1:] - clip[..., :-1]],
            dim=-1,
        )
        clip = clip + self._shelf_gain * shelf

        # ── FF1: clean bass path (parallel 106 Hz LPF) ─────────────────── #
        ff1 = self._lpf(y, self.alpha_ff1)           # (batch, time)

        # ── FF2: gain-conditioned blend ─────────────────────────────────── #
        # At low gain → mostly FF1 (transparent/clean); at high gain → mostly clip.
        blend = torch.sigmoid(self._ff2_blend_w * gk + self._ff2_blend_b)
        # Broadcast blend over time axis when gk has a batch dimension
        if blend.dim() > 0:
            blend = blend.unsqueeze(-1)              # (batch, 1)

        out = blend * clip + (1 - blend) * ff1
        return out * self.output_level

    # ── Condition helpers ─────────────────────────────────────────────────── #

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
            cond = torch.tensor(DEFAULT_CONDITION, device=device)
        else:
            cond = condition.to(device=device, dtype=torch.float32)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return cond.unsqueeze(1).expand(batch, time, N_KNOBS)

    def _extract_gain_knob(
        self,
        condition: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Extract the gain knob scalar (or batch vector) from condition.

        Returns None only when condition is None (caller uses _NINE_OCLOCK default).
        """
        if condition is None:
            return None
        cond = condition.to(device=device, dtype=torch.float32)
        if cond.dim() == 1:
            return cond[0]       # scalar tensor
        return cond[:, 0]        # (batch,)

    # ── Forward pass ─────────────────────────────────────────────────────── #

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
        device = x_2d.device

        # Stage 1: DDSP circuit (with FF1 + FF2)
        gain_knob = self._extract_gain_knob(condition, device)
        ddsp_out = self._ddsp_stage(x_2d, gain_knob)  # (batch, time)

        # Stage 2: GRU on [dry, ddsp_out.detach(), condition]
        # Detach ddsp_out so DDSP parameters get gradients only via the direct
        # loss → output → ddsp_out path, not through the GRU's input path.
        cond_exp = self._expand_condition(condition, batch, time, device)
        gru_in = torch.cat(
            [
                x_2d.unsqueeze(-1),                  # (batch, time, 1) dry
                ddsp_out.detach().unsqueeze(-1),      # (batch, time, 1) context
                cond_exp,                             # (batch, time, N_KNOBS)
            ],
            dim=-1,
        )  # → (batch, time, 2 + N_KNOBS)

        gru_out, new_hidden = self.gru(gru_in, hidden)
        correction = self.gru_fc(gru_out)            # (batch, time, 1)

        output = ddsp_out.unsqueeze(-1) + correction
        return output, new_hidden

    # ── Inference ────────────────────────────────────────────────────────── #

    def predict(
        self,
        x: np.ndarray,
        chunk_size: int = 16384,
        condition: np.ndarray | None = None,
    ) -> np.ndarray:
        """Chunked inference with GRU hidden state carried across chunks.

        Args:
            x: Dry input, shape (N,), float32.
            chunk_size: Samples per chunk. Must be ≤ 32768 to stay within cuDNN's
                sequence-length ceiling when running on CUDA.
            condition: Knob positions [gain, treble, volume] ∈ [0, 1], shape (3,).
                None → 9 o'clock defaults (all ≈ 0.273).

        Returns:
            Predicted wet signal, shape (N,), float32.
        """
        self.eval()
        device = next(self.parameters()).device
        N = len(x)
        out = np.zeros(N, dtype=np.float32)
        hidden = None
        cond_t = (
            torch.tensor(condition, dtype=torch.float32).to(device)
            if condition is not None
            else None
        )
        with torch.no_grad():
            for start in range(0, N, chunk_size):
                chunk = x[start : start + chunk_size]
                t = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
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
            DDSP-stage output (including FF1 + FF2), shape (N,), float32.
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
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
