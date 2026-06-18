"""Differentiable Wiener-Hammerstein model: HPF1 → drive → NL → HPF2 → Shelf → level.

Also provides LinearBiquadModel for purely linear effects (EQ pedals), which
shares the same Stage 1 / Stage 3 interface so the fitting pipeline works
without topology-specific code.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from pedal_model.ode.filters import HighShelfBiquad, PeakBiquad, SecondOrderHighpass
from pedal_model.ode.nonlinearity import OddSymmetricMLP, UnconstrainedMLP


class WHODEModel(nn.Module):
    """Physics-informed differentiable Wiener-Hammerstein model.

    Implements the shared fitter interface (stage1_parameters, freeze_nl,
    unfreeze_nl, is_linear) so KlonSimFitter works without topology-specific code.


    Signal chain:
        dry → HPF1(fc₁) → ×drive → NL(·) → HPF2(fc₂) → Shelf(fs, A) → ×level → ŷ

    All parameters are continuous-time and trainable. Filter cutoffs are
    parameterised as log(ω₀) to enforce positivity without hard constraints.
    Drive and output_level are parameterised as log values for the same reason.

    Args:
        fc1_init: Initial pre-filter cutoff in Hz.
        fc2_init: Initial post-filter cutoff in Hz.
        shelf_freq_init: Initial shelf frequency in Hz.
        shelf_gain_db_init: Initial shelf gain in dB.
        drive_init: Initial drive multiplier.
        output_level_init: Initial output level scalar.
        nl_hidden: Hidden size for the nonlinearity MLP.
        use_odd_symmetric_nl: If True, use OddSymmetricMLP (for symmetric clipping).
            If False, use UnconstrainedMLP (for asymmetric clipping).
    """

    is_linear: bool = False  # fitter interface: WHODEModel always has a NL stage

    def __init__(
        self,
        fc1_init: float = 100.0,
        fc2_init: float = 500.0,
        shelf_freq_init: float = 3000.0,
        shelf_gain_db_init: float = 2.0,
        drive_init: float = 1.0,
        output_level_init: float = 0.5,
        nl_hidden: int = 8,
        use_odd_symmetric_nl: bool = True,
    ) -> None:
        super().__init__()
        self.hpf1 = SecondOrderHighpass(fc_init=fc1_init)
        self.hpf2 = SecondOrderHighpass(fc_init=fc2_init)
        self.shelf = HighShelfBiquad(
            freq_init=shelf_freq_init,
            gain_db_init=shelf_gain_db_init,
        )
        if use_odd_symmetric_nl:
            self.nl: OddSymmetricMLP | UnconstrainedMLP = OddSymmetricMLP(hidden_size=nl_hidden)
        else:
            self.nl = UnconstrainedMLP(hidden_size=nl_hidden)

        self.log_drive = nn.Parameter(torch.tensor(math.log(drive_init)))
        self.log_output_level = nn.Parameter(torch.tensor(math.log(output_level_init)))

    @property
    def drive(self) -> float:
        """Current drive scalar."""
        return float(torch.exp(self.log_drive).item())

    @property
    def output_level(self) -> float:
        """Current output level scalar."""
        return float(torch.exp(self.log_output_level).item())

    def forward(self, dry: Tensor, dt: float) -> Tensor:
        """Full Wiener-Hammerstein forward pass.

        Args:
            dry: Dry input signal, shape (N,), float32.
            dt: Sample timestep in seconds (= 1/sr).

        Returns:
            Model output ŷ, shape (N,).
        """
        x = self.hpf1(dry, dt)
        x = x * torch.exp(self.log_drive)
        x = self.nl(x)
        x = self.hpf2(x, dt)
        x = self.shelf(x, dt)
        return x * torch.exp(self.log_output_level)

    def pre_filter_output(self, dry: Tensor, dt: float) -> Tensor:
        """Apply only HPF1 (no drive) to the dry signal.

        Used in Stage 2 to construct the NL input signal.

        Args:
            dry: Dry input signal, shape (N,), float32.
            dt: Sample timestep in seconds.

        Returns:
            Pre-filtered signal x_pre, shape (N,).
        """
        with torch.no_grad():
            return self.hpf1(dry, dt)

    def combined_log_magnitude(
        self, freqs_rad: Tensor, dt: float
    ) -> Tensor:
        """Analytical log|H_model(f)| = log|G| + log|H1| + log|H2| + log|H_shelf|.

        Used in Stage 1 frequency-domain fitting (no ODE integration).

        Args:
            freqs_rad: Angular frequencies in rad/s, shape (M,).
            dt: Sample timestep in seconds.

        Returns:
            log|H_model|, shape (M,), differentiable w.r.t. all filter parameters
            and the combined gain G = drive × output_level.
        """
        log_G = self.log_drive + self.log_output_level
        log_H1 = self.hpf1.frequency_response_log_mag(freqs_rad)
        log_H2 = self.hpf2.frequency_response_log_mag(freqs_rad)
        log_Hs = self.shelf.frequency_response_log_mag(freqs_rad, dt)
        return log_G + log_H1 + log_H2 + log_Hs

    def get_params_dict(self) -> dict[str, float]:
        """Return all physical parameter values as a plain dict.

        Returns:
            Dict with keys: fc1_hz, fc2_hz, shelf_freq_hz, shelf_gain_db,
            drive, output_level.
        """
        return {
            "fc1_hz": self.hpf1.fc_hz,
            "fc2_hz": self.hpf2.fc_hz,
            "shelf_freq_hz": self.shelf.freq_hz,
            "shelf_gain_db": self.shelf.gain_db,
            "drive": self.drive,
            "output_level": self.output_level,
        }

    def log_params(self, prefix: str = "") -> None:
        """Print current physical parameter values to stdout.

        Args:
            prefix: Optional label prefix (e.g. stage name).
        """
        p = self.get_params_dict()
        label = f"[{prefix}] " if prefix else ""
        print(
            f"{label}fc1={p['fc1_hz']:.1f} Hz | "
            f"fc2={p['fc2_hz']:.1f} Hz | "
            f"shelf={p['shelf_freq_hz']:.0f} Hz {p['shelf_gain_db']:+.2f} dB | "
            f"drive={p['drive']:.3f} | "
            f"level={p['output_level']:.3f}"
        )

    # ── Fitter interface ──────────────────────────────────────────────────────

    def stage1_parameters(self) -> list[nn.Parameter]:
        """Parameters optimised in Stage 1 (filter cutoffs + combined gain).

        NL weights are excluded — they are frozen during Stage 1.
        """
        return (
            list(self.hpf1.parameters())
            + list(self.hpf2.parameters())
            + [self.log_drive, self.log_output_level]
        )

    def freeze_nl(self) -> None:
        """Freeze NL weights for Stage 1 (linear-regime TF fitting)."""
        for p in self.nl.parameters():
            p.requires_grad_(False)

    def unfreeze_nl(self) -> None:
        """Restore NL gradients after Stage 1."""
        for p in self.nl.parameters():
            p.requires_grad_(True)

    def freeze_filters(self) -> None:
        """Freeze HPF1, HPF2, and shelf parameters (Stage 2 — only fit NL + drive/level)."""
        for p in list(self.hpf1.parameters()) + list(self.hpf2.parameters()) + list(self.shelf.parameters()):
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (Stage 3 — joint refinement)."""
        for p in self.parameters():
            p.requires_grad_(True)

    @classmethod
    def from_params_dict(
        cls,
        p: dict[str, float],
        nl_hidden: int = 8,
        use_odd_symmetric_nl: bool = True,
    ) -> "WHODEModel":
        """Construct model initialised to the given physical parameter values.

        Args:
            p: Dict with keys: fc1_hz, fc2_hz, shelf_freq_hz, shelf_gain_db,
               drive, output_level.
            nl_hidden: Hidden size for the nonlinearity MLP.
            use_odd_symmetric_nl: Whether to use the odd-symmetric NL variant.

        Returns:
            WHODEModel with parameters set to the given values.
        """
        return cls(
            fc1_init=p["fc1_hz"],
            fc2_init=p["fc2_hz"],
            shelf_freq_init=p["shelf_freq_hz"],
            shelf_gain_db_init=p["shelf_gain_db"],
            drive_init=p["drive"],
            output_level_init=p["output_level"],
            nl_hidden=nl_hidden,
            use_odd_symmetric_nl=use_odd_symmetric_nl,
        )


# ── LinearBiquadModel ─────────────────────────────────────────────────────────


#: Type alias for any biquad section module supported by LinearBiquadModel.
BiquadSection = Union[SecondOrderHighpass, HighShelfBiquad, PeakBiquad]


class LinearBiquadModel(nn.Module):
    """Purely linear cascade of N biquad sections — for EQ pedals and linear effects.

    Implements the same fitter interface as WHODEModel so KlonSimFitter runs
    without topology-specific branching:
        stage1_parameters(), freeze_nl(), unfreeze_nl(), is_linear,
        combined_log_magnitude(), forward(), get_params_dict(), log_params(),
        freeze_filters(), unfreeze_all()

    Signal chain: dry → section[0] → section[1] → … → section[N-1] → ŷ

    Args:
        sections: Ordered list of biquad section modules. Each must implement
            forward(u, dt) and frequency_response_log_mag(freqs_rad, dt).
    """

    is_linear: bool = True  # gates Stage 2 skip in KlonSimFitter.run()

    def __init__(self, sections: list[BiquadSection]) -> None:
        super().__init__()
        self.biquads = nn.ModuleList(sections)
        # Dummy log_drive / log_output_level so log_params() and get_params_dict()
        # work unchanged — fixed at 0 (drive=1, output_level=1).
        self.log_drive = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.log_output_level = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ── Fitter interface ──────────────────────────────────────────────────────

    def stage1_parameters(self) -> list[nn.Parameter]:
        """All biquad section parameters — linear model trains everything in Stage 1."""
        return list(self.parameters())

    def freeze_nl(self) -> None:
        """No-op — linear model has no NL."""

    def unfreeze_nl(self) -> None:
        """No-op — linear model has no NL."""

    def freeze_filters(self) -> None:
        """Freeze all section parameters (used in Stage 2 — skipped for linear models)."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Restore gradients on all parameters."""
        for p in self.parameters():
            p.requires_grad_(True)

    # ── Forward passes ────────────────────────────────────────────────────────

    def combined_log_magnitude(self, freqs_rad: Tensor, dt: float) -> Tensor:
        """Sum of log|H_k(e^{jω})| over all biquad sections.

        Args:
            freqs_rad: Angular frequencies in rad/s, shape (M,).
            dt: Sample timestep in seconds.

        Returns:
            log|H_total|, shape (M,).
        """
        result = torch.zeros_like(freqs_rad)
        for section in self.biquads:
            result = result + section.frequency_response_log_mag(freqs_rad, dt)
        return result

    def forward(self, dry: Tensor, dt: float) -> Tensor:
        """Cascade all biquad sections on the dry signal.

        Args:
            dry: Dry input signal, shape (N,), float32.
            dt: Sample timestep in seconds.

        Returns:
            Filtered output ŷ, shape (N,).
        """
        x = dry
        for section in self.biquads:
            x = section(x, dt)
        return x

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_params_dict(self) -> dict[str, float]:
        """Return per-section parameter values as a flat dict."""
        params: dict[str, float] = {}
        for i, sec in enumerate(self.biquads):
            prefix = f"s{i}"
            if hasattr(sec, "fc_hz"):
                params[f"{prefix}_fc_hz"] = float(sec.fc_hz)  # type: ignore[attr-defined]
            if hasattr(sec, "freq_hz"):
                params[f"{prefix}_freq_hz"] = float(sec.freq_hz)  # type: ignore[attr-defined]
            if hasattr(sec, "gain_db") and isinstance(sec.gain_db, (int, float)):
                params[f"{prefix}_gain_db"] = float(sec.gain_db)  # type: ignore[attr-defined]
            elif hasattr(sec, "gain_db") and hasattr(sec.gain_db, "item"):
                params[f"{prefix}_gain_db"] = float(sec.gain_db.item())  # type: ignore[attr-defined]
            if hasattr(sec, "Q"):
                params[f"{prefix}_Q"] = float(sec.Q)  # type: ignore[attr-defined]
        return params

    def log_params(self, prefix: str = "") -> None:
        """Print current parameter values to stdout."""
        p = self.get_params_dict()
        label = f"[{prefix}] " if prefix else ""
        print(label + " | ".join(f"{k}={v:.3f}" for k, v in p.items()))
