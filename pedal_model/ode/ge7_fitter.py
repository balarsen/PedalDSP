"""Fit the GE-7 block model to measured complex responses (frequency domain, many slider settings).

The data are H(jω) ratios measured on a frequency grid, one row per slider setting. The residual is
the complex log of model ÷ measured, ``log|r| + j·angle(r)``, stacked as real numbers: log-magnitude
in nepers (1 Np = 8.69 dB) and phase in radians. Using the ratio keeps the phase wrap-safe while the
model is close.

Decision (A0): Levenberg–Marquardt instead of the plan's Adam → L-BFGS. The problem is a small
nonlinear least-squares (~35 parameters), where LM converges in tens of iterations and gives JᵀJ,
the Fisher information, as a by-product. Revisit if the parameter count grows or the fit moves to
the time domain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from pedal_model.ode import blocks
from pedal_model.ode.ge7_model import (
    BYPASS_SPECS,
    RATIO_SPECS,
    SLIDER_KEYS,
    SPECS_BY_NAME,
    bypass_response,
    pack,
    ratio_response,
    raw_labels,
    unpack,
)

RATIO_NAMES: tuple[str, ...] = tuple(p.name for p in RATIO_SPECS)
BYPASS_NAMES: tuple[str, ...] = tuple(p.name for p in BYPASS_SPECS)


def log_ratio_residual(h_model: Tensor, h_meas: Tensor, weight: Tensor | None, phase_weight: float):
    """Complex log of model ÷ measured as a real vector [w·log|r|, w·λ·angle(r)]."""
    lr = torch.log(h_model / h_meas)
    w = 1.0 if weight is None else weight
    return torch.cat([(w * lr.real).reshape(-1), (w * phase_weight * lr.imag).reshape(-1)])


# ── Levenberg–Marquardt ──────────────────────────────────────────────────────


@dataclass
class LMResult:
    theta: Tensor
    cost: float  # 0.5 Σ r²
    iterations: int
    converged: bool
    jacobian: Tensor
    residual: Tensor
    history: list[float] = field(default_factory=list)


def levenberg_marquardt(
    fun: Callable[[Tensor], Tensor],
    theta0: Tensor,
    *,
    max_iter: int = 200,
    ftol: float = 1e-15,
    xtol: float = 1e-12,
    mu0: float = 1e-3,
) -> LMResult:
    """Minimise ½‖fun(θ)‖² with Marquardt's diagonal scaling; Jacobian by forward-mode autodiff."""
    jac = torch.func.jacfwd(fun)
    theta = theta0.detach().clone().to(torch.float64)
    r = fun(theta)
    cost = 0.5 * float(r @ r)
    mu = mu0
    history = [cost]
    converged = False
    it = 0
    J = jac(theta)
    for it in range(1, max_iter + 1):
        A = J.T @ J
        g = J.T @ r
        d = torch.diag(A).clamp_min(1e-12 * float(torch.diag(A).max()))
        improved = False
        for _ in range(30):
            try:
                step = torch.linalg.solve(A + mu * torch.diag(d), -g)
            except RuntimeError:
                mu *= 10.0
                continue
            r_new = fun(theta + step)
            cost_new = 0.5 * float(r_new @ r_new)
            if np.isfinite(cost_new) and cost_new <= cost:
                theta = theta + step
                rel = (cost - cost_new) / max(cost, 1e-300)
                r, cost = r_new, cost_new
                mu = max(mu / 3.0, 1e-12)
                improved = True
                break
            mu *= 4.0
        history.append(cost)
        if not improved:
            converged = True  # no downhill step at any damping: at a minimum to working precision
            break
        J = jac(theta)
        if rel < ftol or float(step.abs().max()) < xtol:
            converged = True
            break
    return LMResult(theta, cost, it, converged, J, r, history)


# ── Identifiability ──────────────────────────────────────────────────────────


@dataclass
class IdentifiabilityReport:
    """Linearised uncertainty at the optimum: cov = s² (JᵀJ)⁻¹ with s the residual noise per
    point."""

    labels: list[str]
    raw_std: np.ndarray  # std of each raw parameter (log params: ≈ relative std)
    log_param: np.ndarray  # True where the raw parameter is a log
    eigenvalues: np.ndarray  # of the scaled JᵀJ, ascending
    weakest: list[tuple[float, list[tuple[str, float]]]]  # (eigenvalue, top components)
    condition: float

    def identifiable(self, rel_tol: float = 0.05) -> np.ndarray:
        """Mask of parameters pinned to better than ``rel_tol`` (relative for log params, absolute
        else)."""
        return self.raw_std < rel_tol

    def table(self, values: dict[str, Tensor] | None = None) -> str:
        lines = [f"{'parameter':<18}{'value':>12}{'±1σ':>12}"]
        flat = None
        if values is not None:
            names = _names_from_labels(self.labels)
            flat = [v for n in names for v in values[n].tolist()]
        for i, lab in enumerate(self.labels):
            v = "" if flat is None else f"{flat[i]:12.5g}"
            sd = (
                f"{100 * self.raw_std[i]:10.3g} %"
                if self.log_param[i]
                else f"{self.raw_std[i]:12.3g}"
            )
            lines.append(f"{lab:<18}{v:>12}{sd:>12}")
        lines.append(f"condition number (scaled JᵀJ): {self.condition:.3g}")
        for ev, comps in self.weakest:
            terms = " ".join(f"{c:+.2f}·{n}" for n, c in comps)
            lines.append(f"weak direction (eig {ev:.3g}): {terms}")
        return "\n".join(lines)


def _names_from_labels(labels: list[str]) -> list[str]:
    names: list[str] = []
    for lab in labels:
        n = lab.split("[")[0]
        if not names or names[-1] != n:
            names.append(n)
    return names


def identifiability(J: Tensor, names: tuple[str, ...], noise_std: float, n_weak: int = 3):
    """Fisher-information report for raw parameters ``names`` given residual noise ``noise_std``.

    ``noise_std`` is the std of one residual element (nepers for magnitude), i.e. the measurement
    noise the report should assume; pass the fit's own RMS residual for a real-data fit.
    """
    J = J.detach().numpy()
    A = J.T @ J
    scale = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    As = A / np.outer(scale, scale)
    evals, evecs = np.linalg.eigh(As)
    inv = (evecs / np.clip(evals, 1e-300, None)) @ evecs.T
    cov = noise_std**2 * inv / np.outer(scale, scale)
    labels = raw_labels(names)
    log_param = np.array([SPECS_BY_NAME[lab.split("[")[0]].log for lab in labels])
    weakest = []
    for k in range(min(n_weak, len(evals))):
        v = evecs[:, k]
        top = np.argsort(-np.abs(v))[:4]
        weakest.append((float(evals[k]), [(labels[i], float(v[i])) for i in top]))
    return IdentifiabilityReport(
        labels=labels,
        raw_std=np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        log_param=log_param,
        eigenvalues=evals,
        weakest=weakest,
        condition=float(evals[-1] / max(evals[0], 1e-300)),
    )


# ── Fits ─────────────────────────────────────────────────────────────────────


@dataclass
class RatioData:
    """Measured engaged ÷ bypassed responses: sliders (N, 8) panel dB, freqs (F,), H (N, F)
    complex."""

    sliders: Tensor
    freqs_hz: Tensor
    H: Tensor
    weight: Tensor | None = None  # (N, F) or (F,), e.g. from coherence


@dataclass
class FitResult:
    params: dict[str, Tensor]
    names: tuple[str, ...]
    lm: LMResult
    rms_db: float  # RMS log-magnitude error over all points, dB
    rms_phase_deg: float

    def report(self, noise_std: float | None = None) -> IdentifiabilityReport:
        """Identifiability at this optimum; default noise = the fit's own RMS residual."""
        if noise_std is None:
            r = self.lm.residual
            noise_std = float(torch.sqrt((r @ r) / max(r.numel() - self.lm.theta.numel(), 1)))
        return identifiability(self.lm.jacobian, self.names, noise_std)


def _fit(residual_fn, names, init, *, max_iter) -> tuple[dict[str, Tensor], LMResult]:
    theta0 = pack(init, names)
    lm = levenberg_marquardt(residual_fn, theta0, max_iter=max_iter)
    return unpack(lm.theta, names, init), lm


def _rms(r: Tensor) -> tuple[float, float]:
    half = r.numel() // 2
    mag, ph = r[:half], r[half:]
    return (
        float(torch.sqrt((mag**2).mean())) * 20.0 / np.log(10.0),
        float(torch.sqrt((ph**2).mean())) * 180.0 / np.pi,
    )


def fit_ratio(
    data: RatioData,
    init: dict[str, Tensor],
    *,
    names: tuple[str, ...] = RATIO_NAMES,
    phase_weight: float = 1.0,
    max_iter: int = 200,
) -> FitResult:
    """Fit LEVEL + EQ parameters ``names`` to engaged ÷ bypassed data; others stay at ``init``."""
    s = blocks.s_axis(data.freqs_hz)

    def residual(theta: Tensor) -> Tensor:
        p = unpack(theta, names, init)
        h = ratio_response(p, data.sliders, s)
        return log_ratio_residual(h, data.H, data.weight, phase_weight)

    params, lm = _fit(residual, names, init, max_iter=max_iter)
    rms_db, rms_ph = _rms(residual(lm.theta) if phase_weight != 1.0 else lm.residual)
    return FitResult(params, names, lm, rms_db, rms_ph)


def fit_bypass(
    freqs_hz: Tensor,
    H: Tensor,
    init: dict[str, Tensor],
    *,
    names: tuple[str, ...] = BYPASS_NAMES,
    weight: Tensor | None = None,
    max_iter: int = 200,
) -> FitResult:
    """Fit the input/output blocks to bypassed ÷ chain data (F,) complex."""
    s = blocks.s_axis(freqs_hz)

    def residual(theta: Tensor) -> Tensor:
        p = unpack(theta, names, init)
        return log_ratio_residual(bypass_response(p, s), H, weight, 1.0)

    params, lm = _fit(residual, names, init, max_iter=max_iter)
    rms_db, rms_ph = _rms(lm.residual)
    return FitResult(params, names, lm, rms_db, rms_ph)


# ── Settings for known-answer runs (mirrors the R2 capture plan) ─────────────


def a0_settings(n_random: int = 24, seed: int = 0) -> Tensor:
    """All-centre, the slider map (each slider alone at ±15, ±7.5), adjacent-band pairs (both up,
    and opposite), and random settings. Shape (N, 8) panel dB."""
    rows = [[0.0] * len(SLIDER_KEYS)]
    for i in range(len(SLIDER_KEYS)):
        for v in (15.0, 7.5, -7.5, -15.0):
            row = [0.0] * len(SLIDER_KEYS)
            row[i] = v
            rows.append(row)
    for i in range(6):
        for a, b in ((15.0, 15.0), (15.0, -15.0)):
            row = [0.0] * len(SLIDER_KEYS)
            row[i], row[i + 1] = a, b
            rows.append(row)
    rng = np.random.default_rng(seed)
    rows += rng.uniform(-15.0, 15.0, size=(n_random, len(SLIDER_KEYS))).tolist()
    return torch.tensor(rows, dtype=torch.float64)


def log_grid(f_lo: float = 20.0, f_hi: float = 19000.0, n: int = 200) -> Tensor:
    return torch.logspace(np.log10(f_lo), np.log10(f_hi), n, dtype=torch.float64)
