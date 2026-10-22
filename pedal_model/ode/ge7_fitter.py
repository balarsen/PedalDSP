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

import numpy as np
import torch
from torch import Tensor

from pedal_model.ode.ge7_model import (
    BYPASS_NAMES,
    PARAMS,
    RATIO_NAMES,
    SLIDER_KEYS,
    bypass_response,
    pack,
    ratio_response,
    theta_labels,
    unpack,
)

NEPER_TO_DB = 20.0 / np.log(10.0)


def log_ratio_residual(h_model: Tensor, h_meas: Tensor, weight: Tensor | None, phase_weight: float):
    """Complex log of model ÷ measured as a real vector [w·log|r|, w·λ·angle(r)]."""
    lr = torch.log(h_model / h_meas)
    w = 1.0 if weight is None else weight
    return torch.cat([(w * lr.real).reshape(-1), (w * phase_weight * lr.imag).reshape(-1)])


def levenberg_marquardt(
    fun: Callable[[Tensor], Tensor],
    theta0: Tensor,
    *,
    max_iter: int = 200,
    ftol: float = 1e-15,
    xtol: float = 1e-12,
    mu0: float = 1e-3,
) -> dict:
    """Minimise ½‖r(θ)‖² with r = ``fun(θ)``: Levenberg–Marquardt with Marquardt's scaling.

    Each step solves (JᵀJ + μ·diag(JᵀJ)) δ = −Jᵀr. A step that lowers the cost is taken and μ
    shrinks (towards Gauss–Newton); otherwise μ grows (towards a short gradient step). J comes from
    forward-mode autodiff, exact to machine precision.

    Returns:
        dict with theta, cost (½Σr²), iterations, converged, jacobian and residual at the optimum.
    """
    jac = torch.func.jacfwd(fun)
    theta = theta0.detach().clone().to(torch.float64)
    r = fun(theta)
    cost = 0.5 * float(r @ r)
    mu = mu0
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
        if not improved:
            converged = True  # no downhill step at any damping: at a minimum to working precision
            break
        J = jac(theta)
        if rel < ftol or float(step.abs().max()) < xtol:
            converged = True
            break
    return {
        "theta": theta,
        "cost": cost,
        "iterations": it,
        "converged": converged,
        "jacobian": J,
        "residual": r,
    }


def identifiability(fit: dict, noise_std: float | None = None, n_weak: int = 3) -> dict:
    """How well the data pin each parameter, at a fit's optimum: covariance cov = s²·(JᵀJ)⁻¹.

    ``noise_std`` (s) is the std of one residual element (nepers for magnitude): the measurement
    noise to assume, or by default the fit's own residual, s² = Σr² / (n_residuals − n_params).
    JᵀJ is rescaled to unit diagonal before the eigen-decomposition, so the condition number and
    the weakest directions do not depend on the parameters' units.

    Args:
        fit: A ``fit_ratio`` / ``fit_bypass`` result (uses jacobian, residual, theta, names).

    Returns:
        dict: labels (one per θ element), std (1σ of each θ element: for log parameters that is the
        relative uncertainty), is_log, condition, weakest [(eigenvalue, [(label, weight), ...])].
    """
    if noise_std is None:
        r, n_par = fit["residual"], fit["theta"].numel()
        noise_std = float(torch.sqrt((r @ r) / max(r.numel() - n_par, 1)))
    J = fit["jacobian"].detach().numpy()
    A = J.T @ J
    scale = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    evals, evecs = np.linalg.eigh(A / np.outer(scale, scale))
    inv = (evecs / np.clip(evals, 1e-300, None)) @ evecs.T
    cov = noise_std**2 * inv / np.outer(scale, scale)
    labels = theta_labels(fit["names"])
    weakest = []
    for k in range(min(n_weak, len(evals))):
        v = evecs[:, k]
        top = np.argsort(-np.abs(v))[:4]
        weakest.append((float(evals[k]), [(labels[i], float(v[i])) for i in top]))
    return {
        "labels": labels,
        "std": np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        "is_log": np.array([PARAMS[lab.split("[")[0]][1] for lab in labels]),
        "condition": float(evals[-1] / max(evals[0], 1e-300)),
        "weakest": weakest,
    }


def report_table(rep: dict, theta: Tensor | None = None) -> str:
    """The identifiability report as text: θ value, 1σ (% for log parameters), weak directions."""
    lines = [f"{'parameter':<18}{'value':>12}{'±1σ':>12}"]
    for i, lab in enumerate(rep["labels"]):
        value = ""
        if theta is not None:
            t = float(theta[i])
            value = f"{np.exp(t) if rep['is_log'][i] else t:12.5g}"
        sd = rep["std"][i]
        sd_text = f"{100 * sd:10.3g} %" if rep["is_log"][i] else f"{sd:12.3g}"
        lines.append(f"{lab:<18}{value:>12}{sd_text:>12}")
    lines.append(f"condition number (scaled JᵀJ): {rep['condition']:.3g}")
    for ev, comps in rep["weakest"]:
        terms = " ".join(f"{c:+.2f}·{n}" for n, c in comps)
        lines.append(f"weak direction (eig {ev:.3g}): {terms}")
    return "\n".join(lines)


def fit_ratio(
    sliders: Tensor,
    freqs_hz: Tensor,
    H: Tensor,
    init: dict[str, Tensor],
    *,
    names: tuple[str, ...] = RATIO_NAMES,
    weight: Tensor | None = None,
    phase_weight: float = 1.0,
    max_iter: int = 200,
) -> dict:
    """Fit the LEVEL + EQ parameters to measured engaged ÷ bypassed responses.

    Args:
        sliders: Panel positions, (N, 8) dB, one row per measured setting.
        freqs_hz: Frequency grid, (F,).
        H: Measured responses, (N, F) complex.
        init: Starting parameters; those not in ``names`` stay fixed at these values.
        weight: Optional per-point weight, (N, F) or (F,) (e.g. from coherence).
        phase_weight: Weight of the phase residual relative to log-magnitude.

    Returns:
        The ``levenberg_marquardt`` dict plus params (fitted, physical units), names, rms_db and
        rms_phase_deg (unweighted RMS errors over all points).
    """

    def residual(theta: Tensor) -> Tensor:
        p = unpack(theta, names, init)
        return log_ratio_residual(ratio_response(p, freqs_hz, sliders), H, weight, phase_weight)

    fit = levenberg_marquardt(residual, pack(init, names), max_iter=max_iter)
    fit["params"] = unpack(fit["theta"], names, init)
    fit["names"] = names
    err = torch.log(ratio_response(fit["params"], freqs_hz, sliders) / H)
    fit["rms_db"] = float(torch.sqrt((err.real**2).mean())) * NEPER_TO_DB
    fit["rms_phase_deg"] = float(torch.sqrt((err.imag**2).mean())) * 180.0 / np.pi
    return fit


def fit_bypass(
    freqs_hz: Tensor,
    H: Tensor,
    init: dict[str, Tensor],
    *,
    names: tuple[str, ...] = BYPASS_NAMES,
    weight: Tensor | None = None,
    max_iter: int = 200,
) -> dict:
    """Fit the input/output blocks to the measured bypassed ÷ chain response H, (F,) complex."""

    def residual(theta: Tensor) -> Tensor:
        p = unpack(theta, names, init)
        return log_ratio_residual(bypass_response(p, freqs_hz), H, weight, 1.0)

    fit = levenberg_marquardt(residual, pack(init, names), max_iter=max_iter)
    fit["params"] = unpack(fit["theta"], names, init)
    fit["names"] = names
    err = torch.log(bypass_response(fit["params"], freqs_hz) / H)
    fit["rms_db"] = float(torch.sqrt((err.real**2).mean())) * NEPER_TO_DB
    fit["rms_phase_deg"] = float(torch.sqrt((err.imag**2).mean())) * 180.0 / np.pi
    return fit


# ── Settings for known-answer runs (mirrors the R2 capture plan) ─────────────


def a0_settings(n_random: int = 24, seed: int = 0) -> Tensor:
    """All-centre, the slider map (each slider alone at ±15, ±10), adjacent-band pairs (both up,
    and opposite), and random settings. Shape (N, 8) panel dB."""
    rows = [[0.0] * len(SLIDER_KEYS)]
    for i in range(len(SLIDER_KEYS)):
        for v in (15.0, 10.0, -10.0, -15.0):
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
