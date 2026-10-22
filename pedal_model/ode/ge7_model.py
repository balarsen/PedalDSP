"""Boss GE-7 as its block diagram, with physical block parameters.

Plan: recordings/data/ge7/GE7_PEDALODE_PLAN.md.

    input stage ─► LEVEL ─► EQ summing stage (7 branches on slider pots) ─► output stage
    (gain, 2.2 kHz zero, HP)  (pot split)                             (2.2 kHz pole, HP)

Two responses are modelled, matching the two references the rig measures:

- ``ratio``: engaged ÷ bypassed = gain · delay · H_level · H_eq. The input/output blocks and the
  capture chain cancel, so this is what the slider data constrains.
- ``bypass``: bypassed ÷ chain = gain · zero(f_z) · pole(f_p) · HP(f_hp_in) · HP(f_hp_out). The
  input stage's ×10 and the output divider are known only as one product, so they are one gain here.

The engaged pedal is ``bypass · ratio``. Parameters are continuous-time (Hz, Q, ratios), so one
parameter set serves any sample rate (Daisy at 48 kHz, a plugin at 44.1–192 kHz).

Sliders are given as the panel's dB labels, −15…+15, in the order ``SLIDER_KEYS`` (the rig profile's
control keys). The labels are only positions: what a slider does in dB is what the model predicts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pedal_model.ode import blocks

SLIDER_KEYS: tuple[str, ...] = (
    "hz_100", "hz_200", "hz_400", "hz_800", "hz_1600", "hz_3200", "hz_6400", "level",
)  # fmt: skip
PANEL_BAND_HZ: tuple[float, ...] = (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
SLIDER_RANGE_DB = 15.0

# Every block parameter: name -> (number of values, fitted as log?, unit, meaning).
# Positive parameters are fitted as their logarithm, so a step in the fit is a relative change.
PARAMS: dict[str, tuple[int, bool, str, str]] = {
    # engaged ÷ bypassed: LEVEL + EQ. The LEVEL pot shares the EQ pots' end resistance (decision:
    # the LEVEL data alone, 5 positions of one scalar gain, cannot pin a separate one).
    "band_f": (6, True, "Hz", "band-pass branch centre frequencies, bands 1-6"),
    "band_q": (6, True, "", "band-pass branch Q (branch alone, before pot loading)"),
    "band_sigma": (6, True, "1/R_pot", "band-pass branch peak admittance"),
    "shelf_f": (1, True, "Hz", "band 7 shelf branch corner (H7a: s/(s+w))"),
    "shelf_sigma": (1, True, "1/R_pot", "band 7 shelf branch high-frequency admittance"),
    "eq_rho_in": (1, True, "R_pot", "EQ input resistor / pot resistance (cut)"),
    "eq_rho_fb": (1, True, "R_pot", "EQ feedback resistor / pot resistance (boost)"),
    "pot_end": (1, True, "R_pot", "end resistance of each pot track / pot resistance"),
    "eq_gamma": (1, True, "", "EQ slider pot law exponent (1 = linear)"),
    "lvl_kappa": (1, True, "1/R_pot", "LEVEL branch admittance"),
    "lvl_f": (1, True, "Hz", "LEVEL wiper RC corner"),
    "lvl_rho_in": (1, True, "R_pot", "LEVEL stage input ratio (sets max cut)"),
    "lvl_rho_fb": (1, True, "R_pot", "LEVEL stage feedback ratio (sets max boost)"),
    "lvl_gamma": (1, True, "", "LEVEL pot law exponent"),
    "ratio_gain": (1, True, "", "engaged/bypass gain at all-centre, apart from LEVEL"),
    "ratio_delay_us": (1, False, "us", "residual delay of the engaged path vs bypass"),
    # bypassed ÷ chain: the input and output blocks, in both paths.
    "bypass_gain": (1, True, "", "low-frequency gain, bypassed pedal vs a cable"),
    "in_fz": (1, True, "Hz", "input stage pre-emphasis zero"),
    "out_fp": (1, True, "Hz", "output stage de-emphasis pole"),
    "in_fhp": (1, True, "Hz", "input coupling high-pass"),
    "out_fhp": (1, True, "Hz", "output coupling high-pass"),
}
BYPASS_NAMES: tuple[str, ...] = ("bypass_gain", "in_fz", "out_fp", "in_fhp", "out_fhp")
RATIO_NAMES: tuple[str, ...] = tuple(n for n in PARAMS if n not in BYPASS_NAMES)


def _t(values) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float64).reshape(-1)


def initial_guesses() -> dict[str, Tensor]:
    """Starting values (plan, "Parameters and starting guesses"). Guesses, not measurements.

    EQ strengths: the neighbouring branches load both nodes, so a band's boost is (1+A+B)/(1+A), not
    1+B. rho 40 and sigma 1.0 (pot_end 0.01, linear pots) were chosen so the middle bands give about
    ±15 dB at full and ±8.5 dB at ±10, the coarse Studio One-era slider readings (not re-verified).
    LEVEL: rho_in 0.914 / rho_fb 1.545 with kappa 4 give +14 / −15 dB at the ends (the teardown's
    figures); ratio_gain 0.753 makes all-centre 0 dB.
    """
    return {
        "band_f": _t(PANEL_BAND_HZ),
        "band_q": _t([2.0] * 6),
        "band_sigma": _t([1.0] * 6),
        "shelf_f": _t(6400.0),
        "shelf_sigma": _t(1.0),
        "eq_rho_in": _t(40.0),
        "eq_rho_fb": _t(40.0),
        "pot_end": _t(0.01),
        "eq_gamma": _t(1.0),
        "lvl_kappa": _t(4.0),
        "lvl_f": _t(7.2),
        "lvl_rho_in": _t(0.914),
        "lvl_rho_fb": _t(1.545),
        "lvl_gamma": _t(1.0),
        "ratio_gain": _t(0.753),
        "ratio_delay_us": _t(0.0),
        "bypass_gain": _t(1.0),
        "in_fz": _t(2200.0),
        "out_fp": _t(2200.0),
        "in_fhp": _t(10.0),
        "out_fhp": _t(5.0),
    }


# Component values from the Boss GE-7 schematic (as read 2026-10-22; the band centres they give,
# 96/204/394/790/1549/3303 Hz, land on the panel labels, which checks the reading). Ohms, farads.
R_POT = 10e3  # "10KG x 8": the eight slider pots (the G taper is not specified: fitted)
SCHEMATIC = {
    # band k: series cap, gyrator cap, gyrator large R, gyrator small R (series R of the branch)
    "bands": (
        (1.5e-6, 0.056e-6, 100e3, 330.0),  # 100 Hz: C10, C13, R16, R17
        (0.68e-6, 0.033e-6, 82e3, 330.0),  # 200 Hz: C9, C8, R9, R8
        (0.33e-6, 0.015e-6, 100e3, 330.0),  # 400 Hz: C7, C14, R18, R19
        (0.15e-6, 0.0082e-6, 100e3, 330.0),  # 800 Hz: C5, C6, R7, R6
        (0.1e-6, 0.0039e-6, 82e3, 330.0),  # 1.6 kHz: C4, C15, R21, R22
        (0.039e-6, 0.0022e-6, 82e3, 330.0),  # 3.2 kHz: C2, C3, R5, R4
    ),
    "shelf": (0.047e-6, 820.0),  # 6.4 kHz: C1 in series with R3 (a first-order RC branch)
    "eq_r_in": 3.3e3,  # R24
    "eq_r_fb": 3.3e3,  # R25
    "lvl_branch": (10e-6, 2.2e3),  # LEVEL wiper branch: C16, R2
    "lvl_r_in": 10e3,  # R27
    "lvl_r_fb": 10e3,  # R26
    "pre_emphasis": (0.015e-6, 4.7e3, 470.0),  # input stage C20, R31, R28
    "de_emphasis": (0.015e-6, 4.7e3, 470.0),  # output C27, R33, R36
    "in_coupling": (0.047e-6, 470e3),  # C22, R29
    "out_coupling": (0.047e-6, 1e6),  # C26, R34
}
# Prior widths for training from few settings: a relative 1-sigma for log parameters (component
# tolerances: ~10 % for frequencies and Qs from capacitor values, 5 % for resistor ratios), wide
# where the schematic says nothing (the pot taper, the end resistance, the engaged-path level
# that the schematic does not explain: flat reads -1.2 to -1.6 dB), absolute (us) for the delay.
PRIOR_SIGMA: dict[str, float] = {
    "band_f": 0.10,
    "band_q": 0.20,
    "band_sigma": 0.15,
    "shelf_f": 0.10,
    "shelf_sigma": 0.10,
    "eq_rho_in": 0.05,
    "eq_rho_fb": 0.05,
    "pot_end": 1.5,
    "eq_gamma": 1.0,
    "lvl_kappa": 0.10,
    "lvl_f": 0.25,
    "lvl_rho_in": 0.05,
    "lvl_rho_fb": 0.05,
    "lvl_gamma": 1.0,
    "ratio_gain": 0.5,
    "ratio_delay_us": 50.0,
    "bypass_gain": 0.5,
    "in_fz": 0.10,
    "out_fp": 0.10,
    "in_fhp": 0.5,
    "out_fhp": 0.5,
}


def schematic_values() -> dict[str, Tensor]:
    """Block parameters from the schematic's component values (units of R_POT for admittances).

    Gyrator band branch (series C_s, simulated inductor L = R_big·R_small·C_g, series R_small):

        L = R_big·R_small·C_g,   f = 1 / (2π√(L·C_s)),   Q = 2πf·L / R_small,   σ = R_pot / R_small

    6.4 kHz branch (C1 + R3 in series): admittance (1/R3)·s/(s + 1/(R3·C1)), so f = 1/(2π·R3·C1),
    σ = R_pot / R3 (hypothesis H7a). LEVEL branch likewise, κ = R_pot / R2. ρ = resistor / R_pot.
    Pre-emphasis 1 + s·R31·C20 / (1 + s·R28·C20) has its zero at 1/(2π(R31 + R28)C20); the
    output de-emphasis mirrors it. What the schematic does not give (the G taper, the end
    resistance, the engaged-path level, the delay) stays at the plain guesses.
    """
    c = SCHEMATIC
    f, q, sig = [], [], []
    for cs, cg, rb, rs in c["bands"]:
        ind = rb * rs * cg
        w0 = 1.0 / math.sqrt(ind * cs)
        f.append(w0 / (2 * math.pi))
        q.append(w0 * ind / rs)
        sig.append(R_POT / rs)
    cap, res = c["shelf"]
    lc, lr = c["lvl_branch"]
    pc, pr, pr2 = c["pre_emphasis"]
    dc, dr, dr2 = c["de_emphasis"]
    p = initial_guesses()
    p.update(
        {
            "band_f": _t(f),
            "band_q": _t(q),
            "band_sigma": _t(sig),
            "shelf_f": _t(1 / (2 * math.pi * res * cap)),
            "shelf_sigma": _t(R_POT / res),
            "eq_rho_in": _t(c["eq_r_in"] / R_POT),
            "eq_rho_fb": _t(c["eq_r_fb"] / R_POT),
            "eq_gamma": _t(1.0),
            "lvl_kappa": _t(R_POT / lr),
            "lvl_f": _t(1 / (2 * math.pi * lr * lc)),
            "lvl_rho_in": _t(c["lvl_r_in"] / R_POT),
            "lvl_rho_fb": _t(c["lvl_r_fb"] / R_POT),
            "lvl_gamma": _t(1.0),
            "ratio_gain": _t(1.0),
            "in_fz": _t(1 / (2 * math.pi * (pr + pr2) * pc)),
            "out_fp": _t(1 / (2 * math.pi * (dr + dr2) * dc)),
            "in_fhp": _t(1 / (2 * math.pi * c["in_coupling"][1] * c["in_coupling"][0])),
            "out_fhp": _t(1 / (2 * math.pi * c["out_coupling"][1] * c["out_coupling"][0])),
        }
    )
    return p


def prior_sigma_vector(names: tuple[str, ...]) -> Tensor:
    """PRIOR_SIGMA expanded to one value per element of θ (``pack`` order)."""
    return torch.cat([_t([PRIOR_SIGMA[n]] * PARAMS[n][0]) for n in names])


def pack(params: dict[str, Tensor], names: tuple[str, ...]) -> Tensor:
    """Physical parameters → one vector θ for the fitter (log of the positive ones)."""
    parts = []
    for n in names:
        v = params[n].to(torch.float64).reshape(-1)
        parts.append(torch.log(v) if PARAMS[n][1] else v)
    return torch.cat(parts)


def unpack(theta: Tensor, names: tuple[str, ...], base: dict[str, Tensor]) -> dict[str, Tensor]:
    """θ → physical parameters; parameters not in ``names`` are taken from ``base``."""
    out = dict(base)
    i = 0
    for n in names:
        size, is_log = PARAMS[n][0], PARAMS[n][1]
        v = theta[i : i + size]
        out[n] = torch.exp(v) if is_log else v
        i += size
    return out


def params_to_json(params: dict[str, Tensor]) -> dict[str, list[float]]:
    """Block parameters as plain lists (for the versioned parameter file)."""
    return {n: [float(v) for v in params[n].reshape(-1)] for n in PARAMS}


def params_from_json(values: dict[str, list[float]]) -> dict[str, Tensor]:
    """Block parameters from a parameter file; any missing name keeps its starting guess."""
    out = initial_guesses()
    out.update({n: _t(v) for n, v in values.items() if n in PARAMS})
    return out


def theta_labels(names: tuple[str, ...]) -> list[str]:
    """One label per element of θ, e.g. ``band_f[2]``."""
    labels = []
    for n in names:
        size = PARAMS[n][0]
        labels += [n] if size == 1 else [f"{n}[{k}]" for k in range(size)]
    return labels


def wiper(slider_db: Tensor, gamma: Tensor) -> Tensor:
    """Panel label (−15…+15 dB) → wiper x in [0, 1]; +15 = toward the feedback node = boost.

        v = slider / 15 (clamped to ±1),   x = ½ + ½·sign(v)·|v|^γ

    γ = 1 is a linear pot; γ > 1 keeps the wiper nearer the centre for small moves (gentler near
    the detent), γ < 1 the opposite. Symmetric, so the centre is always x = ½.
    """
    v = (slider_db / SLIDER_RANGE_DB).clamp(-1.0, 1.0)
    return 0.5 + 0.5 * torch.sign(v) * torch.abs(v).clamp_min(1e-12) ** gamma


def ratio_response(p: dict[str, Tensor], freqs_hz: Tensor, sliders: Tensor) -> Tensor:
    """Engaged ÷ bypassed: H = gain · delay · H_level · H_eq.

    Args:
        p: Parameters (``initial_guesses()`` layout).
        freqs_hz: Frequencies, shape (F,).
        sliders: Panel positions in dB, shape (N, 8), in ``SLIDER_KEYS`` order.

    Returns:
        H, shape (N, F) complex.
    """
    s = blocks.s_axis(freqs_hz)
    sliders = sliders.to(torch.float64)
    # EQ stage: six band-pass branches and the shelf branch, each on its slider's pot.
    bands = blocks.bandpass2(s, p["band_f"].unsqueeze(-1), p["band_q"].unsqueeze(-1))  # (6, F)
    shelf = blocks.highpass1(s, p["shelf_f"]).unsqueeze(0)  # shelf branch s/(s+ω): (1, F)
    sigma = torch.cat([p["band_sigma"], p["shelf_sigma"]]).unsqueeze(-1)
    branches = sigma * torch.cat([bands, shelf], dim=0)  # admittances y_k(s), (7, F)
    x_eq = wiper(sliders[:, :7], p["eq_gamma"])
    h_eq = blocks.pot_split_stage(x_eq, branches, p["eq_rho_in"], p["eq_rho_fb"], p["pot_end"])
    # LEVEL stage: the same pot-split idea with one branch (the wiper RC).
    y_lvl = (p["lvl_kappa"] * blocks.highpass1(s, p["lvl_f"])).unsqueeze(0)  # (1, F)
    x_lvl = wiper(sliders[:, 7:8], p["lvl_gamma"])
    h_lvl = blocks.pot_split_stage(x_lvl, y_lvl, p["lvl_rho_in"], p["lvl_rho_fb"], p["pot_end"])
    delay = torch.exp(-s * p["ratio_delay_us"] * 1e-6)
    return p["ratio_gain"] * delay * h_lvl * h_eq


def bypass_response(p: dict[str, Tensor], freqs_hz: Tensor) -> Tensor:
    """Bypassed ÷ chain, shape (F,) complex: gain · zero(f_z) · pole(f_p) · two coupling HPs."""
    s = blocks.s_axis(freqs_hz)
    return (
        p["bypass_gain"]
        * blocks.zero1(s, p["in_fz"])
        * blocks.lowpass1(s, p["out_fp"])
        * blocks.highpass1(s, p["in_fhp"])
        * blocks.highpass1(s, p["out_fhp"])
    ).reshape(-1)


def sliders_tensor(settings: list[dict[str, float]]) -> Tensor:
    """Rig-style slider dicts (missing keys = centre) → (N, 8) tensor in ``SLIDER_KEYS`` order."""
    return torch.tensor(
        [[float(d.get(k, 0.0)) for k in SLIDER_KEYS] for d in settings], dtype=torch.float64
    )
