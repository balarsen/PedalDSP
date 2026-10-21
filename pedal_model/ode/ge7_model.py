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


def theta_labels(names: tuple[str, ...]) -> list[str]:
    """One label per element of θ, e.g. ``band_f[2]``."""
    labels = []
    for n in names:
        size = PARAMS[n][0]
        labels += [n] if size == 1 else [f"{n}[{k}]" for k in range(size)]
    return labels


def wiper(slider_db: Tensor, gamma: Tensor) -> Tensor:
    """Panel position (−15…+15) → wiper x in [0, 1]; +15 = toward the feedback node = boost."""
    u = (slider_db + SLIDER_RANGE_DB) / (2.0 * SLIDER_RANGE_DB)
    return blocks.pot_law(u.clamp(0.0, 1.0), gamma)


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
    shelf = blocks.shelf1(s, p["shelf_f"]).unsqueeze(0)  # (1, F)
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
