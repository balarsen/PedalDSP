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

from dataclasses import dataclass

import torch
from torch import Tensor

from pedal_model.ode import blocks

SLIDER_KEYS: tuple[str, ...] = (
    "hz_100", "hz_200", "hz_400", "hz_800", "hz_1600", "hz_3200", "hz_6400", "level",
)  # fmt: skip
PANEL_BAND_HZ: tuple[float, ...] = (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
SLIDER_RANGE_DB = 15.0


@dataclass(frozen=True)
class ParamSpec:
    """One block parameter: name, element count, whether it is fitted in log space, unit,
    meaning."""

    name: str
    size: int
    log: bool
    unit: str
    doc: str


# engaged ÷ bypassed (LEVEL + EQ). The LEVEL pot shares the EQ pots' end resistance (decision: the
# LEVEL data alone, 5 positions of one scalar gain, cannot pin a separate one).
RATIO_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("band_f", 6, True, "Hz", "band-pass branch centre frequencies, bands 1-6"),
    ParamSpec("band_q", 6, True, "", "band-pass branch Q (branch alone, before pot loading)"),
    ParamSpec("band_sigma", 6, True, "1/R_pot", "band-pass branch peak admittance"),
    ParamSpec("shelf_f", 1, True, "Hz", "band 7 shelf branch corner (H7a: s/(s+w))"),
    ParamSpec("shelf_sigma", 1, True, "1/R_pot", "band 7 shelf branch high-frequency admittance"),
    ParamSpec("eq_rho_in", 1, True, "R_pot", "EQ input resistor / pot resistance (cut)"),
    ParamSpec("eq_rho_fb", 1, True, "R_pot", "EQ feedback resistor / pot resistance (boost)"),
    ParamSpec("pot_end", 1, True, "R_pot", "end resistance of each pot track / pot resistance"),
    ParamSpec("eq_gamma", 1, True, "", "EQ slider pot law exponent (1 = linear)"),
    ParamSpec("lvl_kappa", 1, True, "1/R_pot", "LEVEL branch admittance"),
    ParamSpec("lvl_f", 1, True, "Hz", "LEVEL wiper RC corner"),
    ParamSpec("lvl_rho_in", 1, True, "R_pot", "LEVEL stage input ratio (sets max cut)"),
    ParamSpec("lvl_rho_fb", 1, True, "R_pot", "LEVEL stage feedback ratio (sets max boost)"),
    ParamSpec("lvl_gamma", 1, True, "", "LEVEL pot law exponent"),
    ParamSpec("ratio_gain", 1, True, "", "engaged/bypass gain at all-centre, apart from LEVEL"),
    ParamSpec("ratio_delay_us", 1, False, "us", "residual delay of the engaged path vs bypass"),
)

# bypassed ÷ chain (input and output blocks, both paths).
BYPASS_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("bypass_gain", 1, True, "", "low-frequency gain, bypassed pedal vs a cable"),
    ParamSpec("in_fz", 1, True, "Hz", "input stage pre-emphasis zero"),
    ParamSpec("out_fp", 1, True, "Hz", "output stage de-emphasis pole"),
    ParamSpec("in_fhp", 1, True, "Hz", "input coupling high-pass"),
    ParamSpec("out_fhp", 1, True, "Hz", "output coupling high-pass"),
)

ALL_SPECS: tuple[ParamSpec, ...] = RATIO_SPECS + BYPASS_SPECS
SPECS_BY_NAME: dict[str, ParamSpec] = {p.name: p for p in ALL_SPECS}


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
    """Physical parameters → one raw vector (log for positive parameters)."""
    parts = []
    for n in names:
        v = params[n].to(torch.float64).reshape(-1)
        parts.append(torch.log(v) if SPECS_BY_NAME[n].log else v)
    return torch.cat(parts)


def unpack(theta: Tensor, names: tuple[str, ...], base: dict[str, Tensor]) -> dict[str, Tensor]:
    """Raw vector → physical parameters; parameters not in ``names`` come from ``base``."""
    out = dict(base)
    i = 0
    for n in names:
        spec = SPECS_BY_NAME[n]
        v = theta[i : i + spec.size]
        out[n] = torch.exp(v) if spec.log else v
        i += spec.size
    return out


def raw_labels(names: tuple[str, ...]) -> list[str]:
    """One label per raw-vector element, e.g. ``band_f[2]``."""
    labels = []
    for n in names:
        size = SPECS_BY_NAME[n].size
        labels += [n] if size == 1 else [f"{n}[{k}]" for k in range(size)]
    return labels


def wiper(slider_db: Tensor, gamma: Tensor) -> Tensor:
    """Panel position (−15…+15) → wiper x in [0, 1]; +15 = toward the feedback node = boost."""
    u = (slider_db + SLIDER_RANGE_DB) / (2.0 * SLIDER_RANGE_DB)
    return blocks.pot_law(u.clamp(0.0, 1.0), gamma)


def ratio_response(p: dict[str, Tensor], sliders: Tensor, s: Tensor) -> Tensor:
    """Engaged ÷ bypassed, shape (N, F) complex, for sliders (N, 8) in panel dB and s (F,)."""
    sliders = sliders.to(torch.float64)
    bands = blocks.bandpass2(s, p["band_f"].unsqueeze(-1), p["band_q"].unsqueeze(-1))
    shelf = blocks.shelf1(s, p["shelf_f"]).unsqueeze(0)
    sigma = torch.cat([p["band_sigma"], p["shelf_sigma"]]).unsqueeze(-1)
    branches = sigma * torch.cat([bands, shelf], dim=0)  # (7, F)
    x_eq = wiper(sliders[:, :7], p["eq_gamma"])
    h_eq = blocks.pot_split_stage(x_eq, branches, p["eq_rho_in"], p["eq_rho_fb"], p["pot_end"])

    y_lvl = (p["lvl_kappa"] * blocks.highpass1(s, p["lvl_f"])).unsqueeze(0)  # (1, F)
    x_lvl = wiper(sliders[:, 7:8], p["lvl_gamma"])
    h_lvl = blocks.pot_split_stage(x_lvl, y_lvl, p["lvl_rho_in"], p["lvl_rho_fb"], p["pot_end"])

    delay = torch.exp(-s * p["ratio_delay_us"] * 1e-6)
    return p["ratio_gain"] * delay * h_lvl * h_eq


def bypass_response(p: dict[str, Tensor], s: Tensor) -> Tensor:
    """Bypassed ÷ chain, shape (F,) complex: the input and output blocks."""
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


class Ge7BlockModel:
    """A GE-7 parameter set and its responses on a frequency grid."""

    def __init__(self, params: dict[str, Tensor] | None = None):
        self.params = {k: v.clone() for k, v in (params or initial_guesses()).items()}

    def ratio(self, freqs_hz: Tensor, sliders: Tensor) -> Tensor:
        return ratio_response(self.params, sliders, blocks.s_axis(freqs_hz))

    def bypass(self, freqs_hz: Tensor) -> Tensor:
        return bypass_response(self.params, blocks.s_axis(freqs_hz))

    def engaged(self, freqs_hz: Tensor, sliders: Tensor) -> Tensor:
        """The whole pedal, engaged, relative to a cable."""
        return self.bypass(freqs_hz).unsqueeze(0) * self.ratio(freqs_hz, sliders)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: v.tolist() for k, v in self.params.items()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> Ge7BlockModel:
        return cls({k: _t(v) for k, v in d.items()})
