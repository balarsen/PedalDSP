"""Training the GE-7 block model on a dataset (pedal_model/ode/ge7_train.py): split, recovery,
prior, history and the parameter file. Synthetic datasets from the model itself (no audio)."""

from __future__ import annotations

import json

import numpy as np
import torch

from pedal_model.ode import ge7_fitter as F
from pedal_model.ode import ge7_model as M
from pedal_model.ode import ge7_train as T
from pedal_model.rig.profiles import load_profile

PROFILE = load_profile("ge7").raw


def _truth() -> dict:
    """A 'pedal' near the schematic: parts within tolerance, and a G-taper pot law (γ < 1)."""
    p = M.schematic_values()
    p["band_f"] = p["band_f"] * torch.tensor(
        [1.03, 0.97, 1.02, 0.98, 1.04, 0.99], dtype=torch.float64
    )
    p["band_q"] = p["band_q"] * 1.1
    p["eq_gamma"] = torch.tensor([0.4], dtype=torch.float64)
    p["ratio_gain"] = torch.tensor([0.85], dtype=torch.float64)
    return p


def _dataset(settings: list[tuple[str, dict]], truth: dict) -> dict:
    freqs = F.log_grid()
    h = M.ratio_response(truth, freqs, M.sliders_tensor([c for _, c in settings])).numpy()
    takes = [
        {
            "preset": name,
            "role": "sweep",
            "controls": c,
            "h_re": h[i].real.tolist(),
            "h_im": h[i].imag.tolist(),
            "coherence": [1.0] * len(freqs),
        }
        for i, (name, c) in enumerate(settings)
    ]
    return {"freqs_hz": freqs.tolist(), "takes": takes}


def _record_like() -> list[tuple[str, dict]]:
    """Slider map at ±15/±10, the 6.4k and LEVEL included, plus the profile's random settings."""
    out = [("flat", {})]
    for k in M.SLIDER_KEYS:
        for v in (15.0, 10.0, -10.0, -15.0):
            out.append((f"map_{k}_{v:+g}", {k: v}))
    prof = load_profile("ge7")
    out += [(p.name, p.controls) for p in prof.random_presets("fit")]
    out += [(p.name, p.controls) for p in prof.random_presets("val")]
    out += [("smile", prof.preset("smile").controls), ("all_up_pad15", {"level": 15.0})]
    return out


def test_split_puts_each_take_in_its_group():
    ds = _dataset(_record_like(), M.schematic_values())
    g = T.groups(ds, PROFILE)
    names = lambda key: {ds["takes"][i]["preset"] for i in g[key]}  # noqa: E731
    assert names("test") == {"smile"}
    assert names("excluded") == {"all_up_pad15"}
    assert names("validation") == {f"rand_val_{i:02d}" for i in range(1, 9)}
    assert "flat" in names("fit") and "rand_fit_01" in names("fit")


def test_training_recovers_a_pedal_near_the_schematic():
    truth = _truth()
    fit = T.fit_dataset(_dataset(_record_like(), truth), PROFILE)
    assert fit["rms_fit_db"] < 0.05 and fit["rms_validation_db"] < 0.05
    got = fit["params"]
    np.testing.assert_allclose(got["band_f"].numpy(), truth["band_f"].numpy(), rtol=0.01)
    assert abs(float(got["eq_gamma"]) - 0.4) < 0.02
    assert abs(float(got["ratio_gain"]) - 0.85) < 0.01


def test_one_take_leaves_the_parameters_at_the_schematic_prior():
    """flat alone says almost nothing about the bands: they stay at the schematic, as uncertain
    as the prior, while the overall gain is pinned at once."""
    fit = T.fit_dataset(_dataset([("flat", {})], _truth()), PROFILE)
    sch = M.schematic_values()
    np.testing.assert_allclose(fit["params"]["band_f"].numpy(), sch["band_f"].numpy(), rtol=1e-3)
    std = dict(zip(fit["labels"], fit["std"]))
    assert std["band_f[3]"] > 0.9 * M.PRIOR_SIGMA["band_f"]
    assert std["ratio_gain"] < 0.2 * M.PRIOR_SIGMA["ratio_gain"]


def test_history_and_parameter_file(tmp_path, monkeypatch):
    import pedal_model.rig.qc as qc

    monkeypatch.setattr(T, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(qc, "GE7_MODELS_DIR", tmp_path / "models")
    fit = T.fit_dataset(_dataset(_record_like(), _truth()), PROFILE)
    hist = T.append_history("s1", fit)
    hist = T.append_history("s1", fit)
    assert len(hist) == 2 and hist[-1]["n_fit"] == fit["n_fit"]
    assert json.loads(T.history_path("s1").read_text())[0]["params"]["band_f"]
    p1 = T.write_parameter_file(fit, ["s1"], tmp_path / "models")
    p2 = T.write_parameter_file(fit, ["s1"], tmp_path / "models")
    assert (p1.name, p2.name) == ("ge7_pedalode_v1.json", "ge7_pedalode_v2.json")
    model = qc.expected_model("boss_ge7")  # the QC hook now finds the trained model
    assert model["name"] == "ge7_pedalode_v2" and model["fitted"]
    grid = np.geomspace(20, 19_000, 50)
    db = qc.expected_response("boss_ge7", {"hz_800": 10.0}, 48_000, grid)
    h = M.ratio_response(fit["params"], torch.as_tensor(grid), M.sliders_tensor([{"hz_800": 10.0}]))
    np.testing.assert_allclose(db, 20 * np.log10(np.abs(h[0].numpy())), atol=1e-6)


def test_schematic_band_centres_land_on_the_panel_labels():
    """The component values as read from the schematic give the panel's band centres (within
    5 %) and a 4.13 kHz first-order 6.4k branch: a check on the reading."""
    p = M.schematic_values()
    np.testing.assert_allclose(p["band_f"].numpy(), M.PANEL_BAND_HZ, rtol=0.05)
    assert abs(float(p["shelf_f"]) - 4130) < 10
    assert abs(float(p["lvl_f"]) - 7.23) < 0.01
