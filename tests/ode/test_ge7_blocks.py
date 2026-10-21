"""GE-7 block model and fitter: structural properties and A0 known-answer recovery (no hardware)."""

import numpy as np
import pytest
import torch

from pedal_model.ode import blocks
from pedal_model.ode import ge7_fitter as F
from pedal_model.ode import ge7_model as M

NOISE_NP = 0.05 / 8.686  # 0.05 dB, in nepers


def _db(h):
    return 20.0 * torch.log10(h.abs())


def _symmetric():
    """Guesses with equal input/feedback ratios, so centre is flat and boost/cut mirror."""
    p = M.initial_guesses()
    p["lvl_rho_in"] = p["lvl_rho_fb"].clone()
    p["ratio_gain"] = torch.ones(1, dtype=torch.float64)
    return p


def _truth(seed=1):
    """A 'pedal' drawn around the guesses (the fitter starts from the guesses)."""
    rng = np.random.default_rng(seed)
    g = M.initial_guesses()
    t = dict(g)
    t["band_f"] = g["band_f"] * torch.tensor(rng.uniform(0.85, 1.15, 6))
    t["band_q"] = torch.tensor(rng.uniform(1.2, 3.0, 6))
    t["band_sigma"] = g["band_sigma"] * torch.tensor(rng.uniform(0.7, 1.3, 6))
    t["shelf_f"] = g["shelf_f"] * float(rng.uniform(0.8, 1.25))
    t["shelf_sigma"] = g["shelf_sigma"] * float(rng.uniform(0.7, 1.3))
    for k in ("eq_rho_in", "eq_rho_fb", "lvl_rho_in", "lvl_rho_fb", "lvl_kappa"):
        t[k] = g[k] * float(rng.uniform(0.8, 1.25))
    t["pot_end"] = torch.tensor([rng.uniform(0.005, 0.03)], dtype=torch.float64)
    t["eq_gamma"] = torch.tensor([rng.uniform(0.8, 1.5)], dtype=torch.float64)
    t["lvl_gamma"] = torch.tensor([rng.uniform(0.8, 1.5)], dtype=torch.float64)
    t["lvl_f"] = g["lvl_f"] * float(rng.uniform(0.7, 1.4))
    t["ratio_gain"] = torch.tensor([rng.uniform(0.9, 1.1)], dtype=torch.float64)
    t["ratio_delay_us"] = torch.tensor([rng.uniform(-20.0, 20.0)], dtype=torch.float64)
    t["bypass_gain"] = g["bypass_gain"] * float(rng.uniform(0.8, 1.25))
    t["in_fz"] = g["in_fz"] * float(rng.uniform(0.8, 1.25))
    t["out_fp"] = g["out_fp"] * float(rng.uniform(0.8, 1.25))
    t["in_fhp"] = g["in_fhp"] * float(rng.uniform(0.6, 1.6))
    t["out_fhp"] = g["out_fhp"] * float(rng.uniform(0.6, 1.6))
    return t


def _one(slider: str, value: float):
    return M.sliders_tensor([{slider: value}])


class TestBlocks:
    def test_bandpass_peak_is_one_at_fc(self):
        s = blocks.s_axis(torch.tensor([800.0]))
        h = blocks.bandpass2(s, torch.tensor(800.0, dtype=torch.float64), torch.tensor(2.0))
        assert abs(complex(h[0]) - 1.0) < 1e-12

    def test_pot_law_centre_and_ends(self):
        u = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        for gamma in (0.7, 1.0, 1.6):
            x = blocks.pot_law(u, torch.tensor(gamma, dtype=torch.float64))
            assert torch.allclose(x, u, atol=1e-12)


class TestGe7Structure:
    def test_centre_is_flat(self):
        p = _symmetric()
        h = M.ratio_response(p, F.log_grid(), M.sliders_tensor([{}]))
        assert float(_db(h).abs().max()) < 1e-9

    @pytest.mark.parametrize("slider", M.SLIDER_KEYS)
    @pytest.mark.parametrize("value", [15.0, 7.5])
    def test_boost_and_cut_mirror(self, slider, value):
        p = _symmetric()
        fr = F.log_grid()
        up, down = (
            M.ratio_response(p, fr, _one(slider, value)),
            M.ratio_response(p, fr, _one(slider, -value)),
        )
        assert float((_db(up) + _db(down)).abs().max()) < 1e-9

    def test_guesses_are_plausible(self):
        """Full boost about +15 dB per band, LEVEL ends +14 / -15 dB, all-centre 0 dB."""
        p = M.initial_guesses()
        boosts = []
        for i, f in enumerate(M.PANEL_BAND_HZ):
            h = M.ratio_response(
                p, torch.tensor([f], dtype=torch.float64), _one(M.SLIDER_KEYS[i], 15.0)
            )
            boosts.append(float(_db(h)))
        assert min(boosts) > 11.0 and max(boosts) < 17.0
        assert abs(np.mean(boosts) - 15.0) < 1.0
        f1k = torch.tensor([1000.0], dtype=torch.float64)
        assert abs(float(_db(M.ratio_response(p, f1k, M.sliders_tensor([{}]))))) < 0.01
        assert abs(float(_db(M.ratio_response(p, f1k, _one("level", 15.0)))) - 14.0) < 0.1
        assert abs(float(_db(M.ratio_response(p, f1k, _one("level", -15.0)))) + 15.0) < 0.1

    def test_q_narrows_toward_full_boost(self):
        """Pot loading: a mild setting is broader (relative to its height) than full boost."""
        p = _symmetric()
        fr = F.log_grid(50.0, 3000.0, 600)

        def rel_width(v):
            db = _db(M.ratio_response(p, fr, _one("hz_400", v)))[0]
            above = fr[db > 0.5 * float(db.max())]
            return float(above[-1] / above[0])

        assert rel_width(7.5) > rel_width(15.0) * 1.05

    def test_adjacent_bands_interact(self):
        p = _symmetric()
        fr = F.log_grid()
        both = _db(M.ratio_response(p, fr, M.sliders_tensor([{"hz_400": 15.0, "hz_800": 15.0}])))
        add = _db(M.ratio_response(p, fr, _one("hz_400", 15.0))) + _db(
            M.ratio_response(p, fr, _one("hz_800", 15.0))
        )
        assert float((both - add).abs().max()) > 0.5


class TestKnownAnswer:
    """Rung A0: fit from the guesses to data generated by a perturbed 'pedal'."""

    def setup_method(self):
        self.truth = _truth()
        self.sliders = F.a0_settings()
        self.freqs = F.log_grid()
        self.H = M.ratio_response(self.truth, self.freqs, self.sliders)

    def _errors(self, res):
        return (res["theta"] - M.pack(self.truth, res["names"])).numpy()

    def test_noise_free_recovers_every_parameter(self):
        res = F.fit_ratio(self.sliders, self.freqs, self.H, M.initial_guesses())
        assert res["converged"]
        rep = F.fit_report(res, noise_std=NOISE_NP)
        assert (rep["std"] < 0.05).all(), F.report_table(rep)
        # raw log parameters: error ≈ relative error; delay in µs
        assert np.abs(self._errors(res)).max() < 1e-3
        assert res["rms_db"] < 1e-4

    def test_noisy_within_reported_uncertainty(self):
        gen = torch.Generator().manual_seed(7)
        noise = NOISE_NP * torch.complex(
            torch.randn(self.H.shape, generator=gen, dtype=torch.float64),
            torch.randn(self.H.shape, generator=gen, dtype=torch.float64),
        )
        noisy = self.H * torch.exp(noise)
        res = F.fit_ratio(self.sliders, self.freqs, noisy, M.initial_guesses())
        rep = F.fit_report(res)  # noise estimated from the residual
        z = np.abs(self._errors(res)) / rep["std"]
        assert z.max() < 4.5, F.report_table(rep, res["theta"])
        assert abs(res["rms_db"] - 0.05) < 0.005

    def test_bypass_blocks_recovered(self):
        Hb = M.bypass_response(self.truth, self.freqs)
        res = F.fit_bypass(self.freqs, Hb, M.initial_guesses())
        err = (res["theta"] - M.pack(self.truth, res["names"])).numpy()
        assert np.abs(err).max() < 1e-3
