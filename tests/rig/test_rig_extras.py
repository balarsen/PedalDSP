"""Capture rig: slider map, band fit, sweep signal, self-describing file names, dated configs."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from pedal_model.rig.backends import SimulatedBackend
from pedal_model.rig.bandfit import fit_band, peak_db, q_to_bw_oct, shelf_db
from pedal_model.rig.calibrate import run_chain_test, run_loopback_test
from pedal_model.rig.config import RigConfig
from pedal_model.rig.engine import CaptureEngine
from pedal_model.rig.profiles import KIND_MAP, Preset, load_profile
from pedal_model.rig.qc import check_take
from pedal_model.rig.session import Session
from pedal_model.rig.signals import prepare_signal
from pedal_model.signals.manifest import Manifest

SR = 48_000
GRID = np.geomspace(20, 20_000, 240)


# ── band fit ────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("fc,q,gain", [(800.0, 1.4, 12.0), (150.0, 0.9, -9.0), (3200.0, 2.5, 6.0)])
def test_fit_band_recovers_a_known_peak(fc, q, gain):
    measured = peak_db(GRID, fc, q, gain, SR)
    fit = fit_band(GRID, measured, SR, nominal_hz=fc * 1.3, nominal_db=gain * 0.8)
    assert fit.model == "peak"
    assert fit.fc_hz == pytest.approx(fc, rel=0.01)
    assert fit.q == pytest.approx(q, rel=0.02)
    assert fit.gain_db == pytest.approx(gain, abs=0.05)
    assert fit.bw_oct == pytest.approx(q_to_bw_oct(q), rel=0.02)


def test_fit_band_prefers_a_shelf_when_the_data_is_a_shelf():
    measured = shelf_db(GRID, 6000.0, 10.0, SR)
    fit = fit_band(GRID, measured, SR, nominal_hz=6400.0, nominal_db=10.0, try_shelf=True)
    assert fit.model == "shelf"
    assert fit.fc_hz == pytest.approx(6000.0, rel=0.02)
    assert fit.gain_db == pytest.approx(10.0, abs=0.1)


def test_fit_band_level_control_is_a_flat_gain():
    fit = fit_band(GRID, np.full_like(GRID, -7.5), SR, nominal_hz=None, nominal_db=-7.5)
    assert fit.model == "gain" and fit.gain_db == pytest.approx(-7.5)


def test_q_to_bw_matches_the_cookbook():
    assert q_to_bw_oct(np.sqrt(2)) == pytest.approx(1.0, rel=1e-6)  # Q = sqrt(2) is one octave


# ── profile: slider map ─────────────────────────────────────────────────── #


def test_slider_map_is_every_control_at_every_position():
    prof = load_profile("ge7")
    maps = prof.map_presets()
    assert len(maps) == 8 * 4
    assert all(p.kind == KIND_MAP and p.roles == ("sweep",) for p in maps)
    p = prof.preset("map_hz_800_p7p5")
    assert p.controls["hz_800"] == 7.5
    assert sum(abs(v) for k, v in p.controls.items() if k != "hz_800") == 0
    assert prof.preset("map_level_m15").controls["level"] == -15


def test_every_folder_preset_has_notes():
    prof = load_profile("ge7")
    assert all(p.notes for p in prof.presets())


def test_custom_preset_names_are_checked(tmp_path, monkeypatch):
    prof = load_profile("ge7")
    monkeypatch.setitem(prof.raw, "presets_dir", str(tmp_path))
    with pytest.raises(ValueError):
        prof.save_custom_preset(Preset("Bad Name!", "", {"hz_100": 0.0}))
    path = prof.save_custom_preset(Preset("my_mid_hump", "mid hump", {"hz_800": 6.0}, notes="test"))
    assert json.loads(path.read_text())["sliders"] == {"hz_800": 6.0}


# ── signals ─────────────────────────────────────────────────────────────── #


def test_sweep_signal_is_sweeps_only(masters, tmp_path):
    sig = prepare_signal("sweep", "full", SR, tmp_path)
    m = Manifest(sig.manifest_path)
    assert sig.manifest_path.stem == "sweep_48k"
    assert m.types() == ["log_sine_sweep"]


# ── session files ───────────────────────────────────────────────────────── #


def _engine():
    return CaptureEngine(SimulatedBackend(speed=0), block_size=1024)


def test_map_capture_writes_named_files_and_a_band_fit(masters):
    prof = load_profile("ge7")
    preset = prof.preset("map_hz_800_p15")
    s = Session.create(prof, SR, "quick", RigConfig(), "sim", True, [preset])
    eng = _engine()
    cal_path = s.set_calibration(
        run_loopback_test(eng, SR).to_dict(), run_chain_test(eng, SR, -10.0, -2.0).to_dict()
    )
    assert cal_path.name.startswith("calibration_") and cal_path.is_file()

    from pedal_model.synthetic.ge7 import Ge7Sim, Ge7Sliders

    sim = Ge7Sim.from_sliders(Ge7Sliders.from_dict(preset.controls))
    eng.backend.set_pedal(lambda x, sr: sim.process(x, sr))
    sig = s.signal("sweep")
    m = Manifest(sig.manifest_path)
    dry = m.load_all()
    take = eng.run(dry, SR, send_level_db=-10.0)
    rep, wet = check_take(
        take,
        dry,
        m,
        pedal=prof.pedal,
        controls=preset.controls,
        calibration=s.data["calibration"],
        qc_limits=prof.qc,
    )
    fit = fit_band(rep.response.freqs_hz, rep.response.measured_db, SR, 800.0, 15.0)
    s.save_capture(preset, "sweep", take, wet, rep.to_dict(), band_fit=fit.to_dict())

    t = s.take(preset.name, "sweep")
    assert t["wet_wav"].startswith("ge7_map_hz_800_p15_sweep_wet_48k_") and t["wet_wav"].endswith(
        ".wav"
    )
    assert (s.preset_dir(preset.name) / t["wet_wav"]).is_file()
    params = json.loads((s.preset_dir(preset.name) / "params.json").read_text())
    assert set(params["captures"]) == {"sweep"}
    assert params["preset_kind"] == KIND_MAP and params["sample_rate"] == SR
    assert params["rig"]["band_fit"]["sweep"]["model"] == "peak"
    # Ge7Sim's calibrated 800 Hz band sits a little off the panel label: the fit must find it.
    assert 500 < params["rig"]["band_fit"]["sweep"]["fc_hz"] < 1200
    assert s.data["presets"][preset.name]["status"] == "captured"
    wet_file, sr = sf.read(str(s.preset_dir(preset.name) / t["wet_wav"]))
    assert sr == SR and len(wet_file) == sig.total_samples


def test_rig_snapshot_is_dated(tmp_path):
    path = RigConfig().save_snapshot(tmp_path)
    assert path.name.startswith("rig_setup_") and path.suffix == ".json"
    doc = json.loads(path.read_text())
    assert "saved" in doc and RigConfig.from_dict(doc).asio_device == "Studio USB ASIO Driver"


def test_fit_band_recovers_a_gentle_shelf_and_its_slope():
    measured = shelf_db(GRID, 6400.0, 10.0, SR, slope=0.7)
    fit = fit_band(GRID, measured, SR, nominal_hz=6400.0, nominal_db=10.0, try_shelf=True)
    assert fit.model == "shelf"
    assert fit.slope == pytest.approx(0.7, abs=0.02)
    assert fit.fc_hz == pytest.approx(6400.0, rel=0.02)


@pytest.mark.parametrize("key,fc,model", [("hz_800", 792.0, "peak"), ("hz_6400", 6400.0, "shelf")])
def test_fit_band_on_ge7sim_returns_its_calibrated_bands(key, fc, model):
    """The fit applied to the GE-7 model must give back the model's own band parameters."""
    from pedal_model.rig.qc import expected_response

    controls = {
        k: 0.0
        for k in ("hz_100", "hz_200", "hz_400", "hz_800", "hz_1600", "hz_3200", "hz_6400", "level")
    }
    controls[key] = 15.0
    exp = expected_response("boss_ge7", controls, SR, GRID)
    fit = fit_band(
        GRID, exp, SR, nominal_hz=float(key[3:]), nominal_db=15.0, try_shelf=key == "hz_6400"
    )
    assert fit.model == model
    assert fit.fc_hz == pytest.approx(fc, rel=0.03)
    assert fit.rms_err_db < 0.1
