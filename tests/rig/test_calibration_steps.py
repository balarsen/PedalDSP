"""Guided calibration C1–C4 (plan R1), simulated: references, move checks, QC normalisation."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from pedal_model.rig.backends import SimulatedBackend, SimulatedRig
from pedal_model.rig.calibrate import (
    CAL_STEPS,
    Reference,
    run_chain_test,
    run_loopback_test,
    run_step,
    simulate_guided_calibration,
)
from pedal_model.rig.config import RigConfig, code_version
from pedal_model.rig.engine import CaptureEngine
from pedal_model.rig.profiles import Preset, load_profile
from pedal_model.rig.qc import check_take
from pedal_model.rig.session import Session
from pedal_model.rig.signals import prepare_signal
from pedal_model.signals.manifest import Manifest
from pedal_model.synthetic.ge7 import Ge7Sim, Ge7Sliders

SR = 48_000
FLAT = {k: 0.0 for k in ("hz_100", "hz_200", "hz_400", "hz_800", "hz_1600", "hz_3200")}
FLAT |= {"hz_6400": 0.0, "level": 0.0}


def _engine(rig=None):
    return CaptureEngine(SimulatedBackend(rig, speed=0), block_size=1024)


def _ge7(controls):
    sim = Ge7Sim.from_sliders(Ge7Sliders.from_dict(controls))
    return lambda x, sr: sim.process(x, sr)


def _calibrate(eng):
    return simulate_guided_calibration(
        eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0, engaged_pedal=_ge7(FLAT)
    )


def _texts(res):
    return " ".join(a.text for a in res.advice)


def _shelf_db(f, rig):
    """The simulated chain's tilt, |H(f)| in dB (the analog shelf before discretisation)."""
    g = 10 ** (rig.reamp_tilt_db / 20)
    w = 2 * np.pi * rig.reamp_tilt_hz
    wz, wp = w / np.sqrt(g), w * np.sqrt(g)
    s = 2j * np.pi * f
    return 20 * np.log10(np.abs((wp / wz) * (s + wz) / (s + wp)))


def test_all_steps_pass_in_order():
    cal, results = _calibrate(_engine())
    assert [r.step for r in results] == [s.id for s in CAL_STEPS]
    assert all(r.status == "ok" for r in results), [(r.step, _texts(r)) for r in results]
    assert set(cal["references"]) == {"chain", "bypass", "engaged_flat"}
    assert cal["chain"]["lag_samples"] == cal["loopback"]["lag_samples"] == 47


def test_references_measure_the_chain_bypass_and_engaged_paths():
    rig = SimulatedRig()
    cal, _ = _calibrate(_engine(rig))
    refs = {k: Reference.from_dict(v) for k, v in cal["references"].items()}
    f = np.array([100.0, 1000.0, 10_000.0])
    # The chain's tilt is measured, not assumed flat (what the old scalar normalisation missed).
    chain = refs["chain"].db_at(f)
    expected = _shelf_db(f, rig)
    assert np.allclose(chain - chain[0], expected - expected[0], atol=0.15)
    band = np.geomspace(100, 8000, 30)
    bypass = refs["bypass"].db_at(band) - refs["chain"].db_at(band)
    assert np.allclose(bypass, rig.bypass_gain_db, atol=0.1)
    engaged = refs["engaged_flat"].db_at(band) - refs["bypass"].db_at(band)
    assert np.allclose(engaged, rig.engaged_gain_db, atol=0.1)


def test_c3_warns_when_the_cables_were_not_moved():
    eng = _engine()
    eng.backend.set_pedal_state("cable")
    chain = run_step("C2", eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0, references={})
    c3 = run_step(
        "C3",
        eng,
        SR,
        send_level_db=-10.0,
        train_peak_dbfs=-2.0,
        references={"chain": chain.reference},
    )  # still "cable": the pedal never went back into the chain
    assert c3.status == "warn" and "Identical to the chain" in _texts(c3)


def _bypass_reference(eng):
    """Do C2 and C3 properly; return the references."""
    refs = {}
    for step, state in (("C2", "cable"), ("C3", "bypass")):
        eng.backend.set_pedal_state(state)
        res = run_step(step, eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0, references=refs)
        refs[res.reference.name] = res.reference
    return refs


def test_c3_needs_the_chain_reference_first():
    eng = _engine()
    eng.backend.set_pedal_state("bypass")
    c3 = run_step("C3", eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0, references={})
    assert c3.status == "fail" and "run C2 first" in _texts(c3)


def test_c4_warns_when_the_footswitch_is_off():
    eng = _engine()
    refs = _bypass_reference(eng)  # pedal left bypassed for C4
    c4 = run_step("C4", eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0, references=refs)
    assert c4.status == "warn" and "footswitch" in _texts(c4)


def test_c4_warns_when_the_controls_are_not_flat():
    eng = _engine()
    refs = _bypass_reference(eng)
    eng.backend.set_pedal(_ge7({**FLAT, "hz_800": 12.0}))
    eng.backend.set_pedal_state("engaged")
    c4 = run_step("C4", eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0, references=refs)
    assert c4.status == "warn" and "calibration preset" in _texts(c4)


def test_c4_fails_when_c2_was_measured_through_the_engaged_pedal():
    """The 2026-10-18 mistake: cables not moved for C2, so the 'chain' was the engaged pedal."""
    eng = _engine()
    eng.backend.set_pedal(_ge7(FLAT))
    refs = {}
    for step, state in (("C2", "engaged"), ("C3", "bypass"), ("C4", "engaged")):
        eng.backend.set_pedal_state(state)
        res = run_step(step, eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0, references=refs)
        refs[res.reference.name] = res.reference
    assert res.status == "fail" and "Redo C2" in _texts(res)


def test_chain_offset_comes_from_one_capture():
    cal, _ = _calibrate(_engine())
    chain = cal["references"]["chain"]
    assert chain["return_lag_samples"] - chain["lag_samples"] == 0  # simulated chain: no delay


def test_a_missing_return_fails_and_stops_the_sequence():
    cal, results = _calibrate(_engine(SimulatedRig(return_gain_db=-200.0)))
    assert results[-1].step == "C2" and results[-1].status == "fail"
    assert cal["references"] == {}


def _flat_take_response(masters, tmp_path, cal):
    sig = prepare_signal("train", "full", SR, tmp_path / "sig")
    m = Manifest(sig.manifest_path)
    dry = m.load_all()
    eng = _engine()
    eng.backend.set_pedal(_ge7(FLAT))
    take = eng.run(dry, SR, send_level_db=-10.0)
    rep, _ = check_take(
        take,
        dry,
        m,
        pedal="boss_ge7",
        controls=FLAT,
        calibration=cal,
        qc_limits=load_profile("ge7").qc,
    )
    r = rep.response
    sel = (r.freqs_hz >= 100) & (r.freqs_hz <= 8000)
    return rep, r.measured_db[sel]


def test_qc_divides_by_the_bypass_reference(masters, tmp_path):
    """GE-7 at flat ÷ bypassed is flat (+ the engaged-path gain); the chain tilt is gone."""
    cal, _ = _calibrate(_engine())
    rep, db = _flat_take_response(masters, tmp_path, cal)
    assert rep.response.reference == "bypass"
    assert np.ptp(db) < 0.3
    assert abs(np.median(db) - SimulatedRig().engaged_gain_db) < 0.15
    assert next(i for i in rep.items if i.name == "Normalised by").status == "ok"
    # The pedal's own flat level (C4) is the reference for the level check, not 0 dB.
    level = next(i for i in rep.items if i.name == "Level vs calibration")
    assert level.status == "ok" and abs(float(level.value.split()[0])) < 0.1


def test_scalar_calibration_still_works_but_warns(masters, tmp_path):
    eng = _engine()
    cal = {
        "chain": run_chain_test(eng, SR, -10.0, -2.0).to_dict(),
        "loopback": run_loopback_test(eng, SR).to_dict(),
    }
    rep, db = _flat_take_response(masters, tmp_path, cal)
    assert rep.response.reference == "chain_gain"
    assert np.ptp(db) > 2.0  # the chain's tilt is still in the curve
    assert next(i for i in rep.items if i.name == "Normalised by").status == "warn"


def test_session_stores_references_and_their_audio(masters, tmp_path):
    profile = load_profile("ge7")
    s = Session.create(profile, SR, "quick", RigConfig(), "sim", True, [Preset("flat", "", FLAT)])
    cal, results = _calibrate(_engine())
    files = {}
    for r in results:
        if r.reference is not None:
            files[r.reference.name] = s.save_reference(r.reference.name, r.take)
    path = s.set_calibration(cal["loopback"], cal["chain"], cal["references"], cal["steps"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved["references"]) == {"chain", "bypass", "engaged_flat"}
    assert saved["steps"]["C4"]["status"] == "ok"
    reloaded = Session.load(s.root).data["calibration"]
    assert Reference.from_dict(reloaded["references"]["bypass"]).gain_1k_db < 0
    assert saved["code_version"] == code_version() != "unknown"
    assert all(e["code"] == code_version() for e in Session.load(s.root).data["events"])
    for name, rel in files.items():
        with sf.SoundFile(str(s.root / rel)) as f:
            assert code_version() in f.software and name in f.comment
        audio, sr = sf.read(str(s.root / rel))
        assert sr == SR and audio.shape[1] == 2
        assert rel.startswith("calibration/") and f"_ref_{name}_raw_48k_" in rel
