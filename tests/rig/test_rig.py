"""Capture rig, end to end on the simulated backend: signals, engine, calibration, QC and session
files.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf

from pedal_model.rig.analysis import aligned
from pedal_model.rig.backends import SimulatedBackend, SimulatedRig
from pedal_model.rig.calibrate import run_chain_test, run_loopback_test
from pedal_model.rig.config import RigConfig
from pedal_model.rig.engine import CaptureEngine
from pedal_model.rig.profiles import Preset, load_profile
from pedal_model.rig.qc import check_take
from pedal_model.rig.session import Session
from pedal_model.rig.signals import QUICK_TYPES, prepare_signal
from pedal_model.signals.manifest import Manifest
from pedal_model.synthetic.ge7 import Ge7Sim, Ge7Sliders

SMILE = {
    "hz_100": 8.0,
    "hz_200": 4.0,
    "hz_400": -2.0,
    "hz_800": -6.0,
    "hz_1600": -6.0,
    "hz_3200": 4.0,
    "hz_6400": 8.0,
    "level": 0.0,
}
FLAT = {k: 0.0 for k in SMILE}


def _ge7(controls):
    sim = Ge7Sim.from_sliders(Ge7Sliders.from_dict(controls))
    return lambda x, sr: sim.process(x, sr)


def _engine(rig=None):
    return CaptureEngine(SimulatedBackend(rig, speed=0), block_size=1024)


# ── signals ─────────────────────────────────────────────────────────────── #


def test_quick_excerpt_keeps_only_quick_types(masters, tmp_path):
    sig = prepare_signal("train", "quick", 96_000, tmp_path / "s")
    m = Manifest(sig.manifest_path)
    assert {s.type for s in m.sections} <= set(QUICK_TYPES)
    assert "impulse_train" not in m.types()
    src = Manifest(masters["train"])
    audio, src_audio = m.load_all(), src.load_all()
    for label in ("sweep_low", "tone_b", "pink"):
        np.testing.assert_array_equal(
            m.slice_section(label, audio), src.slice_section(label, src_audio)
        )


def test_48k_signal_has_rescaled_sections(masters, tmp_path):
    s96 = Manifest(prepare_signal("train", "full", 96_000, tmp_path / "a").manifest_path)
    s48 = Manifest(prepare_signal("train", "full", 48_000, tmp_path / "b").manifest_path)
    assert s48.sample_rate == 48_000
    assert s48.total_samples == s96.total_samples // 2
    for a, b in zip(s96.sections, s48.sections):
        assert b.start_sample == a.start_sample // 2
        assert b.start_s == pytest.approx(a.start_s)


def test_val_is_always_full(masters, tmp_path):
    assert prepare_signal("val", "quick", 48_000, tmp_path).manifest_path.stem == "val_full_48k"


# ── engine + calibration ────────────────────────────────────────────────── #


@pytest.mark.parametrize("sr,expected_lag", [(48_000, 47), (96_000, 94)])
def test_loopback_lag_recovered_exactly(sr, expected_lag):
    res = run_loopback_test(_engine(), sr)
    assert res.lag_samples == expected_lag
    assert res.gain_db == pytest.approx(SimulatedRig().loop_gain_db, abs=0.05)
    assert res.ok


def test_chain_gain_and_linearity():
    rig = SimulatedRig()
    res = run_chain_test(_engine(rig), 48_000, send_level_db=-10.0, train_peak_dbfs=-2.0)
    assert res.lag_samples == 47
    assert res.gain_db == pytest.approx(rig.reamp_db + rig.return_gain_db, abs=0.05)
    assert res.linearity_spread_db < 0.05
    assert res.ok


def test_chain_test_reports_missing_return():
    res = run_chain_test(_engine(SimulatedRig(return_gain_db=-200.0)), 48_000, -10.0, -2.0)
    assert not res.ok
    assert any("No return" in a.text for a in res.advice)


def test_chain_test_flags_hot_return():
    res = run_chain_test(_engine(SimulatedRig(return_gain_db=30.0)), 48_000, -10.0, -2.0)
    assert any(a.status in ("warn", "fail") and "In 1 gain" in a.text for a in res.advice)


# ── QC ──────────────────────────────────────────────────────────────────── #


def _capture(masters, tmp_path, pedal_controls, preset_controls, rig=None, sr=48_000):
    sig = prepare_signal("train", "full", sr, tmp_path / "sig")
    m = Manifest(sig.manifest_path)
    dry = m.load_all()
    eng = _engine(rig)
    eng.backend.set_pedal(_ge7(FLAT))
    cal = {
        "chain": run_chain_test(eng, sr, -10.0, -2.0).to_dict(),
        "loopback": run_loopback_test(eng, sr).to_dict(),
    }
    eng.backend.set_pedal(_ge7(pedal_controls))
    take = eng.run(dry, sr, send_level_db=-10.0)
    profile = load_profile("ge7")
    rep, _ = check_take(
        take,
        dry,
        m,
        pedal="boss_ge7",
        controls=preset_controls,
        calibration=cal,
        qc_limits=profile.qc,
    )
    return rep, take, dry


def _item(rep, name):
    return next(i for i in rep.items if i.name == name)


def test_qc_passes_correct_preset(masters, tmp_path):
    rep, take, dry = _capture(masters, tmp_path, SMILE, SMILE)
    assert rep.lag == 47 and rep.loop_lag == 47
    assert _item(rep, "Response vs model").status == "ok"
    assert _item(rep, "Level vs calibration").status == "ok"
    assert rep.verdict == "ok", [vars(i) for i in rep.items if i.status != "ok"]


def test_qc_wet_is_aligned_to_dry(masters, tmp_path):
    rep, take, dry = _capture(masters, tmp_path, FLAT, FLAT)
    wet = aligned(take.rec[:, 0], take.pre_roll, rep.lag, take.signal_len)
    gain = 10 ** ((-10.0 + SimulatedRig().reamp_db + SimulatedRig().return_gain_db) / 20)
    sweep = slice(int(0.2 * 48_000), int(1.2 * 48_000))
    corr = np.corrcoef(dry[sweep] * gain, wet[sweep])[0, 1]
    assert corr > 0.999


def test_qc_catches_wrong_knob_settings(masters, tmp_path):
    rep, _, _ = _capture(masters, tmp_path, FLAT, SMILE)
    assert _item(rep, "Response vs model").status == "fail"


def test_qc_flags_clipping(masters, tmp_path):
    rep, _, _ = _capture(masters, tmp_path, FLAT, FLAT, rig=SimulatedRig(return_gain_db=40.0))
    assert _item(rep, "Return peak").status == "fail"


# ── session files ───────────────────────────────────────────────────────── #


def test_session_params_load_like_the_harness(masters, tmp_path):
    profile = load_profile("ge7")
    preset = Preset("smile", "V shape", SMILE)
    s = Session.create(profile, 48_000, "quick", RigConfig(), "sim", True, [preset])
    eng = _engine()
    eng.backend.set_pedal(_ge7(SMILE))
    for role in ("train", "val"):
        sig = s.signal(role)
        m = Manifest(sig.manifest_path)
        dry = m.load_all()
        take = eng.run(dry, 48_000, send_level_db=-10.0)
        rep, wet = check_take(
            take, dry, m, pedal="boss_ge7", controls=SMILE, calibration=None, qc_limits=profile.qc
        )
        s.save_capture(preset, role, take, wet, rep.to_dict())

    d = s.preset_dir("smile")
    params = json.loads((d / "params.json").read_text())
    # Same resolution as pedal_model/eval/harness.py run_zoo().
    for role in ("train", "val"):
        cap = params["captures"][role]
        dry_manifest = (d / cap["dry_manifest"]).resolve()
        assert dry_manifest.with_suffix(".wav").is_file()
        wet, sr = sf.read(str(d / cap["wet_wav"]))
        assert sr == cap["sample_rate"] == 48_000
        assert len(wet) == cap["n_samples"] == Manifest(dry_manifest).total_samples
    assert params["rig"]["qc"]["train"]["verdict"] in ("ok", "warn")
    reloaded = Session.load(s.root)
    assert reloaded.data["presets"]["smile"]["status"] == "captured"
    assert os.path.isfile(s.root / "session.json")


def test_alignment_keeps_the_pedals_own_delay(masters, tmp_path):
    """A pedal's own delay (EQ group delay, a digital pedal's buffer) must stay in the aligned data.

    The simulated pedal here is a pure 10-sample delay: the correlation peak moves by 10, the
    alignment
    (interface latency from the loopback) must not, and QC reports the difference as the pedal's
    delay.
    """
    sr, pedal_delay = 48_000, 10
    sig = prepare_signal("train", "full", sr, tmp_path / "sig")
    m = Manifest(sig.manifest_path)
    dry = m.load_all()
    eng = _engine()
    cal = {
        "chain": run_chain_test(eng, sr, -10.0, -2.0).to_dict(),
        "loopback": run_loopback_test(eng, sr).to_dict(),
    }
    eng.backend.set_pedal(
        lambda x, _sr: np.concatenate([np.zeros(pedal_delay, np.float32), x[:-pedal_delay]])
    )
    take = eng.run(dry, sr, send_level_db=-10.0)
    rep, wet = check_take(
        take,
        dry,
        m,
        pedal="boss_ge7",
        controls=FLAT,
        calibration=cal,
        qc_limits=load_profile("ge7").qc,
    )
    assert rep.lag == rep.loop_lag == 47
    assert rep.xcorr_lag == 47 + pedal_delay
    assert _item(rep, "Pedal delay").value.startswith(f"+{pedal_delay} ")
    # The delay is still in the wet file: it lines up with the dry signal shifted by pedal_delay.
    sweep = slice(int(0.2 * sr), int(1.2 * sr))
    shifted = slice(sweep.start + pedal_delay, sweep.stop + pedal_delay)
    assert np.corrcoef(dry[sweep], wet[shifted])[0, 1] > 0.999


def test_uncalibrated_take_falls_back_to_correlation_and_warns(masters, tmp_path):
    sig = prepare_signal("train", "full", 48_000, tmp_path / "sig")
    m = Manifest(sig.manifest_path)
    dry = m.load_all()
    eng = _engine()
    take = eng.run(dry, 48_000, send_level_db=-10.0)
    rep, _ = check_take(
        take,
        dry,
        m,
        pedal="boss_ge7",
        controls=FLAT,
        calibration=None,
        qc_limits=load_profile("ge7").qc,
    )
    assert rep.lag == rep.xcorr_lag == 47
    assert _item(rep, "Alignment").status == "warn"
