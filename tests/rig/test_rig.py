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
from pedal_model.rig.calibrate import (
    run_chain_test,
    run_loopback_test,
    simulate_guided_calibration,
    worst_status,
)
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


# A chain with no tone of its own (no re-amp tilt, transparent buffers): for tests of alignment and
# level, where the chain's frequency response would only blur the check.
IDEAL = dict(reamp_tilt_db=0.0, bypass_gain_db=0.0, bypass_hp_hz=0.0, engaged_gain_db=0.0)


def _ok(res: dict) -> bool:
    return worst_status(res["advice"]) != "fail"


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
    assert res["lag_samples"] == expected_lag
    assert res["gain_db"] == pytest.approx(SimulatedRig().loop_gain_db, abs=0.05)
    assert _ok(res)


def test_loopback_clipping_fails_c1():
    """The 1824c's In 2 clipped far below full scale (2026-10-18); C1 must catch that."""
    res = run_loopback_test(_engine(SimulatedRig(loop_clip_dbfs=-50.0)), 48_000, hottest_dbfs=-10.0)
    assert not _ok(res) and res["linearity_spread_db"] > 0.5
    assert any("not linear" in a["text"] for a in res["advice"])
    ok = run_loopback_test(_engine(), 48_000, hottest_dbfs=-10.0)
    assert _ok(ok) and ok["linearity_spread_db"] < 0.1 and len(ok["level_gains_db"]) == 3


def test_loopback_is_sent_quieter_than_the_pedal_send():
    take = _engine().run(np.ones(100, np.float32) * 0.5, 48_000, send_level_db=-10.0)
    ratio_db = 20 * np.log10(take.play[take.pre_roll, 1] / take.play[take.pre_roll, 0])
    assert ratio_db == pytest.approx(-30.0, abs=1e-4)


def test_long_interface_latency_is_found():
    """The real 1824c round trip is 3722 samples (77.5 ms) at 48 kHz with 1024-sample blocks. A
    ±50 ms search once returned −2105 for it."""
    eng = _engine(SimulatedRig(latency_ms=3722 / 48))
    assert run_loopback_test(eng, 48_000)["lag_samples"] == 3722
    assert run_chain_test(eng, 48_000, -10.0, -2.0)["lag_samples"] == 3722


def test_latency_beyond_the_search_fails():
    res = run_loopback_test(_engine(SimulatedRig(latency_ms=600.0)), 48_000)
    assert not _ok(res) and any("longer than" in a["text"] for a in res["advice"])


def test_chain_gain_and_linearity():
    rig = SimulatedRig(**IDEAL)
    res = run_chain_test(_engine(rig), 48_000, send_level_db=-10.0, train_peak_dbfs=-2.0)
    assert res["lag_samples"] == 47
    assert res["gain_db"] == pytest.approx(rig.reamp_db + rig.return_gain_db, abs=0.05)
    assert res["linearity_spread_db"] < 0.05
    assert _ok(res)


def test_chain_test_fails_a_return_that_flattens_below_full_scale():
    """2026-10-20: In 1 preamp at maximum gain flattened the return at about -15 dBFS."""
    res = run_chain_test(_engine(SimulatedRig(clip_dbfs=-15.0)), 48_000, -10.0, -3.0)
    text = " ".join(a["text"] for a in res["advice"])
    assert not _ok(res) and "stops rising" in text


def test_chain_test_judges_linearity_only_over_take_levels():
    """Compression above the loudest take level is headroom information, not a failure."""
    res = run_chain_test(_engine(SimulatedRig(**IDEAL)), 48_000, -10.0, -3.0)
    assert _ok(res) and any("Headroom" in a["text"] for a in res["advice"])


def test_chain_test_reports_missing_return():
    res = run_chain_test(_engine(SimulatedRig(return_gain_db=-200.0)), 48_000, -10.0, -2.0)
    assert not _ok(res)
    assert any("Nothing measurable reached In 1" in a["text"] for a in res["advice"])


def test_chain_test_flags_hot_return():
    res = run_chain_test(_engine(SimulatedRig(return_gain_db=30.0)), 48_000, -10.0, -2.0)
    assert any(a["status"] in ("warn", "fail") and "In 1 gain" in a["text"] for a in res["advice"])


# ── QC ──────────────────────────────────────────────────────────────────── #


def _capture(masters, tmp_path, pedal_controls, preset_controls, rig=None, sr=48_000):
    sig = prepare_signal("train", "full", sr, tmp_path / "sig")
    m = Manifest(sig.manifest_path)
    dry = m.load_all()
    eng = _engine(rig)
    cal, _ = simulate_guided_calibration(
        eng, sr, send_level_db=-10.0, train_peak_dbfs=-2.0, engaged_pedal=_ge7(FLAT)
    )
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
    assert _item(rep, "Audio integrity (loopback)").status == "ok"
    assert rep.verdict == "ok", [vars(i) for i in rep.items if i.status != "ok"]


def test_qc_wet_is_aligned_to_dry(masters, tmp_path):
    rig = SimulatedRig(**IDEAL)
    rep, take, dry = _capture(masters, tmp_path, FLAT, FLAT, rig=rig)
    wet = aligned(take.rec[:, 0], take.pre_roll, rep.lag, take.signal_len)
    gain = 10 ** ((-10.0 + rig.reamp_db + rig.return_gain_db) / 20)
    sweep = slice(int(0.2 * 48_000), int(1.2 * 48_000))
    corr = np.corrcoef(dry[sweep] * gain, wet[sweep])[0, 1]
    assert corr > 0.999


def test_qc_catches_wrong_knob_settings(masters, tmp_path):
    rep, _, _ = _capture(masters, tmp_path, FLAT, SMILE)
    assert _item(rep, "Response vs model").status == "fail"


def test_qc_catches_silently_lost_audio(masters, tmp_path):
    """Blocks lost without a driver dropout flag (seen on the 1824c) must fail QC."""
    rep, take, _ = _capture(masters, tmp_path, FLAT, FLAT, rig=SimulatedRig(dropout_prob=0.05))
    assert take.xruns == 0  # the driver says nothing
    item = _item(rep, "Audio integrity (loopback)")
    assert item.status == "fail" and not item.value.startswith("0 of")


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
    alignment (interface latency from the loopback) must not, and QC reports the difference as the
    pedal's delay.
    """
    sr, pedal_delay = 48_000, 10
    sig = prepare_signal("train", "full", sr, tmp_path / "sig")
    m = Manifest(sig.manifest_path)
    dry = m.load_all()
    eng = _engine(SimulatedRig(**IDEAL))
    cal = {
        "chain": run_chain_test(eng, sr, -10.0, -2.0),
        "loopback": run_loopback_test(eng, sr),
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
