"""The GE-7 record run (plan R2): its step list, the sweeps signal, per-preset train signals."""

from __future__ import annotations

import json

import numpy as np

from pedal_model.rig import dataset as D
from pedal_model.rig.backends import SimulatedBackend, SimulatedRig
from pedal_model.rig.calibrate import simulate_guided_calibration
from pedal_model.rig.config import RigConfig
from pedal_model.rig.engine import CaptureEngine
from pedal_model.rig.profiles import KIND_MAP, KIND_MODEL, Preset, load_profile
from pedal_model.rig.qc import check_take
from pedal_model.rig.session import Session
from pedal_model.rig.signals import prepare_signal
from pedal_model.signals.manifest import Manifest

SR = 48_000


def test_record_run_steps_in_order():
    p = load_profile("ge7")
    run = p.record_run()
    names = [s.name for s in run]
    assert len(names) == len(set(names))
    assert names[:8] == [
        "flat", "flat_r2", "flat_r3", "smile", "smile_r2", "smile_r3", "flat_replug",
        "smile_replug",
    ]  # fmt: skip
    kinds = {s.name: s.kind for s in run}
    assert kinds["flat"] == kinds["smile"] == KIND_MODEL and kinds["flat_r2"] == KIND_MAP
    assert sum(n.startswith("map_") for n in names) == 32
    assert sum(n.startswith("pair_") for n in names) == 16  # 6 adjacent + 2 distant, 2 settings
    assert not {"all_up_2", "all_up_pad15"} & set(names)
    assert [n for n in names if n.endswith("_full")] == ["flat_full", "smile_full", "all_up_3_full"]
    assert all(s.signal == "full" for s in run if s.name.endswith("_full"))
    assert all(s.signal is None for s in run if not s.name.endswith("_full"))
    test = load_profile("ge7").raw["record_run"]["split"]["test"]
    assert set(test) <= set(names)


def test_random_settings_are_seeded_on_a_5_db_grid():
    p = load_profile("ge7")
    fit, val = p.random_presets("fit"), p.random_presets("val")
    assert [s.controls for s in fit] == [s.controls for s in p.random_presets("fit")]
    assert len(fit) == len(val) == 8 and fit[0].controls != val[0].controls
    values = np.array([v for s in fit + val for v in s.controls.values()])
    assert np.all(values % 5 == 0) and values.min() >= -15 and values.max() <= 15


def test_sweeps_signal_is_the_sweeps_only(masters, tmp_path):
    sig = prepare_signal("train", "sweeps", SR, tmp_path)
    m = Manifest(sig.manifest_path)
    assert sig.manifest_path.stem == "train_sweeps_48k"
    assert {s.type for s in m.sections} == {"log_sine_sweep"}


def test_a_preset_can_play_its_own_train_signal(masters):
    """A sweeps session; one preset asks for the full train signal. The take records which
    signal it played, and the dataset reads that one.
    """
    profile = load_profile("ge7")
    flat = profile.preset("flat")
    full = Preset("flat_full", "", flat.controls, KIND_MODEL, "", "full")
    s = Session.create(profile, SR, "sweeps", RigConfig(), "sim", True, [flat, full])
    eng = CaptureEngine(SimulatedBackend(SimulatedRig(), speed=0), block_size=1024)
    cal, results = simulate_guided_calibration(eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0)
    for r in results:
        if r["reference"] is not None:
            name = r["reference"]["name"]
            cal["references"][name]["file"] = s.save_reference(name, r["take"])
    s.set_calibration(cal["loopback"], cal["chain"], cal["references"], cal["steps"])
    for preset in (flat, full):
        sig = s.signal("train", preset.signal)
        m = Manifest(sig.manifest_path)
        dry = m.load_all()
        take = eng.run(dry, SR, send_level_db=-10.0)
        rep, wet = check_take(
            take, dry, m, pedal="boss_ge7", controls=preset.controls,
            calibration=s.data["calibration"], qc_limits=profile.qc,
        )  # fmt: skip
        s.save_capture(preset, "train", take, wet, rep)
    data = json.loads((s.root / "session.json").read_text(encoding="utf-8"))
    assert data["presets"]["flat"]["takes"][0]["signal"] == "train"
    assert data["presets"]["flat_full"]["takes"][0]["signal"] == "train_full"
    assert data["signals"]["train"].endswith("train_sweeps_48k.json")
    assert data["signals"]["train_full"].endswith("train_full_48k.json")
    params = json.loads((s.root / "flat_full" / "params.json").read_text(encoding="utf-8"))
    assert params["captures"]["train"]["dry_signal"] == "train_full_48k"
    ds = D.session_dataset(s.root)
    used = {t["preset"]: t["sections_used"] for t in ds["takes"]}
    assert used["flat"] == used["flat_full"] == ["sweep_low", "sweep_high"]
