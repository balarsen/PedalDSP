"""GE-7 fitting dataset (pedal_model/rig/dataset.py): known delays and responses, simulated rig."""

from __future__ import annotations

import numpy as np
from scipy.signal import chirp, sosfreqz

from pedal_model.rig import dataset as D
from pedal_model.rig.backends import SimulatedBackend, SimulatedRig
from pedal_model.rig.calibrate import simulate_guided_calibration
from pedal_model.rig.config import RigConfig
from pedal_model.rig.engine import CaptureEngine, Take
from pedal_model.rig.profiles import Preset, load_profile
from pedal_model.rig.qc import check_take
from pedal_model.rig.session import Session
from pedal_model.signals.manifest import Manifest
from pedal_model.synthetic.ge7 import Ge7Sim, Ge7Sliders

SR = 48_000
KEYS = ("hz_100", "hz_200", "hz_400", "hz_800", "hz_1600", "hz_3200", "hz_6400", "level")
FLAT = dict.fromkeys(KEYS, 0.0)
SMILE = FLAT | {"hz_100": 9.0, "hz_200": 5.0, "hz_800": -6.0, "hz_6400": 8.0}
IDEAL = dict(reamp_tilt_db=0.0, bypass_gain_db=0.0, bypass_hp_hz=0.0)
BAND = (D.GRID_HZ >= 50) & (D.GRID_HZ <= 10_000)


def test_band_sums_recover_a_pure_delay_exactly():
    """Return = send delayed by d samples behind the loopback: |H| = 1, phase = −ω·d/sr."""
    n, pre, loop, d = SR, 1000, 300, 7
    t = np.arange(n) / SR
    x = (0.1 * chirp(t, 20, 1.0, 20_000, method="logarithmic")).astype(np.float32)
    total = pre + n + SR // 2
    play = np.zeros((total, 2), np.float32)
    play[pre : pre + n, 0] = play[pre : pre + n, 1] = x
    rec = np.zeros_like(play)
    rec[loop:, 1] = play[: total - loop, 1]
    rec[loop + d :, 0] = play[: total - loop - d, 0]
    take = Take(SR, play, rec, pre, n, 0.0, 0, False, 0.0)
    lags = D.take_lags(take, 0, n)
    assert lags == (loop, loop + d)
    sums = D.band_sums(take, 0, n, loop, d, D.n_fft_for(n + SR, SR), D.swept_range(20, 20_000))
    h, coh = D.response_from_sums(sums, d, SR)
    ok = np.isfinite(h) & BAND
    np.testing.assert_allclose(np.abs(h[ok]), 1.0, atol=1e-4)
    np.testing.assert_allclose(
        np.angle(h[ok] * np.exp(2j * np.pi * D.GRID_HZ[ok] * d / SR)), 0.0, atol=1e-3
    )
    assert coh[ok].min() > 0.999


def _session(masters, rig, presets, clip_dbfs_after_calibration=None):
    """A simulated session: guided calibration with saved references, then one train take each."""
    profile = load_profile("ge7")
    s = Session.create(profile, SR, "quick", RigConfig(), "sim", True, [p for p, _ in presets])
    eng = CaptureEngine(SimulatedBackend(rig, speed=0), block_size=1024)
    flat = Ge7Sim.from_sliders(Ge7Sliders.from_dict(FLAT))
    cal, results = simulate_guided_calibration(
        eng,
        SR,
        send_level_db=-10.0,
        train_peak_dbfs=-2.0,
        engaged_pedal=lambda x, sr: flat.process(x, sr),
    )
    for r in results:
        if r["reference"] is not None:
            cal["references"][r["reference"]["name"]]["file"] = s.save_reference(
                r["reference"]["name"], r["take"]
            )
    s.set_calibration(cal["loopback"], cal["chain"], cal["references"], cal["steps"])
    if clip_dbfs_after_calibration is not None:
        rig.clip_dbfs = clip_dbfs_after_calibration
    m = Manifest(s.signal("train").manifest_path)
    dry = m.load_all()
    for preset, controls in presets:
        sim = Ge7Sim.from_sliders(Ge7Sliders.from_dict(controls))
        eng.backend.set_pedal(lambda x, sr, sim=sim: sim.process(x, sr))
        take = eng.run(dry, SR, send_level_db=-10.0)
        rep, wet = check_take(
            take,
            dry,
            m,
            pedal="boss_ge7",
            controls=controls,
            calibration=s.data["calibration"],
            qc_limits=profile.qc,
        )
        s.save_capture(preset, "train", take, wet, rep)
    return s


def _h(take: dict) -> np.ndarray:
    return np.asarray(take["h_re"], dtype=float) + 1j * np.asarray(take["h_im"], dtype=float)


def _ge7_complex(controls: dict) -> np.ndarray:
    """Ge7Sim's complex response on the grid (its biquads in series, times the LEVEL gain)."""
    from pedal_model.synthetic import ge7

    p = Ge7Sim.from_sliders(Ge7Sliders.from_dict(controls)).params
    h = np.full(len(D.GRID_HZ), ge7._level_linear(p.sliders.level), dtype=complex)
    for f0, g, scale in zip(
        ge7._BAND_HZ_CALIBRATED, p.sliders.gains_db()[:6], ge7._BAND_GAIN_FACTOR
    ):
        if abs(g * scale) > 1e-4:
            h *= sosfreqz(ge7._peaking_eq_sos(f0, g * scale, p.band_q, SR), D.GRID_HZ, fs=SR)[1]
    shelf = p.sliders.gains_db()[6] * ge7._SHELF_GAIN_FACTOR
    if abs(shelf) > 1e-4:
        h *= sosfreqz(
            ge7._high_shelf_sos(ge7._SHELF_FREQ_HZ, shelf, ge7._SHELF_SLOPE, SR), D.GRID_HZ, fs=SR
        )[1]
    return h


def test_session_dataset_gives_engaged_over_bypassed(masters):
    rig = SimulatedRig(**IDEAL)
    s = _session(
        masters, rig, [(Preset("flat", "", FLAT), FLAT), (Preset("smile", "", SMILE), SMILE)]
    )
    ds = D.session_dataset(s.root)
    assert [t["preset"] for t in ds["takes"]] == ["flat", "smile"]
    by = {t["preset"]: t for t in ds["takes"]}
    for name, controls in (("flat", FLAT), ("smile", SMILE)):
        t = by[name]
        assert t["sections_dropped"] == [] and abs(t["delay_samples"]) <= 2
        h = _h(t)
        ok = np.isfinite(h) & BAND
        # the simulated engaged path is the pedal times engaged_gain_db; the chain cancels
        expected = _ge7_complex(controls) * 10 ** (rig.engaged_gain_db / 20)
        np.testing.assert_allclose(20 * np.log10(np.abs(h[ok] / expected[ok])), 0.0, atol=0.05)
        np.testing.assert_allclose(np.angle(h[ok] / expected[ok]), 0.0, atol=0.02)
    b = np.asarray(ds["bypass_over_chain"]["h_re"], dtype=float)
    assert np.nanmax(np.abs(20 * np.log10(np.abs(b[BAND])))) < 0.05  # IDEAL: bypass is a wire


def test_session_dataset_drops_a_clipped_sweep(masters):
    """The return clips at −20 dBFS after calibration: the −3 dBFS sweep (return ≈ −13) is
    dropped, the −24 dBFS one kept.
    """
    rig = SimulatedRig(**IDEAL)
    s = _session(
        masters, rig, [(Preset("flat", "", FLAT), FLAT)], clip_dbfs_after_calibration=-20.0
    )
    t = D.session_dataset(s.root)["takes"][0]
    assert t["sections_used"] == ["sweep_low"]
    assert [d["label"] for d in t["sections_dropped"]] == ["sweep_high"]


def test_each_take_keeps_its_own_calibration(masters):
    """Recalibrating in a later sitting (here: the return gain moved 1 dB, then C1-C4 again) must
    not change the takes made before it: each is divided by the calibration it was recorded with."""
    rig = SimulatedRig(**IDEAL)
    s = _session(masters, rig, [(Preset("flat", "", FLAT), FLAT)])
    first_cal = s.take("flat", "train")["calibration_file"]
    rig.return_gain_db += 1.0
    eng = CaptureEngine(SimulatedBackend(rig, speed=0), block_size=1024)
    flat = Ge7Sim.from_sliders(Ge7Sliders.from_dict(FLAT))
    cal, results = simulate_guided_calibration(
        eng, SR, send_level_db=-10.0, train_peak_dbfs=-2.0,
        engaged_pedal=lambda x, sr: flat.process(x, sr),
    )  # fmt: skip
    for r in results:
        if r["reference"] is not None:
            cal["references"][r["reference"]["name"]]["file"] = s.save_reference(
                r["reference"]["name"], r["take"]
            )
    s.set_calibration(cal["loopback"], cal["chain"], cal["references"], cal["steps"])
    assert s.data["calibration"]["file"] != first_cal
    t = D.session_dataset(s.root)["takes"][0]
    assert t["calibration_file"] == first_cal
    h = _h(t)
    ok = np.isfinite(h) & BAND
    # divided by its own (pre-move) bypass: still the pedal alone, 0.4 dB engaged gain
    np.testing.assert_allclose(20 * np.log10(np.abs(h[ok])), rig.engaged_gain_db, atol=0.05)
