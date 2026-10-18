"""The out-of-process stream: shared-memory buffers, position, abort and errors (simulated rig)."""

from __future__ import annotations

import numpy as np
import pytest

from pedal_model.rig.backends import ProcessBackend, SimulatedBackend, SimulatedRig
from pedal_model.rig.calibrate import run_loopback_test
from pedal_model.rig.engine import CaptureEngine

SR = 48_000


def _engine(speed: float = 0.0, **rig) -> CaptureEngine:
    backend = ProcessBackend(
        SimulatedBackend, {"rig": SimulatedRig(**rig), "speed": speed}, "sim in child", False
    )
    return CaptureEngine(backend, block_size=1024)


def test_take_through_a_child_process_matches_in_process():
    rng = np.random.default_rng(3)
    sig = (0.1 * rng.standard_normal(SR)).astype(np.float32)
    rig = dict(reamp_tilt_db=0.0, noise_dbfs=-200.0)
    child = _engine(**rig).run(sig, SR, send_level_db=-10.0)
    local = CaptureEngine(SimulatedBackend(SimulatedRig(**rig), speed=0), 1024).run(sig, SR, -10.0)
    assert not child.aborted and child.xruns == 0
    np.testing.assert_allclose(child.rec, local.rec, atol=1e-6)
    np.testing.assert_array_equal(child.play, local.play)


def test_loopback_lag_through_a_child_process():
    assert run_loopback_test(_engine(), SR).lag_samples == 47


def test_abort_stops_the_child():
    eng = _engine(speed=1.0)  # real time: a 20 s signal would take 20 s
    import threading

    threading.Timer(1.0, eng.abort).start()
    take = eng.run(np.zeros(20 * SR, np.float32), SR, send_level_db=0.0)
    assert take.aborted and take.elapsed_s < 10


def test_child_errors_reach_the_caller():
    backend = ProcessBackend(SimulatedBackend, {"no_such_arg": 1}, "broken", False)
    with pytest.raises(RuntimeError, match="no_such_arg"):
        CaptureEngine(backend, 1024).run(np.zeros(SR, np.float32), SR, 0.0)
