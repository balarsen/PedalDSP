"""Tests for pedal_model.synthetic.pedal."""
from __future__ import annotations

import numpy as np
import pytest

from pedal_model.metrics.harmonic import compute_thd
from pedal_model.synthetic.pedal import KlonSim, KlonSimParams

SR = 48_000


def _sine(
    freq: float = 440.0,
    sr: int = SR,
    duration: float = 0.5,
    amp: float = 0.5,
) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


# ── Output shape / dtype / range ──────────────────────────────────────────────


def test_output_shape_matches_input():
    audio = _sine()
    out = KlonSim().process(audio, SR)
    assert out.shape == audio.shape


def test_output_dtype_float32():
    audio = _sine()
    out = KlonSim().process(audio, SR)
    assert out.dtype == np.float32


def test_output_within_unity_range():
    audio = _sine(amp=0.95)
    out = KlonSim().process(audio, SR)
    assert np.max(np.abs(out)) <= 1.0


def test_output_not_silence():
    audio = _sine()
    out = KlonSim().process(audio, SR)
    assert np.max(np.abs(out)) > 0.01


# ── Reproducibility ───────────────────────────────────────────────────────────


def test_seeded_runs_are_identical():
    audio = _sine()
    p = KlonSimParams(noise_seed=99)
    out_a = KlonSim(p).process(audio, SR)
    out_b = KlonSim(p).process(audio, SR)
    np.testing.assert_array_equal(out_a, out_b)


def test_different_seeds_produce_different_output():
    audio = _sine()
    out_a = KlonSim(KlonSimParams(noise_seed=1)).process(audio, SR)
    out_b = KlonSim(KlonSimParams(noise_seed=2)).process(audio, SR)
    assert not np.array_equal(out_a, out_b)


# ── Nonlinearity ──────────────────────────────────────────────────────────────


def test_higher_drive_adds_more_distortion():
    """More pre-gain → more harmonics → higher THD."""
    audio = _sine(amp=0.3, duration=2.0)
    p_low = KlonSimParams(drive=1.5, noise_floor_dbfs=-90)
    p_high = KlonSimParams(drive=8.0, noise_floor_dbfs=-90)
    thd_low = compute_thd(KlonSim(p_low).process(audio, SR), 440.0, SR)
    thd_high = compute_thd(KlonSim(p_high).process(audio, SR), 440.0, SR)
    assert thd_high > thd_low


def test_soft_clipping_is_nonlinear():
    """RMS ratio of loud/quiet output must be < the linear ratio (tanh compresses)."""
    t = np.arange(SR, dtype=np.float32) / SR
    quiet = (0.1 * np.sin(2.0 * np.pi * 440 * t)).astype(np.float32)
    loud = (0.9 * np.sin(2.0 * np.pi * 440 * t)).astype(np.float32)
    p = KlonSimParams(noise_floor_dbfs=-90)
    rms = lambda x: float(np.sqrt(np.mean(KlonSim(p).process(x, SR) ** 2)))
    # Linear ratio would be 9×; tanh saturation reduces it significantly
    assert rms(loud) / rms(quiet) < 5.0


# ── Noise floor ───────────────────────────────────────────────────────────────


def test_noise_floor_level_at_silence():
    """Feed silence → output RMS ≈ noise_floor_dbfs (within ±3 dB)."""
    silence = np.zeros(SR * 3, dtype=np.float32)
    p = KlonSimParams(noise_floor_dbfs=-70.0)
    out = KlonSim(p).process(silence, SR)
    out_rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(out ** 2))) + 1e-12)
    assert out_rms_db == pytest.approx(-70.0, abs=3.0)


# ── Filtering ─────────────────────────────────────────────────────────────────


def test_sub_bass_attenuated():
    """A 30 Hz tone must be strongly attenuated by the input HPF at 80 Hz."""
    tone = _sine(freq=30.0, duration=1.0, amp=0.5)
    p = KlonSimParams(noise_floor_dbfs=-90)
    out = KlonSim(p).process(tone, SR)
    in_rms = float(np.sqrt(np.mean(tone ** 2)))
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    # 30 Hz is well below the 80 Hz cutoff — expect >20 dB attenuation
    assert out_rms < in_rms * 0.15


def test_mid_frequencies_pass_through():
    """A 1 kHz tone (above all HPF cutoffs) must survive the chain."""
    tone = _sine(freq=1000.0, duration=0.5, amp=0.3)
    p = KlonSimParams(noise_floor_dbfs=-90, drive=1.0, output_level=1.0)
    out = KlonSim(p).process(tone, SR)
    assert float(np.sqrt(np.mean(out ** 2))) > 0.05


# ── Serialisation ─────────────────────────────────────────────────────────────


def test_to_dict_contains_pedal_type():
    d = KlonSim().to_dict()
    assert d["pedal_type"] == "klon_sim"


def test_roundtrip_serialisation():
    p = KlonSimParams(drive=6.0, clip_hpf_hz=200.0, noise_seed=77)
    sim = KlonSim(p)
    sim2 = KlonSim.from_dict(sim.to_dict())
    assert sim2.params.drive == pytest.approx(p.drive)
    assert sim2.params.clip_hpf_hz == pytest.approx(p.clip_hpf_hz)
    assert sim2.params.noise_seed == p.noise_seed


def test_roundtrip_produces_identical_output():
    audio = _sine()
    p = KlonSimParams(drive=5.0, noise_seed=7)
    sim = KlonSim(p)
    sim2 = KlonSim.from_dict(sim.to_dict())
    np.testing.assert_array_equal(sim.process(audio, SR), sim2.process(audio, SR))
