"""Fixtures for the capture-rig tests: tiny synthetic master signals in tmp_path, a fast simulated
rig.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import chirp

import pedal_model.rig.session as session_mod
import pedal_model.rig.signals as signals_mod

MASTER_SR = 96_000


def _section(index, label, kind, start, end, sr, dbfs):
    return {
        "index": index,
        "label": label,
        "type": kind,
        "start_sample": start,
        "end_sample": end,
        "start_s": start / sr,
        "end_s": end / sr,
        "amplitude_dbfs": dbfs,
        "params": {},
    }


def _write_master(path, name, seed):
    """A 5.6 s master: gap, log sweeps at −24 and −3 dBFS, two stepped tones, pink-ish noise, an
    impulse block, gap.
    """
    sr = MASTER_SR
    rng = np.random.default_rng(seed)
    gap = np.zeros(int(0.2 * sr), np.float32)
    t = np.arange(sr) / sr
    unit_sweep = chirp(t, 20, 1.0, 20_000, method="logarithmic")
    sweep = (10 ** (-24 / 20) * unit_sweep).astype(np.float32)
    sweep_high = (10 ** (-3 / 20) * unit_sweep).astype(np.float32)
    tt = np.arange(int(0.2 * sr)) / sr
    tone1 = (10 ** (-18 / 20) * np.sin(2 * np.pi * 440 * tt)).astype(np.float32)
    tone2 = (10 ** (-12 / 20) * np.sin(2 * np.pi * 440 * tt)).astype(np.float32)
    noise = (10 ** (-18 / 20) * rng.standard_normal(sr)).astype(np.float32)
    imp = np.zeros(int(0.5 * sr), np.float32)
    imp[::4800] = 0.25
    parts = [
        ("sweep_low", "log_sine_sweep", sweep, -24),
        ("sweep_high", "log_sine_sweep", sweep_high, -3),
        ("tone_a", "stepped_sine_tone", tone1, -18),
        ("tone_b", "stepped_sine_tone", tone2, -12),
        ("pink", "pink_noise", noise, -18),
        ("imp", "impulse_train", imp, -12),
    ]
    audio, sections, cur = [gap], [], len(gap)
    for i, (label, kind, x, db) in enumerate(parts):
        if kind != "stepped_sine_tone" or label == "tone_a":
            if i:
                audio.append(gap)
                cur += len(gap)
        sections.append(_section(i, label, kind, cur, cur + len(x), sr, db))
        audio.append(x)
        cur += len(x)
    audio.append(gap)
    a = np.concatenate(audio)
    sf.write(str(path.with_suffix(".wav")), a, sr, subtype="FLOAT")
    raw = {
        "schema_version": "1.0",
        "generator_version": "test",
        "signal_name": name,
        "date_created": None,
        "sample_rate": sr,
        "bit_depth": 32,
        "format": "float",
        "seed": seed,
        "total_samples": len(a),
        "total_duration_s": len(a) / sr,
        "params": {},
        "sections": sections,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture
def masters(tmp_path, monkeypatch):
    """Point the rig at synthetic masters and a tmp data root."""
    src = tmp_path / "masters"
    src.mkdir()
    m = {
        "train": _write_master(src / "train_master.json", "train_master", 1),
        "val": _write_master(src / "val_master.json", "val_master", 2),
    }
    monkeypatch.setattr(signals_mod, "SOURCE_MANIFESTS", m)
    monkeypatch.setattr(signals_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(session_mod, "DATA_ROOT", tmp_path / "data")
    return m
