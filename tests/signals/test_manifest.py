"""Tests for pedal_model.signals.manifest.Manifest."""

import json

import numpy as np
import pytest
import soundfile as sf

from pedal_model.signals.generate import ValParams, generate
from pedal_model.signals.manifest import Manifest, Section

SR = 8_000


@pytest.fixture()
def val_pair(tmp_path):
    """Generate a short val signal and return (wav_path, json_path, Manifest)."""
    p = ValParams(
        sample_rate=SR,
        seed=55,
        silence_gap_s=0.05,
        stepped_sweep_amplitudes_dbfs=[-12.0],
        stepped_sweep_duration_s=0.3,
        stepped_sweep_f_start_hz=100.0,
        stepped_sweep_f_end_hz=3_500.0,
        val_stepped_freqs_hz=[220.0, 440.0],
        val_stepped_levels_dbfs=[-12.0],
        val_stepped_tone_duration_s=0.1,
        val_pink_amplitudes_dbfs=[-12.0],
        val_pink_duration_s=0.3,
        val_two_tone_pairs=[[220.0, 440.0]],
        val_two_tone_duration_s=0.25,
        val_impulse_duration_s=0.25,
        val_impulse_amplitude_dbfs=-6.0,
        val_impulse_spacings_s=[0.04, 0.08],
        val_pluck_freqs_hz=[220.0],
        val_pluck_decay_tau_s=0.08,
        val_pluck_amplitude_dbfs=-6.0,
    )
    wav, js = generate("val", tmp_path, val_params=p)
    return wav, js, Manifest(js)


# ── Construction ──────────────────────────────────────────────────────────────


def test_manifest_repr(val_pair):
    _, _, m = val_pair
    r = repr(m)
    assert "val_signal_v1" in r
    assert "sections" in r


def test_manifest_attributes(val_pair):
    _, js, m = val_pair
    raw = json.loads(js.read_text())
    assert m.sample_rate == SR
    assert m.seed == 55
    assert m.total_samples == raw["total_samples"]
    assert m.total_duration_s == pytest.approx(raw["total_duration_s"], abs=1e-4)
    assert m.schema_version == raw["schema_version"]
    assert m.generator_version == raw["generator_version"]


def test_date_created_attribute_present(val_pair):
    _, _, m = val_pair
    assert m.date_created is not None
    assert "T" in m.date_created  # ISO 8601


def test_date_created_none_for_old_manifest(tmp_path, val_pair):
    _, js, _ = val_pair
    raw = json.loads(js.read_text())
    del raw["date_created"]
    old_js = tmp_path / "old.json"
    old_js.write_text(json.dumps(raw))
    m = Manifest(old_js)
    assert m.date_created is None


# ── Section count and structure ───────────────────────────────────────────────


def test_sections_list_nonempty(val_pair):
    _, _, m = val_pair
    assert len(m.sections) > 0


def test_sections_are_section_instances(val_pair):
    _, _, m = val_pair
    for s in m.sections:
        assert isinstance(s, Section)


def test_section_n_samples(val_pair):
    _, _, m = val_pair
    for s in m.sections:
        assert s.n_samples == s.end_sample - s.start_sample


def test_section_duration_s(val_pair):
    _, _, m = val_pair
    for s in m.sections:
        assert s.duration_s == pytest.approx(s.end_s - s.start_s, abs=1e-6)


# ── get_section ───────────────────────────────────────────────────────────────


def test_get_section_returns_correct_label(val_pair):
    _, _, m = val_pair
    label = m.sections[0].label
    sec = m.get_section(label)
    assert sec.label == label


def test_get_section_start_end_match_manifest(val_pair):
    _, js, m = val_pair
    raw_sections = json.loads(js.read_text())["sections"]
    for raw in raw_sections:
        sec = m.get_section(raw["label"])
        assert sec.start_sample == raw["start_sample"]
        assert sec.end_sample == raw["end_sample"]


def test_get_section_unknown_raises_keyerror(val_pair):
    _, _, m = val_pair
    with pytest.raises(KeyError, match="not found"):
        m.get_section("does_not_exist_xyz")


def test_get_section_keyerror_lists_available(val_pair):
    _, _, m = val_pair
    with pytest.raises(KeyError, match="Available labels"):
        m.get_section("does_not_exist_xyz")


# ── sections_of_type ─────────────────────────────────────────────────────────


def test_sections_of_type_returns_list(val_pair):
    _, _, m = val_pair
    result = m.sections_of_type("log_sine_sweep")
    assert isinstance(result, list)


def test_sections_of_type_all_match(val_pair):
    _, _, m = val_pair
    for type_name in m.types():
        result = m.sections_of_type(type_name)
        assert all(s.type == type_name for s in result)


def test_sections_of_type_unknown_returns_empty(val_pair):
    _, _, m = val_pair
    assert m.sections_of_type("nonexistent_type") == []


# ── labels / types ────────────────────────────────────────────────────────────


def test_labels_returns_all_in_order(val_pair):
    _, _, m = val_pair
    assert m.labels() == [s.label for s in m.sections]


def test_types_no_duplicates(val_pair):
    _, _, m = val_pair
    t = m.types()
    assert len(t) == len(set(t))


# ── load_section ─────────────────────────────────────────────────────────────


def test_load_section_shape(val_pair):
    _, _, m = val_pair
    label = m.sections[0].label
    sec = m.get_section(label)
    audio = m.load_section(label)
    assert audio.shape == (sec.n_samples,)


def test_load_section_dtype(val_pair):
    _, _, m = val_pair
    audio = m.load_section(m.sections[0].label)
    assert audio.dtype == np.float32


def test_load_section_matches_wav_slice(val_pair):
    wav, _, m = val_pair
    label = m.sections[0].label
    sec = m.get_section(label)

    full, _ = sf.read(str(wav), dtype="float32", always_2d=False)
    expected = full[sec.start_sample : sec.end_sample]
    actual = m.load_section(label)

    np.testing.assert_array_equal(actual, expected)


def test_load_section_missing_wav_raises(tmp_path, val_pair):
    _, js, _ = val_pair
    # Copy only the JSON; leave the WAV absent
    new_js = tmp_path / "orphan.json"
    raw = json.loads(js.read_text())
    raw["signal_name"] = "orphan"
    new_js.write_text(json.dumps(raw))
    m = Manifest(new_js)
    with pytest.raises(FileNotFoundError, match="generate_signal.py"):
        m.load_section(m.sections[0].label)


# ── load_all ──────────────────────────────────────────────────────────────────


def test_load_all_shape(val_pair):
    _, _, m = val_pair
    audio = m.load_all()
    assert audio.shape == (m.total_samples,)


def test_load_all_dtype(val_pair):
    _, _, m = val_pair
    assert m.load_all().dtype == np.float32


# ── slice_section ─────────────────────────────────────────────────────────────


def test_slice_section_matches_load_section(val_pair):
    _, _, m = val_pair
    label = m.sections[0].label
    full = m.load_all()
    sliced = m.slice_section(label, full)
    direct = m.load_section(label)
    np.testing.assert_array_equal(sliced, direct)


def test_slice_section_returns_view(val_pair):
    _, _, m = val_pair
    label = m.sections[0].label
    full = m.load_all()
    sliced = m.slice_section(label, full)
    # A view shares memory with the original
    assert np.shares_memory(sliced, full)
