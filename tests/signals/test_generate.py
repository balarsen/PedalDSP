"""Tests for pedal_model.signals.generate.

All signals use a low sample rate (8 kHz) and minimal durations so the
suite runs in a few seconds on CPU.  The bit-for-bit round-trip test is
marked as a regression guard: any change that breaks sample-level
reproducibility will fail it.
"""

import json
import warnings

import numpy as np
import pytest
import soundfile as sf

from pedal_model.signals.generate import (
    TrainParams,
    ValParams,
    from_manifest,
    generate,
)

SR = 8_000  # low rate keeps tests fast


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def short_train_params() -> TrainParams:
    """Minimal TrainParams that produces ~2–3 s of signal at 8 kHz."""
    return TrainParams(
        sample_rate=SR,
        seed=7,
        silence_between_s=0.05,
        sweep_f_start_hz=100.0,
        sweep_f_end_hz=3_500.0,
        sweep_duration_s=0.3,
        sweep_amplitudes_dbfs=[-12.0],
        stepped_freqs_hz=[220.0, 440.0],
        stepped_levels_dbfs=[-12.0],
        stepped_tone_duration_s=0.1,
        pink_amplitudes_dbfs=[-12.0],
        pink_duration_s=0.3,
        white_amplitudes_dbfs=[-12.0],
        white_duration_s=0.2,
        ramp_freqs_hz=[220.0],
        ramp_duration_s=0.2,
        ramp_start_dbfs=-30.0,
        ramp_end_dbfs=-6.0,
        impulse_duration_s=0.25,
        impulse_amplitude_dbfs=-6.0,
        impulse_spacings_s=[0.04, 0.08],
        am_chirp_duration_s=0.3,
        am_chirp_amplitude_dbfs=-9.0,
        am_chirp_f_carrier_hz=440.0,
        am_chirp_f_mod_start_hz=1.0,
        am_chirp_f_mod_end_hz=8.0,
        am_chirp_mod_depth=0.5,
        pluck_freqs_hz=[220.0],
        pluck_decay_tau_s=0.08,
        pluck_amplitude_dbfs=-6.0,
        pluck_total_duration_s=0.25,
        two_tone_fixed_freqs_hz=[220.0],
        two_tone_sweep_start_hz=100.0,
        two_tone_sweep_end_hz=2_000.0,
        two_tone_subsection_duration_s=0.25,
        two_tone_amplitude_dbfs=-9.0,
        multitone_freqs_hz=[220.0, 440.0, 880.0],
        multitone_amplitudes_dbfs=[-9.0],
        multitone_duration_s=0.25,
        layered_duration_s=0.25,
    )


@pytest.fixture()
def short_val_params() -> ValParams:
    """Minimal ValParams that produces ~2 s of signal at 8 kHz."""
    return ValParams(
        sample_rate=SR,
        seed=99,
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


# ── Basic output contract ─────────────────────────────────────────────────────


def test_generate_train_returns_two_paths(tmp_path, short_train_params):
    wav, js = generate("train", tmp_path, train_params=short_train_params)
    assert wav.suffix == ".wav"
    assert js.suffix == ".json"
    assert wav.exists()
    assert js.exists()


def test_generate_val_returns_two_paths(tmp_path, short_val_params):
    wav, js = generate("val", tmp_path, val_params=short_val_params)
    assert wav.suffix == ".wav"
    assert js.suffix == ".json"
    assert wav.exists()
    assert js.exists()


def test_generate_unknown_signal_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown signal"):
        generate("test", tmp_path)


# ── Sample count ──────────────────────────────────────────────────────────────


def test_train_wav_length_matches_manifest(tmp_path, short_train_params):
    wav, js = generate("train", tmp_path, train_params=short_train_params)
    manifest = json.loads(js.read_text())
    audio, _ = sf.read(str(wav), dtype="float32", always_2d=False)
    assert len(audio) == manifest["total_samples"]


def test_val_wav_length_matches_manifest(tmp_path, short_val_params):
    wav, js = generate("val", tmp_path, val_params=short_val_params)
    manifest = json.loads(js.read_text())
    audio, _ = sf.read(str(wav), dtype="float32", always_2d=False)
    assert len(audio) == manifest["total_samples"]


def test_total_duration_consistent_with_sample_count(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    m = json.loads(js.read_text())
    expected = round(m["total_samples"] / m["sample_rate"], 6)
    assert m["total_duration_s"] == pytest.approx(expected, abs=1e-4)


# ── Audio quality ─────────────────────────────────────────────────────────────


def test_train_audio_float32(tmp_path, short_train_params):
    wav, _ = generate("train", tmp_path, train_params=short_train_params)
    audio, _ = sf.read(str(wav), dtype="float32", always_2d=False)
    assert audio.dtype == np.float32


def test_val_audio_no_clipping(tmp_path, short_val_params):
    wav, _ = generate("val", tmp_path, val_params=short_val_params)
    audio, _ = sf.read(str(wav), dtype="float32", always_2d=False)
    assert np.max(np.abs(audio)) <= 1.0


def test_train_audio_not_silent(tmp_path, short_train_params):
    wav, _ = generate("train", tmp_path, train_params=short_train_params)
    audio, _ = sf.read(str(wav), dtype="float32", always_2d=False)
    assert np.max(np.abs(audio)) > 0.0


# ── Manifest schema ───────────────────────────────────────────────────────────


def test_manifest_required_fields_present(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    m = json.loads(js.read_text())
    for key in (
        "schema_version", "generator_version", "signal_name",
        "date_created", "sample_rate", "bit_depth", "format",
        "seed", "total_samples", "total_duration_s", "params", "sections",
    ):
        assert key in m, f"Missing key: {key!r}"


def test_manifest_sample_rate_matches_params(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    m = json.loads(js.read_text())
    assert m["sample_rate"] == SR
    assert m["params"]["sample_rate"] == SR


def test_manifest_seed_matches_params(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    m = json.loads(js.read_text())
    assert m["seed"] == short_val_params.seed
    assert m["params"]["seed"] == short_val_params.seed


def test_date_created_is_utc_iso8601(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    m = json.loads(js.read_text())
    # Expect format 2026-06-06T19:00:00Z
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", m["date_created"])


# ── Section ordering and bounds ───────────────────────────────────────────────


def test_sections_not_empty(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    sections = json.loads(js.read_text())["sections"]
    assert len(sections) > 0


def test_sections_in_order(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    sections = json.loads(js.read_text())["sections"]
    starts = [s["start_sample"] for s in sections]
    assert starts == sorted(starts)


def test_sections_no_overlap(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    sections = json.loads(js.read_text())["sections"]
    for i in range(len(sections) - 1):
        assert sections[i]["end_sample"] <= sections[i + 1]["start_sample"], (
            f"Section {sections[i]['label']!r} overlaps with "
            f"{sections[i+1]['label']!r}"
        )


def test_sections_within_total_bounds(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    m = json.loads(js.read_text())
    total = m["total_samples"]
    for s in m["sections"]:
        assert s["start_sample"] >= 0, f"{s['label']} start < 0"
        assert s["end_sample"] <= total, f"{s['label']} end > total_samples"
        assert s["start_sample"] < s["end_sample"], f"{s['label']} zero-length"


def test_section_indices_sequential(tmp_path, short_val_params):
    _, js = generate("val", tmp_path, val_params=short_val_params)
    sections = json.loads(js.read_text())["sections"]
    for i, s in enumerate(sections):
        assert s["index"] == i


# ── Output naming ─────────────────────────────────────────────────────────────


def test_default_train_output_name(tmp_path, short_train_params):
    wav, js = generate("train", tmp_path, train_params=short_train_params)
    assert wav.stem == "train_signal_v1"
    assert js.stem == "train_signal_v1"


def test_custom_output_name(tmp_path, short_val_params):
    wav, js = generate("val", tmp_path, val_params=short_val_params, output_name="my_take")
    assert wav.stem == "my_take"
    assert js.stem == "my_take"
    m = json.loads(js.read_text())
    assert m["signal_name"] == "my_take"


# ── Bit-for-bit reproducibility regression ────────────────────────────────────
# REGRESSION GUARD: any change that breaks sample-level reproduction will fail
# these tests.  The round-trip check compares numpy arrays element-by-element,
# not raw WAV bytes (which differ in the libsndfile PEAK-chunk timestamp).


def test_train_roundtrip_samples_identical(tmp_path, short_train_params):
    wav_orig, js_orig = generate("train", tmp_path / "orig", train_params=short_train_params)
    wav_repro, _ = from_manifest(js_orig, output_dir=tmp_path / "repro")

    orig, _ = sf.read(str(wav_orig), dtype="float32", always_2d=False)
    repro, _ = sf.read(str(wav_repro), dtype="float32", always_2d=False)

    assert np.array_equal(orig, repro), (
        "from_manifest did not reproduce identical samples for train signal. "
        "Check RNG seeding or param reconstruction."
    )


def test_val_roundtrip_samples_identical(tmp_path, short_val_params):
    wav_orig, js_orig = generate("val", tmp_path / "orig", val_params=short_val_params)
    wav_repro, _ = from_manifest(js_orig, output_dir=tmp_path / "repro")

    orig, _ = sf.read(str(wav_orig), dtype="float32", always_2d=False)
    repro, _ = sf.read(str(wav_repro), dtype="float32", always_2d=False)

    assert np.array_equal(orig, repro), (
        "from_manifest did not reproduce identical samples for val signal. "
        "Check RNG seeding or param reconstruction."
    )


def test_roundtrip_manifest_sample_count_matches(tmp_path, short_val_params):
    """total_samples in reproduced manifest must equal the original."""
    _, js_orig = generate("val", tmp_path / "orig", val_params=short_val_params)
    _, js_repro = from_manifest(js_orig, output_dir=tmp_path / "repro")

    orig_count = json.loads(js_orig.read_text())["total_samples"]
    repro_count = json.loads(js_repro.read_text())["total_samples"]
    assert orig_count == repro_count


def test_roundtrip_section_count_matches(tmp_path, short_val_params):
    _, js_orig = generate("val", tmp_path / "orig", val_params=short_val_params)
    _, js_repro = from_manifest(js_orig, output_dir=tmp_path / "repro")

    orig_secs = json.loads(js_orig.read_text())["sections"]
    repro_secs = json.loads(js_repro.read_text())["sections"]
    assert len(orig_secs) == len(repro_secs)


def test_roundtrip_custom_name(tmp_path, short_val_params):
    _, js_orig = generate("val", tmp_path / "orig", val_params=short_val_params)
    wav_repro, js_repro = from_manifest(
        js_orig, output_dir=tmp_path / "repro", output_name="custom_repro"
    )
    assert wav_repro.stem == "custom_repro"
    assert json.loads(js_repro.read_text())["signal_name"] == "custom_repro"


# ── Warning paths ─────────────────────────────────────────────────────────────


def test_from_manifest_warns_on_missing_field(tmp_path, short_val_params):
    _, js = generate("val", tmp_path / "orig", val_params=short_val_params)
    m = json.loads(js.read_text())
    del m["params"]["val_pink_duration_s"]
    bad_js = tmp_path / "bad.json"
    bad_js.write_text(json.dumps(m))

    with pytest.warns(UserWarning, match="absent from the manifest"):
        from_manifest(bad_js, output_dir=tmp_path / "repro")


def test_from_manifest_warns_on_extra_field(tmp_path, short_val_params):
    _, js = generate("val", tmp_path / "orig", val_params=short_val_params)
    m = json.loads(js.read_text())
    m["params"]["future_param_xyz"] = 42
    bad_js = tmp_path / "bad.json"
    bad_js.write_text(json.dumps(m))

    with pytest.warns(UserWarning, match="not known to the current"):
        from_manifest(bad_js, output_dir=tmp_path / "repro")


def test_from_manifest_unknown_signal_name_raises(tmp_path, short_val_params):
    _, js = generate("val", tmp_path / "orig", val_params=short_val_params)
    m = json.loads(js.read_text())
    m["signal_name"] = "mystery_signal_v1"
    bad_js = tmp_path / "bad.json"
    bad_js.write_text(json.dumps(m))

    with pytest.raises(ValueError, match="Cannot determine signal type"):
        from_manifest(bad_js, output_dir=tmp_path / "repro")
