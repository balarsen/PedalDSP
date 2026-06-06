"""Tests for pedal_model.metrics.suite — full suite and manifest-aware per-section."""
from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from pedal_model.metrics.suite import compute_all_metrics, compute_per_section
from pedal_model.signals.generate import ValParams, generate
from pedal_model.signals.manifest import Manifest

SR = 8_000


@pytest.fixture(scope="module")
def suite_signal(tmp_path_factory):
    """Generate a short val signal once; return (manifest, dry, target, predicted, sr)."""
    d = tmp_path_factory.mktemp("suite_metrics")
    p = ValParams(
        sample_rate=SR,
        seed=55,
        silence_gap_s=0.05,
        stepped_sweep_amplitudes_dbfs=[-12.0],
        stepped_sweep_duration_s=0.3,
        stepped_sweep_f_start_hz=100.0,
        stepped_sweep_f_end_hz=3_500.0,
        val_stepped_freqs_hz=[196.0, 293.7],
        val_stepped_levels_dbfs=[-12.0],
        val_stepped_tone_duration_s=0.5,
        val_pink_amplitudes_dbfs=[-12.0],
        val_pink_duration_s=0.3,
        val_two_tone_pairs=[[196.0, 293.7]],
        val_two_tone_duration_s=0.3,
        val_impulse_duration_s=0.2,
        val_impulse_amplitude_dbfs=-6.0,
        val_impulse_spacings_s=[0.04, 0.07],
        val_pluck_freqs_hz=[196.0],
        val_pluck_decay_tau_s=0.06,
        val_pluck_amplitude_dbfs=-6.0,
    )
    wav, js = generate("val", d, val_params=p)
    audio, _ = sf.read(str(wav), dtype="float32", always_2d=False)
    manifest = Manifest(js)
    dry = audio
    target = audio.copy()
    # Mild distortion model: tanh waveshaper with 3× pre-gain
    rng = np.random.default_rng(55)
    predicted = np.tanh(audio * 3.0).astype(np.float32)
    return manifest, dry, target, predicted


# ── compute_all_metrics ───────────────────────────────────────────────────────


class TestComputeAllMetrics:
    def test_returns_dict(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        sec = manifest.sections[0]
        a, b = sec.start_sample, sec.end_sample
        result = compute_all_metrics(target[a:b], predicted[a:b], dry[a:b], SR)
        assert isinstance(result, dict)

    def test_required_keys_present(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        sec = manifest.sections[0]
        a, b = sec.start_sample, sec.end_sample
        result = compute_all_metrics(target[a:b], predicted[a:b], dry[a:b], SR)
        for key in ("ESR", "null_depth_dB", "MSE", "STFT", "FR_err_dB",
                    "THD_target", "THD_pred", "THD_pattern_dist", "HP_sim", "MCD"):
            assert key in result, f"Missing key: {key}"

    def test_all_values_are_float(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        sec = manifest.sections[0]
        a, b = sec.start_sample, sec.end_sample
        result = compute_all_metrics(target[a:b], predicted[a:b], dry[a:b], SR)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is {type(v)}"

    def test_identical_signals_esr_near_zero(self, suite_signal):
        manifest, dry, target, _ = suite_signal
        sec = manifest.sections[0]
        a, b = sec.start_sample, sec.end_sample
        result = compute_all_metrics(target[a:b], target[a:b], dry[a:b], SR)
        assert result["ESR"] == pytest.approx(0.0, abs=1e-6)

    def test_identical_signals_null_depth_inf(self, suite_signal):
        manifest, dry, target, _ = suite_signal
        sec = manifest.sections[0]
        a, b = sec.start_sample, sec.end_sample
        result = compute_all_metrics(target[a:b], target[a:b], dry[a:b], SR)
        assert math.isinf(result["null_depth_dB"])


# ── compute_per_section ───────────────────────────────────────────────────────


class TestComputePerSection:
    def test_returns_all_sections(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        results = compute_per_section(manifest, dry, target, predicted, SR)
        assert set(results.keys()) == {s.label for s in manifest.sections}

    def test_core_keys_in_every_section(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        results = compute_per_section(manifest, dry, target, predicted, SR)
        for label, row in results.items():
            for key in ("ESR", "null_depth_dB", "STFT", "FR_err_dB"):
                assert key in row, f"Section {label!r} missing key {key!r}"

    def test_all_values_are_floats(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        results = compute_per_section(manifest, dry, target, predicted, SR)
        for label, row in results.items():
            for key, val in row.items():
                assert isinstance(val, float), f"{label}.{key} is {type(val)}"

    def test_identity_esr_is_zero(self, suite_signal):
        """Identical target and predicted → ESR = 0 for every section."""
        manifest, dry, target, _ = suite_signal
        results = compute_per_section(manifest, dry, target, target, SR)
        for label, row in results.items():
            assert row["ESR"] == pytest.approx(0.0, abs=1e-6), (
                f"Section {label!r}: ESR={row['ESR']}"
            )

    def test_identity_null_depth_inf(self, suite_signal):
        """Identical target and predicted → null depth = inf."""
        manifest, dry, target, _ = suite_signal
        results = compute_per_section(manifest, dry, target, target, SR)
        for label, row in results.items():
            assert math.isinf(row["null_depth_dB"]), (
                f"Section {label!r}: null_depth_dB={row['null_depth_dB']}"
            )

    def test_distorted_esr_positive(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        results = compute_per_section(manifest, dry, target, predicted, SR)
        for label, row in results.items():
            assert row["ESR"] > 0.0, f"Section {label!r}: ESR={row['ESR']}"

    def test_distorted_null_depth_finite(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        results = compute_per_section(manifest, dry, target, predicted, SR)
        for label, row in results.items():
            assert math.isfinite(row["null_depth_dB"]), (
                f"Section {label!r}: null_depth_dB={row['null_depth_dB']}"
            )

    def test_tone_sections_have_harmonic_keys(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        results = compute_per_section(manifest, dry, target, predicted, SR)
        tone_labels = [s.label for s in manifest.sections_of_type("stepped_sine_tone")]
        assert tone_labels, "Fixture must contain at least one stepped_sine_tone section"
        for label in tone_labels:
            row = results[label]
            for key in ("THD_target", "THD_pred", "THD_pattern_dist", "HP_sim"):
                assert key in row, f"Tone section {label!r} missing key {key!r}"

    def test_non_tone_sections_lack_harmonic_keys(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        results = compute_per_section(manifest, dry, target, predicted, SR)
        non_tone = [s for s in manifest.sections if s.type != "stepped_sine_tone"]
        assert non_tone, "Fixture must have at least one non-tone section"
        for sec in non_tone:
            assert "THD_pattern_dist" not in results[sec.label]

    def test_label_filter_subsets_sections(self, suite_signal):
        manifest, dry, target, predicted = suite_signal
        first_two = [s.label for s in manifest.sections[:2]]
        results = compute_per_section(
            manifest, dry, target, predicted, SR, section_labels=first_two
        )
        assert list(results.keys()) == first_two

    def test_esr_worse_than_identity(self, suite_signal):
        """Distorted model must have higher ESR than the identity."""
        manifest, dry, target, predicted = suite_signal
        res_identity = compute_per_section(manifest, dry, target, target, SR)
        res_distorted = compute_per_section(manifest, dry, target, predicted, SR)
        for label in res_identity:
            assert res_distorted[label]["ESR"] > res_identity[label]["ESR"], (
                f"Section {label!r}: distorted ESR not larger than identity"
            )
