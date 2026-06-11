"""Tests for pedal_model.eval.diagnostics — pure helpers and plot smoke tests."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pedal_model.eval.diagnostics import (
    BG,
    C_TARGET,
    _aweight_db,
    _bandpass_sos,
    _harmonic_amps,
    _save,
    _smooth_log_octave,
    _wiener_ir,
    _wiener_tf,
    dark_ax,
    dark_fig,
    extract_drive_level,
)

# ── Synthetic signal helpers ──────────────────────────────────────────────── #

_SR = 16_000  # fast enough, Nyquist at 8 kHz covers all test tones


def _sine(f0: float, amp: float = 0.5, dur: float = 1.0, sr: int = _SR) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)


def _noise(dur: float = 1.0, sr: int = _SR, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * dur)).astype(np.float32) * 0.3


def _three(dur: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dry, target, predicted) triple of broadband noise signals."""
    dry = _noise(dur, seed=0)
    target = np.tanh(dry * 3.0)
    predicted = np.tanh(dry * 2.8)
    return dry, target, predicted


# ── extract_drive_level ───────────────────────────────────────────────────── #


def test_extract_drive_level_positive():
    assert extract_drive_level("section_l12_foo") == 12


def test_extract_drive_level_negative():
    assert extract_drive_level("val_stepped_tone_f220_l-24") == -24


def test_extract_drive_level_zero():
    assert extract_drive_level("train_l0") == 0


def test_extract_drive_level_no_match_returns_none():
    assert extract_drive_level("no_level_here") is None


def test_extract_drive_level_picks_first_match():
    # Only the first _lN pattern is returned
    result = extract_drive_level("a_l5_b_l10")
    assert result == 5


# ── _wiener_tf ────────────────────────────────────────────────────────────── #


def test_wiener_tf_identity_magnitude_near_one():
    """H(f) ≈ 1 when dry == wet for a broadband signal."""
    sig = _noise(dur=0.5)
    H = _wiener_tf(sig, sig)
    # All bins should be real-valued magnitude ≈ 1
    mag = np.abs(H)
    np.testing.assert_allclose(np.mean(mag), 1.0, atol=0.01)


def test_wiener_tf_gain_shift():
    """H(f) ≈ 2 when wet = 2 * dry (pure gain)."""
    dry = _noise(dur=0.5, seed=7)
    wet = (dry * 2.0).astype(np.float32)
    H = _wiener_tf(dry, wet)
    mag = np.abs(H)
    # Bins where |X|² >> noise_floor will have |H| ≈ 2
    bright_bins = mag[mag > 1.5]
    assert len(bright_bins) > len(mag) // 2
    np.testing.assert_allclose(np.median(mag), 2.0, atol=0.05)


def test_wiener_tf_output_shape_matches_rfft():
    from scipy.fft import rfftfreq
    sig = _noise(dur=0.25)
    H = _wiener_tf(sig, sig)
    n = len(sig)
    assert len(H) == n // 2 + 1


# ── _wiener_ir ────────────────────────────────────────────────────────────── #


def test_wiener_ir_identity_is_delta():
    """IR for identity system should be a near-unit impulse at t=0."""
    sig = _noise(dur=0.25)
    ir = _wiener_ir(sig, sig)
    assert ir.dtype == np.float32
    # Peak should be at sample 0 and much larger than all others
    assert ir[0] == pytest.approx(1.0, abs=0.05)
    assert float(np.max(np.abs(ir[1:]))) < 0.1


def test_wiener_ir_length_matches_input():
    sig = _noise(dur=0.1)
    ir = _wiener_ir(sig, sig)
    assert len(ir) == len(sig)


# ── _aweight_db ───────────────────────────────────────────────────────────── #


def test_aweight_db_zero_at_1khz():
    freqs = np.array([1000.0])
    np.testing.assert_allclose(_aweight_db(freqs), [0.0], atol=0.01)


def test_aweight_db_negative_below_1khz():
    freqs = np.array([100.0, 500.0])
    assert np.all(_aweight_db(freqs) < 0.0)


def test_aweight_db_negative_above_10khz():
    freqs = np.array([16_000.0])
    assert float(_aweight_db(freqs)[0]) < 0.0


def test_aweight_db_at_zero_is_finite():
    freqs = np.array([0.0, 100.0, 1000.0])
    result = _aweight_db(freqs)
    assert np.all(np.isfinite(result))


def test_aweight_db_shape_preserved():
    freqs = np.linspace(20, 20_000, 500)
    result = _aweight_db(freqs)
    assert result.shape == freqs.shape


# ── _harmonic_amps ─────────────────────────────────────────────────────────── #


def test_harmonic_amps_pure_sine_fundamental_dominates():
    """For a pure 440 Hz sine the 1st harmonic should dominate all others."""
    np.random.seed(0)
    sig = _sine(440.0, amp=0.8, dur=1.0)
    amps = _harmonic_amps(sig, _SR, f0=440.0, n_harmonics=4)
    assert amps[0] > 10 * max(amps[1], amps[2], amps[3], 1e-10)


def test_harmonic_amps_returns_correct_count():
    sig = _noise(dur=0.5)
    amps = _harmonic_amps(sig, _SR, f0=440.0, n_harmonics=6)
    assert amps.shape == (6,)
    assert np.all(amps >= 0.0)


def test_harmonic_amps_clipped_signal_has_odd_harmonics():
    """Hard clipping introduces odd harmonics (3rd, 5th …)."""
    sig = _sine(440.0, amp=0.95, dur=1.0)
    clipped = np.clip(sig, -0.3, 0.3)
    amps = _harmonic_amps(clipped.astype(np.float32), _SR, f0=440.0, n_harmonics=5)
    # 3rd harmonic (index 2) should be meaningfully present
    assert amps[2] > 1e-4


# ── _smooth_log_octave ────────────────────────────────────────────────────── #


def test_smooth_log_octave_flat_input_unchanged():
    freqs = np.linspace(100, 8_000, 200)
    vals = np.ones_like(freqs)
    smoothed = _smooth_log_octave(freqs, vals)
    np.testing.assert_allclose(smoothed, 1.0, atol=0.01)


def test_smooth_log_octave_reduces_spike():
    freqs = np.linspace(100, 8_000, 200)
    vals = np.ones_like(freqs)
    # Inject a spike at index 100
    vals[100] = 100.0
    smoothed = _smooth_log_octave(freqs, vals)
    assert smoothed[100] < vals[100]


def test_smooth_log_octave_short_input_returns_copy():
    freqs = np.array([100.0, 200.0, 400.0])
    vals = np.array([1.0, 2.0, 3.0])
    out = _smooth_log_octave(freqs, vals)
    np.testing.assert_array_equal(out, vals)


def test_smooth_log_octave_shape_preserved():
    freqs = np.geomspace(20, 20_000, 300)
    vals = np.random.default_rng(0).standard_normal(300)
    out = _smooth_log_octave(freqs, vals)
    assert out.shape == vals.shape


# ── _bandpass_sos ──────────────────────────────────────────────────────────── #


def test_bandpass_sos_passes_center_frequency():
    from scipy.signal import sosfilt
    f_center = 1000.0
    sr = _SR
    sos = _bandpass_sos(f_center, sr)
    # Impulse response: pass a 1 kHz sine through the filter
    sig = _sine(f_center, amp=1.0, dur=0.5, sr=sr)
    filtered = sosfilt(sos, sig)
    # Energy should be largely preserved at the center frequency
    assert float(np.sqrt(np.mean(filtered ** 2))) > 0.3


def test_bandpass_sos_rejects_distant_frequency():
    from scipy.signal import sosfilt
    sr = _SR
    sos = _bandpass_sos(1000.0, sr)
    # 100 Hz should be strongly attenuated
    sig = _sine(100.0, amp=1.0, dur=0.5, sr=sr)
    filtered = sosfilt(sos, sig)
    rms_out = float(np.sqrt(np.mean(filtered ** 2)))
    rms_in = float(np.sqrt(np.mean(sig ** 2)))
    assert rms_out < 0.1 * rms_in


def test_bandpass_sos_returns_ndarray():
    sos = _bandpass_sos(440.0, _SR)
    assert isinstance(sos, np.ndarray)
    assert sos.ndim == 2  # SOS format: (n_sections, 6)


# ── dark_fig / dark_ax ─────────────────────────────────────────────────────── #


def test_dark_fig_returns_fig_and_axes():
    fig, axes = dark_fig(nrows=1, ncols=2)
    assert isinstance(fig, plt.Figure)
    assert fig.get_facecolor() == pytest.approx(
        plt.matplotlib.colors.to_rgba(BG), abs=0.01
    )


def test_dark_ax_sets_dark_background():
    fig, ax = plt.subplots()
    returned = dark_ax(ax)
    assert returned is ax
    assert ax.get_facecolor() == pytest.approx(
        plt.matplotlib.colors.to_rgba(BG), abs=0.01
    )


def test_dark_fig_multi_row_creates_correct_shape():
    fig, axes = dark_fig(nrows=2, ncols=3)
    assert axes.shape == (2, 3)


# ── _save ──────────────────────────────────────────────────────────────────── #


def test_save_creates_file(tmp_path):
    fig, _ = dark_fig()
    out = tmp_path / "sub" / "fig.png"
    _save(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_none_path_does_nothing(tmp_path):
    fig, _ = dark_fig()
    _save(fig, None)  # must not raise


# ── Plot smoke tests ──────────────────────────────────────────────────────── #
# Each test verifies the function runs without error and returns a Figure.
# Use short signals at _SR=16000 to keep tests fast.


def test_plot_static_transfer_curve_returns_figure():
    from pedal_model.eval.diagnostics import plot_static_transfer_curve
    dry, target, predicted = _three(dur=0.5)
    fig = plot_static_transfer_curve(dry, target, predicted, sr=_SR)
    assert isinstance(fig, plt.Figure)


def test_plot_describing_function_returns_figure():
    from pedal_model.eval.diagnostics import plot_describing_function
    dry, target, predicted = _three(dur=1.0)
    fig = plot_describing_function(dry, target, predicted, sr=_SR)
    assert isinstance(fig, plt.Figure)


def test_plot_lag_error_returns_figure():
    from pedal_model.eval.diagnostics import plot_lag_error
    dry, target, predicted = _three(dur=0.5)
    fig = plot_lag_error(dry, target, predicted, sr=_SR)
    assert isinstance(fig, plt.Figure)


def test_plot_transfer_function_returns_figure():
    from pedal_model.eval.diagnostics import plot_transfer_function
    dry, target, predicted = _three(dur=0.5)
    fig = plot_transfer_function(dry, target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_harmonic_profile_returns_figure():
    from pedal_model.eval.diagnostics import plot_harmonic_profile
    dry, target, predicted = _three(dur=1.0)
    # signature: (dry, target, predicted, sr, fundamentals=None, ...)
    fig = plot_harmonic_profile(dry, target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_waveform_morphology_returns_figure():
    from pedal_model.eval.diagnostics import plot_waveform_morphology
    # signature: (dry, wet, sr, ...) — uses wet (target), not predicted
    dry, wet, _ = _three(dur=1.0)
    fig = plot_waveform_morphology(dry, wet, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_spectrogram_overlay_returns_figure():
    from pedal_model.eval.diagnostics import plot_spectrogram_overlay
    # signature: (target, predicted, sr, ...) — no dry parameter
    _, target, predicted = _three(dur=2.0)
    fig = plot_spectrogram_overlay(target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_error_spectrogram_returns_figure():
    from pedal_model.eval.diagnostics import plot_error_spectrogram
    # signature: (target, predicted, sr, ...) — no dry parameter
    _, target, predicted = _three(dur=2.0)
    fig = plot_error_spectrogram(target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_gain_reduction_returns_figure():
    from pedal_model.eval.diagnostics import plot_gain_reduction
    # signature: (dry, wet, sr, ...) — uses wet (target), not predicted
    dry, wet, _ = _three(dur=0.5)
    fig = plot_gain_reduction(dry, wet, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_save_path_creates_file(tmp_path):
    """Verify save_path kwarg works for at least one plot function."""
    from pedal_model.eval.diagnostics import plot_static_transfer_curve
    dry, target, predicted = _three(dur=0.2)
    out = tmp_path / "transfer.png"
    fig = plot_static_transfer_curve(dry, target, predicted, sr=_SR, save_path=out)
    assert out.exists()


# ── Utility helper tests ──────────────────────────────────────────────────── #


def test_synth_sine_length():
    from pedal_model.eval.diagnostics import _synth_sine
    out = _synth_sine(440.0, _SR, 0.5, 0.8)
    assert len(out) == int(_SR * 0.5)


def test_synth_sine_dtype_float32():
    from pedal_model.eval.diagnostics import _synth_sine
    out = _synth_sine(440.0, _SR, 0.1, 1.0)
    assert out.dtype == np.float32


def test_synth_sine_amplitude():
    from pedal_model.eval.diagnostics import _synth_sine
    out = _synth_sine(440.0, _SR, 1.0, 0.5)
    assert abs(float(np.max(np.abs(out))) - 0.5) < 0.01


def test_fft_amp_at_pure_sine():
    """_fft_amp_at must return ~amp for a pure sine at f0."""
    from pedal_model.eval.diagnostics import _fft_amp_at, _synth_sine
    sig = _synth_sine(1000.0, _SR, 2.0, 0.7)
    amp = _fft_amp_at(sig, _SR, 1000.0)
    assert abs(amp - 0.7) < 0.05


def test_fft_amp_at_silence_is_near_zero():
    from pedal_model.eval.diagnostics import _fft_amp_at
    sig = np.zeros(4096, dtype=np.float32)
    amp = _fft_amp_at(sig, _SR, 440.0)
    assert amp < 1e-6


# ── Additional plot smoke tests ───────────────────────────────────────────── #


def test_plot_impulse_response_returns_figure():
    from pedal_model.eval.diagnostics import plot_impulse_response
    dry, target, predicted = _three(dur=1.0)
    fig = plot_impulse_response(dry, target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_step_response_returns_figure():
    from pedal_model.eval.diagnostics import plot_step_response
    # Step response needs transitions — ramp signal provides those
    np.random.seed(0)
    n = int(_SR * 1.0)
    dry = np.sign(np.random.randn(n)).astype(np.float32) * 0.5
    target = np.tanh(dry * 3.0).astype(np.float32)
    predicted = np.tanh(dry * 2.8).astype(np.float32)
    fig = plot_step_response(dry, target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_group_delay_returns_figure():
    from pedal_model.eval.diagnostics import plot_group_delay
    dry, target, predicted = _three(dur=1.0)
    fig = plot_group_delay(dry, target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_coherence_returns_figure():
    from pedal_model.eval.diagnostics import plot_coherence
    dry, target, predicted = _three(dur=2.0)
    fig = plot_coherence(dry, target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_imd_returns_figure():
    from pedal_model.eval.diagnostics import plot_imd
    dry, target, predicted = _three(dur=2.0)
    fig = plot_imd(dry, target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_level_dependent_fr_returns_figure():
    from pedal_model.eval.diagnostics import plot_level_dependent_fr
    dry, target, predicted = _three(dur=2.0)
    fig = plot_level_dependent_fr(dry, target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_nonlinear_frequency_map_returns_figure():
    from pedal_model.eval.diagnostics import plot_nonlinear_frequency_map
    dry, target, predicted = _three(dur=2.0)
    fig = plot_nonlinear_frequency_map(dry, target, predicted, _SR, n_tones=4)
    assert isinstance(fig, plt.Figure)


def test_plot_aweighted_error_returns_figure():
    from pedal_model.eval.diagnostics import plot_aweighted_error
    _, target, predicted = _three(dur=1.0)
    fig = plot_aweighted_error(target, predicted, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_envelope_comparison_returns_figure():
    from pedal_model.eval.diagnostics import plot_envelope_comparison
    dry, wet, _ = _three(dur=1.0)
    fig = plot_envelope_comparison(dry, wet, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_phase_portrait_returns_figure():
    from pedal_model.eval.diagnostics import plot_phase_portrait
    dry, wet, _ = _three(dur=1.0)
    fig = plot_phase_portrait(dry, wet, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_odd_even_harmonic_ratio_returns_figure():
    from pedal_model.eval.diagnostics import plot_odd_even_harmonic_ratio
    dry, wet, _ = _three(dur=2.0)
    fig = plot_odd_even_harmonic_ratio(dry, wet, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_frequency_smearing_matrix_returns_figure():
    from pedal_model.eval.diagnostics import plot_frequency_smearing_matrix
    dry, wet, _ = _three(dur=2.0)
    fig = plot_frequency_smearing_matrix(dry, wet, _SR, n_tones=4)
    assert isinstance(fig, plt.Figure)


def test_plot_coherence_nonlinearity_returns_figure():
    from pedal_model.eval.diagnostics import plot_coherence_nonlinearity
    dry, wet, _ = _three(dur=2.0)
    fig = plot_coherence_nonlinearity(dry, wet, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_dynamic_transfer_curve_returns_figure():
    from pedal_model.eval.diagnostics import plot_dynamic_transfer_curve
    dry, wet, _ = _three(dur=1.0)
    fig = plot_dynamic_transfer_curve(dry, wet, _SR)
    assert isinstance(fig, plt.Figure)


# ── Zoo-results analysis functions ────────────────────────────────────────── #


def _make_zoo():
    """Minimal ZooResults with two models and two sections each."""
    from pedal_model.eval.results import ModelResult, ZooResults

    def _mr(name, esr, null_depth, stft=0.1, fit_t=1.0, pred_t=0.1):
        mr = ModelResult(
            name=name, model_type="neural",
            prediction=np.zeros(1000, dtype=np.float32),
            fit_time_s=fit_t, predict_time_s=pred_t,
        )
        mr.metrics_per_section = {
            "sec_a": {"ESR": esr, "null_depth_dB": null_depth, "STFT": stft},
            "sec_b": {"ESR": esr * 0.9, "null_depth_dB": null_depth + 1.0, "STFT": stft * 0.9},
        }
        return mr

    zoo = ZooResults(
        config={}, date_run="2026-01-01", sr=16_000,
        dry_val=np.zeros(1000, dtype=np.float32),
        wet_val=np.zeros(1000, dtype=np.float32),
    )
    zoo.models["fir"]  = _mr("fir",  esr=0.02, null_depth=25.0, fit_t=0.3,  pred_t=0.01)
    zoo.models["lstm"] = _mr("lstm", esr=1.20, null_depth=3.0,  fit_t=30.0, pred_t=0.5)
    return zoo


def test_compute_viable_models_classifies_correctly():
    from pedal_model.eval.diagnostics import compute_viable_models
    zoo = _make_zoo()
    result = compute_viable_models(zoo, ["fir", "lstm"],
                                   esr_threshold=0.5, null_depth_threshold=10.0)
    assert set(result.keys()) == {"viable", "marginal", "not_viable"}
    viable_names = [n for n, _, _ in result["viable"]]
    not_viable_names = [n for n, _, _ in result["not_viable"]]
    assert "fir" in viable_names
    assert "lstm" in not_viable_names


def test_compute_viable_models_all_viable():
    from pedal_model.eval.diagnostics import compute_viable_models
    zoo = _make_zoo()
    result = compute_viable_models(zoo, ["fir", "lstm"],
                                   esr_threshold=2.0, null_depth_threshold=-100.0)
    assert len(result["not_viable"]) == 0


def test_compute_viable_models_tuple_fields():
    from pedal_model.eval.diagnostics import compute_viable_models
    zoo = _make_zoo()
    result = compute_viable_models(zoo, ["fir"])
    for bucket in result.values():
        for item in bucket:
            name, esr, null_d = item
            assert isinstance(name, str)
            assert isinstance(esr, float)
            assert isinstance(null_d, float)


def test_plot_esr_null_scatter_returns_figure():
    from pedal_model.eval.diagnostics import plot_esr_null_scatter
    zoo = _make_zoo()
    fig = plot_esr_null_scatter(zoo, ["fir", "lstm"])
    assert isinstance(fig, plt.Figure)


def test_plot_esr_stft_scatter_returns_figure():
    from pedal_model.eval.diagnostics import plot_esr_stft_scatter
    zoo = _make_zoo()
    fig = plot_esr_stft_scatter(zoo, ["fir", "lstm"])
    assert isinstance(fig, plt.Figure)


def test_plot_timing_chart_returns_figure():
    from pedal_model.eval.diagnostics import plot_timing_chart
    zoo = _make_zoo()
    fig = plot_timing_chart(zoo, ["fir", "lstm"])
    assert isinstance(fig, plt.Figure)


def test_plot_timing_chart_empty_returns_none():
    from pedal_model.eval.diagnostics import plot_timing_chart
    zoo = _make_zoo()
    result = plot_timing_chart(zoo, [])
    assert result is None


def test_build_perceptual_heatmap_df_shape():
    from pedal_model.eval.diagnostics import build_perceptual_heatmap_df
    zoo = _make_zoo()
    perc_cache = {
        "fir":  {"CDPAM": 0.8},
        "lstm": {"CDPAM": 0.5},
    }
    df = build_perceptual_heatmap_df(zoo, ["fir", "lstm"],
                                      perc_cache, ["ESR", "STFT", "CDPAM"])
    assert df.shape == (2, 3)
    assert list(df.index) == ["fir", "lstm"]
    assert "ESR" in df.columns and "CDPAM" in df.columns


def test_build_perceptual_heatmap_df_values():
    from pedal_model.eval.diagnostics import build_perceptual_heatmap_df
    zoo = _make_zoo()
    perc_cache = {"fir": {"CDPAM": 0.9}, "lstm": {"CDPAM": 0.4}}
    df = build_perceptual_heatmap_df(zoo, ["fir", "lstm"],
                                      perc_cache, ["ESR", "CDPAM"])
    assert abs(df.loc["fir", "CDPAM"] - 0.9) < 1e-6
    # ESR for fir: mean of sec_a=0.02, sec_b=0.018 → ~0.019
    assert df.loc["fir", "ESR"] < 0.05
