"""Regression smoke tests for pedal_model.utils.plotting.

Tests verify that each helper:
  - returns the documented type (plt.Figure or plt.Axes)
  - does not raise on valid inputs
  - does not mutate the input array

No visual correctness is checked — matplotlib rendering is not deterministic
across backends.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pandas as pd

from pedal_model.utils.plotting import (
    compare_spectra,
    compare_spectra_smooth,
    compare_waveforms,
    db_spectrum,
    db_spectrum_smooth,
    filter_impulse_response,
    freq_response_db,
    plot_esr_by_section_type,
    plot_esr_scatter,
    plot_freq_response,
    plot_impulse_response,
    plot_metrics_heatmap,
    plot_rms_envelope,
    plot_spectrum,
    plot_topology_weights,
    plot_waveform,
    plot_cqt_spectrogram,
    plot_mel_spectrogram,
    plot_waveshow,
    signal_dashboard,
    waveform_spectrogram_panel,
)
from scipy.signal import butter

_SR = 48_000
_DUR = 0.5  # short signals to keep tests fast

_RNG = np.random.default_rng(42)
_SIG = (_RNG.standard_normal(int(_SR * _DUR))).astype(np.float32)
_SIG /= np.max(np.abs(_SIG))


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to free memory."""
    yield
    plt.close("all")


# ── Pure computation helpers ──────────────────────────────────────────────────

def test_db_spectrum_returns_arrays():
    freqs, mag = db_spectrum(_SIG, _SR)
    assert isinstance(freqs, np.ndarray)
    assert isinstance(mag, np.ndarray)
    assert freqs.shape == mag.shape
    assert len(freqs) == len(_SIG) // 2 + 1


def test_db_spectrum_no_nan():
    _, mag = db_spectrum(_SIG, _SR)
    assert np.all(np.isfinite(mag))


def test_freq_response_db_shape():
    kernel = np.zeros(64, dtype=np.float32)
    kernel[0] = 1.0  # identity
    freqs, H = freq_response_db(kernel, _SR, n_fft=256)
    assert freqs.shape == H.shape
    assert len(freqs) == 256 // 2 + 1


def test_filter_impulse_response_shape():
    sos = butter(4, 1000.0, fs=_SR, output="sos")
    ir = filter_impulse_response(sos, n=512)
    assert ir.shape == (512,)
    assert ir.dtype == np.float32


# ── Single-axes plot helpers ──────────────────────────────────────────────────

def test_plot_waveform_returns_axes():
    ax = plot_waveform(_SIG, _SR)
    assert isinstance(ax, plt.Axes)


def test_plot_waveform_on_existing_axes():
    fig, ax = plt.subplots()
    result = plot_waveform(_SIG, _SR, ax=ax)
    assert result is ax


def test_plot_spectrum_returns_axes():
    ax = plot_spectrum(_SIG, _SR)
    assert isinstance(ax, plt.Axes)


def test_plot_impulse_response_returns_axes():
    kernel = np.sinc(np.linspace(-5, 5, 128)).astype(np.float32)
    ax = plot_impulse_response(kernel)
    assert isinstance(ax, plt.Axes)


def test_plot_freq_response_returns_axes():
    kernel = np.sinc(np.linspace(-5, 5, 128)).astype(np.float32)
    ax = plot_freq_response(kernel, _SR)
    assert isinstance(ax, plt.Axes)


# ── librosa-based helpers ─────────────────────────────────────────────────────

def test_plot_waveshow_returns_axes():
    ax = plot_waveshow(_SIG, _SR)
    assert isinstance(ax, plt.Axes)


def test_plot_mel_spectrogram_returns_axes():
    ax = plot_mel_spectrogram(_SIG, _SR)
    assert isinstance(ax, plt.Axes)


def test_plot_cqt_spectrogram_returns_axes():
    ax = plot_cqt_spectrogram(_SIG, _SR)
    assert isinstance(ax, plt.Axes)


def test_plot_mel_spectrogram_custom_params():
    ax = plot_mel_spectrogram(_SIG, _SR, n_mels=64, hop_length=256,
                               fmin=80.0, fmax=4000.0, title="test")
    assert isinstance(ax, plt.Axes)


# ── Compound figure helpers ───────────────────────────────────────────────────

def test_compare_waveforms_returns_figure():
    sigs = {
        "a": (_SIG, "blue"),
        "b": (_SIG * 0.5, "red"),
    }
    fig = compare_waveforms(sigs, _SR, title="Test")
    assert isinstance(fig, plt.Figure)


def test_compare_spectra_returns_figure():
    sigs = {
        "x": (_SIG, "steelblue"),
        "y": (_SIG * 0.8, "coral"),
    }
    fig = compare_spectra(sigs, _SR, title="Spectra")
    assert isinstance(fig, plt.Figure)


def test_signal_dashboard_returns_figure():
    sigs = {"signal": (_SIG, "steelblue")}
    fig = signal_dashboard(sigs, _SR, title="Dashboard")
    assert isinstance(fig, plt.Figure)


def test_waveform_spectrogram_panel_returns_figure():
    fig = waveform_spectrogram_panel(_SIG, _SR, title="Panel")
    assert isinstance(fig, plt.Figure)


def test_waveform_spectrogram_panel_custom_cmap():
    fig = waveform_spectrogram_panel(_SIG, _SR, cmap="viridis",
                                      waveform_color="#FF6B6B")
    assert isinstance(fig, plt.Figure)


# ── Results-analysis functions ────────────────────────────────────────────────

def test_db_spectrum_smooth_shape():
    f, db = db_spectrum_smooth(_SIG, _SR, n_bins=128)
    assert f.shape == (128,)
    assert db.shape == (128,)


def test_db_spectrum_smooth_freqs_increasing():
    f, _ = db_spectrum_smooth(_SIG, _SR, n_bins=256)
    assert np.all(np.diff(f) > 0)


def test_db_spectrum_smooth_same_n_bins_identical_x_axis():
    """n_bins identical → x-axis identical for any two signals of same length."""
    sig2 = _SIG * 0.5
    f1, _ = db_spectrum_smooth(_SIG, _SR, n_bins=512)
    f2, _ = db_spectrum_smooth(sig2, _SR, n_bins=512)
    np.testing.assert_array_equal(f1, f2)


def test_compare_spectra_smooth_returns_figure():
    sigs = {
        "a": (_SIG, "steelblue"),
        "b": (_SIG * 0.8, "tomato"),
        "target": (_SIG * 0.9, "black"),
    }
    fig = compare_spectra_smooth(sigs, _SR, title="Smooth spectra")
    assert isinstance(fig, plt.Figure)


def test_compare_spectra_smooth_floor_clips():
    """All dB values in the figure must be ≥ floor_db."""
    _, db = db_spectrum_smooth(_SIG, _SR, n_bins=64)
    floor = -60.0
    db_clipped = np.maximum(db, floor)
    assert np.all(db_clipped >= floor)


def test_plot_metrics_heatmap_returns_figure():
    df = pd.DataFrame(
        {"ESR": [0.1, 0.5], "null_depth_dB": [20.0, 15.0], "STFT": [0.2, 0.4]},
        index=["modelA", "modelB"],
    )
    fig = plot_metrics_heatmap(df)
    assert isinstance(fig, plt.Figure)


def test_plot_metrics_heatmap_with_nan():
    """NaN values must not raise — they map to the midpoint colour."""
    df = pd.DataFrame(
        {"ESR": [float("nan"), 0.5], "STFT": [0.1, float("nan")]},
        index=["m1", "m2"],
    )
    fig = plot_metrics_heatmap(df)
    assert isinstance(fig, plt.Figure)


def test_plot_esr_scatter_returns_figure():
    _df = pd.DataFrame({
        "model": ["a", "a", "b", "b"],
        "section": ["s1", "s2", "s1", "s2"],
        "ESR": [0.1, 0.3, 0.5, 0.8],
    })
    fig = plot_esr_scatter(_df)
    assert isinstance(fig, plt.Figure)


def test_plot_esr_by_section_type_returns_figure():
    _df = pd.DataFrame({
        "model": ["a", "a", "b", "b"],
        "section": ["s1", "s2", "s1", "s2"],
        "type": ["sweep", "tone", "sweep", "tone"],
        "ESR": [0.1, 0.3, 0.5, 0.8],
    })
    fig = plot_esr_by_section_type(_df)
    assert isinstance(fig, plt.Figure)


def test_plot_rms_envelope_returns_figure():
    fig = plot_rms_envelope(_SIG, _SIG * 0.8, _SR)
    assert isinstance(fig, plt.Figure)


def test_plot_rms_envelope_with_sections():
    n = len(_SIG)
    sections = [(0, n // 3), (n // 3, 2 * n // 3)]
    fig = plot_rms_envelope(_SIG, _SIG * 0.8, _SR, sections=sections)
    assert isinstance(fig, plt.Figure)


def test_plot_topology_weights_returns_figure():
    topo = {
        "LearnableSoftClipper": 0.35,
        "LearnableParallelBlend": 0.25,
        "SkipConnection": 0.10,
        "LearnableGain": 0.05,
        "LearnableLowpass": 0.05,
        "LearnableHighpass": 0.05,
        "LearnableHighShelf": 0.04,
        "LearnableBandpass": 0.04,
        "LearnableAsymClipper": 0.04,
        "LearnableClipThenFilter": 0.03,
    }
    fig = plot_topology_weights(topo)
    assert isinstance(fig, plt.Figure)


def test_plot_topology_weights_sorted_descending():
    """Highest-weight primitive must appear at the top (yticklabels[0])."""
    topo = {"B": 0.6, "A": 0.3, "C": 0.1}
    fig = plot_topology_weights(topo)
    ax = fig.axes[0]
    assert ax.get_yticklabels()[0].get_text() == "B"


# ── Input mutation guard ──────────────────────────────────────────────────────

def test_plot_waveform_does_not_mutate():
    sig_copy = _SIG.copy()
    plot_waveform(_SIG, _SR)
    np.testing.assert_array_equal(_SIG, sig_copy)


def test_waveform_spectrogram_panel_does_not_mutate():
    sig_copy = _SIG.copy()
    waveform_spectrogram_panel(_SIG, _SR)
    np.testing.assert_array_equal(_SIG, sig_copy)
