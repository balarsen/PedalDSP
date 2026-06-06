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

from pedal_model.utils.plotting import (
    compare_spectra,
    compare_waveforms,
    db_spectrum,
    filter_impulse_response,
    freq_response_db,
    plot_freq_response,
    plot_impulse_response,
    plot_spectrum,
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


# ── Input mutation guard ──────────────────────────────────────────────────────

def test_plot_waveform_does_not_mutate():
    sig_copy = _SIG.copy()
    plot_waveform(_SIG, _SR)
    np.testing.assert_array_equal(_SIG, sig_copy)


def test_waveform_spectrogram_panel_does_not_mutate():
    sig_copy = _SIG.copy()
    waveform_spectrogram_panel(_SIG, _SR)
    np.testing.assert_array_equal(_SIG, sig_copy)
