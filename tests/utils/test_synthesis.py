"""Tests for pedal_model.utils.synthesis."""
import numpy as np
import pytest

from pedal_model.utils.synthesis import g_major_chord, guitar_note, pure_sine, white_noise_id


_SR = 48_000


def test_guitar_note_shape():
    sig = guitar_note(440.0, sr=_SR, duration=0.1)
    assert sig.shape == (int(_SR * 0.1),)
    assert sig.dtype == np.float32


def test_guitar_note_peak_amplitude():
    sig = guitar_note(440.0, sr=_SR, duration=0.5, amp=0.8)
    assert np.max(np.abs(sig)) == pytest.approx(0.8, rel=0.05)


def test_guitar_note_reproducible_with_seed():
    a = guitar_note(220.0, sr=_SR, duration=0.1, seed=7)
    b = guitar_note(220.0, sr=_SR, duration=0.1, seed=7)
    np.testing.assert_array_equal(a, b)


def test_guitar_note_different_seed():
    a = guitar_note(220.0, sr=_SR, duration=0.1, seed=1)
    b = guitar_note(220.0, sr=_SR, duration=0.1, seed=2)
    assert not np.allclose(a, b)


def test_g_major_chord_shape():
    chord = g_major_chord(sr=_SR, duration=0.1)
    assert chord.shape == (int(_SR * 0.1),)
    assert chord.dtype == np.float32


def test_g_major_chord_not_silent():
    chord = g_major_chord(sr=_SR, duration=0.1)
    assert np.max(np.abs(chord)) > 1e-3


def test_white_noise_shape():
    sig = white_noise_id(sr=_SR, duration=0.1)
    assert sig.shape == (int(_SR * 0.1),)
    assert sig.dtype == np.float32


def test_white_noise_rms_approx_amplitude():
    amp = 0.3
    sig = white_noise_id(sr=_SR, duration=2.0, amplitude=amp, seed=42)
    rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2)))
    assert rms == pytest.approx(amp, rel=0.05)


def test_white_noise_reproducible():
    a = white_noise_id(seed=99)
    b = white_noise_id(seed=99)
    np.testing.assert_array_equal(a, b)


def test_pure_sine_shape():
    sig = pure_sine(440.0, sr=_SR, duration=0.1)
    assert sig.shape == (int(_SR * 0.1),)
    assert sig.dtype == np.float32


def test_pure_sine_peak():
    amp = 0.7
    sig = pure_sine(440.0, sr=_SR, duration=1.0, amp=amp)
    assert np.max(np.abs(sig)) == pytest.approx(amp, rel=1e-4)


def test_pure_sine_frequency():
    """Dominant FFT bin should be at f0."""
    f0 = 1000.0
    sig = pure_sine(f0, sr=_SR, duration=1.0)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / _SR)
    mag = np.abs(np.fft.rfft(sig))
    peak_f = freqs[np.argmax(mag)]
    assert abs(peak_f - f0) < 2.0  # within 2 Hz
