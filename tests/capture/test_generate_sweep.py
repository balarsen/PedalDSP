"""Tests for pedal_model.capture.generate_sweep."""
import numpy as np
import pytest

from pedal_model.capture.generate_sweep import (
    build_capture_signal,
    generate_alignment_click,
    generate_log_sweep,
)

SR = 48000


def test_sweep_length():
    duration = 5.0
    sweep = generate_log_sweep(20, 20000, duration, SR)
    assert len(sweep) == int(SR * duration)


def test_sweep_dtype():
    sweep = generate_log_sweep(20, 20000, 1.0, SR)
    assert sweep.dtype == np.float32


def test_sweep_amplitude_within_bounds():
    amp = 0.5
    sweep = generate_log_sweep(20, 20000, 2.0, SR, amplitude=amp)
    assert np.max(np.abs(sweep)) <= amp + 1e-5


def test_alignment_click_spike_at_zero():
    click = generate_alignment_click(SR)
    assert click[0] == pytest.approx(0.9)
    # rest should be zero
    assert np.all(click[1:] == 0.0)


def test_alignment_click_length():
    duration = 0.25
    click = generate_alignment_click(SR, duration)
    assert len(click) == int(SR * duration)


def test_capture_signal_float32():
    signal = build_capture_signal(SR)
    assert signal.dtype == np.float32


def test_capture_signal_no_clipping():
    signal = build_capture_signal(SR)
    assert np.max(np.abs(signal)) < 1.0


def test_capture_signal_has_click():
    signal = build_capture_signal(SR)
    # Click starts after 1s of silence
    click_region = signal[SR : SR + int(SR * 0.25)]
    assert np.max(np.abs(click_region)) > 0.8
