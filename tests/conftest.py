"""Shared pytest fixtures for the PedalDSP test suite."""
import numpy as np
import pytest
import torch


@pytest.fixture
def sr() -> int:
    return 48000


@pytest.fixture
def short_noise(sr) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(sr).astype(np.float32) * 0.3


@pytest.fixture
def sine_440(sr) -> np.ndarray:
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def sine_1k(sr) -> np.ndarray:
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)


@pytest.fixture
def dry_wet_gain(short_noise) -> tuple[np.ndarray, np.ndarray]:
    """Dry/wet pair where wet = dry * 0.7 (linear gain change only)."""
    return short_noise, (short_noise * 0.7).astype(np.float32)


@pytest.fixture
def dry_wet_clip(short_noise) -> tuple[np.ndarray, np.ndarray]:
    """Dry/wet pair where wet = soft-clipped dry (nonlinear)."""
    wet = np.tanh(short_noise * 4.0).astype(np.float32)
    return short_noise, wet
