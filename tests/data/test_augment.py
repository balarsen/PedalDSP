"""Tests for pedal_model.data.augment."""
import numpy as np
import pytest

from pedal_model.data.augment import AddNoise, GainJitter


def test_gain_jitter_shape_preserved():
    rng = np.random.default_rng(0)
    dry = rng.standard_normal(200).astype(np.float32)
    wet = rng.standard_normal(200).astype(np.float32)
    aug = GainJitter(min_db=-6.0, max_db=6.0)
    d_aug, w_aug = aug(dry, wet)
    assert d_aug.shape == dry.shape
    assert w_aug.shape == wet.shape
    assert d_aug.dtype == np.float32


def test_gain_jitter_applies_same_gain_to_both():
    """Dry/wet ratio must be preserved after gain jitter."""
    rng = np.random.default_rng(1)
    dry = rng.standard_normal(500).astype(np.float32) + 0.1  # avoid zeros
    wet = dry * 2.0  # known ratio
    aug = GainJitter(min_db=-3.0, max_db=3.0)
    d_aug, w_aug = aug(dry, wet)
    # ratio wet/dry should be preserved
    ratio_before = wet / dry
    ratio_after = w_aug / d_aug
    np.testing.assert_allclose(ratio_after, ratio_before, rtol=1e-5)


def test_gain_jitter_stays_in_db_range():
    """Applied gain should be within [min_db, max_db]."""
    rng = np.random.default_rng(2)
    dry = np.ones(100, dtype=np.float32)
    wet = np.ones(100, dtype=np.float32)
    aug = GainJitter(min_db=-6.0, max_db=6.0)
    d_aug, _ = aug(dry, wet)
    applied_gain = float(d_aug[0])
    applied_db = 20.0 * np.log10(applied_gain)
    assert -6.0 <= applied_db <= 6.0


def test_add_noise_wet_unchanged():
    """AddNoise modifies dry only — wet must be identical."""
    rng = np.random.default_rng(3)
    dry = rng.standard_normal(300).astype(np.float32)
    wet = rng.standard_normal(300).astype(np.float32)
    aug = AddNoise(noise_floor_db=-80.0)
    d_aug, w_aug = aug(dry, wet)
    np.testing.assert_array_equal(w_aug, wet)


def test_add_noise_dry_modified():
    rng = np.random.default_rng(4)
    dry = rng.standard_normal(300).astype(np.float32)
    wet = np.zeros(300, dtype=np.float32)
    aug = AddNoise(noise_floor_db=-60.0)
    d_aug, _ = aug(dry, wet)
    assert not np.array_equal(d_aug, dry)


def test_add_noise_level_is_below_signal():
    """Noise floor at -60 dBFS should be at least 40 dB below a 0 dBFS signal."""
    signal = np.ones(10_000, dtype=np.float32)  # 0 dBFS
    aug = AddNoise(noise_floor_db=-60.0)
    d_aug, _ = aug(signal.copy(), signal.copy())
    noise = d_aug - signal
    noise_rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(noise ** 2))) + 1e-12)
    assert noise_rms_db < -50.0  # at least 50 dB below signal


def test_add_noise_dtype_preserved():
    dry = np.zeros(100, dtype=np.float32)
    wet = np.zeros(100, dtype=np.float32)
    aug = AddNoise()
    d_aug, w_aug = aug(dry, wet)
    assert d_aug.dtype == np.float32
    assert w_aug.dtype == np.float32
