"""Tests for pedal_model.metrics.perceptual."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pedal_model.metrics.perceptual import (
    compute_lsd,
    compute_mcd,
    composite_perceptual,
    equal_loudness_weighted,
    guitar_prior_weighted,
    harmonic_weighted,
    masking_weighted,
    mel_axis_weighted,
    run_all,
    signal_adaptive_weighted,
)

SR = 48_000
RNG = np.random.default_rng(0)


def _noise(seed: int = 0, n: int = SR * 2) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.3


def _sine(freq: float = 440.0, sr: int = SR, n: int = SR * 2, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _clipped(signal: np.ndarray, clip: float = 0.5) -> np.ndarray:
    return np.clip(signal * 4.0, -clip, clip).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Original metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_mcd_identical_signals():
    signal = _noise(0)
    mcd = compute_mcd(signal, signal, SR)
    assert mcd == pytest.approx(0.0, abs=1e-4)


def test_mcd_non_negative():
    a, b = _noise(0), _noise(1)
    assert compute_mcd(a, b, SR) >= 0.0


def test_mcd_increases_with_distortion():
    signal = _noise(0)
    slight = (signal + 0.01 * _noise(1)).astype(np.float32)
    heavy = (signal + 1.0 * _noise(2)).astype(np.float32)
    assert compute_mcd(signal, slight, SR) < compute_mcd(signal, heavy, SR)


def test_lsd_identical():
    inp = _noise(0)
    out = inp * 0.8
    lsd = compute_lsd(out, out, inp, SR)
    assert lsd == pytest.approx(0.0, abs=1e-4)


def test_lsd_non_negative():
    inp = _noise(0)
    t = (inp * 0.8).astype(np.float32)
    p = (inp * 0.5).astype(np.float32)
    assert compute_lsd(t, p, inp, SR) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Shared contract tests (run against all six new metrics)
# ─────────────────────────────────────────────────────────────────────────────

_SIX = [
    mel_axis_weighted,
    equal_loudness_weighted,
    masking_weighted,
    signal_adaptive_weighted,
    guitar_prior_weighted,
    harmonic_weighted,
]
_SIX_IDS = [
    "mel_axis",
    "equal_loudness",
    "masking",
    "signal_adaptive",
    "guitar_prior",
    "harmonic",
]


@pytest.mark.parametrize("fn", _SIX, ids=_SIX_IDS)
def test_new_metric_returns_required_keys(fn):
    dry = _noise(0)
    target = _clipped(dry)
    pred = _noise(1)
    result = fn(pred, target, dry, sr=SR)
    for key in ("score", "label", "unit", "metadata", "arrays"):
        assert key in result, f"'{key}' missing from {fn.__name__} result"


@pytest.mark.parametrize("fn", _SIX, ids=_SIX_IDS)
def test_new_metric_identical_pred_target_near_zero(fn):
    """Identical pred and target should produce score ≈ 0."""
    dry = _noise(0)
    target = _clipped(dry)
    result = fn(target, target, dry, sr=SR)
    assert result["score"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("fn", _SIX, ids=_SIX_IDS)
def test_new_metric_non_negative(fn):
    dry = _noise(0)
    target = _clipped(dry)
    pred = _noise(1)
    result = fn(pred, target, dry, sr=SR)
    score = result["score"]
    assert math.isfinite(score) and score >= 0.0


@pytest.mark.parametrize("fn", _SIX, ids=_SIX_IDS)
def test_new_metric_length_mismatch_raises(fn):
    dry = _noise(0)
    target = dry.copy()
    pred = _noise(1, n=SR)  # wrong length
    with pytest.raises(ValueError, match="length"):
        fn(pred, target, dry, sr=SR)


@pytest.mark.parametrize("fn", _SIX, ids=_SIX_IDS)
def test_new_metric_silent_input_returns_zero(fn):
    n = SR * 2
    silent = np.zeros(n, dtype=np.float32)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = fn(silent, silent, silent, sr=SR)
    assert result["score"] == 0.0


@pytest.mark.parametrize("fn", _SIX, ids=_SIX_IDS)
def test_new_metric_short_signal_returns_nan(fn):
    short = _noise(0, n=1024)  # below 4096 threshold
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = fn(short, short, short, sr=SR)
    assert math.isnan(result["score"])


@pytest.mark.parametrize("fn", _SIX, ids=_SIX_IDS)
def test_new_metric_larger_error_gives_larger_score(fn):
    """Adding more noise should increase the score.

    masking_weighted uses a quiet target so that the masking threshold is low
    enough for noise to poke above it.
    """
    dry = _noise(0)
    if fn is masking_weighted:
        # Clipped target creates a very high masking threshold that hides both
        # small and large noise.  Use a quiet sine instead.
        target = (0.05 * _sine(440.0)).astype(np.float32)
    else:
        target = _clipped(dry)
    small_noise = (target + 0.01 * _noise(1)).astype(np.float32)
    big_noise   = (target + 0.50 * _noise(2)).astype(np.float32)
    r_small = fn(small_noise, target, dry, sr=SR)["score"]
    r_big   = fn(big_noise,   target, dry, sr=SR)["score"]
    assert r_small < r_big


# ─────────────────────────────────────────────────────────────────────────────
# Metric-specific unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_mel_axis_error_map_shape():
    dry = _noise(0)
    target = _clipped(dry)
    result = mel_axis_weighted(target + 0.05 * _noise(1), target, dry, sr=SR)
    em = result["arrays"]["error_map"]
    assert em.ndim == 2
    assert em.shape[0] == 128  # n_mels default


def test_equal_loudness_a_weights_peak_around_4khz():
    dry = _noise(0)
    target = _clipped(dry)
    result = equal_loudness_weighted(_noise(1), target, dry, sr=SR)
    freqs = np.array(result["metadata"]["freqs_hz"])
    weights_db = np.array(result["metadata"]["a_weights_db"])
    peak_freq = freqs[np.argmax(weights_db)]
    # A-weighting peaks around 3.5–4 kHz
    assert 2000.0 < peak_freq < 6000.0


def test_masking_masked_fraction_in_01():
    dry = _noise(0)
    target = _clipped(dry)
    result = masking_weighted(_noise(1), target, dry, sr=SR)
    mf = result["metadata"]["masked_fraction"]
    assert 0.0 <= mf <= 1.0


def test_signal_adaptive_effective_snr_shape():
    dry = _noise(0)
    target = _clipped(dry)
    result = signal_adaptive_weighted(_noise(1), target, dry, sr=SR)
    snr = np.array(result["metadata"]["effective_snr_db"])
    # Should have n_fft//2+1 = 1025 bins for n_fft=2048
    assert snr.shape == (2048 // 2 + 1,)


def test_guitar_prior_metadata_keys():
    dry = _noise(0)
    target = _clipped(dry)
    result = guitar_prior_weighted(_noise(1), target, dry, sr=SR)
    for key in ("prior_weights_db", "freqs_hz", "per_bin_weighted_error"):
        assert key in result["metadata"]


def test_guitar_prior_weights_peak_in_guitar_range():
    """Prior should be highest between 80 Hz and 5 kHz."""
    dry = _noise(0)
    target = _clipped(dry)
    result = guitar_prior_weighted(_noise(1), target, dry, sr=SR, n_fft=4096)
    freqs = np.array(result["metadata"]["freqs_hz"])
    prior_db = np.array(result["metadata"]["prior_weights_db"])
    in_range = (freqs >= 80.0) & (freqs <= 5000.0)
    out_range = (freqs < 40.0) | (freqs > 15000.0)
    assert prior_db[in_range].mean() > prior_db[out_range].mean()


def test_harmonic_metadata_keys():
    dry = _sine(220.0)
    target = _clipped(dry)
    result = harmonic_weighted(_noise(1), target, dry, sr=SR)
    for key in (
        "f0_hz", "harmonic_freqs", "pred_harmonic_amps",
        "target_harmonic_amps", "harmonic_amp_error_db",
        "even_harmonic_error_db", "odd_harmonic_error_db",
    ):
        assert key in result["metadata"]


def test_harmonic_f0_override():
    """Supplying f0 directly should be used without estimation."""
    dry = _sine(440.0)
    target = _clipped(dry)
    result = harmonic_weighted(_noise(1), target, dry, sr=SR, f0=440.0)
    assert abs(result["metadata"]["f0_hz"] - 440.0) < 1.0


def test_harmonic_weight_mask_shape():
    dry = _sine(220.0)
    target = _clipped(dry)
    result = harmonic_weighted(_noise(1), target, dry, sr=SR, f0=220.0, n_fft=2048)
    mask = result["arrays"]["weight_mask"]
    assert mask.shape == (2048 // 2 + 1,)
    assert mask.mean() == pytest.approx(1.0, rel=0.05)  # normalized to mean ≈ 1


# ─────────────────────────────────────────────────────────────────────────────
# Composite
# ─────────────────────────────────────────────────────────────────────────────

def test_composite_required_keys():
    dry = _noise(0)
    target = _clipped(dry)
    result = composite_perceptual(_noise(1), target, dry, sr=SR)
    for key in ("composite_score", "individual_scores", "individual_results", "weights_used"):
        assert key in result


def test_composite_individual_scores_finite():
    dry = _noise(0)
    target = _clipped(dry)
    result = composite_perceptual(_noise(1), target, dry, sr=SR)
    for name, score in result["individual_scores"].items():
        assert math.isfinite(score), f"{name} score is not finite"


def test_run_all_equals_composite():
    dry = _noise(0)
    target = _clipped(dry)
    pred = _noise(1)
    r_all = run_all(pred, target, dry, sr=SR)
    r_comp = composite_perceptual(pred, target, dry, sr=SR)
    assert r_all["composite_score"] == pytest.approx(r_comp["composite_score"], rel=1e-6)
