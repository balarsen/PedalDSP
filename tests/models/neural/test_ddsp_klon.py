"""Tests for pedal_model.models.neural.ddsp_klon — KlonDDSP hybrid model."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pedal_model.models.neural.ddsp_klon import (
    DEFAULT_CONDITION,
    N_KNOBS,
    KlonDDSP,
)

torch.manual_seed(0)
np.random.seed(0)

SR = 16_000
N = 2048
BATCH = 2
CHUNK = 512


def _dry(n: int = N) -> np.ndarray:
    """440 Hz sine, float32."""
    t = np.arange(n, dtype=np.float32) / SR
    return (0.5 * np.sin(2.0 * np.pi * 440 * t)).astype(np.float32)


def _dry_t(batch: int = BATCH, time: int = CHUNK) -> torch.Tensor:
    """Random (batch, time, 1) tensor."""
    torch.manual_seed(1)
    return torch.randn(batch, time, 1) * 0.3


# ── Forward pass shapes ───────────────────────────────────────────────────── #


def test_forward_output_shape_3d_input():
    model = KlonDDSP()
    x = _dry_t()
    out, hidden = model(x)
    assert out.shape == (BATCH, CHUNK, 1)
    assert hidden.shape == (1, BATCH, model.gru_hidden)


def test_forward_output_shape_2d_input():
    model = KlonDDSP()
    x = torch.randn(BATCH, CHUNK)
    out, hidden = model(x)
    assert out.shape == (BATCH, CHUNK, 1)


def test_forward_with_explicit_condition():
    model = KlonDDSP()
    x = _dry_t()
    cond = torch.tensor([0.5, 0.5, 0.5])
    out, _ = model(x, condition=cond)
    assert out.shape == (BATCH, CHUNK, 1)


def test_forward_different_conditions_produce_different_output():
    """Two different knob settings must (eventually) produce different outputs."""
    torch.manual_seed(42)
    model = KlonDDSP(gru_hidden=32)
    # Randomise weights so GRU actually uses the condition
    for p in model.parameters():
        nn.init.uniform_(p, -0.1, 0.1)
    x = _dry_t()
    cond_low = torch.zeros(N_KNOBS)
    cond_high = torch.ones(N_KNOBS)
    out_low, _ = model(x, condition=cond_low)
    out_high, _ = model(x, condition=cond_high)
    # Outputs may differ after weights are initialised non-trivially
    # With uniform random weights the condition *should* produce different outputs
    assert not torch.allclose(out_low, out_high, atol=1e-6)


import torch.nn as nn  # needed for the test above


def test_forward_none_condition_uses_default():
    """None condition and explicit DEFAULT_CONDITION should give the same output."""
    model = KlonDDSP()
    x = _dry_t()
    out_none, _ = model(x, condition=None)
    cond = torch.tensor(DEFAULT_CONDITION)
    out_explicit, _ = model(x, condition=cond)
    torch.testing.assert_close(out_none, out_explicit)


# ── Hidden state carryover ────────────────────────────────────────────────── #


def test_hidden_state_carried_across_chunks():
    model = KlonDDSP()
    x = torch.randn(1, CHUNK * 2)
    _, h1 = model(x[:, :CHUNK])
    _, h2 = model(x[:, CHUNK:], hidden=h1)
    assert h2.shape == (1, 1, model.gru_hidden)


# ── DDSP-only stage ───────────────────────────────────────────────────────── #


def test_predict_ddsp_shape():
    model = KlonDDSP()
    out = model.predict_ddsp(_dry())
    assert out.shape == (_dry().shape,)[0]
    assert out.dtype == np.float32


def test_predict_ddsp_not_silence():
    model = KlonDDSP()
    out = model.predict_ddsp(_dry())
    assert np.max(np.abs(out)) > 1e-4


def test_predict_ddsp_output_bounded():
    """tanh clamp + output_level sigmoid ensures output never blows up."""
    model = KlonDDSP()
    loud = np.ones(N, dtype=np.float32)
    out = model.predict_ddsp(loud)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) < 5.0


# ── Full predict ─────────────────────────────────────────────────────────── #


def test_predict_shape_and_dtype():
    model = KlonDDSP()
    out = model.predict(_dry())
    assert out.shape == (N,)
    assert out.dtype == np.float32


def test_predict_with_explicit_condition():
    model = KlonDDSP()
    cond = np.array([0.3, 0.5, 0.7], dtype=np.float32)
    out = model.predict(_dry(), condition=cond)
    assert out.shape == (N,)
    assert np.isfinite(out).all()


# ── Gradients ────────────────────────────────────────────────────────────── #


def test_gradients_flow_through_ddsp_params():
    model = KlonDDSP()
    x = _dry_t()
    out, _ = model(x)
    loss = out.mean()
    loss.backward()
    assert model._log_drive.grad is not None
    assert torch.isfinite(model._log_drive.grad)


def test_gradients_flow_through_gru():
    model = KlonDDSP()
    x = _dry_t()
    out, _ = model(x)
    loss = out.mean()
    loss.backward()
    for name, p in model.gru.named_parameters():
        assert p.grad is not None, f"No gradient on GRU param {name}"


def test_gradients_flow_through_ff1_alpha():
    """FF1 LPF coefficient must receive gradients from the loss."""
    model = KlonDDSP()
    x = _dry_t()
    out, _ = model(x)
    out.mean().backward()
    assert model._alpha_ff1_raw.grad is not None
    assert torch.isfinite(model._alpha_ff1_raw.grad)


def test_gradients_flow_through_ff2_blend():
    """FF2 blend parameters must receive gradients from the loss."""
    model = KlonDDSP()
    x = _dry_t()
    out, _ = model(x)
    out.mean().backward()
    assert model._ff2_blend_w.grad is not None
    assert model._ff2_blend_b.grad is not None


def test_ff1_attenuates_high_freq():
    """FF1 (106 Hz LPF) should pass a 50 Hz tone and attenuate a 5 kHz tone."""
    import math
    model = KlonDDSP()
    sr = 48000
    N = 4096
    t = torch.arange(N, dtype=torch.float32)

    low = torch.sin(2 * math.pi * 50 / sr * t).unsqueeze(0)
    high = torch.sin(2 * math.pi * 5000 / sr * t).unsqueeze(0)

    with torch.no_grad():
        a = model.alpha_ff1
        low_out = model._lpf(low, a)
        high_out = model._lpf(high, a)

    # Skip the first 256 samples (LPF transient), measure steady-state
    low_rms = low_out[:, 256:].pow(2).mean().sqrt().item()
    high_rms = high_out[:, 256:].pow(2).mean().sqrt().item()
    assert low_rms > high_rms * 10, (
        f"FF1 should attenuate 5 kHz by >20 dB vs 50 Hz: "
        f"low_rms={low_rms:.4f}, high_rms={high_rms:.4f}"
    )


def test_ff2_blend_differs_by_gain():
    """Different gain knob positions must produce different DDSP outputs."""
    torch.manual_seed(0)
    model = KlonDDSP()
    x = torch.randn(1, CHUNK) * 0.3

    with torch.no_grad():
        gain_low = torch.tensor(0.0)
        gain_high = torch.tensor(1.0)
        out_low = model._ddsp_stage(x, gain_low)
        out_high = model._ddsp_stage(x, gain_high)

    assert not torch.allclose(out_low, out_high, atol=1e-6), (
        "FF2 blend must produce different outputs at gain=0 vs gain=1"
    )


def test_gru_input_includes_ddsp_context():
    """GRU input size must be 2 + N_KNOBS (dry + ddsp context + knobs)."""
    from pedal_model.models.neural.ddsp_klon import _GRU_INPUT_SIZE
    model = KlonDDSP()
    assert model.gru.input_size == _GRU_INPUT_SIZE
    assert _GRU_INPUT_SIZE == 2 + N_KNOBS


# ── DDSP fraction analysis ────────────────────────────────────────────────── #


def test_analyse_ddsp_fraction_returns_expected_keys():
    model = KlonDDSP()
    dry = _dry()
    wet = _dry() * 0.5 + 0.1
    result = model.analyse_ddsp_fraction(dry, wet)
    for key in ("ddsp_esr", "full_esr", "correction_rms_ratio", "ddsp_fraction"):
        assert key in result, f"Missing key: {key}"


def test_analyse_ddsp_fraction_in_range():
    model = KlonDDSP()
    dry = _dry()
    wet = _dry() * 0.8
    result = model.analyse_ddsp_fraction(dry, wet)
    assert 0.0 <= result["ddsp_fraction"] <= 1.0
    assert result["correction_rms_ratio"] >= 0.0
    assert result["ddsp_esr"] >= 0.0


# ── Initialisation sanity ────────────────────────────────────────────────── #


def test_initial_correction_is_small_relative_to_ddsp():
    """gru_fc must be initialised with small weights so the model starts near-DDSP-only.

    If gru_fc starts with large random weights (PyTorch default Kaiming-uniform),
    the GRU correction immediately dominates over the physics stage, pushing
    full_esr > 1.0 from epoch 0 and stalling training.
    """
    torch.manual_seed(0)
    model = KlonDDSP()
    x = _dry_t(batch=1, time=CHUNK)
    with torch.no_grad():
        out, _ = model(x)  # (1, CHUNK, 1)
        ddsp_only = model._ddsp_stage(x.squeeze(-1))  # (1, CHUNK)
        correction = out.squeeze(-1) - ddsp_only  # (1, CHUNK)
    ddsp_rms = float(ddsp_only.pow(2).mean().sqrt())
    correction_rms = float(correction.pow(2).mean().sqrt())
    # Correction should be at least 100× quieter than the physics stage at init
    assert correction_rms < ddsp_rms * 0.01, (
        f"correction_rms={correction_rms:.4f} too large vs ddsp_rms={ddsp_rms:.4f}"
    )


# ── Knob defaults ─────────────────────────────────────────────────────────── #


def test_default_condition_is_nine_oclock():
    nine = 3.0 / 11.0
    assert DEFAULT_CONDITION == pytest.approx([nine, nine, nine], abs=1e-6)


def test_n_knobs_is_three():
    assert N_KNOBS == 3


# ── fit raises ────────────────────────────────────────────────────────────── #


def test_fit_raises_not_implemented():
    model = KlonDDSP()
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros(100), np.zeros(100), SR)


# ── name / receptive_field / state_dict ──────────────────────────────────── #


def test_name():
    assert "Klon" in KlonDDSP().name or "DDSP" in KlonDDSP().name


def test_receptive_field():
    assert KlonDDSP().receptive_field == 4096


def test_state_dict_roundtrip(tmp_path):
    """Save weights, load into fresh model, outputs must be identical."""
    torch.manual_seed(0)
    model = KlonDDSP()
    x = _dry_t(batch=1, time=CHUNK)
    with torch.no_grad():
        out_orig, _ = model(x)

    ckpt = tmp_path / "klon.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = KlonDDSP()
    model2.load_state_dict(torch.load(ckpt, weights_only=True))
    with torch.no_grad():
        out_loaded, _ = model2(x)

    torch.testing.assert_close(out_orig, out_loaded, atol=0.0, rtol=0.0)
