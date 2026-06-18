"""Tests for pedal_model.ode.filters."""
import math

import numpy as np
import pytest
import scipy.signal
import torch

from pedal_model.ode.filters import HighShelfBiquad, SecondOrderHighpass

SR = 48_000
DT = 1.0 / SR


# ── SecondOrderHighpass ───────────────────────────────────────────────────────


def test_hpf_cutoff_at_minus3db():
    """Analytical HPF |H(jω₀)| must be -3 dB (±0.15 dB) at the specified cutoff."""
    for fc in (80.0, 200.0, 350.0):
        hpf = SecondOrderHighpass(fc_init=fc)
        omega_0 = torch.tensor([2.0 * math.pi * fc])
        with torch.no_grad():
            log_H = hpf.frequency_response_log_mag(omega_0).item()
        db = 20.0 * log_H / math.log(10)
        assert abs(db - (-3.01)) < 0.15, (
            f"fc={fc} Hz: |H(jω₀)| = {db:.3f} dB, expected -3.01 dB"
        )


def test_hpf_passband_flat():
    """HPF must be within 0.2 dB of 0 dB one decade above its cutoff."""
    for fc in (80.0, 200.0):
        hpf = SecondOrderHighpass(fc_init=fc)
        omega_hi = torch.tensor([2.0 * math.pi * 10 * fc])  # 1 decade above fc
        with torch.no_grad():
            log_H = hpf.frequency_response_log_mag(omega_hi).item()
        db = 20.0 * log_H / math.log(10)
        assert abs(db) < 0.2, (
            f"fc={fc} Hz: passband at 10×fc = {db:.3f} dB, expected ~0 dB"
        )


def test_hpf_stopband_40db_per_decade():
    """HPF must have > 35 dB attenuation one decade below its cutoff (2nd-order slope)."""
    for fc in (200.0, 350.0):
        hpf = SecondOrderHighpass(fc_init=fc)
        omega_lo = torch.tensor([2.0 * math.pi * fc / 10])  # 1 decade below fc
        with torch.no_grad():
            log_H = hpf.frequency_response_log_mag(omega_lo).item()
        db = 20.0 * log_H / math.log(10)
        assert db < -35.0, (
            f"fc={fc} Hz: stopband at fc/10 = {db:.1f} dB, expected < -35 dB"
        )


def test_hpf_gradient_flows_to_log_omega():
    """Gradient must flow from output loss through the HPF to log_omega."""
    hpf = SecondOrderHighpass(fc_init=80.0)
    assert hpf.log_omega.requires_grad

    x = torch.randn(512)
    y = hpf(x, DT)
    loss = y.sum()
    loss.backward()

    assert hpf.log_omega.grad is not None
    assert torch.isfinite(hpf.log_omega.grad)
    assert hpf.log_omega.grad.abs() > 0


def test_hpf_fc_hz_property():
    """fc_hz must match the initialised frequency within 0.1%."""
    for fc_init in (80.0, 350.0, 1000.0):
        hpf = SecondOrderHighpass(fc_init=fc_init)
        assert abs(hpf.fc_hz - fc_init) / fc_init < 1e-3, (
            f"fc_hz={hpf.fc_hz:.2f} != init {fc_init}"
        )


def test_hpf_output_shape_and_dtype():
    """forward() must return float32 with the same length as input."""
    hpf = SecondOrderHighpass(fc_init=100.0)
    x = torch.randn(1024)
    with torch.no_grad():
        y = hpf(x, DT)
    assert y.shape == (1024,)
    assert y.dtype == torch.float32


def test_hpf_attenuates_dc():
    """A constant (DC) input through the HPF must produce near-zero output after settling."""
    hpf = SecondOrderHighpass(fc_init=100.0)
    n_settle = int(0.5 * SR)
    x = torch.ones(n_settle)
    with torch.no_grad():
        y = hpf(x, DT)
    # After settling, the HPF output should be near zero for DC input
    assert float(y[-256:].abs().mean()) < 0.01


def test_hpf_parameter_update_changes_output():
    """Manually updating log_omega must change the filter output."""
    hpf = SecondOrderHighpass(fc_init=100.0)
    x = torch.randn(2048)
    with torch.no_grad():
        y1 = hpf(x, DT).clone()
        # Change fc to 500 Hz
        hpf.log_omega.copy_(torch.tensor(math.log(2.0 * math.pi * 500.0)))
        y2 = hpf(x, DT)
    assert not torch.allclose(y1, y2), "Changing log_omega must change the output"


# ── HighShelfBiquad ───────────────────────────────────────────────────────────


def test_shelf_matches_scipy_lfilter():
    """Shelf output must be close to scipy.signal lfilter (same EQ Cookbook formula)."""
    from pedal_model.synthetic.pedal import _high_shelf_biquad

    freq_hz, gain_db = 3000.0, 2.0
    np.random.seed(1)
    x_np = np.random.randn(8 * SR).astype(np.float32) * 0.3

    b, a = _high_shelf_biquad(freq_hz, gain_db, SR)
    y_scipy = scipy.signal.lfilter(b, a, x_np).astype(np.float32)

    shelf = HighShelfBiquad(freq_init=freq_hz, gain_db_init=gain_db)
    with torch.no_grad():
        y_model = shelf(torch.from_numpy(x_np), DT).numpy()

    skip = int(0.05 * SR)
    np.testing.assert_allclose(
        y_model[skip:], y_scipy[skip:], rtol=1e-4, atol=1e-5,
        err_msg="HighShelfBiquad deviates from scipy reference beyond tolerance",
    )


def test_shelf_gradient_flows():
    """Gradient must flow to both log_omega and log_A."""
    shelf = HighShelfBiquad(freq_init=3000.0, gain_db_init=2.0)
    x = torch.randn(512)
    y = shelf(x, DT)
    y.sum().backward()

    assert shelf.log_omega.grad is not None and torch.isfinite(shelf.log_omega.grad)
    assert shelf.log_A.grad is not None and torch.isfinite(shelf.log_A.grad)


def test_shelf_property_values():
    """freq_hz and gain_db properties must round-trip through log parameters."""
    for freq in (1000.0, 3000.0, 8000.0):
        for gain in (-3.0, 0.0, 2.0, 6.0):
            shelf = HighShelfBiquad(freq_init=freq, gain_db_init=gain)
            assert abs(shelf.freq_hz - freq) / freq < 1e-3
            assert abs(shelf.gain_db - gain) < 0.01


def test_shelf_zero_gain_is_allpass():
    """A shelf with 0 dB gain must pass the signal unmodified."""
    shelf = HighShelfBiquad(freq_init=3000.0, gain_db_init=0.0)
    x = torch.randn(4096)
    with torch.no_grad():
        y = shelf(x, DT)
    np.testing.assert_allclose(
        y.numpy(), x.numpy(), rtol=1e-4, atol=1e-5,
        err_msg="0 dB shelf must be an all-pass",
    )


def test_shelf_output_shape_and_dtype():
    """forward() must return float32 with same length as input."""
    shelf = HighShelfBiquad(freq_init=3000.0, gain_db_init=2.0)
    x = torch.randn(2048)
    with torch.no_grad():
        y = shelf(x, DT)
    assert y.shape == (2048,)
    assert y.dtype == torch.float32


# ── _FastBiquadFn gradient correctness ───────────────────────────────────────


def test_fast_biquad_gradcheck():
    """torch.autograd.gradcheck must pass for _FastBiquadFn (all three inputs).

    Coefficients are a stable Butterworth HPF (fc=1000 Hz, sr=48000), giving
    poles at |z| ≈ 0.91 < 1. Unstable filters (|z| > 1) cause h[n] → ∞ and
    swamp finite-difference Jacobians in float-point noise — so stability matters.
    """
    from pedal_model.ode.filters import _FastBiquadFn, SecondOrderHighpass

    torch.manual_seed(0)
    # Derive stable Butterworth HPF coefficients from the actual class.
    hpf = SecondOrderHighpass(fc_init=1000.0)
    with torch.no_grad():
        b0, b1, b2, a1, a2 = hpf._compute_hpf_biquad_coeffs(SR)  # SR = 48000

    x = torch.randn(32, dtype=torch.float64, requires_grad=True)
    b = torch.stack([b0, b1, b2]).double().detach().requires_grad_(True)
    a = torch.stack([a1, a2]).double().detach().requires_grad_(True)

    assert torch.autograd.gradcheck(
        _FastBiquadFn.apply,
        (x, b, a),
        eps=1e-5,
        atol=1e-4,
        rtol=1e-3,
    ), "Gradient check failed for _FastBiquadFn"


def test_hpf_fast_matches_scipy():
    """SecondOrderHighpass.forward must match scipy lfilter with the same biquad."""
    import scipy.signal

    fc = 200.0
    np.random.seed(7)
    x_np = np.random.randn(4096).astype(np.float32)

    hpf = SecondOrderHighpass(fc_init=fc)
    with torch.no_grad():
        b0, b1, b2, a1, a2 = hpf._compute_hpf_biquad_coeffs(SR)
        b_np = np.array([b0.item(), b1.item(), b2.item()])
        a_np = np.array([1.0, a1.item(), a2.item()])

    y_scipy = scipy.signal.lfilter(b_np, a_np, x_np.astype(np.float64)).astype(np.float32)
    y_model = hpf(torch.from_numpy(x_np), DT).detach().numpy()

    np.testing.assert_allclose(
        y_model, y_scipy, rtol=1e-4, atol=1e-5,
        err_msg="SecondOrderHighpass.forward must match scipy lfilter reference",
    )
