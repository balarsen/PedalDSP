"""Tests for pedal_model.models.neural.ddsp — LearnableFIR, LearnableWaveshaper,
LearnableIIR, and DDSPModel."""
import numpy as np
import pytest
import torch

from pedal_model.models.neural.ddsp import (
    DDSPModel,
    LearnableFIR,
    LearnableIIR,
    LearnableWaveshaper,
)


# ── LearnableIIR ──────────────────────────────────────────────────────────── #


def test_learnable_iir_output_shape():
    iir = LearnableIIR()
    x = torch.randn(3, 1, 256)
    y = iir(x)
    assert y.shape == x.shape


def test_learnable_iir_near_identity_at_zero_coefficient():
    """a1=0 → y[n] = x[n] — output should equal input."""
    iir = LearnableIIR()
    with torch.no_grad():
        iir.a1.fill_(0.0)  # tanh(0) = 0 → identity
    x = torch.randn(1, 1, 128)
    y = iir(x)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=0)


def test_learnable_iir_fir_matches_exact_iir():
    """FIR-approximated IIR must match the exact sample-by-sample recurrence."""
    torch.manual_seed(0)
    iir = LearnableIIR()
    with torch.no_grad():
        iir.a1.fill_(0.8)
        a1_val = torch.tanh(iir.a1).item()
        x = torch.randn(2, 1, 512)
        y_fir = iir(x)
        # Reference: exact recurrence y[n] = x[n] - a1*y[n-1]
        y_ref = torch.zeros_like(x)
        prev = torch.zeros(2, 1, 1)
        for t in range(512):
            yt = x[:, :, t : t + 1] - a1_val * prev
            y_ref[:, :, t : t + 1] = yt
            prev = yt
    diff = (y_fir - y_ref).abs().max().item()
    assert diff < 1e-4, f"FIR approx error {diff:.2e} exceeds 1e-4"


def test_learnable_iir_gradient_flows_to_a1():
    """Gradient must reach self.a1 through the FIR convolution path."""
    iir = LearnableIIR()
    x = torch.randn(1, 1, 64)
    y = iir(x)
    loss = y.sum()
    loss.backward()
    assert iir.a1.grad is not None
    assert iir.a1.grad.abs().item() > 0


def test_learnable_iir_stable_for_large_a1_raw():
    """tanh constrains a1 to (-1,1) regardless of raw parameter magnitude."""
    iir = LearnableIIR()
    with torch.no_grad():
        iir.a1.fill_(10.0)  # tanh(10) ≈ 1.0 — filter near marginal stability
    x = torch.randn(1, 1, 256) * 0.1
    y = iir(x)
    assert torch.isfinite(y).all()


# ── LearnableFIR ──────────────────────────────────────────────────────────── #


def test_learnable_fir_output_shape():
    fir = LearnableFIR(n_taps=64)
    x = torch.randn(2, 1, 256)
    y = fir(x)
    assert y.shape == x.shape


def test_learnable_fir_identity_init():
    """Dirac init acts as identity on a DC signal after the initial transient.

    nn.init.dirac_ places 1 at kernel[n_taps//2], which introduces a delay of
    n_taps//2 - 1 samples.  A constant DC input (all ones) is unaffected by
    delay, so every output past the transient should equal 1.
    """
    fir = LearnableFIR(n_taps=32)
    x = torch.ones(1, 1, 128)
    y = fir(x)
    # After the n_taps-1 = 31 sample transient, DC output must equal DC input
    torch.testing.assert_close(y[:, :, 31:], x[:, :, 31:], atol=1e-5, rtol=0)


# ── LearnableWaveshaper ───────────────────────────────────────────────────── #


def test_learnable_waveshaper_output_shape():
    ws = LearnableWaveshaper(hidden=16)
    x = torch.randn(3, 1, 128)
    y = ws(x)
    assert y.shape == x.shape


def test_learnable_waveshaper_output_finite():
    ws = LearnableWaveshaper(hidden=8)
    x = torch.randn(2, 1, 64) * 5  # large input to stress tanh activations
    y = ws(x)
    assert torch.isfinite(y).all()


def test_learnable_waveshaper_initial_output_small():
    """Output layer must be near-zero at init to prevent large initial ESR.

    Default Kaiming-uniform on Linear(16,1) produces outputs up to ±4 RMS,
    giving ESR ≈ 8 from epoch 0 and saturating tanh gradients.
    """
    torch.manual_seed(0)
    ws = LearnableWaveshaper(hidden=16)
    x = torch.randn(1, 1, 4096) * 0.3  # typical dry RMS
    with torch.no_grad():
        y = ws(x)
    input_rms = float(x.pow(2).mean().sqrt())
    output_rms = float(y.pow(2).mean().sqrt())
    # Initial output must be at least 50× quieter than input
    assert output_rms < input_rms * 0.02, (
        f"output_rms={output_rms:.4f} too large vs input_rms={input_rms:.4f}"
    )


# ── DDSPModel ─────────────────────────────────────────────────────────────── #


def test_ddsp_model_forward_shape():
    model = DDSPModel(n_fir_taps=32)
    x = torch.randn(2, 1, 256)
    y = model(x)
    assert y.shape == x.shape


def test_ddsp_model_forward_finite():
    torch.manual_seed(0)
    model = DDSPModel(n_fir_taps=32)
    x = torch.randn(2, 1, 128)
    y = model(x)
    assert torch.isfinite(y).all()


def test_ddsp_model_predict_shape():
    model = DDSPModel(n_fir_taps=32)
    x = np.random.randn(1000).astype(np.float32) * 0.3
    out = model.predict(x)
    assert out.shape == (1000,)
    assert out.dtype == np.float32


def test_ddsp_model_predict_not_silence():
    torch.manual_seed(0)
    model = DDSPModel(n_fir_taps=32)
    x = np.random.randn(500).astype(np.float32) * 0.5
    out = model.predict(x)
    assert np.abs(out).max() > 1e-6


def test_ddsp_model_fit_raises():
    model = DDSPModel()
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), 48000)


def test_ddsp_model_gradient_end_to_end():
    """Loss must backprop to all learnable parameters."""
    model = DDSPModel(n_fir_taps=16)
    x = torch.randn(1, 1, 64)
    target = torch.randn(1, 1, 64)
    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    assert model.iir.a1.grad is not None
    assert model.fir.kernel.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ddsp_model_forward_on_gpu():
    model = DDSPModel(n_fir_taps=16).cuda()
    x = torch.randn(2, 1, 64, device="cuda")
    y = model(x)
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()


# ── DDSPModel name / receptive_field / state_dict ────────────────────────── #


def test_ddsp_model_name():
    assert "DDSP" in DDSPModel(n_fir_taps=32).name


def test_ddsp_model_receptive_field_equals_n_fir_taps():
    assert DDSPModel(n_fir_taps=32).receptive_field == 32
    assert DDSPModel(n_fir_taps=64).receptive_field == 64


def test_ddsp_model_state_dict_roundtrip(tmp_path):
    """Save weights, load into fresh model, outputs must be identical."""
    torch.manual_seed(0)
    model = DDSPModel(n_fir_taps=32)
    x = torch.randn(1, 1, 128)
    with torch.no_grad():
        out_orig = model(x)

    ckpt = tmp_path / "ddsp.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = DDSPModel(n_fir_taps=32)
    model2.load_state_dict(torch.load(ckpt, weights_only=True))
    with torch.no_grad():
        out_loaded = model2(x)

    torch.testing.assert_close(out_orig, out_loaded, atol=0.0, rtol=0.0)
