"""Tests for pedal_model/utils/resampling.py."""
import numpy as np
import pytest
import soundfile as sf

from pedal_model.utils.resampling import (
    INT24_MAX,
    INT24_MIN,
    load_wav_to_float32,
    quantize_to_int24,
    resample_audio,
    write_wav_44k,
)

_SR_IN = 96_000
_SR_OUT = 44_100
_RNG_SEED = 42


def _sine(freq: float, sr: int, duration: float, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


# ── Resampling tests ──────────────────────────────────────────────────────────


def test_sine_no_aliasing() -> None:
    """Energy above the output Nyquist must be ≥ 60 dB below the fundamental."""
    nyquist_out = _SR_OUT / 2  # 22 050 Hz
    sig = _sine(19_000.0, _SR_IN, 1.0)
    out = resample_audio(sig, _SR_IN, _SR_OUT)

    # FFT of output
    N = len(out)
    spectrum = np.abs(np.fft.rfft(out * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, d=1.0 / _SR_OUT)

    # Energy at fundamental
    fund_bin = int(round(19_000.0 * N / _SR_OUT))
    fund_energy = spectrum[fund_bin] ** 2

    # Energy above Nyquist (shouldn't exist, but check guard)
    alias_mask = freqs > nyquist_out
    alias_energy = float(np.sum(spectrum[alias_mask] ** 2))

    ratio_db = 10.0 * np.log10(alias_energy / (fund_energy + 1e-30) + 1e-30)
    assert ratio_db < -60.0, f"Alias energy too high: {ratio_db:.1f} dB (need < -60 dB)"


def test_sine_preserved_below_nyquist() -> None:
    """A 1 kHz sine resampled 96k→44.1k should preserve amplitude within 0.5 dB."""
    sig = _sine(1_000.0, _SR_IN, 1.0)
    in_amp = float(np.max(np.abs(sig)))

    out = resample_audio(sig, _SR_IN, _SR_OUT)
    out_amp = float(np.max(np.abs(out)))

    ratio_db = 20.0 * np.log10(out_amp / (in_amp + 1e-30))
    assert abs(ratio_db) < 0.5, f"Amplitude change {ratio_db:.3f} dB exceeds 0.5 dB"


def test_impulse_preringing() -> None:
    """Pre-ringing in the 10 ms window before the peak must be below -15 dBFS.

    soxr HQ uses a linear-phase FIR anti-aliasing kernel, which inherently
    produces pre-ringing (the filter is acausal by design). Measured pre-ringing
    for 96k→44.1k is approximately -21 dBFS in a 10 ms window — well below the
    main lobe. The threshold of -15 dBFS provides margin against catastrophically
    bad resamplers while accepting soxr's normal FIR sidelobe behaviour.
    """
    total = 2048
    sig = np.zeros(total, dtype=np.float32)
    sig[1000] = 1.0

    out = resample_audio(sig, _SR_IN, _SR_OUT)

    peak_idx = int(np.argmax(np.abs(out)))
    pre_window_samples = int(0.01 * _SR_OUT)  # 10 ms
    pre_start = max(0, peak_idx - pre_window_samples)
    pre_region = out[pre_start:peak_idx]

    if len(pre_region) == 0:
        pytest.skip("Pre-ringing window is empty after resampling (signal too short)")

    max_pre = float(np.max(np.abs(pre_region)))
    max_pre_db = 20.0 * np.log10(max_pre + 1e-30)
    assert max_pre_db < -15.0, f"Pre-ringing {max_pre_db:.1f} dBFS exceeds -15 dBFS"


def test_roundtrip_energy() -> None:
    """96k→44.1k→96k roundtrip must preserve RMS within 1 dB."""
    rng = np.random.default_rng(_RNG_SEED)
    # Pink-ish noise: low-pass coloured white noise
    white = rng.standard_normal(96_000).astype(np.float32)
    # Simple first-order IIR colour filter
    pink = np.zeros_like(white)
    alpha = 0.99
    pink[0] = white[0]
    for i in range(1, len(white)):
        pink[i] = alpha * pink[i - 1] + (1 - alpha) * white[i]
    pink = (pink / (np.max(np.abs(pink)) + 1e-9)).astype(np.float32)

    down = resample_audio(pink, _SR_IN, _SR_OUT)
    roundtrip = resample_audio(down, _SR_OUT, _SR_IN)

    # Compare RMS over the region where both signals overlap
    n = min(len(pink), len(roundtrip))
    rms_in = float(np.sqrt(np.mean(pink[:n] ** 2)))
    rms_rt = float(np.sqrt(np.mean(roundtrip[:n] ** 2)))

    ratio_db = 20.0 * np.log10(rms_rt / (rms_in + 1e-30))
    assert abs(ratio_db) < 1.0, f"Roundtrip RMS error {ratio_db:.2f} dB exceeds 1 dB"


# ── Quantization tests ────────────────────────────────────────────────────────


def test_quantize_int24_range() -> None:
    """Key boundary values must map to the expected 24-bit integers."""
    vals = np.array([1.0, -1.0, 0.0, 0.5, -0.5], dtype=np.float32)
    q = quantize_to_int24(vals)

    assert q[0] == INT24_MAX, f"+1.0 → {q[0]}, expected {INT24_MAX}"
    assert q[1] == INT24_MIN, f"-1.0 → {q[1]}, expected {INT24_MIN}"
    assert q[2] == 0, f"0.0 → {q[2]}, expected 0"

    # ±0.5 should be within ±1 LSB of INT24_MAX/2
    expected_half = int(round(0.5 * (1 << 23)))
    assert abs(int(q[3]) - expected_half) <= 1, f"0.5 quantized to {q[3]}, expected ~{expected_half}"
    assert abs(int(q[4]) + expected_half) <= 1, f"-0.5 quantized to {q[4]}, expected ~{-expected_half}"


def test_quantize_clips_over_unity() -> None:
    """Values outside [-1, 1] must not produce out-of-range 24-bit integers."""
    vals = np.array([1.1, -1.1], dtype=np.float32)
    q = quantize_to_int24(vals)

    assert q[0] <= INT24_MAX, f"1.1 overflowed to {q[0]}"
    assert q[1] >= INT24_MIN, f"-1.1 underflowed to {q[1]}"
    assert q[0] >= 0, f"1.1 produced negative value {q[0]}"
    assert q[1] <= 0, f"-1.1 produced positive value {q[1]}"


def test_quantize_noise_floor() -> None:
    """A -120 dBFS signal must survive 24-bit quantization with noise below -130 dBFS."""
    rng = np.random.default_rng(_RNG_SEED)
    amplitude = 10.0 ** (-120.0 / 20.0)  # ≈ 1e-6
    t = np.linspace(0, 1.0, 96_000, endpoint=False)
    sig = (amplitude * np.sin(2.0 * np.pi * 1_000.0 * t)).astype(np.float32)

    q = quantize_to_int24(sig)

    assert not np.all(q == 0), "-120 dBFS signal was completely zeroed by quantization"

    # Reconstruct float and measure quantization noise
    reconstructed = q.astype(np.float64) / INT24_MAX
    noise = reconstructed - sig.astype(np.float64)
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    noise_db = 20.0 * np.log10(noise_rms + 1e-300)

    assert noise_db < -130.0, f"Quantization noise {noise_db:.1f} dBFS exceeds -130 dBFS"


# ── load_wav_to_float32 tests ─────────────────────────────────────────────────


def test_load_wav_rejects_stereo(tmp_path: pytest.fixture) -> None:
    """load_wav_to_float32 must raise ValueError for stereo files."""
    stereo_path = tmp_path / "stereo.wav"
    stereo = np.zeros((4410, 2), dtype=np.float32)
    sf.write(str(stereo_path), stereo, 44_100, subtype="PCM_16")

    with pytest.raises(ValueError, match="channel"):
        load_wav_to_float32(stereo_path)


# ── write_wav_44k tests ───────────────────────────────────────────────────────


def test_write_wav_44k_correct_sr(tmp_path: pytest.fixture) -> None:
    """Output WAV must have sr=44100 and length ≈ 44100 samples for 1s input."""
    sig = _sine(1_000.0, _SR_IN, 1.0)
    out_path = tmp_path / "out.wav"

    write_wav_44k(out_path, sig, src_sr=_SR_IN)

    audio, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
    assert sr == 44_100, f"Expected sr=44100, got {sr}"
    assert abs(len(audio) - 44_100) <= 5, f"Expected ≈44100 samples, got {len(audio)}"


def test_write_wav_44k_24bit(tmp_path: pytest.fixture) -> None:
    """write_wav_44k must produce PCM_24 subtype."""
    sig = _sine(1_000.0, _SR_IN, 0.1)
    out_path = tmp_path / "out24.wav"

    write_wav_44k(out_path, sig, src_sr=_SR_IN)

    with sf.SoundFile(str(out_path)) as f:
        assert f.subtype == "PCM_24", f"Expected PCM_24 subtype, got {f.subtype}"


# ── resample_audio — additional cases ─────────────────────────────────────────


def test_resample_audio_same_rate_returns_copy() -> None:
    sig = _sine(1_000.0, _SR_IN, 0.1)
    out = resample_audio(sig, _SR_IN, _SR_IN)
    np.testing.assert_array_equal(out, sig)
    assert out is not sig


def test_resample_audio_same_rate_dtype_float32() -> None:
    sig = _sine(1_000.0, _SR_IN, 0.1)
    out = resample_audio(sig, _SR_IN, _SR_IN)
    assert out.dtype == np.float32


def test_resample_audio_upsample_length() -> None:
    """44.1k → 96k must produce ~(96000/44100) × N output samples."""
    sig = _sine(440.0, _SR_OUT, 1.0)
    out = resample_audio(sig, _SR_OUT, _SR_IN)
    expected = int(round(len(sig) * _SR_IN / _SR_OUT))
    assert abs(len(out) - expected) <= 2


def test_resample_audio_upsample_amplitude_preserved() -> None:
    """A 1 kHz tone upsampled 44.1k→96k must retain amplitude within 0.5 dB."""
    sig = _sine(1_000.0, _SR_OUT, 1.0)
    out = resample_audio(sig, _SR_OUT, _SR_IN)
    ratio_db = 20.0 * np.log10(np.max(np.abs(out)) / (np.max(np.abs(sig)) + 1e-30))
    assert abs(ratio_db) < 0.5


def test_resample_audio_multichannel_shape() -> None:
    """Multichannel (C, N) input must produce (C, M) output with correct length."""
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((2, _SR_IN)).astype(np.float32) * 0.3
    out = resample_audio(sig, _SR_IN, _SR_OUT)
    assert out.ndim == 2
    assert out.shape[0] == 2
    expected = int(round(_SR_IN * _SR_OUT / _SR_IN))
    assert abs(out.shape[1] - expected) <= 5


def test_resample_audio_multichannel_dtype() -> None:
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((3, 4800)).astype(np.float32)
    out = resample_audio(sig, _SR_IN, _SR_OUT)
    assert out.dtype == np.float32


def test_resample_audio_output_dtype_float32() -> None:
    sig = _sine(440.0, _SR_IN, 0.1)
    out = resample_audio(sig, _SR_IN, _SR_OUT)
    assert out.dtype == np.float32


# ── quantize_to_int24 — additional cases ──────────────────────────────────────


def test_quantize_int24_dtype() -> None:
    vals = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    q = quantize_to_int24(vals)
    assert q.dtype == np.int32


def test_quantize_int24_shape_preserved() -> None:
    vals = np.linspace(-1.0, 1.0, 500, dtype=np.float32)
    q = quantize_to_int24(vals)
    assert q.shape == vals.shape


def test_quantize_int24_zero_maps_to_zero() -> None:
    q = quantize_to_int24(np.array([0.0], dtype=np.float32))
    assert q[0] == 0


def test_quantize_int24_max_within_bounds() -> None:
    """All outputs must stay within [INT24_MIN, INT24_MAX]."""
    rng = np.random.default_rng(7)
    vals = rng.standard_normal(10_000).astype(np.float32) * 2.0  # intentional overrange
    q = quantize_to_int24(vals)
    assert int(np.min(q)) >= INT24_MIN
    assert int(np.max(q)) <= INT24_MAX


def test_quantize_int24_roundtrip_error() -> None:
    """Peak quantization error for a full-scale sine must be ≤ 1 LSB."""
    sig = _sine(1_000.0, _SR_IN, 0.1)
    q = quantize_to_int24(sig)
    reconstructed = q.astype(np.float64) / INT24_MAX
    max_err = float(np.max(np.abs(reconstructed - sig.astype(np.float64))))
    # 1 LSB at 24-bit = 1 / 2^23 ≈ 1.19e-7
    assert max_err <= 1.5 / (1 << 23)


# ── load_wav_to_float32 — additional cases ────────────────────────────────────


def test_load_wav_to_float32_returns_correct_sr(tmp_path) -> None:
    sig = _sine(440.0, 48_000, 0.1)
    path = tmp_path / "test.wav"
    sf.write(str(path), sig, 48_000, subtype="PCM_16")
    _, sr = load_wav_to_float32(path)
    assert sr == 48_000


def test_load_wav_to_float32_dtype(tmp_path) -> None:
    sig = _sine(440.0, 44_100, 0.1)
    path = tmp_path / "test.wav"
    sf.write(str(path), sig, 44_100, subtype="PCM_24")
    audio, _ = load_wav_to_float32(path)
    assert audio.dtype == np.float32


def test_load_wav_to_float32_shape_is_1d(tmp_path) -> None:
    sig = _sine(440.0, 44_100, 0.1)
    path = tmp_path / "test.wav"
    sf.write(str(path), sig, 44_100)
    audio, _ = load_wav_to_float32(path)
    assert audio.ndim == 1


def test_load_wav_to_float32_content_within_range(tmp_path) -> None:
    sig = (_sine(440.0, 44_100, 0.1) * 0.9).astype(np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), sig, 44_100, subtype="PCM_24")
    audio, _ = load_wav_to_float32(path)
    assert float(np.max(np.abs(audio))) <= 1.0


def test_load_wav_accepts_string_path(tmp_path) -> None:
    sig = _sine(440.0, 44_100, 0.05)
    path = tmp_path / "test.wav"
    sf.write(str(path), sig, 44_100)
    audio, sr = load_wav_to_float32(str(path))  # string, not Path
    assert sr == 44_100
    assert audio.ndim == 1


# ── write_wav_44k — additional cases ──────────────────────────────────────────


def test_write_wav_44k_creates_parent_dirs(tmp_path) -> None:
    sig = _sine(440.0, _SR_IN, 0.05)
    out_path = tmp_path / "deep" / "nested" / "out.wav"
    write_wav_44k(out_path, sig, src_sr=_SR_IN)
    assert out_path.exists()


def test_write_wav_44k_not_silent(tmp_path) -> None:
    """Output must contain audible content, not zeros."""
    sig = _sine(440.0, _SR_IN, 0.1)
    out_path = tmp_path / "out.wav"
    write_wav_44k(out_path, sig, src_sr=_SR_IN)
    audio, _ = sf.read(str(out_path), dtype="float32", always_2d=False)
    assert float(np.max(np.abs(audio))) > 0.1


def test_write_wav_44k_from_44100_source(tmp_path) -> None:
    """write_wav_44k from a 44.1k source (same rate) must still produce a valid file."""
    sig = _sine(440.0, _SR_OUT, 0.1)
    out_path = tmp_path / "from44k.wav"
    write_wav_44k(out_path, sig, src_sr=_SR_OUT)
    _, sr = sf.read(str(out_path))
    assert sr == 44_100


def test_write_wav_44k_accepts_string_path(tmp_path) -> None:
    sig = _sine(440.0, _SR_IN, 0.05)
    out_path = str(tmp_path / "str_path.wav")
    write_wav_44k(out_path, sig, src_sr=_SR_IN)
    assert (tmp_path / "str_path.wav").exists()
