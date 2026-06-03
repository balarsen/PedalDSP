"""Perceptual audio quality metrics."""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, f_min: float = 0.0, f_max: float | None = None) -> np.ndarray:
    """Build a triangular mel filterbank matrix.

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT size (number of positive-frequency bins = n_fft // 2 + 1).
        n_mels: Number of mel filters.
        f_min: Lowest frequency in Hz.
        f_max: Highest frequency in Hz. Defaults to sr / 2.

    Returns:
        Filterbank matrix, shape (n_mels, n_fft // 2 + 1).
    """
    if f_max is None:
        f_max = sr / 2.0

    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    freqs = rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        f_l, f_c, f_r = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (freqs - f_l) / (f_c - f_l + 1e-12)
        falling = (f_r - freqs) / (f_r - f_c + 1e-12)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
    return fb


def compute_mcd(
    target: np.ndarray,
    predicted: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    n_mels: int = 40,
    hop: int = 512,
) -> float:
    """Mel Cepstral Distortion in dB (lower is better).

    Args:
        target: Reference audio, shape (N,), float32.
        predicted: Model output, same shape.
        sr: Sample rate in Hz.
        n_fft: FFT window size.
        n_mels: Number of mel filterbank bands.
        hop: Hop size in samples.

    Returns:
        MCD in dB. < 2 dB is good; captures tonal character independently of phase.
    """
    fb = _mel_filterbank(sr, n_fft, n_mels)
    window = get_window("hann", n_fft)

    def mcc(x: np.ndarray) -> np.ndarray:
        n_frames = max(1, (len(x) - n_fft) // hop + 1)
        frames = np.zeros((n_frames, n_fft))
        for i in range(n_frames):
            chunk = x[i * hop : i * hop + n_fft]
            if len(chunk) < n_fft:
                chunk = np.pad(chunk, (0, n_fft - len(chunk)))
            frames[i] = chunk * window
        mag = np.abs(np.fft.rfft(frames, axis=1))  # (n_frames, n_fft//2+1)
        mel = np.dot(fb, mag.T).T + 1e-8           # (n_frames, n_mels)
        log_mel = np.log(mel)
        from scipy.fft import dct
        return dct(log_mel, axis=1, norm="ortho")  # (n_frames, n_mels)

    c_target = mcc(target)
    c_pred = mcc(predicted)
    diff = c_target - c_pred
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(np.sum(diff ** 2, axis=1)))
    return float(mcd)


def compute_lsd(
    target: np.ndarray,
    predicted: np.ndarray,
    input_signal: np.ndarray,
    sr: int,
    n_fft: int = 2048,
) -> float:
    """Log Spectral Distance in dB between transfer functions.

    Args:
        target: Reference wet audio, shape (N,).
        predicted: Model wet output, same shape.
        input_signal: Dry input that generated both, same shape.
        sr: Sample rate in Hz.
        n_fft: FFT size.

    Returns:
        LSD in dB. < 1 dB is excellent.
    """
    n = min(len(input_signal), n_fft)
    X = np.abs(rfft(input_signal[:n])) + 1e-8
    H_target = np.abs(rfft(target[:n])) / X
    H_pred = np.abs(rfft(predicted[:n])) / X

    db_t = 10.0 * np.log10(H_target ** 2 + 1e-12)
    db_p = 10.0 * np.log10(H_pred ** 2 + 1e-12)
    k = len(db_t)
    return float(np.sqrt(np.sum((db_t - db_p) ** 2) / k))
