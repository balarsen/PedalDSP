"""Perceptual audio quality metrics."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window

_MIN_SAMPLES = 4096


# ──────────────────────────────────────────────────────────────────────────────
# Original helpers (compute_mcd, compute_lsd)
# ──────────────────────────────────────────────────────────────────────────────

def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 40,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
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


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers for the six diagnostic metrics
# ──────────────────────────────────────────────────────────────────────────────

def _validate_inputs(
    pred: np.ndarray,
    target: np.ndarray,
) -> tuple[float | None, str | None]:
    """Check lengths, silence, and minimum sample count.

    Returns (override_score, reason) if an edge case is hit, else (None, None).
    Raises ValueError on length mismatch.
    """
    if len(pred) != len(target):
        raise ValueError(
            f"pred and target must have the same length "
            f"(got {len(pred)} vs {len(target)})"
        )
    if len(pred) < _MIN_SAMPLES:
        warnings.warn(
            f"Signal shorter than {_MIN_SAMPLES} samples; returning NaN.",
            stacklevel=3,
        )
        return float("nan"), "too_short"
    if np.max(np.abs(target)) < 1e-10 and np.max(np.abs(pred)) < 1e-10:
        warnings.warn("Both pred and target are silent; returning 0.0.", stacklevel=3)
        return 0.0, "silent"
    return None, None


def _stft_mag(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Short-time magnitude spectrogram using a Hann window.

    Args:
        signal: 1-D audio array.
        n_fft: FFT size.
        hop: Hop size in samples.

    Returns:
        Shape (n_fft // 2 + 1, n_frames), float64.
    """
    window = get_window("hann", n_fft)
    sig = signal.astype(np.float64)
    n_frames = max(1, (len(sig) - n_fft) // hop + 1)
    n_freqs = n_fft // 2 + 1
    out = np.empty((n_frames, n_freqs), dtype=np.float64)
    for i in range(n_frames):
        chunk = sig[i * hop : i * hop + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        out[i] = np.abs(rfft(chunk * window))
    return out.T  # (n_freqs, n_frames)


def _a_weighting_db(freqs_hz: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting curve in dB, 0 dB reference at 1 kHz.

    Args:
        freqs_hz: Frequency array in Hz, shape (K,).

    Returns:
        A-weighting in dB, shape (K,).
    """
    f = np.maximum(freqs_hz, 1.0)
    f2 = f ** 2
    Ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    return 20.0 * np.log10(Ra + 1e-30) + 2.00


def _build_guitar_prior(freqs_hz: np.ndarray) -> np.ndarray:
    """Fixed guitar spectral importance prior, normalized so mean = 1.0.

    Piecewise linear in dB using breakpoints that reflect the electric guitar
    register: full weight 80 Hz–3 kHz, presence boost at 3–5 kHz, roll-off
    above 10 kHz and below 80 Hz.

    Args:
        freqs_hz: Frequency array in Hz, shape (K,).

    Returns:
        Linear weights, shape (K,), mean = 1.0.
    """
    breakpoints_hz = [20.0,  80.0,  3000.0,  5000.0, 10000.0, 20000.0]
    breakpoints_db = [-18.0,  0.0,     0.0,    +3.0,    -6.0,   -18.0]
    weight_db = np.interp(freqs_hz, breakpoints_hz, breakpoints_db)
    weights = 10.0 ** (weight_db / 20.0)
    weights /= weights.mean() + 1e-10
    return weights


def _estimate_f0(signal: np.ndarray, sr: int) -> tuple[float, float]:
    """Autocorrelation-based F0 estimate.

    Analyses 4096-sample frames, finds the peak lag in the range 60–1200 Hz,
    and returns the median across frames.

    Args:
        signal: 1-D audio array.
        sr: Sample rate in Hz.

    Returns:
        (f0_hz, confidence) — falls back to 110.0 Hz when confidence < 0.5.
    """
    frame_size = 4096
    hop = 2048
    lag_min = max(1, int(sr / 1200))
    lag_max = min(len(signal) - 1, int(sr / 60))

    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    f0s: list[float] = []
    confs: list[float] = []

    for i in range(n_frames):
        frame = signal[i * hop : i * hop + frame_size].astype(np.float64)
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        n_fft2 = 2 * frame_size
        F = np.fft.rfft(frame, n=n_fft2)
        ac = np.fft.irfft(F * np.conj(F))[:frame_size].real
        norm = ac[0] + 1e-10
        ac = ac / norm

        if lag_max < lag_min + 1:
            continue
        region = ac[lag_min : lag_max + 1]
        peak_idx = int(np.argmax(region))
        peak_val = float(region[peak_idx])
        mean_val = float(np.mean(np.abs(region))) + 1e-10
        confidence = peak_val / mean_val
        f0s.append(sr / (lag_min + peak_idx))
        confs.append(confidence)

    if not f0s:
        return 110.0, 0.0

    med_conf = float(np.median(confs))
    if med_conf < 0.5:
        return 110.0, med_conf
    return float(np.median(f0s)), med_conf


# ──────────────────────────────────────────────────────────────────────────────
# Metric 1 — mel_axis_weighted
# ──────────────────────────────────────────────────────────────────────────────

def mel_axis_weighted(
    pred: np.ndarray,
    target: np.ndarray,
    dry: np.ndarray,
    sr: int = 96000,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict:
    """Spectral error on the mel frequency axis (cochlear geometry warping).

    Warps the frequency axis to match cochlear resolution — fine spacing below
    ~500 Hz, coarse above — without applying loudness sensitivity weighting.
    Catches errors that are spectrally concentrated in perceptually resolved
    bands.

    Catches: models that smear or shift formants/harmonics in the guitar midrange
    in ways that warp differently on a linear vs mel scale.

    Args:
        pred: Model output, float32 1-D array.
        target: Pedal wet signal, same shape.
        dry: Dry input signal (unused; reserved for API consistency).
        sr: Sample rate in Hz.
        n_mels: Number of mel filterbank bands.
        fmin: Lowest mel band center in Hz.
        fmax: Highest mel band center in Hz.
        n_fft: STFT window size.
        hop: STFT hop size.

    Returns:
        {'score', 'label', 'unit', 'metadata', 'arrays'} — score is mean
        absolute error in mel-log-magnitude space (dB), lower is better.
    """
    override, reason = _validate_inputs(pred, target)
    if override is not None:
        return {
            "score": override, "label": "mel_axis_weighted",
            "unit": "dB", "metadata": {"edge_case": reason}, "arrays": {},
        }

    fb = _mel_filterbank(sr, n_fft, n_mels=n_mels, f_min=fmin, f_max=min(fmax, sr / 2.0))

    S_pred = _stft_mag(pred, n_fft, hop)    # (n_freqs, n_frames)
    S_tgt  = _stft_mag(target, n_fft, hop)

    mel_pred = fb @ S_pred + 1e-8  # (n_mels, n_frames)
    mel_tgt  = fb @ S_tgt  + 1e-8

    db_pred = 20.0 * np.log10(mel_pred)
    db_tgt  = 20.0 * np.log10(mel_tgt)

    error_map = np.abs(db_tgt - db_pred)               # (n_mels, n_frames)
    per_band  = error_map.mean(axis=1)                  # (n_mels,)
    score     = float(error_map.mean())

    # Mel band centre frequencies (Hz)
    def _mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + min(fmax, sr / 2.0) / 700.0)
    mel_ctrs = np.linspace(mel_min, mel_max, n_mels + 2)[1:-1]
    mel_freqs = _mel_to_hz(mel_ctrs)

    return {
        "score": score,
        "label": "mel_axis_weighted",
        "unit": "dB",
        "metadata": {
            "per_band_error": per_band.tolist(),
            "mel_freqs": mel_freqs.tolist(),
        },
        "arrays": {
            "error_map": error_map,
            "per_band_error": per_band,
            "mel_freqs": mel_freqs,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metric 2 — equal_loudness_weighted
# ──────────────────────────────────────────────────────────────────────────────

def equal_loudness_weighted(
    pred: np.ndarray,
    target: np.ndarray,
    dry: np.ndarray,
    sr: int = 96000,
    spl_db: float = 83.0,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict:
    """Spectral magnitude error weighted by A-weighting (IEC 61672).

    Penalises errors in the 2–4 kHz sensitivity peak more than errors in the
    low-end and high-shelf where hearing is less acute.

    Catches: models that faithfully reproduce bass frequencies but mis-represent
    the 2–5 kHz presence region that defines pick attack and note definition.

    Args:
        pred: Model output, float32 1-D array.
        target: Pedal wet signal, same shape.
        dry: Dry input signal (unused).
        sr: Sample rate in Hz.
        spl_db: Listening level in dB SPL (reserved for future ISO 226
            level-dependent curves; currently ignored — A-weighting is used).
        n_fft: STFT window size.
        hop: STFT hop size.

    Returns:
        {'score', 'label', 'unit', 'metadata', 'arrays'} — score is mean
        A-weighted magnitude error, lower is better.
    """
    override, reason = _validate_inputs(pred, target)
    if override is not None:
        return {
            "score": override, "label": "equal_loudness_weighted",
            "unit": "dB (A-wtd)", "metadata": {"edge_case": reason}, "arrays": {},
        }

    S_pred = _stft_mag(pred, n_fft, hop)
    S_tgt  = _stft_mag(target, n_fft, hop)
    n_freqs = S_pred.shape[0]

    freqs = rfftfreq(n_fft, d=1.0 / sr)
    a_db  = _a_weighting_db(freqs)
    w_lin = 10.0 ** (a_db / 20.0)
    w_lin = w_lin / (w_lin.max() + 1e-10)  # normalize so max = 1

    raw_error = np.abs(S_pred - S_tgt)                      # (n_freqs, n_frames)
    weighted  = w_lin[:, None] * raw_error
    score = float(weighted.mean())

    per_bin_weighted_error = weighted.mean(axis=1)           # (n_freqs,)

    return {
        "score": score,
        "label": "equal_loudness_weighted",
        "unit": "dB (A-wtd)",
        "metadata": {
            "a_weights_db": a_db.tolist(),
            "freqs_hz": freqs.tolist(),
            "per_bin_weighted_error": per_bin_weighted_error.tolist(),
        },
        "arrays": {
            "a_weights_db": a_db,
            "freqs_hz": freqs,
            "per_bin_weighted_error": per_bin_weighted_error,
            "raw_error_per_bin": raw_error.mean(axis=1),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metric 3 — masking_weighted
# ──────────────────────────────────────────────────────────────────────────────

def masking_weighted(
    pred: np.ndarray,
    target: np.ndarray,
    dry: np.ndarray,
    sr: int = 96000,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict:
    """Audible error fraction after psychoacoustic masking (simultaneous masking).

    Estimates a masking threshold from the TARGET signal via a spectral spreading
    kernel (~25 dB/octave slope) and an absolute hearing floor at −70 dBFS.
    Errors below the threshold are inaudible; errors above it are penalised in
    proportion to how far above they fall.

    Catches: errors that are spectrally far from loud components and therefore
    clearly audible, versus errors hidden under the signal's own masking skirt.

    Args:
        pred: Model output, float32 1-D array.
        target: Pedal wet signal, same shape.
        dry: Dry input signal (unused).
        sr: Sample rate in Hz.
        n_fft: STFT window size.
        hop: STFT hop size.

    Returns:
        {'score', 'label', 'unit', 'metadata', 'arrays'} — score is mean
        visibility (error above masking threshold / threshold), lower is better.
    """
    override, reason = _validate_inputs(pred, target)
    if override is not None:
        return {
            "score": override, "label": "masking_weighted",
            "unit": "visibility", "metadata": {"edge_case": reason}, "arrays": {},
        }

    S_pred = _stft_mag(pred, n_fft, hop)
    S_tgt  = _stft_mag(target, n_fft, hop)
    n_freqs, n_frames = S_tgt.shape

    half_w = 50
    k = np.arange(-half_w, half_w + 1)
    spread_kernel = np.exp(-np.abs(k) * 0.1)
    spread_kernel /= spread_kernel.max()

    masking_threshold = np.array(
        [np.convolve(S_tgt[:, t], spread_kernel, mode="same") for t in range(n_frames)]
    ).T  # (n_freqs, n_frames)

    absolute_floor = 10.0 ** (-70.0 / 20.0)
    masking_threshold = np.maximum(masking_threshold, absolute_floor)

    raw_error  = np.abs(S_pred - S_tgt)
    visibility = np.maximum(raw_error - masking_threshold, 0.0) / (masking_threshold + 1e-10)
    score = float(visibility.mean())

    masked_fraction = float((raw_error <= masking_threshold).mean())

    mean_threshold  = masking_threshold.mean(axis=1)
    mean_error      = raw_error.mean(axis=1)
    mean_visibility = visibility.mean(axis=1)

    return {
        "score": score,
        "label": "masking_weighted",
        "unit": "visibility",
        "metadata": {
            "masked_fraction": masked_fraction,
            "mean_masking_threshold": mean_threshold.tolist(),
            "mean_error": mean_error.tolist(),
            "mean_visibility": mean_visibility.tolist(),
        },
        "arrays": {
            "mean_masking_threshold": mean_threshold,
            "mean_error": mean_error,
            "mean_visibility": mean_visibility,
            "visibility_map": visibility,
            "freqs_hz": rfftfreq(n_fft, d=1.0 / sr),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metric 4 — signal_adaptive_weighted
# ──────────────────────────────────────────────────────────────────────────────

def signal_adaptive_weighted(
    pred: np.ndarray,
    target: np.ndarray,
    dry: np.ndarray,
    sr: int = 96000,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict:
    """Spectral error weighted by the target signal's own power spectrum.

    Errors at frequencies where the target has high energy are penalised more.
    A cheap masking proxy: inaudible (quiet) bins count less.

    Catches: models that fail at dominant frequency regions of a specific signal,
    e.g. a TCN that adds harmonic smear right where the guitar string resonates.

    Args:
        pred: Model output, float32 1-D array.
        target: Pedal wet signal, same shape.
        dry: Dry input signal (unused).
        sr: Sample rate in Hz.
        n_fft: STFT window size.
        hop: STFT hop size.

    Returns:
        {'score', 'label', 'unit', 'metadata', 'arrays'} — score is mean
        signal-power-weighted magnitude error, lower is better.
    """
    override, reason = _validate_inputs(pred, target)
    if override is not None:
        return {
            "score": override, "label": "signal_adaptive_weighted",
            "unit": "ratio", "metadata": {"edge_case": reason}, "arrays": {},
        }

    S_pred = _stft_mag(pred, n_fft, hop)
    S_tgt  = _stft_mag(target, n_fft, hop)

    P_target = S_tgt ** 2
    w = P_target / (P_target.mean() + 1e-10)   # mean weight ≈ 1.0

    raw_error = np.abs(S_pred - S_tgt)
    score = float((w * raw_error).mean())

    S_tgt_mean  = S_tgt.mean(axis=1)
    error_mean  = raw_error.mean(axis=1)
    eff_snr_db  = 20.0 * np.log10(S_tgt_mean / (error_mean + 1e-10))
    mean_wtd_err = (w * raw_error).mean(axis=1)
    mean_pow_db  = 10.0 * np.log10(P_target.mean(axis=1) + 1e-10)
    freqs = rfftfreq(n_fft, d=1.0 / sr)

    return {
        "score": score,
        "label": "signal_adaptive_weighted",
        "unit": "ratio",
        "metadata": {
            "mean_target_power_db": mean_pow_db.tolist(),
            "mean_weighted_error": mean_wtd_err.tolist(),
            "effective_snr_db": eff_snr_db.tolist(),
        },
        "arrays": {
            "mean_target_power_db": mean_pow_db,
            "mean_weighted_error": mean_wtd_err,
            "effective_snr_db": eff_snr_db,
            "freqs_hz": freqs,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metric 5 — guitar_prior_weighted
# ──────────────────────────────────────────────────────────────────────────────

def guitar_prior_weighted(
    pred: np.ndarray,
    target: np.ndarray,
    dry: np.ndarray,
    sr: int = 96000,
    n_fft: int = 2048,
    hop: int = 512,
    prior_source: str = "builtin",
) -> dict:
    """Spectral error weighted by a fixed electric guitar spectral prior.

    Unlike signal_adaptive, the weights come from a FIXED curve representing
    the characteristic spectral envelope of electric guitar regardless of
    what this specific signal does. Catches systematic frequency response
    errors in the guitar register even on test signals that don't strongly
    excite that range.

    Catches: TCN or LSTM that attenuates the 3–5 kHz presence peak (pick
    attack, note definition) or adds excess bass below 80 Hz.

    Args:
        pred: Model output, float32 1-D array.
        target: Pedal wet signal, same shape.
        dry: Dry input signal (unused).
        sr: Sample rate in Hz.
        n_fft: STFT window size.
        hop: STFT hop size.
        prior_source: 'builtin' or path to a .npy file containing a
            (n_freqs,) array of linear weights for n_freqs = n_fft//2+1.

    Returns:
        {'score', 'label', 'unit', 'metadata', 'arrays'} — score is mean
        guitar-prior-weighted magnitude error, lower is better.
    """
    override, reason = _validate_inputs(pred, target)
    if override is not None:
        return {
            "score": override, "label": "guitar_prior_weighted",
            "unit": "dB", "metadata": {"edge_case": reason}, "arrays": {},
        }

    freqs = rfftfreq(n_fft, d=1.0 / sr)

    if prior_source == "builtin":
        weights = _build_guitar_prior(freqs)
    else:
        weights = np.load(prior_source).astype(np.float64)
        if len(weights) != len(freqs):
            raise ValueError(
                f"Loaded prior has {len(weights)} bins but n_fft={n_fft} "
                f"gives {len(freqs)} bins"
            )
        weights = weights / (weights.mean() + 1e-10)

    S_pred = _stft_mag(pred, n_fft, hop)
    S_tgt  = _stft_mag(target, n_fft, hop)

    raw_error = np.abs(S_pred - S_tgt)
    weighted  = weights[:, None] * raw_error
    score = float(weighted.mean())

    per_bin_wtd_err = weighted.mean(axis=1)
    breakpoints_hz  = [20.0, 80.0, 3000.0, 5000.0, 10000.0, 20000.0]
    breakpoints_db  = [-18.0, 0.0, 0.0, 3.0, -6.0, -18.0]
    prior_db = np.interp(freqs, breakpoints_hz, breakpoints_db)

    return {
        "score": score,
        "label": "guitar_prior_weighted",
        "unit": "dB",
        "metadata": {
            "prior_weights_db": prior_db.tolist(),
            "freqs_hz": freqs.tolist(),
            "per_bin_weighted_error": per_bin_wtd_err.tolist(),
            "prior_source": prior_source,
        },
        "arrays": {
            "prior_weights_db": prior_db,
            "freqs_hz": freqs,
            "per_bin_weighted_error": per_bin_wtd_err,
            "raw_error_per_bin": raw_error.mean(axis=1),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metric 6 — harmonic_weighted
# ──────────────────────────────────────────────────────────────────────────────

def harmonic_weighted(
    pred: np.ndarray,
    target: np.ndarray,
    dry: np.ndarray,
    sr: int = 96000,
    n_harmonics: int = 12,
    f0: float | None = None,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict:
    """Spectral error up-weighted at harmonic partial frequencies.

    Builds a Gaussian comb mask centred on harmonics of the estimated (or
    supplied) fundamental; errors at harmonics are penalised 6× more than
    inter-harmonic noise-floor errors. Also reports per-harmonic amplitude
    differences and even/odd harmonic error breakdown.

    Catches: the Big Muff / TCN failure mode where the model produces clean
    even/odd harmonic balance rather than the fuzz pedal's asymmetric
    odd-harmonic signature, even when overall ESR is moderate.

    Args:
        pred: Model output, float32 1-D array.
        target: Pedal wet signal, same shape.
        dry: Dry input signal used for F0 estimation when f0=None.
        sr: Sample rate in Hz.
        n_harmonics: Number of harmonics to track.
        f0: Fundamental in Hz. If None, estimated from dry via autocorrelation.
        n_fft: STFT window size.
        hop: STFT hop size.

    Returns:
        {'score', 'label', 'unit', 'metadata', 'arrays'} — score is mean
        harmonic-weighted magnitude error, lower is better.
    """
    override, reason = _validate_inputs(pred, target)
    if override is not None:
        return {
            "score": override, "label": "harmonic_weighted",
            "unit": "dB", "metadata": {"edge_case": reason}, "arrays": {},
        }

    if f0 is None:
        f0, _conf = _estimate_f0(dry, sr)

    S_pred = _stft_mag(pred, n_fft, hop)
    S_tgt  = _stft_mag(target, n_fft, hop)
    freqs  = rfftfreq(n_fft, d=1.0 / sr)
    n_freqs = len(freqs)

    # Build Gaussian comb weight mask
    harmonic_boost = 5.0
    bandwidth_cents = 50.0
    weights = np.ones(n_freqs, dtype=np.float64)
    harmonic_freqs: list[float] = []

    for n in range(1, n_harmonics + 1):
        f_h = n * f0
        if f_h > sr / 2.0:
            break
        harmonic_freqs.append(f_h)
        cents_dist = 1200.0 * np.abs(
            np.log2((freqs + 1e-10) / (f_h + 1e-10))
        )
        weights += harmonic_boost * np.exp(-0.5 * (cents_dist / bandwidth_cents) ** 2)

    weights /= weights.mean() + 1e-10

    raw_error = np.abs(S_pred - S_tgt)
    score = float((weights[:, None] * raw_error).mean())

    # Per-harmonic amplitude extraction
    def _harmonic_amps(S: np.ndarray, f_list: list[float]) -> list[float]:
        amps = []
        for f_h in f_list:
            idx = int(np.argmin(np.abs(freqs - f_h)))
            amps.append(float(S[idx, :].mean()))
        return amps

    tgt_amps  = _harmonic_amps(S_tgt,  harmonic_freqs)
    pred_amps = _harmonic_amps(S_pred, harmonic_freqs)

    n_actual = len(harmonic_freqs)
    amp_err_db = [
        float(20.0 * np.log10((pred_amps[i] + 1e-10) / (tgt_amps[i] + 1e-10)))
        for i in range(n_actual)
    ]

    # Even harmonics: n = 2, 4, 6, ... (indices 1, 3, 5, ...)
    even_errs = [amp_err_db[i] for i in range(1, n_actual, 2)]
    # Odd harmonics: n = 3, 5, 7, ... (indices 2, 4, 6, ...)
    odd_errs  = [amp_err_db[i] for i in range(2, n_actual, 2)]

    even_err_db = float(np.mean(np.abs(even_errs))) if even_errs else float("nan")
    odd_err_db  = float(np.mean(np.abs(odd_errs)))  if odd_errs  else float("nan")

    return {
        "score": score,
        "label": "harmonic_weighted",
        "unit": "dB",
        "metadata": {
            "f0_hz": float(f0),
            "harmonic_freqs": harmonic_freqs,
            "pred_harmonic_amps": pred_amps,
            "target_harmonic_amps": tgt_amps,
            "harmonic_amp_error_db": amp_err_db,
            "even_harmonic_error_db": even_err_db,
            "odd_harmonic_error_db": odd_err_db,
        },
        "arrays": {
            "weight_mask": weights,
            "freqs_hz": freqs,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Composite
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS: dict[str, float] = {
    "mel_axis": 1.0,
    "equal_loudness": 1.0,
    "masking": 1.0,
    "signal_adaptive": 1.0,
    "guitar_prior": 1.0,
    "harmonic": 1.0,
}


def composite_perceptual(
    pred: np.ndarray,
    target: np.ndarray,
    dry: np.ndarray,
    sr: int = 96000,
    weights: dict[str, float] | None = None,
) -> dict:
    """Combined perceptual score from all six weighted spectral metrics.

    Scores are z-score normalised (across the six values) before combining so
    that metrics with different units don't dominate. Weights default to equal
    until a perception study provides calibration data.

    Args:
        pred: Model output, float32 1-D array.
        target: Pedal wet signal, same shape.
        dry: Dry input signal.
        sr: Sample rate in Hz.
        weights: Per-metric weights dict. Keys: 'mel_axis', 'equal_loudness',
            'masking', 'signal_adaptive', 'guitar_prior', 'harmonic'.
            Missing keys default to 1.0.

    Returns:
        {
            'composite_score': float,
            'individual_scores': {name: score},
            'individual_results': {name: full_result_dict},
            'weights_used': dict,
        }
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}

    results = {
        "mel_axis":       mel_axis_weighted(pred, target, dry, sr),
        "equal_loudness": equal_loudness_weighted(pred, target, dry, sr),
        "masking":        masking_weighted(pred, target, dry, sr),
        "signal_adaptive": signal_adaptive_weighted(pred, target, dry, sr),
        "guitar_prior":   guitar_prior_weighted(pred, target, dry, sr),
        "harmonic":       harmonic_weighted(pred, target, dry, sr),
    }

    scores_arr = np.array([results[k]["score"] for k in w], dtype=np.float64)
    finite_mask = np.isfinite(scores_arr)
    if finite_mask.any():
        mu = scores_arr[finite_mask].mean()
        sigma = scores_arr[finite_mask].std() + 1e-10
        z = np.where(finite_mask, (scores_arr - mu) / sigma, 0.0)
    else:
        z = np.zeros(len(scores_arr))

    weight_vals = np.array([w[k] for k in w], dtype=np.float64)
    composite = float(np.dot(z, weight_vals) / (weight_vals.sum() + 1e-10))

    return {
        "composite_score": composite,
        "individual_scores": {k: results[k]["score"] for k in w},
        "individual_results": results,
        "weights_used": w,
    }


def run_all(
    pred: np.ndarray,
    target: np.ndarray,
    dry: np.ndarray,
    sr: int = 96000,
) -> dict:
    """Run all six perceptual metrics and the composite score.

    Convenience entry point for the benchmark harness.

    Args:
        pred: Model output, float32 1-D array.
        target: Pedal wet signal, same shape.
        dry: Dry input signal.
        sr: Sample rate in Hz.

    Returns:
        Full composite_perceptual result dict.
    """
    return composite_perceptual(pred, target, dry, sr)


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    SR = 96000
    T = 2.0
    N = int(SR * T)
    t = np.linspace(0, T, N, endpoint=False)

    # Dry: 220 Hz sine
    dry = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    # Target: hard-clipped distorted version (many harmonics)
    target = np.clip(dry * 8.0, -0.9, 0.9).astype(np.float32)

    # Pred: low-pass filtered version (sounds cleaner — simulates TCN failure)
    from scipy.signal import butter, sosfilt
    sos = butter(4, 2000 / (SR / 2), btype="low", output="sos")
    pred = sosfilt(sos, target).astype(np.float32)

    print("PedalDSP perceptual metrics — self-test")
    print(f"  dry:    220 Hz sine, {N} samples @ {SR} Hz")
    print(f"  target: hard-clipped (±0.9), RMS={float(np.sqrt(np.mean(target**2))):.3f}")
    print(f"  pred:   LP-filtered target (fc=2 kHz), RMS={float(np.sqrt(np.mean(pred**2))):.3f}")
    print()

    funcs = [
        (mel_axis_weighted,       "mel_axis_weighted",       "dB"),
        (equal_loudness_weighted, "equal_loudness_weighted", "dB (A-wtd)"),
        (masking_weighted,        "masking_weighted",        "visibility"),
        (signal_adaptive_weighted,"signal_adaptive_weighted","ratio"),
        (guitar_prior_weighted,   "guitar_prior_weighted",   "dB"),
        (harmonic_weighted,       "harmonic_weighted",       "dB"),
    ]

    all_ok = True
    fmt = "  {:<30s}  score={:.4f}  unit={}"
    for fn, name, unit in funcs:
        try:
            r = fn(pred, target, dry, sr=SR)
            score = r["score"]
            extra = ""
            if name == "masking_weighted":
                extra = f"  masked_fraction={r['metadata']['masked_fraction']:.2%}"
            elif name == "harmonic_weighted":
                extra = f"  f0={r['metadata']['f0_hz']:.1f} Hz"
            print(fmt.format(name, score, unit) + extra)
            if not (np.isfinite(score) and score >= 0.0):
                print(f"    FAIL: score not finite non-negative")
                all_ok = False
        except Exception as exc:
            print(f"  {name}  FAIL: {exc}")
            all_ok = False

    comp = composite_perceptual(pred, target, dry, sr=SR)
    print(fmt.format("composite_perceptual", comp["composite_score"], "z-score"))

    print()
    if all_ok:
        print("All self-tests passed.")
    else:
        print("SOME SELF-TESTS FAILED.")
        sys.exit(1)
