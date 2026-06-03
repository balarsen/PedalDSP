"""FIR filter model: fits an impulse response from a dry/wet capture pair."""
import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import fftconvolve

from ..base import PedalModel


class FIRModel(PedalModel):
    """Identify and apply an FIR filter from frequency-domain division.

    Best suited for Level-1 pedals: boosts, buffers, passive EQ. Produces
    near-zero ESR and THD error when the pedal is truly linear.
    """

    def __init__(self, n_taps: int = 1024) -> None:
        """
        Args:
            n_taps: Length of the FIR kernel. Longer = more frequency resolution.
        """
        self.n_taps = n_taps
        self._kernel: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"FIR-{self.n_taps}"

    @property
    def receptive_field(self) -> int:
        return self.n_taps

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        """Estimate the FIR kernel via H(ω) = FFT(wet) / FFT(dry).

        Args:
            dry: Dry input, shape (N,), float32.
            wet: Wet output, shape (N,), float32.
            sr: Sample rate in Hz.
        """
        n = len(dry)
        DRY = rfft(dry, n=n)
        WET = rfft(wet, n=n)
        # Regularised frequency-domain division
        H = WET / (DRY + 1e-8 * np.max(np.abs(DRY)))
        h = irfft(H, n=n)
        # Truncate and apply a Hann window to suppress time-aliasing artefacts
        h = h[: self.n_taps]
        # Exponential decay window: preserves h[0]=1 and tapers the tail.
        # A symmetric Hann window would zero out the h[0] (main gain tap).
        h *= np.exp(-np.arange(self.n_taps) * 4.0 / self.n_taps)
        self._kernel = h.astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted FIR filter.

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Filtered output, same shape.
        """
        if self._kernel is None:
            raise RuntimeError("Call fit() before predict().")
        # mode='full' gives the causal convolution; take first len(x) samples.
        # mode='same' would shift output by n_taps//2, misaligning the prediction.
        return fftconvolve(x, self._kernel, mode="full")[: len(x)].astype(np.float32)
