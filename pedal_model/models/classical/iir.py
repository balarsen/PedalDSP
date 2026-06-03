"""IIR filter model: fits a rational transfer function from frequency response data."""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import sosfilt, tf2sos

from ..base import PedalModel


def _fit_iir_ls(H: np.ndarray, w: np.ndarray, n_b: int, n_a: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit digital IIR coefficients to complex frequency response samples via least squares.

    Replacement for scipy.signal.invfreqz (removed in scipy 1.14).
    Solves H(ωₖ)·A(e^{jωₖ}) = B(e^{jωₖ}) as a real-valued linear system.

    Args:
        H: Complex frequency response, shape (K,).
        w: Normalised angular frequencies in [0, π], shape (K,).
        n_b: Numerator order (number of zeros).
        n_a: Denominator order (number of poles). a[0] is fixed to 1.

    Returns:
        (b, a) — 1-D coefficient arrays suitable for scipy.signal.tf2sos.
    """
    z = np.exp(1j * w)
    K = len(w)

    # Denominator columns: -H_k · z_k^{-m}  for m = 1..n_a
    A_cols = np.column_stack([-H * (z ** (-m)) for m in range(1, n_a + 1)])  # (K, n_a)
    # Numerator columns: z_k^{-m}  for m = 0..n_b
    B_cols = np.column_stack([z ** (-m) for m in range(0, n_b + 1)])         # (K, n_b+1)

    mat = np.hstack([A_cols, B_cols])  # (K, n_a + n_b + 1) complex
    # Stack real and imaginary parts so the system is purely real-valued
    mat_r = np.vstack([mat.real, mat.imag])
    rhs_r = np.hstack([H.real, H.imag])

    x, _, _, _ = np.linalg.lstsq(mat_r, rhs_r, rcond=None)

    a = np.concatenate([[1.0], x[:n_a]])
    b = x[n_a:]
    return b, a


class IIRModel(PedalModel):
    """Identifies an IIR filter (poles + zeros) from a dry/wet capture pair.

    Uses second-order sections (SOS) for numerically stable filtering.
    Best for Level-1 pedals and tone stacks.
    """

    def __init__(self, order: int = 8, n_freq_points: int = 512) -> None:
        """
        Args:
            order: Filter order (number of poles = number of zeros).
            n_freq_points: Number of frequency points used for the fit.
        """
        self.order = order
        self.n_freq_points = n_freq_points
        self._sos: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"IIR-{self.order}"

    @property
    def receptive_field(self) -> int:
        return self.order * 10

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        """Estimate IIR coefficients by fitting to the measured frequency response.

        Args:
            dry: Dry input, shape (N,), float32.
            wet: Wet output, shape (N,), float32.
            sr: Sample rate in Hz.
        """
        n = len(dry)
        freqs = rfftfreq(n, d=1.0 / sr)
        DRY = rfft(dry, n=n)
        WET = rfft(wet, n=n)
        H = WET / (DRY + 1e-8 * np.max(np.abs(DRY)))

        idx = np.round(np.linspace(1, len(freqs) - 1, self.n_freq_points)).astype(int)
        w = 2.0 * np.pi * freqs[idx] / sr  # normalised angular frequency in [0, π]
        h = H[idx]

        b, a = _fit_iir_ls(h, w, n_b=self.order, n_a=self.order)
        self._sos = tf2sos(b, a)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted IIR filter using second-order sections.

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Filtered output, same shape.
        """
        if self._sos is None:
            raise RuntimeError("Call fit() before predict().")
        return sosfilt(self._sos, x).astype(np.float32)
