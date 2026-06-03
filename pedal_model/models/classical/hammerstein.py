"""Hammerstein model: static polynomial nonlinearity followed by a linear FIR filter."""
import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import rfft, irfft

from ..base import PedalModel


class HammersteinModel(PedalModel):
    """v[n] = f(x[n]),  y[n] = Σ h[k]·v[n-k].

    Identification:
    - f(·) is a polynomial fit via least squares on the amplitude-dependent
      harmonic content observed in the capture.
    - h[k] is identified from the linear regime (low-amplitude part of the signal).

    Best for Level-2 pedals: mild overdrive, soft clipping (diode-based).
    """

    def __init__(self, poly_order: int = 5, n_taps: int = 512) -> None:
        """
        Args:
            poly_order: Polynomial order for the static nonlinearity.
                Odd-order polynomials model symmetric (diode) clipping.
            n_taps: FIR kernel length for the output filter.
        """
        self.poly_order = poly_order
        self.n_taps = n_taps
        self._poly_coeffs: np.ndarray | None = None
        self._kernel: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"Hammerstein-p{self.poly_order}"

    @property
    def receptive_field(self) -> int:
        return self.n_taps

    def _apply_nl(self, x: np.ndarray) -> np.ndarray:
        assert self._poly_coeffs is not None
        return np.polyval(self._poly_coeffs, x).astype(np.float32)

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        """Identify f(·) and h[k] from a dry/wet capture pair.

        Args:
            dry: Dry input, shape (N,), float32.
            wet: Wet output, shape (N,), float32.
            sr: Sample rate in Hz.
        """
        # --- Identify f(·) via polynomial regression ---
        # Build Vandermonde matrix with odd + even powers up to poly_order
        powers = np.arange(1, self.poly_order + 1)
        X = np.column_stack([dry ** p for p in powers])  # (N, poly_order)
        # Least squares: wet ≈ X @ coeffs
        coeffs, _, _, _ = np.linalg.lstsq(X, wet, rcond=None)
        # Store as np.polyval format (highest power first) with a zero constant
        full = np.zeros(self.poly_order + 1)
        for i, p in enumerate(powers):
            full[self.poly_order - p] = coeffs[i]
        self._poly_coeffs = full

        # --- Identify h[k] from linear regime (low-amplitude samples) ---
        threshold = np.percentile(np.abs(dry), 25)
        mask = np.abs(dry) < threshold
        if mask.sum() < self.n_taps * 4:
            # Fall back to full signal if not enough low-amplitude samples
            mask = np.ones(len(dry), dtype=bool)

        dry_lin = dry[mask]
        # Apply NL to get intermediate signal, then find filter from intermediate→wet
        v_lin = self._apply_nl(dry_lin)
        # Align wet to same samples for kernel estimation
        wet_lin = wet[mask]

        n = len(v_lin)
        V = rfft(v_lin, n=n)
        W = rfft(wet_lin, n=n)
        H = W / (V + 1e-8 * np.max(np.abs(V)))
        h = irfft(H, n=n)[: self.n_taps]
        h *= np.hanning(self.n_taps)
        self._kernel = h.astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply f(·) then convolve with h[k].

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Model output, same shape.
        """
        if self._poly_coeffs is None or self._kernel is None:
            raise RuntimeError("Call fit() before predict().")
        v = self._apply_nl(x)
        return fftconvolve(v, self._kernel, mode="full")[: len(x)].astype(np.float32)
