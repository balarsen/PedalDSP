"""Wiener-Hammerstein model: input filter → static NL → output filter."""
import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import fftconvolve

from ..base import PedalModel


class WienerHammersteinModel(PedalModel):
    """v[n] = Σ h1[k]·x[n-k],  w[n] = f(v[n]),  y[n] = Σ h2[k]·w[n-k].

    Topology matches the Tube Screamer circuit exactly:
    op-amp bandpass filter → diode soft clipper → tone/output filter.

    Identification uses a two-stage approach:
    1. Estimate h1 from the linear regime (low amplitude).
    2. Apply h1, fit f(·) via polynomial regression on the residual.
    3. Apply f(·), estimate h2 from intermediate → wet.
    """

    def __init__(self, poly_order: int = 5, n_taps: int = 512) -> None:
        """
        Args:
            poly_order: Polynomial order for f(·).
            n_taps: FIR kernel length for both h1 and h2.
        """
        self.poly_order = poly_order
        self.n_taps = n_taps
        self._h1: np.ndarray | None = None
        self._h2: np.ndarray | None = None
        self._poly_coeffs: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"WienerHammerstein-p{self.poly_order}"

    @property
    def receptive_field(self) -> int:
        return self.n_taps * 2

    def _fit_fir(self, inp: np.ndarray, out: np.ndarray) -> np.ndarray:
        n = len(inp)
        I = rfft(inp, n=n)
        O = rfft(out, n=n)
        H = O / (I + 1e-8 * np.max(np.abs(I)))
        h = irfft(H, n=n)[: self.n_taps]
        h *= np.hanning(self.n_taps)
        return h.astype(np.float32)

    def _apply_nl(self, x: np.ndarray) -> np.ndarray:
        assert self._poly_coeffs is not None
        return np.polyval(self._poly_coeffs, x).astype(np.float32)

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        """Three-stage identification.

        Args:
            dry: Dry input, shape (N,), float32.
            wet: Wet output, shape (N,), float32.
            sr: Sample rate in Hz.
        """
        # Stage 1: estimate h1 from linear regime (dry → wet at low amplitude)
        threshold = np.percentile(np.abs(dry), 20)
        mask = np.abs(dry) < threshold
        if mask.sum() < self.n_taps * 4:
            mask = np.ones(len(dry), dtype=bool)
        self._h1 = self._fit_fir(dry[mask], wet[mask])

        # Stage 2: apply h1 to full signal, fit f(·)
        v = fftconvolve(dry, self._h1, mode="full")[: len(dry)]
        powers = np.arange(1, self.poly_order + 1)
        X = np.column_stack([v ** p for p in powers])
        coeffs, _, _, _ = np.linalg.lstsq(X, wet, rcond=None)
        full = np.zeros(self.poly_order + 1)
        for i, p in enumerate(powers):
            full[self.poly_order - p] = coeffs[i]
        self._poly_coeffs = full

        # Stage 3: apply f(v), estimate h2
        w = self._apply_nl(v)
        self._h2 = self._fit_fir(w, wet)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """h1 → f(·) → h2.

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Model output, same shape.
        """
        if any(a is None for a in (self._h1, self._h2, self._poly_coeffs)):
            raise RuntimeError("Call fit() before predict().")
        v = fftconvolve(x, self._h1, mode="full")[: len(x)]
        w = self._apply_nl(v)
        return fftconvolve(w, self._h2, mode="full")[: len(x)].astype(np.float32)
