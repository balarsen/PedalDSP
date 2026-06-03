"""2nd-order Volterra series model via least-squares regression."""
import numpy as np

from ..base import PedalModel


class VolterraModel(PedalModel):
    """Discrete Volterra series up to 2nd order.

    y[n] = Σₖ h1[k]·x[n-k]
         + Σₖ Σⱼ h2[k,j]·x[n-k]·x[n-j]

    Identification: least-squares regression with all linear and quadratic
    lag-product features as columns. Memory M controls parameter count:
    - Linear:    M terms
    - Quadratic: M(M+1)/2 unique terms (upper-triangle)

    Practical limit: M ≤ 30 to keep the system matrix tractable.
    """

    def __init__(self, memory: int = 20) -> None:
        """
        Args:
            memory: Number of past samples (M). Total parameters = M + M(M+1)/2.
        """
        self.memory = memory
        self._coeffs: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"Volterra-M{self.memory}"

    @property
    def receptive_field(self) -> int:
        return self.memory

    def _build_features(self, x: np.ndarray) -> np.ndarray:
        """Build the regression feature matrix for signal x.

        Args:
            x: Input signal, shape (N,).

        Returns:
            Feature matrix, shape (N - memory, n_features).
        """
        n = len(x)
        M = self.memory
        rows = n - M
        if rows <= 0:
            raise ValueError(f"Signal too short for memory={M}. Need > {M} samples.")

        # Linear features: x[n], x[n-1], ..., x[n-M]  (lag 0 through M)
        linear_cols = [x[M - k : n - k if k > 0 else n] for k in range(M + 1)]
        # Quadratic features: x[n-k]*x[n-j] for 0 <= k <= j <= M
        quad_cols = []
        for k in range(M + 1):
            for j in range(k, M + 1):
                quad_cols.append(linear_cols[k] * linear_cols[j])

        cols = linear_cols + quad_cols
        return np.column_stack(cols).astype(np.float64)

    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        """Identify Volterra kernels via least-squares.

        Args:
            dry: Dry input, shape (N,), float32.
            wet: Wet output, shape (N,), float32.
            sr: Sample rate in Hz.
        """
        X = self._build_features(dry.astype(np.float64))
        y = wet[self.memory :].astype(np.float64)
        self._coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the Volterra series.

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Predicted output, shape (N - memory,), float32.
            Note: output is shorter than input by `memory` samples.
        """
        if self._coeffs is None:
            raise RuntimeError("Call fit() before predict().")
        X = self._build_features(x.astype(np.float64))
        y = X @ self._coeffs
        # Prepend zeros so output length matches input length
        out = np.concatenate([np.zeros(self.memory), y])
        return out.astype(np.float32)
