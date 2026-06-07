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
        """Identify Volterra kernels via streaming ridge-regularised least-squares.

        Streams through the full signal in 65536-sample blocks, accumulating
        the normal-equation matrices X.T @ X and X.T @ y without ever
        building the full feature matrix.  Mathematically identical to solving
        on the concatenated matrix but requires only O(features²) RAM instead
        of O(N × features).

        Ridge regularisation strength scales with trace(X.T X) so it is
        amplitude-invariant across different gain stages.

        Args:
            dry: Dry input, shape (N,), float32.
            wet: Wet output, shape (N,), float32.
            sr: Sample rate in Hz.
        """
        M = self.memory
        N = len(dry)
        if N <= M:
            raise ValueError(f"Signal too short for memory={M}. Need > {M} samples.")

        xd = dry.astype(np.float64)
        yd = wet.astype(np.float64)

        # Probe feature count using the smallest valid input
        n_feat = self._build_features(xd[: M + 2]).shape[1]

        XTX = np.zeros((n_feat, n_feat), dtype=np.float64)
        XTy = np.zeros(n_feat, dtype=np.float64)

        _BLOCK = 65_536
        pos = M  # first sample with full M-sample context
        n_blocks = 0
        while pos < N:
            end = min(pos + _BLOCK, N)
            # xd[pos-M : end] supplies M context samples + (end-pos) target samples
            X_block = self._build_features(xd[pos - M : end])   # (end-pos, n_feat)
            y_block = yd[pos : end]
            XTX += X_block.T @ X_block
            XTy += X_block.T @ y_block
            pos = end
            n_blocks += 1

        # Ridge: (X.T X + αI) h = X.T y
        alpha = 1e-4 * float(np.trace(XTX)) / n_feat
        self._coeffs = np.linalg.solve(XTX + alpha * np.eye(n_feat), XTy)
        print(f"      Volterra: streamed {n_blocks} blocks, {N/1e6:.2f}M samples "
              f"(100.0% coverage)")

    def predict(self, x: np.ndarray, chunk_size: int = 50_000) -> np.ndarray:
        """Apply the Volterra series in chunks to keep memory bounded.

        Building the full feature matrix for a long signal at once can exceed
        RAM (e.g., 3 M samples × 252 features × 8 bytes ≈ 6 GB). Processing
        in 50 k-sample chunks keeps peak usage under 100 MB.

        Args:
            x: Dry input, shape (N,), float32.
            chunk_size: Samples per chunk. Reduce if memory-constrained.

        Returns:
            Predicted output, shape (N,), float32.
            First `memory` samples are zero (no context for prediction).
        """
        if self._coeffs is None:
            raise RuntimeError("Call fit() before predict().")
        M = self.memory
        N = len(x)
        out = np.zeros(N, dtype=np.float32)
        xd = x.astype(np.float64)
        for start in range(M, N, chunk_size):
            end = min(start + chunk_size, N)
            # Slice includes M context samples so _build_features can form lags
            X_chunk = self._build_features(xd[start - M : end])
            # X_chunk shape: (end - start, n_features)
            out[start:end] = (X_chunk @ self._coeffs).astype(np.float32)
        return out
