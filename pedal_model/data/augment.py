"""Data augmentation transforms for dry/wet audio pairs."""
import numpy as np


class GainJitter:
    """Apply identical random gain to both dry and wet channels.

    Gain is sampled once per call, applied to both channels so the
    dry→wet relationship is preserved.
    """

    def __init__(self, min_db: float = -6.0, max_db: float = 6.0) -> None:
        """
        Args:
            min_db: Minimum gain in dB.
            max_db: Maximum gain in dB.
        """
        self.min_db = min_db
        self.max_db = max_db
        self._rng = np.random.default_rng()

    def __call__(
        self, dry: np.ndarray, wet: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random gain to both channels.

        Args:
            dry: Dry audio, shape (N,), float32.
            wet: Wet audio, shape (N,), float32.

        Returns:
            (dry_aug, wet_aug) — same shape, gain-shifted.
        """
        db = self._rng.uniform(self.min_db, self.max_db)
        gain = 10.0 ** (db / 20.0)
        return (dry * gain).astype(np.float32), (wet * gain).astype(np.float32)


class AddNoise:
    """Add low-level white noise to the dry channel only.

    Simulates the noise floor of the interface or pedal.
    """

    def __init__(self, noise_floor_db: float = -60.0) -> None:
        """
        Args:
            noise_floor_db: Noise level relative to 0 dBFS.
        """
        self.noise_amp = 10.0 ** (noise_floor_db / 20.0)
        self._rng = np.random.default_rng()

    def __call__(
        self, dry: np.ndarray, wet: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        noise = self._rng.standard_normal(len(dry)).astype(np.float32) * self.noise_amp
        return (dry + noise).astype(np.float32), wet
