"""Abstract base class for all pedal models."""
from abc import ABC, abstractmethod

import numpy as np


class PedalModel(ABC):
    @abstractmethod
    def fit(self, dry: np.ndarray, wet: np.ndarray, sr: int) -> None:
        """Identify model parameters from a dry/wet capture pair.

        Args:
            dry: Input signal, shape (N,), float32, range [-1, 1].
            wet: Target output signal, same shape as dry.
            sr: Sample rate in Hz.
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run the model on a dry input signal.

        Args:
            x: Dry input, shape (N,), float32.

        Returns:
            Predicted wet signal, same shape as x.
        """

    @property
    @abstractmethod
    def receptive_field(self) -> int:
        """Number of past samples the model depends on."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier."""
