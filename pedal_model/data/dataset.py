"""PyTorch Dataset: slices aligned dry/wet WAV pairs into training windows."""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..capture.align import load_and_align


class PedalDataset(Dataset):
    """Overlapping-window dataset for sample-by-sample model training.

    Each item is a (dry_window, wet_sample) pair:
    - dry_window: R past dry samples ending at position n.
    - wet_sample: The corresponding wet sample at position n.

    For LSTM/GRU, use chunk_mode=True to return full chunks instead of
    single-sample targets — the trainer handles TBPTT.
    """

    def __init__(
        self,
        dry: np.ndarray,
        wet: np.ndarray,
        receptive_field: int = 1024,
        hop: int = 1,
    ) -> None:
        """
        Args:
            dry: Aligned dry audio, shape (N,), float32.
            wet: Aligned wet audio, shape (N,), float32.
            receptive_field: Context window length R. Must match the model's RF.
            hop: Step size between windows. 1 = maximum overlap (expensive).
                 Use larger values (e.g. 64) to trade coverage for speed.
        """
        assert len(dry) == len(wet), "dry and wet must have the same length."
        self._dry = np.pad(dry, (receptive_field - 1, 0))  # causal padding
        self._wet = wet
        self.rf = receptive_field
        self.hop = hop
        self._indices = np.arange(0, len(wet) - receptive_field + 1, hop, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self._indices[idx])
        window = torch.tensor(self._dry[start : start + self.rf], dtype=torch.float32)
        target = torch.tensor(self._wet[start + self.rf - 1], dtype=torch.float32)
        return window, target

    @classmethod
    def from_wav(
        cls,
        path: Path | str,
        receptive_field: int = 1024,
        hop: int = 1,
    ) -> "PedalDataset":
        """Load and align a stereo capture WAV, return a ready-to-use dataset.

        Args:
            path: Path to stereo WAV (ch0=dry, ch1=wet).
            receptive_field: Context window length in samples.
            hop: Step size between windows.

        Returns:
            PedalDataset instance.
        """
        dry, wet, _ = load_and_align(path)
        return cls(dry, wet, receptive_field=receptive_field, hop=hop)


class ChunkDataset(Dataset):
    """Chunk-based dataset for LSTM/GRU training with TBPTT.

    Each item is a (dry_chunk, wet_chunk) pair of length chunk_size,
    drawn from non-overlapping sequential segments of the audio.
    """

    def __init__(
        self,
        dry: np.ndarray,
        wet: np.ndarray,
        chunk_size: int = 2048,
    ) -> None:
        """
        Args:
            dry: Dry audio, shape (N,), float32.
            wet: Wet audio, shape (N,), float32.
            chunk_size: Samples per chunk. Passed to TBPTT in trainer.
        """
        n_chunks = len(dry) // chunk_size
        self._dry = torch.tensor(
            dry[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, 1),
            dtype=torch.float32,
        )
        self._wet = torch.tensor(
            wet[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, 1),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self._dry)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dry[idx], self._wet[idx]
