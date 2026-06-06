"""PyTorch Dataset: slices aligned dry/wet WAV pairs into training windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..capture.align import load_and_align
from .resample import resample


def _load_sections(
    manifest: "Manifest",  # type: ignore[name-defined]  # noqa: F821
    wav_path: Path | str,
    *,
    section_labels: list[str] | None,
    section_types: list[str] | None,
    target_sr: int | None,
) -> np.ndarray:
    """Load and concatenate selected sections from a WAV file.

    Sections are selected by *section_labels* (exact match) or *section_types*
    (all sections of those types), or both (union). If neither filter is given,
    all sections are loaded. Silence gaps between sections are excluded — only
    the labeled section audio is included.

    Args:
        manifest: Loaded :class:`~pedal_model.signals.manifest.Manifest`.
        wav_path: WAV file to read from (must match manifest).
        section_labels: Exact section labels to include, or ``None``.
        section_types: Section type strings to include, or ``None``.
        target_sr: Resample to this rate after loading, or ``None`` to keep
            the manifest's native sample rate.

    Returns:
        Concatenated audio, shape ``(N,)``, float32.
    """
    if section_labels is None and section_types is None:
        sections = manifest.sections
    else:
        seen: set[str] = set()
        sections = []
        if section_labels:
            for lbl in section_labels:
                sec = manifest.get_section(lbl)
                if sec.label not in seen:
                    sections.append(sec)
                    seen.add(sec.label)
        if section_types:
            for sec in manifest.sections_of_type_multi(section_types):
                if sec.label not in seen:
                    sections.append(sec)
                    seen.add(sec.label)
        sections.sort(key=lambda s: s.index)

    chunks: list[np.ndarray] = []
    for sec in sections:
        audio, _ = sf.read(
            str(wav_path),
            start=sec.start_sample,
            stop=sec.end_sample,
            dtype="float32",
            always_2d=False,
        )
        if target_sr is not None and target_sr != manifest.sample_rate:
            audio = resample(audio, manifest.sample_rate, target_sr)
        chunks.append(audio)

    if not chunks:
        raise ValueError(
            "No sections matched the given filters. "
            f"Available labels: {manifest.labels()}"
        )
    return np.concatenate(chunks)


class PedalDataset(Dataset):
    """Overlapping-window dataset for sample-by-sample model training.

    Each item is a (dry_window, wet_sample) pair:
    - dry_window: R past dry samples ending at position n, shape (R,).
    - wet_sample: The corresponding wet sample at position n, scalar.

    For LSTM/GRU training with TBPTT, use :class:`ChunkDataset` instead.
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
        if len(dry) != len(wet):
            raise ValueError(
                f"dry and wet must be the same length; got {len(dry)} vs {len(wet)}."
            )
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
        """Load and align a stereo capture WAV (ch0=dry, ch1=wet).

        Args:
            path: Path to stereo WAV.
            receptive_field: Context window length in samples.
            hop: Step size between windows.

        Returns:
            :class:`PedalDataset` ready for training.
        """
        dry, wet, _ = load_and_align(path)
        return cls(dry, wet, receptive_field=receptive_field, hop=hop)

    @classmethod
    def from_manifest(
        cls,
        dry_manifest: "Manifest",  # type: ignore[name-defined]  # noqa: F821
        wet_wav: Path | str,
        *,
        section_labels: list[str] | None = None,
        section_types: list[str] | None = None,
        target_sr: int | None = None,
        receptive_field: int = 1024,
        hop: int = 1,
    ) -> "PedalDataset":
        """Build a dataset from a signal manifest and its corresponding wet capture.

        Dry audio is loaded section-by-section from the manifest's WAV file
        (using the manifest's own path). Wet audio is loaded from *wet_wav*
        using the same sample ranges — preserving dry/wet alignment exactly.
        Silence gaps between sections are excluded from both signals.

        Args:
            dry_manifest: Loaded :class:`~pedal_model.signals.manifest.Manifest`
                for the dry (generated) signal.
            wet_wav: Path to the wet WAV — the same signal after passing through
                the pedal, sample-aligned with the dry.
            section_labels: Exact section labels to include. ``None`` = all.
            section_types: Section type strings to include. ``None`` = all.
                Combined with *section_labels* as a union.
            target_sr: Resample both signals to this rate after loading.
                ``None`` keeps the manifest's native sample rate.
            receptive_field: Window length R in samples (at *target_sr* if
                resampling, else at manifest sample rate).
            hop: Step between windows. 1 = max overlap.

        Returns:
            :class:`PedalDataset` ready for training.
        """
        dry = _load_sections(
            dry_manifest,
            dry_manifest.path.with_suffix(".wav"),
            section_labels=section_labels,
            section_types=section_types,
            target_sr=target_sr,
        )
        wet = _load_sections(
            dry_manifest,
            wet_wav,
            section_labels=section_labels,
            section_types=section_types,
            target_sr=target_sr,
        )
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
        if len(dry) != len(wet):
            raise ValueError(
                f"dry and wet must be the same length; got {len(dry)} vs {len(wet)}."
            )
        n_chunks = len(dry) // chunk_size
        self._dry = torch.tensor(
            dry[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, 1),
            dtype=torch.float32,
        )
        self._wet = torch.tensor(
            wet[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, 1),
            dtype=torch.float32,
        )
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return len(self._dry)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dry[idx], self._wet[idx]

    @classmethod
    def from_manifest(
        cls,
        dry_manifest: "Manifest",  # type: ignore[name-defined]  # noqa: F821
        wet_wav: Path | str,
        *,
        section_labels: list[str] | None = None,
        section_types: list[str] | None = None,
        target_sr: int | None = None,
        chunk_size: int = 2048,
    ) -> "ChunkDataset":
        """Build a chunk dataset from a signal manifest and its wet capture.

        Identical filtering logic to :meth:`PedalDataset.from_manifest`.

        Args:
            dry_manifest: Loaded :class:`~pedal_model.signals.manifest.Manifest`.
            wet_wav: Aligned wet WAV path.
            section_labels: Section labels to include. ``None`` = all.
            section_types: Section type strings to include. ``None`` = all.
            target_sr: Resample to this rate, or ``None`` to keep native rate.
            chunk_size: TBPTT chunk length in samples.

        Returns:
            :class:`ChunkDataset` ready for training.
        """
        dry = _load_sections(
            dry_manifest,
            dry_manifest.path.with_suffix(".wav"),
            section_labels=section_labels,
            section_types=section_types,
            target_sr=target_sr,
        )
        wet = _load_sections(
            dry_manifest,
            wet_wav,
            section_labels=section_labels,
            section_types=section_types,
            target_sr=target_sr,
        )
        return cls(dry, wet, chunk_size=chunk_size)
