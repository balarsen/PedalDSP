"""Manifest reader: load a signal JSON and query sections by label or type."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Section:
    """Metadata for one named segment of a signal file.

    Attributes:
        label: Unique section identifier, e.g. ``"val_sweep_l-24"``.
        type: Signal type string, e.g. ``"log_sine_sweep"``.
        index: Position in the manifest sections list (0-based).
        start_sample: First sample index (inclusive).
        end_sample: Last sample index (exclusive).
        start_s: Start time in seconds.
        end_s: End time in seconds.
        amplitude_dbfs: Nominal level in dBFS (peak or RMS per ``params``).
        params: Full per-section param dict from the manifest.
    """

    label: str
    type: str
    index: int
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    amplitude_dbfs: float
    params: dict[str, Any]

    @property
    def n_samples(self) -> int:
        """Number of samples in this section."""
        return self.end_sample - self.start_sample

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.end_s - self.start_s


class Manifest:
    """Parsed signal manifest with section lookup.

    Args:
        path: Path to a ``{signal}_signal_v1.json`` file produced by
            :func:`pedal_model.signals.generate.generate`.

    Attributes:
        path: Resolved path to the JSON file.
        signal_name: e.g. ``"train_signal_v1"``.
        sample_rate: Recording sample rate in Hz.
        seed: RNG seed used during generation.
        total_samples: Total length of the WAV in samples.
        total_duration_s: Total length in seconds.
        schema_version: Manifest schema version string.
        generator_version: Generator code version string.
        params: Raw params dict from the manifest.
        sections: All sections in order.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        raw = json.loads(self.path.read_text())

        self.signal_name: str = raw["signal_name"]
        self.sample_rate: int = raw["sample_rate"]
        self.seed: int = raw["seed"]
        self.total_samples: int = raw["total_samples"]
        self.total_duration_s: float = raw["total_duration_s"]
        self.schema_version: str = raw["schema_version"]
        self.generator_version: str = raw["generator_version"]
        self.params: dict[str, Any] = raw["params"]

        self.sections: list[Section] = [
            Section(
                label=s["label"],
                type=s["type"],
                index=s["index"],
                start_sample=s["start_sample"],
                end_sample=s["end_sample"],
                start_s=s["start_s"],
                end_s=s["end_s"],
                amplitude_dbfs=s["amplitude_dbfs"],
                params=s["params"],
            )
            for s in raw["sections"]
        ]
        self._by_label: dict[str, Section] = {s.label: s for s in self.sections}

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_section(self, label: str) -> Section:
        """Return the Section with the given label.

        Args:
            label: Exact section label, e.g. ``"val_sweep_l-24"``.

        Returns:
            The matching :class:`Section`.

        Raises:
            KeyError: If *label* is not in this manifest.
        """
        try:
            return self._by_label[label]
        except KeyError:
            raise KeyError(
                f"Section {label!r} not found in {self.path.name}. "
                f"Available labels: {sorted(self._by_label)}"
            ) from None

    def sections_of_type(self, type_name: str) -> list[Section]:
        """Return all sections whose ``type`` field matches *type_name*.

        Args:
            type_name: e.g. ``"log_sine_sweep"``, ``"stepped_sine_tone"``.

        Returns:
            List of matching sections in manifest order. Empty if none match.
        """
        return [s for s in self.sections if s.type == type_name]

    def labels(self) -> list[str]:
        """Return all section labels in manifest order."""
        return [s.label for s in self.sections]

    def types(self) -> list[str]:
        """Return the unique section types present in this manifest."""
        seen: list[str] = []
        for s in self.sections:
            if s.type not in seen:
                seen.append(s.type)
        return seen

    # ── Audio loading ─────────────────────────────────────────────────────────

    def _wav_path(self) -> Path:
        """Resolve the WAV file expected alongside this manifest."""
        candidate = self.path.with_suffix(".wav")
        if not candidate.exists():
            raise FileNotFoundError(
                f"WAV file not found at {candidate}. "
                "Regenerate it with: "
                f"python generate_signal.py --from-manifest {self.path}"
            )
        return candidate

    def load_section(self, label: str) -> np.ndarray:
        """Load audio for one named section from the WAV file.

        Args:
            label: Section label to load.

        Returns:
            Audio samples, shape ``(n_samples,)``, float32, range ``[-1, 1]``.

        Raises:
            KeyError: If *label* is not in this manifest.
            FileNotFoundError: If the WAV file does not exist alongside the JSON.
        """
        sec = self.get_section(label)
        audio, _ = sf.read(
            str(self._wav_path()),
            start=sec.start_sample,
            stop=sec.end_sample,
            dtype="float32",
            always_2d=False,
        )
        return audio

    def load_all(self) -> np.ndarray:
        """Load the full WAV into memory.

        Returns:
            Audio samples, shape ``(total_samples,)``, float32.

        Raises:
            FileNotFoundError: If the WAV file does not exist.
        """
        audio, _ = sf.read(str(self._wav_path()), dtype="float32", always_2d=False)
        return audio

    def slice_section(self, label: str, audio: np.ndarray) -> np.ndarray:
        """Slice a pre-loaded audio array to a named section.

        Useful when the full WAV is already in memory and you want to avoid
        repeated disk reads.

        Args:
            label: Section label.
            audio: Full audio array of length ``total_samples``.

        Returns:
            View into *audio* for the requested section, shape ``(n_samples,)``.
        """
        sec = self.get_section(label)
        return audio[sec.start_sample : sec.end_sample]

    def __repr__(self) -> str:
        return (
            f"Manifest({self.signal_name!r}, "
            f"{len(self.sections)} sections, "
            f"{self.total_duration_s:.1f}s, "
            f"sr={self.sample_rate})"
        )
