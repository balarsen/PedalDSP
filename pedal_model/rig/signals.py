"""Capture signals for a session, at the session sample rate.

Every session derives its own signal files from the 96 kHz masters, so a session folder is
self-contained and 48 kHz and 96 kHz sessions play exactly the same material. Each derived WAV gets
a rewritten manifest (same format as ``pedal_model.signals``) with section positions in the new
sample rate.

Roles:
    train: the full training signal, the quick excerpt (sweeps + stepped tones + pink noise),
        or the sweeps only (kind "sweeps": all the GE-7 block-model fit uses).
    val: the validation signal, always in full.
    sweep: the three log sweeps only (~1 min), for measurements such as the GE-7 slider map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from pedal_model.data.resample import resample
from pedal_model.rig.config import PROJECT_ROOT, SIGNALS_DIR

SOURCE_MANIFESTS = {
    "train": SIGNALS_DIR / "train_signal_v2.json",  # 96 kHz master, 642.5 s
    "val": SIGNALS_DIR / "val_signal_v1.json",  # 96 kHz master, 65 s (always played in full)
}

SIGNAL_KINDS = {
    "full": "Full train signal (642.5 s) + val (65 s): for model fitting",
    "quick": "Quick check: sweeps + stepped tones + pink noise (~3.5 min) + val (65 s)",
    "sweeps": "Sweeps only (62 s) + val (65 s): the block-model fit; a preset can ask for full",
}

# Section types kept in each excerpt. Quick: enough for frequency response, level linearity and the
# noise floor. Sweep: frequency response only, at three levels.
EXCERPT_TYPES = {
    "quick": ("log_sine_sweep", "stepped_sine_tone", "pink_noise"),
    "sweep": ("log_sine_sweep",),
}
QUICK_TYPES = EXCERPT_TYPES["quick"]
QUICK_GAP_S = 0.5  # silence between excerpted blocks, as in the generator (silence_between_s)


@dataclass
class SessionSignal:
    """One prepared dry signal: the manifest path (WAV beside it) and its length."""

    role: str
    manifest_path: Path
    sample_rate: int
    total_samples: int

    @property
    def wav_path(self) -> Path:
        return self.manifest_path.with_suffix(".wav")

    @property
    def duration_s(self) -> float:
        return self.total_samples / self.sample_rate


def _excerpt(raw: dict, audio: np.ndarray, types: tuple[str, ...]) -> tuple[np.ndarray, list[dict]]:
    """Concatenate the sections of the given types, with silence gaps, and re-index them."""
    sr = raw["sample_rate"]
    gap = np.zeros(int(round(QUICK_GAP_S * sr)), dtype=np.float32)
    pieces: list[np.ndarray] = [gap]
    sections: list[dict] = []
    cursor = len(gap)
    previous_end = None
    for s in raw["sections"]:
        if s["type"] not in types:
            continue
        seg = audio[s["start_sample"] : s["end_sample"]]
        # Keep runs of adjacent sections (the stepped tones) together; gap between separate blocks.
        if previous_end is not None and s["start_sample"] != previous_end:
            pieces.append(gap)
            cursor += len(gap)
        new = dict(s)
        new["index"] = len(sections)
        new["start_sample"] = cursor
        new["end_sample"] = cursor + len(seg)
        new["start_s"] = cursor / sr
        new["end_s"] = (cursor + len(seg)) / sr
        sections.append(new)
        pieces.append(seg)
        cursor += len(seg)
        previous_end = s["end_sample"]
    pieces.append(gap)
    return np.concatenate(pieces).astype(np.float32), sections


def _rescale_sections(sections: list[dict], sr_in: int, sr_out: int) -> list[dict]:
    """Convert section sample positions from sr_in to sr_out (times in seconds are unchanged)."""
    if sr_in == sr_out:
        return [dict(s) for s in sections]
    out = []
    for s in sections:
        new = dict(s)
        new["start_sample"] = int(round(s["start_sample"] * sr_out / sr_in))
        new["end_sample"] = int(round(s["end_sample"] * sr_out / sr_in))
        out.append(new)
    return out


def signal_name(role: str, kind: str, sr: int) -> str:
    """File stem of a session signal: ``train_quick_48k``, ``val_full_96k``, ``sweep_48k``."""
    if role == "sweep":
        return f"sweep_{sr // 1000}k"
    return f"{role}_{'full' if role == 'val' else kind}_{sr // 1000}k"


def prepare_signal(role: str, kind: str, sr: int, out_dir: Path) -> SessionSignal:
    """Write the role's signal (WAV + manifest) into *out_dir*; skipped if it is already there.

    Args:
        role: "train", "val" or "sweep".
        kind: "full", "quick" or "sweeps" (only affects train; val is always full, sweep always
            sweeps).
        sr: Session sample rate in Hz (48000 or 96000).
        out_dir: Session ``signals/`` folder.

    Returns:
        The prepared signal.
    """
    name = signal_name(role, kind, sr)
    if role == "sweep" or (role == "train" and kind == "sweeps"):
        excerpt = "sweep"
    elif role == "train" and kind == "quick":
        excerpt = "quick"
    else:
        excerpt = None
    manifest_path = out_dir / f"{name}.json"
    if manifest_path.is_file() and manifest_path.with_suffix(".wav").is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SessionSignal(role, manifest_path, raw["sample_rate"], raw["total_samples"])

    src_path = SOURCE_MANIFESTS["val" if role == "val" else "train"]
    raw = json.loads(src_path.read_text(encoding="utf-8"))
    audio, _ = sf.read(str(src_path.with_suffix(".wav")), dtype="float32", always_2d=False)
    sr_src = raw["sample_rate"]

    if excerpt:
        audio, sections = _excerpt(raw, audio, EXCERPT_TYPES[excerpt])
    else:
        sections = [dict(s) for s in raw["sections"]]

    audio = resample(audio, sr_src, sr)
    sections = _rescale_sections(sections, sr_src, sr)

    out = dict(raw)
    out["signal_name"] = name
    out["sample_rate"] = sr
    out["total_samples"] = int(len(audio))
    out["total_duration_s"] = len(audio) / sr
    out["sections"] = sections
    out["derived_from"] = {
        "manifest": src_path.relative_to(PROJECT_ROOT).as_posix(),
        "role": role,
        "excerpt": excerpt,
        "excerpt_types": list(EXCERPT_TYPES[excerpt]) if excerpt else None,
        "source_sample_rate": sr_src,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(manifest_path.with_suffix(".wav")), audio, sr, subtype="FLOAT")
    manifest_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return SessionSignal(role, manifest_path, sr, int(len(audio)))
