"""Compare saved captures: rebuild a preset's response from its files, and line two sessions up (48
vs 96 kHz).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from pedal_model.rig.qc import expected_response, measure_response
from pedal_model.rig.session import Session
from pedal_model.signals.manifest import Manifest

COMPARE_BAND_HZ = (20.0, 20_000.0)
SPLIT_HZ = (
    10_000.0  # report agreement below and above this separately (the plan's 48 vs 96 kHz check)
)


@dataclass
class PresetResponse:
    session: str
    preset: str
    sample_rate: int
    freqs_hz: np.ndarray
    measured_db: np.ndarray
    expected_db: np.ndarray | None


def preset_response(session: Session, preset: str, role: str = "train") -> PresetResponse | None:
    """Response of one saved capture, normalised exactly as QC does (send level and calibration
    chain gain).
    """
    d = session.preset_dir(preset)
    wet_path = d / f"wet_{role}.wav"
    if not wet_path.is_file():
        return None
    sig = session.signal(role)
    manifest = Manifest(sig.manifest_path)
    dry = manifest.load_all()
    wet, sr = sf.read(str(wet_path), dtype="float32")
    nyquist_band = (COMPARE_BAND_HZ[0], min(COMPARE_BAND_HZ[1], sr / 2 * 0.95))
    grid, db = measure_response(dry, wet, manifest, sr, nyquist_band)
    db = db - session.data["send_level_db"]
    chain = (session.data.get("calibration") or {}).get("chain")
    if chain:
        db = db - chain["gain_db"]
    controls = session.data["presets"][preset].get("controls")
    exp = expected_response(session.data["pedal"], controls, sr, grid) if controls else None
    return PresetResponse(session.root.name, preset, sr, grid, db, exp)


def agreement(a: PresetResponse, b: PresetResponse) -> dict[str, float]:
    """RMS difference between two responses on a's grid, below and above SPLIT_HZ (dB)."""
    bb = np.interp(a.freqs_hz, b.freqs_hz, b.measured_db, left=np.nan, right=np.nan)
    diff = a.measured_db - bb
    lo = a.freqs_hz < SPLIT_HZ
    out = {"rms_below_10k_db": float(np.sqrt(np.nanmean(diff[lo] ** 2)))}
    hi = ~lo & ~np.isnan(diff)
    out["rms_above_10k_db"] = float(np.sqrt(np.mean(diff[hi] ** 2))) if np.any(hi) else float("nan")
    out["offset_db"] = float(np.nanmedian(diff))
    return out


def common_presets(a: Session, b: Session) -> list[str]:
    return [
        p
        for p in a.data["presets"]
        if p in b.data["presets"]
        and (a.preset_dir(p) / "wet_train.wav").is_file()
        and (b.preset_dir(p) / "wet_train.wav").is_file()
    ]


def load(path: Path | str) -> Session:
    return Session.load(Path(path))
