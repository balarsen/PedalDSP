"""Capture sessions on disk: folder layout, session.json log, per-preset WAVs, and
harness-ready params.json.

Layout::

    recordings/data/<profile>/sessions/<YYYY-MM-DD_HHMM>_<sr>k/      (simulated runs:
    sessions_demo/)
        session.json                 settings, rig snapshot, calibration, per-preset status, event
        log
        signals/train_<kind>_<sr>k.{wav,json}, val_full_<sr>k.{wav,json}
        <preset>/params.json         loads in scripts/fit_models.py --klon-sim-dir <preset dir>
        <preset>/wet_train.wav       return, aligned to the dry signal (lag and pre-roll removed)
        <preset>/raw_train.wav       unaligned 2-channel recording: [return, loopback]
        <preset>/wet_val.wav, raw_val.wav
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from pedal_model.rig.config import DATA_ROOT, RigConfig
from pedal_model.rig.engine import Take
from pedal_model.rig.profiles import Preset, Profile
from pedal_model.rig.signals import SessionSignal, prepare_signal

SCHEMA_VERSION = "1.0"
WAV_SUBTYPE = "PCM_24"  # the 1824c converters are 24-bit, so this is lossless for hardware takes


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Session:
    """One capture session (one sample rate, one pedal, one signal choice)."""

    def __init__(self, root: Path, data: dict) -> None:
        self.root = root
        self.data = data

    # ── create / load ───────────────────────────────────────────────────── #

    @classmethod
    def create(
        cls,
        profile: Profile,
        sr: int,
        signal_kind: str,
        rig: RigConfig,
        backend_name: str,
        simulated: bool,
        presets: list[Preset],
    ) -> "Session":
        """Make the folder and prepare its signals (resampling takes a few seconds: call off the GUI
        thread).
        """
        base = DATA_ROOT / profile.id / ("sessions_demo" if simulated else "sessions")
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        root = base / f"{stamp}_{sr // 1000}k"
        n = 2
        while root.exists():
            root = base / f"{stamp}_{sr // 1000}k_{n}"
            n += 1
        root.mkdir(parents=True)
        sig_dir = root / "signals"
        signals = {r: prepare_signal(r, signal_kind, sr, sig_dir) for r in ("train", "val")}
        data = {
            "schema_version": SCHEMA_VERSION,
            "created": _now(),
            "profile": profile.id,
            "pedal": profile.pedal,
            "sample_rate": sr,
            "signal_kind": signal_kind,
            "backend": backend_name,
            "simulated": simulated,
            "send_level_db": profile.send_level_db,
            "rig": rig.to_dict(),
            "signals": {
                r: s.manifest_path.relative_to(root).as_posix() for r, s in signals.items()
            },
            "calibration": None,
            "presets": {
                p.name: {
                    "status": "pending",
                    "controls": p.controls,
                    "description": p.description,
                    "takes": [],
                }
                for p in presets
            },
            "events": [],
        }
        s = cls(root, data)
        s.log("session_created")
        return s

    @classmethod
    def load(cls, root: Path) -> "Session":
        return cls(root, json.loads((root / "session.json").read_text(encoding="utf-8")))

    @staticmethod
    def list_sessions(profile_id: str, simulated: bool | None = None) -> list[Path]:
        """Session folders for a profile, newest first."""
        dirs = []
        for sub in ("sessions", "sessions_demo"):
            if simulated is True and sub == "sessions":
                continue
            if simulated is False and sub == "sessions_demo":
                continue
            base = DATA_ROOT / profile_id / sub
            if base.is_dir():
                dirs += [d for d in base.iterdir() if (d / "session.json").is_file()]
        return sorted(dirs, key=lambda d: d.name, reverse=True)

    # ── state ───────────────────────────────────────────────────────────── #

    @property
    def sample_rate(self) -> int:
        return int(self.data["sample_rate"])

    def signal(self, role: str) -> SessionSignal:
        path = self.root / self.data["signals"][role]
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SessionSignal(role, path, raw["sample_rate"], raw["total_samples"])

    def save(self) -> None:
        tmp = self.root / "session.json.tmp"
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(
            tmp, self.root / "session.json"
        )  # atomic: a crash never leaves a half-written log

    def log(self, event: str, **info) -> None:
        """Append to the event log (the future capture/train loop reads this) and save."""
        self.data["events"].append({"t": _now(), "event": event, **info})
        self.save()

    def set_calibration(self, loopback: dict | None, chain: dict | None) -> None:
        self.data["calibration"] = {"loopback": loopback, "chain": chain, "t": _now()}
        self.log("calibrated")

    def set_status(self, preset: str, status: str) -> None:
        self.data["presets"][preset]["status"] = status
        self.log("preset_status", preset=preset, status=status)

    def preset_dir(self, preset: str) -> Path:
        return self.root / preset

    # ── takes ───────────────────────────────────────────────────────────── #

    def save_capture(
        self, preset: Preset, role: str, take: Take, wet: np.ndarray, qc: dict
    ) -> None:
        """Write one role (train or val) of a preset and refresh its params.json."""
        d = self.preset_dir(preset.name)
        d.mkdir(exist_ok=True)
        sr = take.sr
        sf.write(str(d / f"wet_{role}.wav"), wet, sr, subtype=WAV_SUBTYPE)
        sf.write(str(d / f"raw_{role}.wav"), take.rec, sr, subtype=WAV_SUBTYPE)
        entry = {
            "role": role,
            "t": _now(),
            "lag_samples": qc["lag_samples"],
            "loop_lag_samples": qc["loop_lag_samples"],
            "xruns": take.xruns,
            "pre_roll": take.pre_roll,
            "verdict": qc["verdict"],
            "qc": qc["items"],
        }
        p = self.data["presets"][preset.name]
        p["takes"] = [t for t in p["takes"] if t["role"] != role] + [entry]
        p["status"] = (
            "captured" if {t["role"] for t in p["takes"]} >= {"train", "val"} else "partial"
        )
        self.write_params(preset)
        self.log("capture_saved", preset=preset.name, role=role, verdict=qc["verdict"])

    def write_params(self, preset: Preset) -> Path:
        """params.json in the harness format (pedal_model/eval/harness.py) plus a "rig" block."""
        d = self.preset_dir(preset.name)
        p = self.data["presets"][preset.name]
        takes = {t["role"]: t for t in p["takes"]}
        captures = {}
        for role in ("train", "val"):
            sig = self.signal(role)
            captures[role] = {
                "dry_signal": sig.manifest_path.stem,
                "dry_manifest": os.path.relpath(sig.manifest_path, d).replace("\\", "/"),
                "wet_wav": f"wet_{role}.wav",
                "raw_wav": f"raw_{role}.wav",
                "raw_channels": ["return (In 1)", "loopback (In 2)"],
                "sample_rate": sig.sample_rate,
                "n_samples": sig.total_samples,
                "duration_s": sig.duration_s,
                "lag_samples": takes.get(role, {}).get("lag_samples"),
            }
        doc = {
            "schema_version": "1.1",
            "date_created": _now(),
            "source": "simulated" if self.data["simulated"] else "hardware",
            "pedal": self.data["pedal"],
            "preset_name": preset.name,
            "preset_description": preset.description,
            "sliders": preset.controls,
            "captures": captures,
            "rig": {
                "session": self.root.name,
                "backend": self.data["backend"],
                "send_level_db": self.data["send_level_db"],
                "channels": self.data["rig"]["channels"],
                "boxes": self.data["rig"]["boxes"],
                "calibration": self.data["calibration"],
                "qc": {t["role"]: {"verdict": t["verdict"], "items": t["qc"]} for t in p["takes"]},
            },
        }
        path = d / "params.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return path
