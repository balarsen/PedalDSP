"""Capture sessions on disk: folder layout, session.json log, per-preset WAVs and params.json.

Layout (simulated runs go to ``sessions_demo/`` instead of ``sessions/``)::

    recordings/data/<profile>/sessions/<YYYY-MM-DD_HHMM>_<sr>k/
        session.json           settings, rig snapshot, calibration, presets, status, events
        calibration_<stamp>.json            every calibration, also as its own dated file
        calibration/<profile>_ref_<name>_raw_<sr>k_<stamp>.flac  C2-C4 reference captures
        signals/               train_<kind>_<sr>k.*, val_full_<sr>k.*, sweep_<sr>k.* (WAV + JSON)
        <preset>/params.json   loads in scripts/fit_models.py --klon-sim-dir <preset dir>
        <preset>/<profile>_<preset>_<role>_wet_<sr>k_<stamp>.flac  aligned return
        <preset>/<profile>_<preset>_<role>_raw_<sr>k_<stamp>.flac  return + loopback, unaligned

    <stamp> is the local date and time, YYYY-MM-DD_HHMM.

File names carry the profile, preset, role, sample rate and time, so a WAV copied out of its folder
is still identifiable; the same fields are in params.json and session.json. Retakes keep the earlier
files (the session log says which take is current).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from pedal_model.rig.config import DATA_ROOT, RigConfig, code_version, local_stamp
from pedal_model.rig.engine import Take
from pedal_model.rig.profiles import Preset, Profile
from pedal_model.rig.signals import SessionSignal, prepare_signal

SCHEMA_VERSION = "1.1"
# Captures are stored as 24-bit FLAC: the 1824c converters are 24-bit, so this is lossless for
# hardware takes, and on real captures it is 49-64 % of the WAV size (measured 2026-10-18).
# Session signals stay float32 WAV: FLAC stores integers only.
CAPTURE_FORMAT, CAPTURE_SUBTYPE, CAPTURE_EXT = "FLAC", "PCM_24", ".flac"
SIGNAL_ROLES = ("train", "val", "sweep")


def write_capture(path: Path, audio: np.ndarray, sr: int, description: str) -> None:
    """Write a capture as 24-bit FLAC, tagged with the code version and what it holds."""
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    with sf.SoundFile(
        str(path), "w", sr, channels, subtype=CAPTURE_SUBTYPE, format=CAPTURE_FORMAT
    ) as f:
        # Tags must be set before the first frame is written.
        f.software = f"PedalDSP capture rig, code {code_version()}"
        f.comment = description
        f.date = local_stamp(iso=True)
        f.write(audio)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Session:
    """One capture session (one sample rate, one pedal, one train-signal choice)."""

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
        """Make the folder and prepare its signals (takes seconds: call it off the GUI thread)."""
        base = DATA_ROOT / profile.id / ("sessions_demo" if simulated else "sessions")
        stamp = local_stamp()
        root = base / f"{stamp}_{sr // 1000}k"
        n = 2
        while root.exists():
            root = base / f"{stamp}_{sr // 1000}k_{n}"
            n += 1
        root.mkdir(parents=True)
        signals = {r: prepare_signal(r, signal_kind, sr, root / "signals") for r in SIGNAL_ROLES}
        data = {
            "schema_version": SCHEMA_VERSION,
            "created": _utc(),
            "created_local": local_stamp(iso=True),
            "code_version": code_version(),
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
            "presets": {},
            "events": [],
        }
        s = cls(root, data)
        for p in presets:
            s.add_preset(p, save=False)
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

    @property
    def profile_id(self) -> str:
        return self.data["profile"]

    def signal_key(self, role: str, kind: str | None = None) -> str:
        """Key of a signal in session.json "signals": the role, or "train_<kind>" when a preset
        asks for a train signal other than the session's (val and sweep have one kind only).
        """
        if role != "train" or kind in (None, self.data["signal_kind"]):
            return role
        return f"train_{kind}"

    def signal(self, role: str, kind: str | None = None) -> SessionSignal:
        """A session signal; prepared on first use (a preset's own train signal, or a role this
        session predates, e.g. "sweep"). ``kind`` is the preset's train signal, None for the
        session's.
        """
        key = self.signal_key(role, kind)
        if key not in self.data["signals"]:
            sig = prepare_signal(
                role, kind or self.data["signal_kind"], self.sample_rate, self.root / "signals"
            )
            self.data["signals"][key] = sig.manifest_path.relative_to(self.root).as_posix()
            self.save()
        path = self.root / self.data["signals"][key]
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SessionSignal(role, path, raw["sample_rate"], raw["total_samples"])

    def presets(self) -> list[Preset]:
        """The session's presets in plan order, as stored in session.json."""
        return [Preset.from_dict(name, d) for name, d in self.data["presets"].items()]

    def add_preset(self, preset: Preset, save: bool = True) -> None:
        """Add a preset to the session (no-op if it is already there)."""
        if preset.name not in self.data["presets"]:
            self.data["presets"][preset.name] = {
                **preset.to_dict(),
                "status": "pending",
                "takes": [],
            }
            if save:
                self.log("preset_added", preset=preset.name, kind=preset.kind)

    def save(self) -> None:
        tmp = self.root / "session.json.tmp"
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(
            tmp, self.root / "session.json"
        )  # atomic: a crash never leaves a half-written log

    def log(self, event: str, **info) -> None:
        """Append to the event log (the future capture/train loop reads this) and save."""
        self.data["events"].append({"t": _utc(), "event": event, "code": code_version(), **info})
        self.save()

    def set_calibration(
        self,
        loopback: dict | None,
        chain: dict | None,
        references: dict[str, dict] | None = None,
        steps: dict[str, dict] | None = None,
    ) -> Path:
        """Store the calibration in session.json and as its own dated file; returns that file.

        ``references`` are the C2-C4 response curves by name ("chain", "bypass", "engaged_flat");
        ``steps`` the status of each guided step ("C1"... with status, advice, time).
        """
        stamp = local_stamp()
        cal = {
            "loopback": loopback,
            "chain": chain,
            "references": references or {},
            "steps": steps or {},
            "t": _utc(),
            "time_local": local_stamp(iso=True),
            "session": self.root.name,
            "profile": self.profile_id,
            "sample_rate": self.sample_rate,
            "code_version": code_version(),
            "file": f"calibration_{stamp}.json",
        }
        self.data["calibration"] = cal
        path = self.root / f"calibration_{stamp}.json"
        path.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8", newline="\n")
        self.log("calibrated", file=path.name)
        return path

    def save_reference(self, name: str, take: Take) -> str:
        """Write a reference capture (return + loopback, unaligned); returns its session path."""
        d = self.root / "calibration"
        d.mkdir(exist_ok=True)
        fname = f"{self.profile_id}_ref_{name}_raw_{self.sample_rate // 1000}k_{local_stamp()}"
        path = d / (fname + CAPTURE_EXT)
        what = f"{self.root.name} reference {name}: return, loopback"
        write_capture(path, take.rec, take.sr, what)
        rel = path.relative_to(self.root).as_posix()
        self.log("reference_saved", reference=name, file=rel)
        return rel

    def set_status(self, preset: str, status: str) -> None:
        self.data["presets"][preset]["status"] = status
        self.log("preset_status", preset=preset, status=status)

    def preset_dir(self, preset: str) -> Path:
        return self.root / preset

    def take(self, preset: str, role: str) -> dict | None:
        """The current take entry of one role of a preset, or None."""
        for t in self.data["presets"].get(preset, {}).get("takes", []):
            if t["role"] == role:
                return t
        return None

    def wet_path(self, preset: str, role: str) -> Path | None:
        t = self.take(preset, role)
        return self.preset_dir(preset) / t["wet_wav"] if t else None

    # ── takes ───────────────────────────────────────────────────────────── #

    def _file_name(self, preset: str, role: str, what: str, stamp: str) -> str:
        sr_k = self.sample_rate // 1000
        return f"{self.profile_id}_{preset}_{role}_{what}_{sr_k}k_{stamp}{CAPTURE_EXT}"

    def save_capture(
        self,
        preset: Preset,
        role: str,
        take: Take,
        wet: np.ndarray,
        qc: dict,
        band_fit: dict | None = None,
        level_plan: dict | None = None,
    ) -> None:
        """Write one role of a preset (train, val or sweep) and refresh its params.json."""
        d = self.preset_dir(preset.name)
        d.mkdir(exist_ok=True)
        stamp = local_stamp()
        wet_name = self._file_name(preset.name, role, "wet", stamp)
        raw_name = self._file_name(preset.name, role, "raw", stamp)
        what = f"{self.root.name} {preset.name} {role}"
        what += f", send level {take.send_level_db:+.1f} dB"
        write_capture(d / wet_name, wet, take.sr, f"{what}: aligned return")
        write_capture(d / raw_name, take.rec, take.sr, f"{what}: return, loopback (unaligned)")
        entry = {
            "role": role,
            "signal": self.signal_key(role, preset.signal),
            "t": _utc(),
            "time_local": local_stamp(iso=True),
            "wet_wav": wet_name,
            "raw_wav": raw_name,
            "lag_samples": qc["lag_samples"],
            "loop_lag_samples": qc["loop_lag_samples"],
            "xcorr_lag_samples": qc.get("xcorr_lag_samples"),
            "xruns": take.xruns,
            "pre_roll": take.pre_roll,
            "verdict": qc["verdict"],
            "code_version": code_version(),
            # The send level this take actually used (auto-level may lower it per take), the
            # auto-level plan, and the calibration it was checked against.
            "send_level_db": take.send_level_db,
            "auto_level": level_plan,
            "calibration_file": (self.data.get("calibration") or {}).get("file"),
            "qc": qc["items"],
            "band_fit": band_fit,
        }
        self.add_preset(preset, save=False)
        p = self.data["presets"][preset.name]
        p["takes"] = [t for t in p["takes"] if t["role"] != role] + [entry]
        done = {t["role"] for t in p["takes"]} >= set(preset.roles)
        p["status"] = "captured" if done else "partial"
        self.write_params(preset)
        self.log(
            "capture_saved", preset=preset.name, role=role, verdict=qc["verdict"], file=wet_name
        )

    def write_params(self, preset: Preset) -> Path:
        """params.json in the harness format (pedal_model/eval/harness.py) plus a "rig" block.

        Map presets (sweep only) have no val capture, so they are measurements, not fitting input.
        """
        d = self.preset_dir(preset.name)
        p = self.data["presets"][preset.name]
        captures = {}
        for role in preset.roles:
            sig = self.signal(role, preset.signal)
            t = self.take(preset.name, role) or {}
            captures[role] = {
                "dry_signal": sig.manifest_path.stem,
                "dry_manifest": os.path.relpath(sig.manifest_path, d).replace("\\", "/"),
                "wet_wav": t.get("wet_wav"),
                "raw_wav": t.get("raw_wav"),
                "raw_channels": ["return (In 1)", "loopback (In 2)"],
                "sample_rate": sig.sample_rate,
                "n_samples": sig.total_samples,
                "duration_s": sig.duration_s,
                "lag_samples": t.get("lag_samples"),
                "captured": t.get("time_local"),
                "send_level_db": t.get("send_level_db", self.data["send_level_db"]),
                "auto_level": t.get("auto_level"),
                "calibration_file": t.get("calibration_file"),
            }
        doc = {
            "schema_version": "1.2",
            "date_created": _utc(),
            "saved_local": local_stamp(iso=True),
            "code_version": code_version(),
            "source": "simulated" if self.data["simulated"] else "hardware",
            "profile": self.profile_id,
            "pedal": self.data["pedal"],
            "preset_name": preset.name,
            "preset_kind": preset.kind,
            "preset_description": preset.description,
            "preset_notes": preset.notes,
            "sample_rate": self.sample_rate,
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
                "band_fit": {t["role"]: t.get("band_fit") for t in p["takes"] if t.get("band_fit")},
            },
        }
        path = d / "params.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
        return path
