"""Pedal profiles: controls, preset source, send level and QC limits for each pedal the rig can
capture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pedal_model.rig.config import PROJECT_ROOT

PROFILES_DIR = Path(__file__).resolve().parent


@dataclass
class Preset:
    """One pedal setting to capture: a name, a description and the control values."""

    name: str
    description: str
    controls: dict[str, float]


@dataclass
class Profile:
    """A pedal profile loaded from ``profiles/<id>.json``."""

    raw: dict

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def pedal(self) -> str:
        return self.raw["pedal"]

    @property
    def display_name(self) -> str:
        return self.raw["display_name"]

    @property
    def controls(self) -> list[dict]:
        return self.raw["controls"]

    @property
    def send_level_db(self) -> float:
        return float(self.raw["send_level_db"])

    @property
    def qc(self) -> dict:
        return self.raw.get("qc", {})

    def presets(self) -> list[Preset]:
        """All presets from the profile's preset folder, with the baseline preset first."""
        folder = PROJECT_ROOT / self.raw["presets_dir"]
        out: list[Preset] = []
        for f in sorted(folder.glob("preset_*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append(
                Preset(
                    name=d["preset_name"],
                    description=d.get("description", ""),
                    controls={k: float(v) for k, v in d["sliders"].items()},
                )
            )
        first = self.raw.get("first_preset")
        out.sort(key=lambda p: (p.name != first, p.name))
        return out

    def preset(self, name: str) -> Preset:
        """Look up one preset by name."""
        for p in self.presets():
            if p.name == name:
                return p
        raise KeyError(f"preset {name!r} not in profile {self.id!r}")


def available_profiles() -> list[str]:
    """Profile ids found in this folder."""
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def load_profile(profile_id: str) -> Profile:
    """Load ``profiles/<profile_id>.json``."""
    return Profile(json.loads((PROFILES_DIR / f"{profile_id}.json").read_text(encoding="utf-8")))
