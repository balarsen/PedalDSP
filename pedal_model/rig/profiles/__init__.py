"""Pedal profiles: controls, presets, send level and QC limits for each pedal the rig captures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pedal_model.rig.config import PROJECT_ROOT

PROFILES_DIR = Path(__file__).resolve().parent

# Preset kinds. "model": train + val signals, for model fitting. "map": the short sweep-only
# signal, a measurement (for example one GE-7 slider moved, to map its centre frequency and
# bandwidth).
KIND_MODEL = "model"
KIND_MAP = "map"
ROLES = {KIND_MODEL: ("train", "val"), KIND_MAP: ("sweep",)}

NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")  # preset names become folder and file names


@dataclass
class Preset:
    """One pedal setting to capture.

    Attributes:
        name: Short id, also the preset's folder name (lowercase letters, digits, underscores).
        description: One line shown under the name.
        controls: Control key -> value (GE-7: slider dB).
        kind: KIND_MODEL (train + val, for fitting) or KIND_MAP (sweep-only measurement).
        notes: Longer explanation of what the preset does and why it is captured.
    """

    name: str
    description: str
    controls: dict[str, float]
    kind: str = KIND_MODEL
    notes: str = ""

    @property
    def roles(self) -> tuple[str, ...]:
        return ROLES[self.kind]

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "controls": self.controls,
            "kind": self.kind,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Preset":
        return cls(
            name,
            d.get("description", ""),
            {k: float(v) for k, v in d["controls"].items()},
            d.get("kind", KIND_MODEL),
            d.get("notes", ""),
        )


@dataclass
class Profile:
    """A pedal profile loaded from ``profiles/<id>.json``."""

    raw: dict
    _extra: list[Preset] = field(default_factory=list)

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

    @property
    def presets_dir(self) -> Path:
        return PROJECT_ROOT / self.raw["presets_dir"]

    def presets(self) -> list[Preset]:
        """All presets from the profile's preset folder, with the baseline preset first."""
        notes = self.raw.get("preset_notes", {})
        out: list[Preset] = []
        for f in sorted(self.presets_dir.glob("preset_*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            name = d["preset_name"]
            out.append(
                Preset(
                    name,
                    d.get("description", ""),
                    {k: float(v) for k, v in d["sliders"].items()},
                    KIND_MODEL,
                    d.get("notes") or notes.get(name, ""),
                )
            )
        first = self.raw.get("first_preset")
        out.sort(key=lambda p: (p.name != first, p.name))
        return out

    def map_presets(self) -> list[Preset]:
        """The slider-map measurement: each control alone at each map position, the rest at 0."""
        cfg = self.raw.get("map", {})
        positions = cfg.get("positions", [])
        out = []
        for c in self.controls:
            for v in positions:
                controls = {k["key"]: 0.0 for k in self.controls}
                controls[c["key"]] = float(v)
                tag = f"{'p' if v > 0 else 'm'}{abs(v):g}".replace(".", "p")
                band = f"{c['band_hz']:g} Hz band" if c.get("band_hz") else f"{c['label']} control"
                out.append(
                    Preset(
                        f"map_{c['key']}_{tag}",
                        f"Slider map: {c['label']} at {v:+g} {c['unit']}, all others at 0",
                        controls,
                        KIND_MAP,
                        f"Measures the {band} on its own at {v:+g} {c['unit']}: the fit gives "
                        "its real centre frequency, bandwidth (Q) and gain at this position. "
                        "Comparing positions shows whether the bandwidth changes between boost "
                        "and cut. Uses the ~1 min sweep-only signal.",
                    )
                )
        return out

    def preset(self, name: str) -> Preset:
        """Look up one preset by name (folder presets, map presets and presets added this run)."""
        for p in self.presets() + self.map_presets() + self._extra:
            if p.name == name:
                return p
        raise KeyError(f"preset {name!r} not in profile {self.id!r}")

    def save_custom_preset(self, preset: Preset) -> Path:
        """Write a new preset into the profile's preset folder (same format as the others)."""
        if not NAME_PATTERN.match(preset.name):
            raise ValueError("Preset names use lowercase letters, digits and underscores only.")
        path = self.presets_dir / f"preset_{preset.name}.json"
        if path.exists():
            raise FileExistsError(f"{path.name} already exists.")
        doc = {
            "schema_version": "1.0",
            "pedal": self.pedal,
            "preset_name": preset.name,
            "description": preset.description,
            "sliders": preset.controls,
            "notes": preset.notes,
        }
        path.write_text(json.dumps(doc, indent=4) + "\n", encoding="utf-8", newline="\n")
        return path


def available_profiles() -> list[str]:
    """Profile ids found in this folder."""
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def load_profile(profile_id: str) -> Profile:
    """Load ``profiles/<profile_id>.json``."""
    return Profile(json.loads((PROFILES_DIR / f"{profile_id}.json").read_text(encoding="utf-8")))
