"""Pedal profiles: controls, presets, send level and QC limits for each pedal the rig captures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pedal_model.rig.config import PROJECT_ROOT

PROFILES_DIR = Path(__file__).resolve().parent

# Preset kinds. "model": train + val signals, for model fitting. "map": the short sweep-only
# signal, a measurement (for example one GE-7 slider moved, to map its centre frequency and
# bandwidth).
KIND_MODEL = "model"
KIND_MAP = "map"
ROLES = {KIND_MODEL: ("train", "val"), KIND_MAP: ("sweep",)}

NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")  # preset names become folder and file names


def _tag(v: float) -> str:
    """A control value for a preset name: +15 -> p15, -7.5 -> m7p5."""
    return f"{'p' if v > 0 else 'm'}{abs(v):g}".replace(".", "p")


@dataclass
class Preset:
    """One pedal setting to capture.

    Attributes:
        name: Short id, also the preset's folder name (lowercase letters, digits, underscores).
        description: One line shown under the name.
        controls: Control key -> value (GE-7: slider dB).
        kind: KIND_MODEL (train + val, for fitting) or KIND_MAP (sweep-only measurement).
        notes: Longer explanation of what the preset does and why it is captured.
        signal: Train signal for this preset ("full", "quick", "sweeps"), or None for the
            session's choice.
    """

    name: str
    description: str
    controls: dict[str, float]
    kind: str = KIND_MODEL
    notes: str = ""
    signal: str | None = None

    @property
    def roles(self) -> tuple[str, ...]:
        return ROLES[self.kind]

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "controls": self.controls,
            "kind": self.kind,
            "notes": self.notes,
            "signal": self.signal,
        }

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Preset":
        return cls(
            name,
            d.get("description", ""),
            {k: float(v) for k, v in d["controls"].items()},
            d.get("kind", KIND_MODEL),
            d.get("notes", ""),
            d.get("signal"),
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
                    d.get("signal"),
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
                tag = _tag(v)
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

    def interaction_presets(self) -> list[Preset]:
        """Band pairs moved together (plan P0.1), sweep only: every adjacent pair and the profile's
        distant pairs, at each of the profile's settings (record_run.interaction).
        """
        cfg = self.raw.get("record_run", {}).get("interaction", {})
        labels = {c["key"]: c["label"] for c in self.controls}
        bands = [c["key"] for c in self.controls if c.get("band_hz")]
        pairs = list(zip(bands, bands[1:])) + [tuple(p) for p in cfg.get("distant", [])]
        out = []
        for a, b in pairs:
            for va, vb in cfg.get("settings", []):
                controls = {c["key"]: 0.0 for c in self.controls}
                controls[a], controls[b] = float(va), float(vb)
                out.append(
                    Preset(
                        f"pair_{a}_{_tag(va)}_{b}_{_tag(vb)}",
                        f"Interaction: {labels[a]} at {va:+g}, {labels[b]} at {vb:+g}, others 0",
                        controls,
                        KIND_MAP,
                        "Two bands moved together. The block model says they do not simply add "
                        "(they share the summing stage), adjacent bands least of all: this step "
                        "measures by how much. Sweep-only signal.",
                    )
                )
        return out

    def random_presets(self, group: str) -> list[Preset]:
        """Seeded random settings, sweep only (record_run.random): every control uniform over its
        range, rounded to ``step_db`` so it can be set repeatably by eye.

        Args:
            group: "fit" (used for fitting) or "val" (validation: model choices only).
        """
        cfg = self.raw.get("record_run", {}).get("random")
        if not cfg:
            return []
        rng = np.random.default_rng(cfg["seeds"][group])
        step = cfg["step_db"]
        use = (
            "The fit uses these."
            if group == "fit"
            else "Validation: only to choose between model variants, never fitted."
        )
        out = []
        for i in range(cfg["n"]):
            controls = {
                c["key"]: float(step * round(rng.uniform(c["min"], c["max"]) / step))
                for c in self.controls
            }
            out.append(
                Preset(
                    f"rand_{group}_{i + 1:02d}",
                    f"Random setting {i + 1} ({group} group, seed {cfg['seeds'][group]})",
                    controls,
                    KIND_MAP,
                    f"Every slider at a random position ({step:g} dB steps). {use} "
                    "Sweep-only signal.",
                )
            )
        return out

    def record_run(self) -> list[Preset]:
        """The record session (plan R2), in capture order.

        Repeats of the repeatability presets (the first take is the preset itself, then
        sweep-only repeats, then one each after re-plugging), the slider map, the interaction
        pairs, the presets (minus ``exclude``), the random settings (fit, then val), and copies
        of a few presets with the full train signal (for time-domain and neural models).
        """
        cfg = self.raw.get("record_run")
        if not cfg:
            return []
        base = {p.name: p for p in self.presets()}
        rep = cfg["repeats"]
        out: list[Preset] = []
        for name in rep["presets"]:
            out.append(base[name])
            for k in range(2, rep["times"] + 1):
                out.append(
                    Preset(
                        f"{name}_r{k}",
                        f"{name}, repeat {k} (no change: press Ready again)",
                        base[name].controls,
                        KIND_MAP,
                        "The same setting again, untouched: the spread between repeats is the "
                        "measurement's own floor. Sweep-only signal.",
                    )
                )
        for name in rep["presets"]:
            out.append(
                Preset(
                    f"{name}_replug",
                    f"{name} after re-plugging cables A and B",
                    base[name].controls,
                    KIND_MAP,
                    "Before this take, unplug and re-plug cable A (pedal IN) and cable B (pedal "
                    "OUT), and set the sliders again. The difference from the repeats shows "
                    "what re-plugging and re-setting cost. Sweep-only signal.",
                )
            )
        out += self.map_presets() + self.interaction_presets()
        skip = set(rep["presets"]) | set(cfg.get("exclude", []))
        out += [p for name, p in base.items() if name not in skip]
        out += self.random_presets("fit") + self.random_presets("val")
        for name in cfg.get("full_train", []):
            p = base[name]
            out.append(
                Preset(
                    f"{name}_full",
                    f"{name} with the full train signal (~12 min)",
                    p.controls,
                    KIND_MODEL,
                    "The full 642 s train signal (noise, plucks, multitones, ramps) for "
                    "time-domain checks and a neural baseline, should the block model need help.",
                    "full",
                )
            )
        return out

    def preset(self, name: str) -> Preset:
        """Look up one preset by name (folder, map and record-run presets, and ones added now)."""
        for p in self.presets() + self.map_presets() + self.record_run() + self._extra:
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
            "notes": preset.notes or preset.description,  # never empty: Plan shows the notes
        }
        path.write_text(json.dumps(doc, indent=4) + "\n", encoding="utf-8", newline="\n")
        return path


def available_profiles() -> list[str]:
    """Profile ids found in this folder."""
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def load_profile(profile_id: str) -> Profile:
    """Load ``profiles/<profile_id>.json``."""
    return Profile(json.loads((PROFILES_DIR / f"{profile_id}.json").read_text(encoding="utf-8")))
