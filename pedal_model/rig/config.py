"""Rig configuration: audio device, channel map, and the settings of every box in the chain."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RIG_PATH = PROJECT_ROOT / "configs" / "rig.json"
DATA_ROOT = PROJECT_ROOT / "recordings" / "data"
SIGNALS_DIR = DATA_ROOT / "signals"

SAMPLE_RATES = (48_000, 96_000)

# Settings shown (and editable) on each box of the Setup block diagram. Values are free text on
# purpose: most of them are knob positions or switch states that only a human can read off the
# hardware.
DEFAULT_BOXES: dict[str, dict[str, str]] = {
    "interface": {
        "main_knob": "mark it: the Main knob sets the Out 1/2 level (verify)",
        "in1_gain": "set during calibration (note the knob position)",
        "in1_48v": "OFF",
        "in2_gain": "minimum",
        "in2_48v": "OFF",
    },
    "prormp": {
        "level": "note the knob position",
        "ground_lift": "off",
    },
    "pedal": {
        "engaged": "on (LED lit)",
        "power": "9 V supply",
    },
    "prodi": {
        "pad_15db": "off (switch on the input side)",
        "ground_lift": "off (switch on the XLR output side)",
    },
}


@dataclass
class ChannelMap:
    """1-based interface channel numbers, as printed on the 1824c."""

    send_out: int = 1
    loop_out: int = 2
    return_in: int = 1
    loop_in: int = 2


@dataclass
class RigConfig:
    """Everything about the physical rig that a capture session needs to know and record.

    Attributes:
        asio_device: Name of the ASIO driver as PortAudio lists it.
        block_size: Audio callback size in frames; larger is safer against dropouts.
        channels: Which interface outputs/inputs carry send, loopback, and return.
        boxes: Free-text settings per box id ("interface", "prormp", "pedal", "prodi").
    """

    asio_device: str = "Studio USB ASIO Driver"
    block_size: int = 1024
    channels: ChannelMap = field(default_factory=ChannelMap)
    boxes: dict[str, dict[str, str]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_BOXES.items()}
    )

    def to_dict(self) -> dict:
        """Plain dict for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RigConfig":
        """Build from a dict, filling anything missing with defaults."""
        cfg = cls()
        cfg.asio_device = d.get("asio_device", cfg.asio_device)
        cfg.block_size = int(d.get("block_size", cfg.block_size))
        cfg.channels = ChannelMap(**{**asdict(cfg.channels), **d.get("channels", {})})
        for box, settings in d.get("boxes", {}).items():
            cfg.boxes.setdefault(box, {}).update(settings)
        return cfg

    @classmethod
    def load(cls, path: Path = DEFAULT_RIG_PATH) -> "RigConfig":
        """Load from JSON; returns defaults if the file does not exist yet."""
        if not path.is_file():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path = DEFAULT_RIG_PATH) -> None:
        """Write to JSON (pretty-printed so it diffs well in git)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
