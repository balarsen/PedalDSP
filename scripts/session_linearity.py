"""Which presets of a capture session are linear (good data) and which clip (need a retake).

For every preset's train take: the response of the sweeps at -24, -12 and -3 dBFS input (they must
For every preset's train take: the response of the sweeps at -24, -12 and -3 dBFS input (they
session is changed.

    uv run python scripts/session_linearity.py recordings/data/ge7/sessions/<session>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from pedal_model.rig.analysis import frequency_response, log_grid, smooth_octave
from pedal_model.signals.manifest import Manifest

LINEAR_DB = 0.25  # sweeps at -24 and -3 dBFS input differ by less than this: linear
CLIP_DB = 1.0  # more than this: clipping
THD_OK_DB = -50.0


def sweep_db(dry: np.ndarray, wet: np.ndarray, sr: int, s) -> tuple[np.ndarray, np.ndarray]:
    """Gain (dB, 1/6-octave smoothed) of one sweep section, wet over dry."""
    f, h, _ = frequency_response(
        dry[s.start_sample : s.end_sample], wet[s.start_sample : s.end_sample], sr
    )
    k = f > 0
    grid = log_grid(20, 19000)
    return grid, smooth_octave(f[k], 20 * np.log10(np.abs(h[k]) + 1e-12), grid)


def worst_thd_db(wet: np.ndarray, sr: int, m: Manifest) -> float:
    """Highest THD (H2..H5 power over H1, dB) among the loudest stepped tones below 5 kHz."""
    worst = -200.0
    for s in m.sections:
        if s.type != "stepped_sine_tone" or not s.label.endswith("l-3"):
            continue
        f0 = float(s.label.split("_f")[1].split("_")[0])
        if f0 > 5000:
            continue
        a, b = s.start_sample + sr // 10, s.end_sample - sr // 10
        x = np.abs(np.fft.rfft(wet[a:b] * np.hanning(b - a))) ** 2
        fx = np.fft.rfftfreq(b - a, 1 / sr)

        def power(fr: float) -> float:
            i = int(np.argmin(np.abs(fx - fr)))
            return float(x[max(i - 3, 0) : i + 4].sum())

        h1 = power(f0)
        harmonics = sum(power(k * f0) for k in range(2, 6) if k * f0 < 0.95 * sr / 2)
        worst = max(worst, 10 * np.log10(harmonics / h1 + 1e-30))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path)
    root = ap.parse_args().session
    d = json.loads((root / "session.json").read_text(encoding="utf-8"))
    print(f"{'preset':<14}{'level':>6}{'lin dB':>8}{'THD dB':>8}{'peak':>7}  verdict")
    for name, p in d["presets"].items():
        take = next((t for t in p.get("takes", []) if t["role"] == "train"), None)
        if take is None:
            continue
        m = Manifest(root / d["signals"]["train"])
        dry = m.load_all().astype(np.float64)
        wet, sr = sf.read(str(root / name / take["wet_wav"]), dtype="float64")
        lo = next(s for s in m.sections if s.label == "sweep_low")
        hi = next(s for s in m.sections if s.label == "sweep_high")
        grid, g_lo = sweep_db(dry, wet, sr, lo)
        _, g_hi = sweep_db(dry, wet, sr, hi)
        band = (grid >= 100) & (grid <= 8000)
        lin = float(np.max(np.abs(g_hi - g_lo)[band]))
        thd = worst_thd_db(wet, sr, m)
        peak = next((i["value"] for i in take["qc"] if i["name"] == "Return peak"), "?")
        if lin < LINEAR_DB and thd < THD_OK_DB:
            verdict = "good (linear)"
        elif lin > CLIP_DB or thd > -30.0:
            verdict = "CLIPS: retake with auto-level on"
        else:
            verdict = "mild compression: check"
        lvl = p["controls"].get("level", 0.0)
        print(f"{name:<14}{lvl:>+6.0f}{lin:8.2f}{thd:8.1f}{peak:>12}  {verdict}")


if __name__ == "__main__":
    main()
