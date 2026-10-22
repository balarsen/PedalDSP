"""Re-run calibration analysis and QC on a capture session's raw FLACs with the current code.

For debugging: nothing in the session is changed. The raw captures hold both inputs (return and
loopback), and the played signals are reproducible, so every alignment and response can be
recomputed after a fix.

    uv run python scripts/reanalyse_session.py recordings/data/ge7/sessions/<session>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from pedal_model.rig.calibrate import (
    check_bypass,
    check_engaged,
    check_reference,
    ref_db_at,
    reference_from_take,
    reference_signal,
)
from pedal_model.rig.engine import PRE_ROLL_S, Take
from pedal_model.rig.profiles import load_profile
from pedal_model.rig.qc import check_take
from pedal_model.signals.manifest import Manifest

REPORT_HZ = (30, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 15_000)


def rebuild_take(raw_path: Path, signal: np.ndarray, send_level_db: float) -> Take:
    """A Take from a raw capture: the played signal is rebuilt (both outputs, as route "both")."""
    rec, sr = sf.read(str(raw_path), dtype="float32")
    pre = int(round(PRE_ROLL_S * sr))
    play = np.zeros_like(rec)
    gain = np.float32(10 ** (send_level_db / 20))
    play[pre : pre + len(signal), 0] = signal * gain
    play[pre : pre + len(signal), 1] = signal * gain
    return Take(sr, play, rec, pre, len(signal), send_level_db, 0, False, 0.0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console is cp1252: "≥" fails
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path)
    args = ap.parse_args()
    root: Path = args.session
    data = json.loads((root / "session.json").read_text(encoding="utf-8"))
    sr = int(data["sample_rate"])
    profile = load_profile(data["profile"])
    cal_old = data.get("calibration") or {}

    print(f"Session {root.name}: {sr} Hz, send level {data['send_level_db']} dB\n")
    print("References (recomputed from calibration/*.flac):")
    refs: dict[str, dict] = {}
    sweep = np.concatenate([reference_signal(sr), np.zeros(int(0.2 * sr), np.float32)])
    for name, old in (cal_old.get("references") or {}).items():
        take = rebuild_take(root / old["file"], sweep, 0.0)
        ref = reference_from_take(take, name)
        refs[name] = ref
        coh = check_reference(ref)[0]["text"]
        print(
            f"  {name:<13} loop lag {ref['lag_samples']} (stored {old['lag_samples']}), "
            f"In 1 lag {ref['return_lag_samples']}; 1 kHz {ref['gain_1k_db']:+.2f} dB; {coh}"
        )
    if "chain" in refs and "bypass" in refs:
        print("  C3:", check_bypass(refs["bypass"], refs["chain"])[0]["text"])
    if "bypass" in refs and "engaged_flat" in refs:
        c4 = check_engaged(refs["engaged_flat"], refs["bypass"], refs.get("chain"))
        print("  C4:", c4[0]["text"])
    if "chain" in refs:
        r = refs["chain"]
        row = "  ".join(
            f"{f}:{ref_db_at(r, np.array([f]))[0] - r['gain_1k_db']:+.1f}" for f in REPORT_HZ
        )
        print(f"  chain response re 1 kHz (dB): {row}")

    # The stored calibration with its curves recomputed; the rest (C1, C2's return ceiling) is what
    # the session was checked against.
    cal = {**cal_old, "references": refs}

    for name, p in data["presets"].items():
        for t in p.get("takes", []):
            sig_rel = data["signals"][t["role"]]
            m = Manifest(root / sig_rel)
            dry = m.load_all()
            send = t.get("send_level_db", data["send_level_db"])  # per take since auto-level
            take = rebuild_take(root / name / t["raw_wav"], dry, send)
            rep, _ = check_take(
                take,
                dry,
                m,
                pedal=profile.pedal,
                controls=p.get("controls"),
                calibration=cal,
                qc_limits=profile.qc,
            )
            print(
                f"\n{name} / {t['role']}: verdict {rep['verdict']} (stored {t['verdict']}); lag "
                f"{rep['lag_samples']} (stored {t['lag_samples']}), "
                f"loop {rep['loop_lag_samples']}, xcorr {rep['xcorr_lag_samples']}"
            )
            shown = ("Alignment", "Pedal delay", "Response vs model")
            for i in rep["items"]:
                if i["status"] != "ok" or i["name"] in shown:
                    print(f"   [{i['status']}] {i['name']}: {i['value']}  {i['detail']}")
            r = rep["response"]
            if r is not None:
                vals = np.interp(REPORT_HZ, r["freqs_hz"], r["measured_db"])
                row = "  ".join(f"{f}:{v:+.2f}" for f, v in zip(REPORT_HZ, vals))
                print(f"   measured (dB): {row}")


if __name__ == "__main__":
    main()
