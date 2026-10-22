"""Build the GE-7 fitting dataset from a capture session (engaged ÷ bypassed per take).

Recomputes every response from the session's raw FLACs (pedal_model/rig/dataset.py) and writes
results/ge7_dataset/<session>.json, then prints a summary per take.

    uv run python scripts/ge7_dataset.py recordings/data/ge7/sessions/<session>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pedal_model.rig.dataset import session_dataset

ROOT = Path(__file__).resolve().parent.parent
REPORT_HZ = (50, 100, 400, 1000, 3200, 6400, 10_000)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console is cp1252
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ge7_dataset")
    ap.add_argument(
        "--roles", default="train,sweep", help="comma-separated take roles (default train,sweep)"
    )
    args = ap.parse_args()
    ds = session_dataset(args.session, tuple(args.roles.split(",")))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{ds['session']}.json"
    path.write_text(json.dumps(ds, indent=1) + "\n", encoding="utf-8", newline="\n")

    f = np.asarray(ds["freqs_hz"])
    cols = "".join(f"{hz:>8}" for hz in REPORT_HZ)
    print(f"{ds['session']}: {len(ds['takes'])} takes -> {path}")
    print(
        f"engaged / bypassed, dB at Hz:  {'preset':<14}{'role':<6}{cols}  delay  min coh  dropped"
    )
    for t in ds["takes"]:
        h = np.asarray(t["h_re"], dtype=float) + 1j * np.asarray(t["h_im"], dtype=float)
        db = np.interp(REPORT_HZ, f, 20 * np.log10(np.abs(h)))
        coh = np.asarray(t["coherence"], dtype=float)
        band = (f >= 50) & (f <= 10_000)
        dropped = ", ".join(d["label"] for d in t["sections_dropped"]) or "-"
        row = "".join(f"{v:8.2f}" for v in db)
        print(
            f"{'':31}{t['preset']:<14}{t['role']:<6}{row}  {t['delay_samples']:+5d}  "
            f"{np.nanmin(coh[band]):7.4f}  {dropped}"
        )


if __name__ == "__main__":
    main()
