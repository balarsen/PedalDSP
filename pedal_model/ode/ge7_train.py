"""Train the GE-7 block model on a capture session's dataset, as it grows (plan: rung A).

One fit = Levenberg–Marquardt on every "fit" take's engaged ÷ bypassed response (complex, the
dataset's 200-point grid), starting from and pulled towards the schematic's values
(``ge7_model.schematic_values``, prior widths ``PRIOR_SIGMA``). With few takes the prior keeps the
fit sensible; each new take narrows the parameters' posterior widths, which is how the model
"gets set". After each fit the validation takes are scored; the test takes are never touched
here (they are scored once, at the end of rung A).

Split (profile record_run.split): a take is "excluded" if its preset is in record_run.exclude
(e.g. all_up_pad15, a rig diagnostic with the ProDI pad on), "test" if its controls equal a test
preset's (so the repeats of smile are test too), "validation" if its preset name matches the
validation pattern, "fit" otherwise.
"""

from __future__ import annotations

import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pedal_model.ode import ge7_fitter as F
from pedal_model.ode import ge7_model as M
from pedal_model.rig.config import PROJECT_ROOT, code_version

COHERENCE_MIN = 0.9  # grid points below this coherence are left out of the fit
PRIOR_NOISE_NP = 0.01  # the scale s of the prior rows (any constant; the data weight is relative)
# How much one take is worth. The dominant error is how the sliders were set (flat re-set: up to
# 0.5 dB between sessions; repeats untouched: 0.03 dB), and it is smooth across frequency, so a
# take's 200 grid points are not 200 independent measurements. Counting each take as
# N_EFF_PER_TAKE independent numbers with DATA_NOISE_NP error keeps the schematic prior in play
# until enough settings arrive; counting all 200 points let six presets pull the 800 Hz band to
# 294 Hz and two Qs to 70-100 (2026-10-22).
DATA_NOISE_NP = 0.03  # ≈0.26 dB, the re-set floor
N_EFF_PER_TAKE = 20  # ~1/2-octave correlation over ~10 octaves
MAX_ITER = 60
RESULTS_DIR = PROJECT_ROOT / "results" / "ge7_fit"
MODELS_DIR = PROJECT_ROOT / "pedal_model" / "rig" / "models" / "ge7"


def take_group(take: dict, split: dict, test_controls: list[dict], exclude: list[str]) -> str:
    """ "fit", "validation", "test" or "excluded" for one dataset take (module docstring)."""
    if take["preset"] in exclude:
        return "excluded"
    controls = {k: float(v) for k, v in (take.get("controls") or {}).items()}
    if any(all(abs(controls.get(k, 0.0) - v) < 1e-6 for k, v in c.items()) for c in test_controls):
        return "test"
    if fnmatch.fnmatch(take["preset"], split.get("validation", "")):
        return "validation"
    return "fit"


def groups(dataset: dict, profile_raw: dict) -> dict[str, list[int]]:
    """Indices of the dataset's takes in each group."""
    split = profile_raw.get("record_run", {}).get("split", {})
    presets = {}
    for path in (PROJECT_ROOT / profile_raw["presets_dir"]).glob("preset_*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        presets[d["preset_name"]] = d["sliders"]
    test_controls = []
    for name in split.get("test", []):
        full = {k: 0.0 for k in M.SLIDER_KEYS}
        full.update({k: float(v) for k, v in presets.get(name, {}).items()})
        test_controls.append(full)
    exclude = profile_raw.get("record_run", {}).get("exclude", [])
    out: dict[str, list[int]] = {"fit": [], "validation": [], "test": [], "excluded": []}
    for i, t in enumerate(dataset["takes"]):
        out[take_group(t, split, test_controls, exclude)].append(i)
    return out


def arrays(dataset: dict, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fitter inputs for some takes: sliders (N, 8) panel dB, H (N, F) complex, weight (N, F).

    Points no sweep covered (None in the JSON) or with coherence below COHERENCE_MIN get weight
    0 and H = 1 (so the log stays finite).
    """
    takes = [dataset["takes"][i] for i in idx]
    sliders = M.sliders_tensor([t.get("controls") or {} for t in takes])
    re = np.array([[np.nan if v is None else v for v in t["h_re"]] for t in takes], dtype=float)
    im = np.array([[np.nan if v is None else v for v in t["h_im"]] for t in takes], dtype=float)
    coh = np.array([[0.0 if v is None else v for v in t["coherence"]] for t in takes], dtype=float)
    ok = np.isfinite(re) & np.isfinite(im) & (coh >= COHERENCE_MIN)
    h = np.where(ok, re + 1j * im, 1.0 + 0j)
    return (
        sliders,
        torch.as_tensor(h, dtype=torch.complex128),
        torch.as_tensor(ok.astype(float), dtype=torch.float64),
    )


def per_take_rms_db(
    params: dict, freqs: torch.Tensor, dataset: dict, idx: list[int]
) -> list[float]:
    """RMS of 20·log10|model ÷ measured| over each take's used points, dB."""
    if not idx:
        return []
    sliders, h, w = arrays(dataset, idx)
    err = torch.log(M.ratio_response(params, freqs, sliders) / h).real * F.NEPER_TO_DB
    return [
        float(torch.sqrt((e[m > 0] ** 2).mean())) if (m > 0).any() else float("nan")
        for e, m in zip(err, w)
    ]


def fit_dataset(
    dataset: dict, profile_raw: dict, start: dict | None = None, use_prior: bool = True
) -> dict:
    """Fit the block model to the dataset's "fit" takes; score the validation takes.

    Args:
        dataset: ``rig.dataset.session_dataset`` output.
        profile_raw: The pedal profile (its record_run.split).
        start: Starting parameters (the previous fit, for a warm start), or None: the schematic.
        use_prior: Pull towards the schematic (False: data only, for the physics score).

    Returns:
        dict: params, labels, theta, std (1σ per θ element: relative for log parameters),
        n_fit, n_validation, rms_fit_db, rms_validation_db, rms_per_take {name/role: dB},
        groups, converged, condition.
    """
    g = groups(dataset, profile_raw)
    freqs = torch.as_tensor(dataset["freqs_hz"], dtype=torch.float64)
    schematic = M.schematic_values()
    init = start if start is not None else schematic
    if not g["fit"]:
        raise ValueError("no takes to fit yet")
    sliders, h, w = arrays(dataset, g["fit"])
    # Data rows relative to the prior rows: (s / σ_data) · √(N_eff / points per take).
    w = w * (PRIOR_NOISE_NP / DATA_NOISE_NP) * float(np.sqrt(N_EFF_PER_TAKE / h.shape[1]))
    fit = F.fit_ratio(
        sliders,
        freqs,
        h,
        init,
        weight=w,
        max_iter=MAX_ITER,
        prior=schematic if use_prior else None,
        prior_noise_np=PRIOR_NOISE_NP,
    )
    rep = F.identifiability(fit, noise_std=PRIOR_NOISE_NP)
    names = {i: f"{t['preset']}/{t['role']}" for i, t in enumerate(dataset["takes"])}
    rms_fit = per_take_rms_db(fit["params"], freqs, dataset, g["fit"])
    rms_val = per_take_rms_db(fit["params"], freqs, dataset, g["validation"])
    return {
        "params": fit["params"],
        "labels": rep["labels"],
        "theta": fit["theta"].detach().numpy(),
        "std": rep["std"],
        "is_log": rep["is_log"],
        "n_fit": len(g["fit"]),
        "n_validation": len(g["validation"]),
        "rms_fit_db": float(np.sqrt(np.nanmean(np.square(rms_fit)))) if rms_fit else float("nan"),
        "rms_validation_db": float(np.sqrt(np.nanmean(np.square(rms_val))))
        if rms_val
        else float("nan"),
        "rms_per_take": {
            names[i]: v for i, v in zip(g["fit"] + g["validation"], rms_fit + rms_val)
        },
        "groups": {k: [names[i] for i in v] for k, v in g.items()},
        "converged": bool(fit["converged"]),
        "condition": float(rep["condition"]),
    }


def history_record(fit: dict) -> dict:
    """One line of the training history (JSON-ready)."""
    return {
        "t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_version": code_version(),
        "n_fit": fit["n_fit"],
        "n_validation": fit["n_validation"],
        "rms_fit_db": round(fit["rms_fit_db"], 4),
        "rms_validation_db": None
        if np.isnan(fit["rms_validation_db"])
        else round(fit["rms_validation_db"], 4),
        "converged": fit["converged"],
        "condition": fit["condition"],
        "params": M.params_to_json(fit["params"]),
        "std": {lab: round(float(s), 6) for lab, s in zip(fit["labels"], fit["std"])},
    }


def history_path(session: str) -> Path:
    return RESULTS_DIR / session / "history.json"


def append_history(session: str, fit: dict) -> list[dict]:
    """Add a fit to the session's training history file; returns the whole history."""
    path = history_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    hist.append(history_record(fit))
    path.write_text(json.dumps(hist, indent=1) + "\n", encoding="utf-8", newline="\n")
    return hist


def write_parameter_file(fit: dict, sessions: list[str], models_dir: Path = MODELS_DIR) -> Path:
    """Write the next ge7_pedalode_v<N>.json (never overwrites). QC's expected curve then uses it.

    Returns:
        The new file's path.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    versions = [int(p.stem.rsplit("_v", 1)[1]) for p in models_dir.glob("ge7_pedalode_v*.json")]
    n = max(versions, default=0) + 1
    path = models_dir / f"ge7_pedalode_v{n}.json"
    doc = {
        "model": "ge7_blocks",
        "version": n,
        "band7_hypothesis": "H7a (first-order RC branch, from the schematic)",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_version": code_version(),
        "sessions": sessions,
        "params": M.params_to_json(fit["params"]),
        "std": {lab: float(s) for lab, s in zip(fit["labels"], fit["std"])},
        "rms_fit_db": fit["rms_fit_db"],
        "rms_validation_db": fit["rms_validation_db"],
        "groups": fit["groups"],
        "note": "Fitted with the schematic prior; the test settings are not scored yet.",
    }
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path
