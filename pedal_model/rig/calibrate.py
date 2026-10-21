"""Session calibration: the guided steps C1–C4, each a short test signal and a few checks.

C1 loopback (Out 2 → In 2: the interface alone), C2 the chain alone (pedal out of the chain),
C3 the pedal bypassed, C4 the pedal engaged at its calibration preset. C2–C4 each store a
reference response curve, and QC divides every take by the profile's chosen one.

Every result is a plain dict, the same as it is stored in the session's JSON. Advice is a list of
``{"status": "ok" | "warn" | "fail", "text": ...}`` lines.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import chirp

from pedal_model.rig.analysis import (
    MAX_LAG_S,
    find_lag,
    frequency_response,
    log_grid,
    peak_dbfs,
    rms_dbfs,
    smooth_octave,
)
from pedal_model.rig.engine import CaptureEngine, Take

SWEEP_S = 1.0  # log sweep used for the lag measurement
SWEEP_F = (50.0, 15_000.0)
SWEEP_DBFS = -24.0  # quiet: keeps a nonlinear pedal near-linear so the correlation peak is sharp
TONE_HZ = 1000.0
TONE_S = 0.4
TONE_MEASURE_S = 0.2  # middle part of each tone, away from the on/off edges
GAP_S = 0.2
CHAIN_LEVELS_DBFS = (-36.0, -30.0, -24.0, -18.0, -12.0, -9.0)  # tone peaks at the send output
LOOP_TONE_DBFS = -20.0
LOOP_LINEARITY_STEPS_DB = (-20.0, -10.0, 0.0)  # tones relative to the hottest send of a take
LOOP_LINEARITY_LIMIT_DB = 0.5

LINEARITY_LIMIT_DB = 0.25  # docs/ge7_plan.md: flat2 showed <= 0.25 dB gain variation across levels
RETURN_PEAK_TARGET_DBFS = (-18.0, -6.0)  # where the loudest training section should land on In 1
NO_SIGNAL_DBFS = -70.0
CROSSTALK_LIMIT_DBFS = -50.0
LAG_SEARCH_S = MAX_LAG_S
# A delay longer than the search still produces a partial-overlap peak just inside the edge, so
# only lags well inside the search are trusted.
LAG_TRUST_FRACTION = 0.8

REFERENCE_SWEEP_S = 3.0
REFERENCE_DBFS = -20.0
REFERENCE_F_LO = 20.0
REFERENCE_F_HI = 19_000.0  # the sweep ends at 20 kHz, so its last bins are unreliable
REFERENCE_NPERSEG = 16384  # 2.9 Hz bins at 48 kHz
COMPARE_BAND_HZ = (50.0, 10_000.0)  # where references are compared with each other
# Two references closer than this everywhere (50 Hz-10 kHz) are the same path. Real differences seen
# so far: bypassed vs cable 0.34 dB, engaged vs bypassed 1.5 dB (2026-10-18).
SAME_PATH_DB = 0.1
BYPASS_LEVEL_FAIL_DB = 6.0  # bypassed pedal vs cable: a bigger level change means a wiring fault
FLAT_LIMIT_DB = 3.0  # engaged ÷ bypassed at the calibration preset

# The guided steps. Two guitar cables around the pedal get names, printed in the GUI: cable A runs
# ProRMP OUTPUT → pedal IN, cable B runs pedal OUT → ProDI INPUT. "moves" are the exact actions
# ({pedal} is replaced by the pedal's name); "pedal_state" is what the simulated rig does for the
# step (the demo's stand-in for the moves); "cables" are highlighted on the diagram; "reference" is
# the response curve the step stores.
CAL_STEPS: list[dict] = [
    {
        "id": "C1",
        "title": "Loopback",
        "moves": ["No cable moves. Check the loopback cable: Out 2 (rear) → In 2 (front)."],
        "checks": "The interface's own latency, gain and linearity; In 1 must stay silent.",
        "pedal_state": None,
        "cables": ["loop"],
        "reference": None,
    },
    {
        "id": "C2",
        "title": "Chain reference (pedal out of the chain)",
        "moves": [
            "Pull cable A out of {pedal} IN (its other end stays in ProRMP OUTPUT).",
            "Pull cable B out of ProDI INPUT (leave it hanging from {pedal} OUT).",
            "Plug cable A's free end into ProDI INPUT. Now ProRMP OUTPUT → cable A → ProDI INPUT.",
        ],
        "checks": "Chain gain, latency, linearity and return level, then the chain's own frequency "
        "response H_chain(f), which every take is corrected by.",
        "pedal_state": "cable",
        "cables": ["inst_in", "inst_out"],
        "reference": "chain",
    },
    {
        "id": "C3",
        "title": "{pedal} bypassed",
        "moves": [
            "Pull cable A out of ProDI INPUT and plug it back into {pedal} IN.",
            "Plug cable B's free end back into ProDI INPUT.",
            "{pedal} powered, footswitch LED OFF (bypassed).",
        ],
        "checks": "The bypassed pedal ÷ the chain: its buffers, H_bypass(f). It must differ from "
        "the chain reference, or the pedal is not in the chain.",
        "pedal_state": "bypass",
        "cables": ["inst_in", "inst_out"],
        "reference": "bypass",
    },
    {
        "id": "C4",
        "title": "{pedal} engaged, calibration preset",
        "moves": ["Press the footswitch: LED ON.", "Set every control to the calibration preset."],
        "checks": "Engaged ÷ bypassed should be near 0 dB everywhere, and must differ from "
        "bypassed (or the footswitch is still off).",
        "pedal_state": "engaged",
        "cables": ["inst_in", "inst_out"],
        "reference": "engaged_flat",
    },
]


def advice(status: str, text: str) -> dict:
    return {"status": status, "text": text}


def worst_status(lines: list[dict]) -> str:
    """ "fail" if any line fails, else "warn" if any warns, else "ok"."""
    statuses = {a["status"] for a in lines}
    return "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "ok")


def _sweep(sr: int, dbfs: float) -> np.ndarray:
    t = np.arange(int(SWEEP_S * sr)) / sr
    x = chirp(t, SWEEP_F[0], SWEEP_S, SWEEP_F[1], method="logarithmic")
    fade = np.minimum(1.0, np.minimum(t, SWEEP_S - t) / 0.01)  # 10 ms fades: no clicks
    return (10 ** (dbfs / 20) * x * fade).astype(np.float32)


def _tone(sr: int, dbfs: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)
    return (10 ** (dbfs / 20) * np.sin(2 * np.pi * TONE_HZ * t) * fade).astype(np.float32)


def _gap(sr: int, seconds: float = GAP_S) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


DROPOUT_TEXT = "{} dropouts. Raise the block size in rig.json, close other audio apps."
LAG_TOO_LONG_TEXT = (
    f"The delay is longer than the {LAG_SEARCH_S * 1e3:.0f} ms search: is the ASIO buffer in "
    "Universal Control set very large? Lower it and run again."
)


def run_loopback_test(engine: CaptureEngine, sr: int, hottest_dbfs: float = -10.0) -> dict:
    """C1, Out 2 → In 2 only: interface latency, gain and linearity; In 1 must stay quiet.

    Args:
        hottest_dbfs: The loudest peak a take will send (training-signal peak + send level),
            before the engine's loopback trim. The loop is tested at that level and 10 and 20 dB
            below it: a clipping loopback makes alignment and the integrity check unreliable.

    Returns:
        dict: lag_samples, gain_db, loop_noise_dbfs, return_dbfs_while_loop_only,
        linearity_spread_db, level_gains_db, advice.
    """
    sweep = _sweep(sr, LOOP_TONE_DBFS)
    parts, starts, cursor = [sweep, _gap(sr)], [], len(sweep) + len(_gap(sr))
    for step in LOOP_LINEARITY_STEPS_DB:
        tone = _tone(sr, hottest_dbfs + step, TONE_S)
        starts.append(cursor)
        parts += [tone, _gap(sr)]
        cursor += len(tone) + len(_gap(sr))
    take = engine.run(np.concatenate(parts), sr, send_level_db=0.0, route="loop")
    pre = take.pre_roll
    lag = find_lag(take.play[:, 1], take.rec[:, 1], sr, pre, pre + len(sweep), LAG_SEARCH_S)

    a = int((TONE_S - TONE_MEASURE_S) / 2 * sr)
    b = int((TONE_S + TONE_MEASURE_S) / 2 * sr)
    gains = []
    for s0 in starts:
        got = take.rec[pre + s0 + a + lag : pre + s0 + b + lag, 1]
        gains.append(rms_dbfs(got) - rms_dbfs(take.play[pre + s0 + a : pre + s0 + b, 1]))
    gain = gains[0]  # the quietest tone: the least likely to be compressed
    spread = float(max(gains) - min(gains))
    loudest = take.rec[pre + starts[-1] + a + lag : pre + starts[-1] + b + lag, 1]
    noise = rms_dbfs(take.rec[int(0.1 * sr) : pre, 1])
    crosstalk = rms_dbfs(take.rec[:, 0])

    lines = []
    if take.xruns:
        lines.append(advice("fail", DROPOUT_TEXT.format(take.xruns)))
    if lag > LAG_TRUST_FRACTION * LAG_SEARCH_S * sr:
        lines.append(advice("fail", LAG_TOO_LONG_TEXT))
    if rms_dbfs(loudest) < NO_SIGNAL_DBFS:
        lines.append(
            advice(
                "fail",
                "No loopback signal on In 2. Check the Out 2 → In 2 cable, and that In 2 gain is "
                "not fully off.",
            )
        )
    else:
        lines.append(
            advice(
                "ok",
                f"Loopback found: {lag} samples ({lag / sr * 1e3:.2f} ms), gain {gain:+.1f} dB.",
            )
        )
    if crosstalk > CROSSTALK_LIMIT_DBFS:
        lines.append(
            advice(
                "fail",
                f"In 1 hears the loopback ({crosstalk:.0f} dBFS). Cables swapped? Out 1 and Out 2 "
                "must go to different places.",
            )
        )
    if spread > LOOP_LINEARITY_LIMIT_DB:
        lines.append(
            advice(
                "fail",
                f"The loopback is not linear: its gain changes by {spread:.1f} dB over 20 dB of "
                "level, so In 2 is clipping. Turn In 2 gain fully down and use a TRS line "
                "cable; if it persists, lower loop_send_db in configs/rig.json.",
            )
        )
    else:
        lines.append(advice("ok", f"Loopback linear: gain spread {spread:.2f} dB over 20 dB."))
    if noise > -80:
        lines.append(
            advice(
                "warn", f"In 2 noise is high ({noise:.0f} dBFS RMS). Check the cable and ground."
            )
        )
    return {
        "lag_samples": int(lag),
        "gain_db": float(gain),
        "loop_noise_dbfs": noise,
        "return_dbfs_while_loop_only": crosstalk,
        "linearity_spread_db": spread,
        "level_gains_db": [float(g) for g in gains],
        "advice": lines,
    }


def run_chain_test(
    engine: CaptureEngine, sr: int, send_level_db: float, train_peak_dbfs: float
) -> dict:
    """C2, through the chain: gain, latency and linearity, plus advice on the hardware knobs.

    Args:
        engine: Capture engine on the session's backend.
        sr: Session sample rate in Hz.
        send_level_db: The profile's digital send level (applied to every capture signal).
        train_peak_dbfs: Peak of the session's training signal before the send level.

    Returns:
        dict: lag_samples, gain_db (median at 1 kHz), linearity_spread_db, return_noise_dbfs,
        predicted_train_peak_dbfs, steps (send_dbfs, return_dbfs, gain_db per tone), advice.
    """
    sweep = _sweep(sr, SWEEP_DBFS)
    parts, starts, cursor = [sweep, _gap(sr)], [], len(sweep) + len(_gap(sr))
    for lvl in CHAIN_LEVELS_DBFS:
        tone = _tone(sr, lvl, TONE_S)
        starts.append(cursor)
        parts += [tone, _gap(sr)]
        cursor += len(tone) + len(_gap(sr))
    take = engine.run(np.concatenate(parts), sr, send_level_db=0.0, route="both")
    pre = take.pre_roll
    lag = find_lag(take.play[:, 0], take.rec[:, 0], sr, pre, pre + len(sweep), LAG_SEARCH_S)

    steps = []
    a = int((TONE_S - TONE_MEASURE_S) / 2 * sr)
    b = int((TONE_S + TONE_MEASURE_S) / 2 * sr)
    for lvl, s in zip(CHAIN_LEVELS_DBFS, starts):
        send = take.play[pre + s + a : pre + s + b, 0]
        ret = take.rec[pre + s + a + lag : pre + s + b + lag, 0]
        steps.append(
            {
                "send_dbfs": lvl,
                "return_dbfs": rms_dbfs(ret) + 3.01,  # sine peak = RMS + 3.01 dB
                "gain_db": rms_dbfs(ret) - rms_dbfs(send),
            }
        )
    gains = [st["gain_db"] for st in steps if st["return_dbfs"] > NO_SIGNAL_DBFS + 20]
    gain = float(np.median(gains)) if gains else -200.0
    spread = float(max(gains) - min(gains)) if len(gains) > 1 else 0.0
    noise = rms_dbfs(take.rec[int(0.1 * sr) : pre, 0])
    predicted = train_peak_dbfs + send_level_db + gain
    peak_seen = peak_dbfs(take.rec[:, 0])
    result = {
        "lag_samples": int(lag),
        "gain_db": gain,
        "linearity_spread_db": spread,
        "return_noise_dbfs": noise,
        "predicted_train_peak_dbfs": predicted,
        "steps": steps,
        "advice": [],
    }
    lines = result["advice"]

    lo, hi = RETURN_PEAK_TARGET_DBFS
    if take.xruns:
        lines.append(advice("fail", DROPOUT_TEXT.format(take.xruns)))
    if lag > LAG_TRUST_FRACTION * LAG_SEARCH_S * sr:
        lines.append(advice("fail", LAG_TOO_LONG_TEXT))
    if not gains:
        lines.append(
            advice(
                "fail",
                "No return on In 1. Check: pedal powered and engaged, cables ProRMP → pedal → "
                "ProDI → In 1, ProRMP level not at zero, In 1 gain up.",
            )
        )
        return result
    lines.append(
        advice(
            "ok",
            f"Chain found: {lag} samples ({lag / sr * 1e3:.2f} ms), gain {gain:+.1f} dB at 1 kHz.",
        )
    )
    if peak_seen > -1.0:
        lines.append(
            advice(
                "fail",
                f"The return clipped ({peak_seen:+.1f} dBFS). Lower the In 1 gain, or switch "
                "on the ProDI −15 dB pad.",
            )
        )
    if spread > LINEARITY_LIMIT_DB:
        lines.append(
            advice(
                "warn",
                f"Gain changes by {spread:.2f} dB across levels (limit {LINEARITY_LIMIT_DB} dB): "
                "something is saturating. Lower the 1824c Main knob or the ProRMP level, then "
                "raise In 1 gain to compensate.",
            )
        )
    else:
        lines.append(
            advice("ok", f"Linear: gain spread {spread:.2f} dB across {len(gains)} levels.")
        )
    if predicted > hi:
        lines.append(
            advice(
                "warn",
                f"The loudest training section would reach {predicted:+.1f} dBFS on In 1 (target "
                f"{lo:.0f} to {hi:.0f}). Lower the In 1 gain by about "
                f"{predicted - (lo + hi) / 2:.0f} dB.",
            )
        )
    elif predicted < lo:
        lines.append(
            advice(
                "warn",
                f"The loudest training section would only reach {predicted:+.1f} dBFS on In 1 "
                f"(target {lo:.0f} to {hi:.0f}). Raise the In 1 gain by about "
                f"{(lo + hi) / 2 - predicted:.0f} dB.",
            )
        )
    else:
        lines.append(
            advice(
                "ok",
                f"Return level good: the training signal will peak near {predicted:+.1f} dBFS.",
            )
        )
    if noise > -70:
        lines.append(
            advice(
                "warn",
                f"Return noise is high ({noise:.0f} dBFS RMS). Try the ProDI ground lift; check "
                "the pedal power supply.",
            )
        )
    return result


# ── Reference curves (C2–C4) ────────────────────────────────────────────── #


def reference_signal(sr: int) -> np.ndarray:
    """A 3 s log sweep from 20 Hz to 20 kHz (or 0.45·sr), with fades."""
    f_hi = min(20_000.0, 0.45 * sr)
    t = np.arange(int(REFERENCE_SWEEP_S * sr)) / sr
    x = chirp(t, REFERENCE_F_LO, REFERENCE_SWEEP_S, f_hi, method="logarithmic")
    fade = np.minimum(1.0, np.minimum(t, REFERENCE_SWEEP_S - t) / 0.01)
    return (10 ** (REFERENCE_DBFS / 20) * x * fade).astype(np.float32)


def reference_from_take(take: Take, name: str) -> dict:
    """A reference curve H(f) = return ÷ send from a recorded reference sweep (live or from FLAC).

    The return is aligned on the loopback's lag only, so the chain's and the pedal's own delays stay
    in the phase. The return's own lag in the same take, minus the loopback's, is the analog chain's
    delay, free of the few-sample latency jitter between separate stream opens.

    Returns:
        dict: name, lag_samples (loopback), return_lag_samples, freqs_hz (log grid), mag_db
        (1/6-octave smoothed), phase_rad (unwrapped), coherence, noise_dbfs, gain_1k_db, file.
    """
    sr = take.sr
    pre, n = take.pre_roll, len(reference_signal(sr))
    lag = find_lag(take.play[:, 1], take.rec[:, 1], sr, pre, pre + n, LAG_SEARCH_S)
    return_lag = find_lag(take.play[:, 0], take.rec[:, 0], sr, pre, pre + n, LAG_SEARCH_S)
    sent = take.play[pre : pre + n, 0]
    back = take.rec[pre + lag : pre + lag + n, 0]
    f, h, coh = frequency_response(sent, back, sr, nperseg=REFERENCE_NPERSEG)
    grid = log_grid(REFERENCE_F_LO, min(REFERENCE_F_HI, 0.45 * sr))
    keep = f > 0
    mag = smooth_octave(f[keep], 20 * np.log10(np.abs(h[keep]) + 1e-12), grid)
    phase = np.interp(grid, f[keep], np.unwrap(np.angle(h[keep])))
    return {
        "name": name,
        "lag_samples": int(lag),
        "return_lag_samples": int(return_lag),
        "freqs_hz": [round(float(v), 5) for v in grid],
        "mag_db": [round(float(v), 5) for v in mag],
        "phase_rad": [round(float(v), 5) for v in phase],
        "coherence": [round(float(v), 5) for v in np.interp(grid, f[keep], coh[keep])],
        "noise_dbfs": rms_dbfs(take.rec[int(0.1 * sr) : pre, 0]),
        "gain_1k_db": float(np.interp(np.log(1000.0), np.log(grid), mag)),
        "file": None,  # the raw capture, relative to the session folder, once saved
    }


def ref_db_at(ref: dict, freqs_hz: np.ndarray) -> np.ndarray:
    """A reference's magnitude in dB at other frequencies (linear interpolation in log f)."""
    return np.interp(np.log(freqs_hz), np.log(ref["freqs_hz"]), ref["mag_db"])


def _band_diff(a: dict, b: dict) -> np.ndarray:
    """Reference a − reference b, in dB, over COMPARE_BAND_HZ on a's grid."""
    f = np.asarray(a["freqs_hz"])
    sel = (f >= COMPARE_BAND_HZ[0]) & (f <= COMPARE_BAND_HZ[1])
    return (np.asarray(a["mag_db"]) - ref_db_at(b, f))[sel]


def check_reference(ref: dict) -> list[dict]:
    """Checks every reference must pass: a signal came back, and it is clean."""
    if ref["gain_1k_db"] < -60.0:
        return [
            advice(
                "fail",
                "Nothing came back on In 1. Check the cables for this step, the pedal's power, and "
                "the ProRMP level.",
            )
        ]
    f = np.asarray(ref["freqs_hz"])
    sel = (f >= COMPARE_BAND_HZ[0]) & (f <= COMPARE_BAND_HZ[1])
    coh = float(np.min(np.asarray(ref["coherence"])[sel]))
    st = "ok" if coh > 0.98 else "warn"
    note = "" if st == "ok" else " Low: noise, hum or a loose cable."
    return [advice(st, f"Response measured; coherence ≥ {coh:.3f} from 50 Hz to 10 kHz.{note}")]


def check_bypass(bypass: dict, chain: dict) -> list[dict]:
    """C3: the bypassed pedal must be in the chain (differ from C2) and at a sane level."""
    d = _band_diff(bypass, chain)
    med, span = float(np.median(d)), float(np.ptp(d))
    if abs(med) > BYPASS_LEVEL_FAIL_DB:
        return [
            advice(
                "fail",
                f"Bypassed pedal is {med:+.1f} dB from the chain reference. Pedal powered? Cables "
                "A and B fully in?",
            )
        ]
    if np.max(np.abs(d)) < SAME_PATH_DB:
        return [
            advice(
                "warn",
                "Identical to the chain reference: is the pedal really in the chain? Check that "
                "cable A goes into the pedal's IN and cable B from its OUT. (A true-bypass pedal "
                "would look like this; the GE-7 has buffers.)",
            )
        ]
    return [
        advice(
            "ok",
            f"Bypassed pedal vs cable: {med:+.2f} dB, varying {span:.2f} dB from 50 Hz to 10 kHz.",
        )
    ]


def check_engaged(engaged: dict, bypass: dict, chain: dict | None = None) -> list[dict]:
    """C4: engaged must differ from bypassed (footswitch on) and from C2, and be near flat."""
    if chain is not None and np.max(np.abs(_band_diff(engaged, chain))) < SAME_PATH_DB:
        return [
            advice(
                "fail",
                "Identical to the chain reference (C2): the pedal must have been in the chain, "
                "engaged, during C2. Redo C2 with cable A plugged straight into ProDI INPUT, then "
                "C3 and C4.",
            )
        ]
    d = _band_diff(engaged, bypass)
    med, worst = float(np.median(d)), float(np.max(np.abs(d)))
    if worst < SAME_PATH_DB:
        return [
            advice(
                "warn", "Identical to bypassed: is the footswitch LED on? Press it and run again."
            )
        ]
    if worst > FLAT_LIMIT_DB:
        return [
            advice(
                "warn",
                f"Engaged ÷ bypassed reaches {worst:.1f} dB (limit {FLAT_LIMIT_DB:.0f} dB): are "
                "all controls at the calibration preset?",
            )
        ]
    return [
        advice(
            "ok", f"Engaged ÷ bypassed: {med:+.2f} dB, within ±{worst:.2f} dB from 50 Hz to 10 kHz."
        )
    ]


# ── One guided step ─────────────────────────────────────────────────────── #


def run_step(
    step_id: str,
    engine: CaptureEngine,
    sr: int,
    *,
    send_level_db: float,
    train_peak_dbfs: float,
    references: dict[str, dict],
) -> dict:
    """Run one guided step's test. ``references`` holds the curves of the steps already passed.

    Returns:
        dict: step, status (worst advice), advice, loopback (C1), chain (C2), reference (C2–C4),
        take (the reference capture, for saving; not stored in JSON).
    """
    result = {"step": step_id, "loopback": None, "chain": None, "reference": None, "take": None}
    if step_id == "C1":
        result["loopback"] = run_loopback_test(engine, sr, train_peak_dbfs + send_level_db)
        lines = list(result["loopback"]["advice"])
    elif step_id == "C2":
        result["chain"] = run_chain_test(engine, sr, send_level_db, train_peak_dbfs)
        lines = list(result["chain"]["advice"])
        if worst_status(lines) != "fail":
            take = engine.run(_reference_play(sr), sr, send_level_db=0.0, route="both")
            ref = reference_from_take(take, "chain")
            tilt = ref_db_at(ref, np.array([10_000.0]))[0] - ref_db_at(ref, np.array([100.0]))[0]
            lines += check_reference(ref)
            lines.append(
                advice("ok", f"Chain response: {tilt:+.2f} dB at 10 kHz relative to 100 Hz.")
            )
            result["reference"], result["take"] = ref, take
    else:
        name = "bypass" if step_id == "C3" else "engaged_flat"
        take = engine.run(_reference_play(sr), sr, send_level_db=0.0, route="both")
        ref = reference_from_take(take, name)
        lines = check_reference(ref)
        base, earlier = ("chain", "C2") if step_id == "C3" else ("bypass", "C3")
        if base not in references:
            lines.append(advice("fail", f"No {base} reference yet: run {earlier} first."))
        elif worst_status(lines) != "fail":
            if step_id == "C3":
                lines += check_bypass(ref, references["chain"])
            else:
                lines += check_engaged(ref, references["bypass"], references.get("chain"))
        result["reference"], result["take"] = ref, take
    result["advice"] = lines
    result["status"] = worst_status(lines)
    return result


def _reference_play(sr: int) -> np.ndarray:
    return np.concatenate([reference_signal(sr), _gap(sr)])


def calibration_dict(results: list[dict]) -> dict:
    """The calibration as the session stores it, built from step results in order."""
    cal: dict = {"loopback": None, "chain": None, "references": {}, "steps": {}}
    for res in results:
        if res["loopback"] is not None:
            cal["loopback"] = res["loopback"]
        if res["chain"] is not None:
            cal["chain"] = res["chain"]
        if res["reference"] is not None and res["status"] != "fail":
            cal["references"][res["reference"]["name"]] = res["reference"]
        cal["steps"][res["step"]] = {"status": res["status"], "advice": res["advice"]}
    return cal


def simulate_guided_calibration(
    engine: CaptureEngine,
    sr: int,
    *,
    send_level_db: float,
    train_peak_dbfs: float,
    engaged_pedal=None,
) -> tuple[dict, list[dict]]:
    """Run C1–C4 on a simulated backend, doing each step's cable moves in software.

    ``engaged_pedal`` is the pedal model at its calibration preset (None = a wire).
    """
    backend = engine.backend
    backend.set_pedal(engaged_pedal)
    results: list[dict] = []
    refs: dict[str, dict] = {}
    for step in CAL_STEPS:
        if step["pedal_state"] is not None:
            backend.set_pedal_state(step["pedal_state"])
        res = run_step(
            step["id"],
            engine,
            sr,
            send_level_db=send_level_db,
            train_peak_dbfs=train_peak_dbfs,
            references=refs,
        )
        results.append(res)
        if res["status"] == "fail":
            break
        if res["reference"] is not None:
            refs[res["reference"]["name"]] = res["reference"]
    return calibration_dict(results), results
