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
# Tone peaks at the send output. The last two go beyond any take level, to find where the
# return path starts to compress (its ceiling).
CHAIN_LEVELS_DBFS = (-36.0, -30.0, -24.0, -18.0, -12.0, -9.0, -6.0, -3.0)
CEILING_COMPRESSION_DB = 0.5  # a tone this far below the chain gain marks the ceiling
# The ceiling is also measured at a low frequency: the ProRMP and ProDI transformers saturate
# there first (flat's loudest pink noise compressed 0.6 dB at 50 Hz, 2026-10-21).
LOW_TONE_HZ = 60.0
LOW_LEVELS_DBFS = (-30.0, -18.0, -12.0, -9.0, -6.0, -3.0)
DEFAULT_CEILING_DBFS = -15.0  # 1824c In 1, measured 2026-10-21; used until C2 measures it
# Where the chain's limit is. C2 plays its tone steps at 1 kHz and at 60 Hz, where the chain's gain
# differs (~5 dB on the real rig). A limit on the send side starts at the same SENT level at both
# frequencies; one on the return side at the same RETURN level. On 2026-10-22 (In 3) both started
# at -9 dBFS sent while their returns differed by 5 dB: a send-side limit, which the old
# return-only ceiling turned into a needless 10 dB cut in every take's level.
LIMIT_KIND_MIN_GAIN_DIFF_DB = 2.0  # the two frequencies' gains must differ this much to tell
# Return limit when C2 found the send side limits first (the return side was never reached): the
# converter's full scale. The live linearity check and its automatic retakes cover what this misses
# (e.g. the pedal's own output stage on big boosts).
RETURN_FULL_SCALE_DBFS = -1.0
# A reference whose band coherence (1/12 octave, from the one-FFT estimate) is below this was
# disturbed during its sweep. Good references read >= 0.999 (60 Hz hum the worst); the bypass
# reference of 2026-10-22_1600 read 0.984 at 2.9 kHz, which the Welch coherence (0.9997) hid.
REFERENCE_COHERENCE_WARN = 0.995
COHERENCE_BAND_OCT = 1 / 12
LOOP_TONE_DBFS = -20.0
LOOP_LINEARITY_STEPS_DB = (-20.0, -10.0, 0.0)  # tones relative to the hottest send of a take
LOOP_LINEARITY_LIMIT_DB = 0.5

LINEARITY_LIMIT_DB = 0.25  # docs/ge7_plan.md: flat2 showed <= 0.25 dB gain variation across levels
LINEARITY_FAIL_DB = 1.0  # more than this over the take's levels distorts the training data
LINEARITY_MARGIN_DB = 3.0  # linearity is judged up to the loudest take level + this
RETURN_PEAK_TARGET_DBFS = (-18.0, -6.0)  # where the loudest training section should land on In 1
NO_SIGNAL_DBFS = -70.0
HEARD_ABOVE_NOISE_DB = 30.0  # a chain tone this far above the return's noise is measured
TOO_QUIET_FAIL_DB = 12.0  # a return this far below RETURN_PEAK_TARGET_DBFS fails C2
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
SWEPT_FRACTION = 0.975  # use bins up to this fraction of the sweep's top frequency
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


def _tone(sr: int, dbfs: float, seconds: float, freq_hz: float = TONE_HZ) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)
    return (10 ** (dbfs / 20) * np.sin(2 * np.pi * freq_hz * t) * fade).astype(np.float32)


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
    noise = rms_dbfs(take.rec[int(0.1 * sr) : pre, 1])
    crosstalk = rms_dbfs(take.rec[:, 0])
    gains, received = [], []
    for s0 in starts:
        got = rms_dbfs(take.rec[pre + s0 + a + lag : pre + s0 + b + lag, 1])
        received.append(got)
        gains.append(got - rms_dbfs(take.play[pre + s0 + a : pre + s0 + b, 1]))
    # Linearity is only judged on tones well above In 2's noise: a tone lost in noise reads as a
    # gain change that is not there.
    heard = [g for g, r in zip(gains, received) if r > noise + 20.0]
    gain = heard[0] if heard else gains[-1]  # the quietest clearly heard tone
    spread = float(max(heard) - min(heard)) if len(heard) > 1 else 0.0

    lines = []
    if take.xruns:
        lines.append(advice("fail", DROPOUT_TEXT.format(take.xruns)))
    if lag > LAG_TRUST_FRACTION * LAG_SEARCH_S * sr:
        lines.append(advice("fail", LAG_TOO_LONG_TEXT))
    if received[-1] < NO_SIGNAL_DBFS or len(heard) < 2:
        lines.append(
            advice(
                "fail",
                f"Almost nothing reaches In 2 (loudest tone {received[-1]:.0f} dBFS, noise "
                f"{noise:.0f} dBFS; loop gain {gains[-1]:+.0f} dB). Check the Main knob (Out 1/2 "
                "are the Main outputs: turn it fully up), that the cable runs from Main Out R to "
                "In 2, and the UC mixer (DAW 2 unmuted, Main not muted).",
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
    if len(heard) < 2:
        pass  # no level to judge linearity at; reported above
    elif spread > LOOP_LINEARITY_LIMIT_DB:
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


def _onset(heard: list[dict]) -> dict | None:
    """The loudest tone step before the first one compressed by more than CEILING_COMPRESSION_DB
    relative to the quietest heard tone (its send_dbfs, return_dbfs, gain_db); None if none is
    compressed."""
    if not heard:
        return None
    for prev, st in zip(heard, heard[1:]):
        if heard[0]["gain_db"] - st["gain_db"] > CEILING_COMPRESSION_DB:
            return prev
    return None


def limit_kind(onset_1k: dict | None, onset_low: dict | None, gain_diff_db: float) -> str | None:
    """Where the chain's limit sits: "send", "return", "unknown", or None if nothing compressed.

    The two frequencies' onsets are compared in sent and in returned level: the limit is on the
    side where they agree. ``gain_diff_db`` is the chain gain at 1 kHz minus at the low tone; below
    LIMIT_KIND_MIN_GAIN_DIFF_DB the two cannot be told apart ("unknown": both limits apply).
    """
    if onset_1k is None and onset_low is None:
        return None
    if onset_1k is None or onset_low is None or abs(gain_diff_db) < LIMIT_KIND_MIN_GAIN_DIFF_DB:
        return "unknown"
    d_send = abs(onset_1k["send_dbfs"] - onset_low["send_dbfs"])
    d_return = abs(onset_1k["return_dbfs"] - onset_low["return_dbfs"])
    return "send" if d_send < d_return else "return"


def rig_limits(chain: dict | None) -> dict:
    """The limits auto-level plans with, from a C2 result: send_ceiling_dbfs (peak of the sent
    signal, or None) and return_ceiling_dbfs (peak at the return input).

    A calibration from before the limit was located has only a return ceiling (or none:
    DEFAULT_CEILING_DBFS).
    """
    chain = chain or {}
    if "limit_kind" not in chain:
        ceiling = chain.get("return_ceiling_dbfs")
        return {
            "limit_kind": None,
            "send_ceiling_dbfs": None,
            "return_ceiling_dbfs": DEFAULT_CEILING_DBFS if ceiling is None else ceiling,
        }
    kind = chain["limit_kind"]
    ret = chain.get("return_ceiling_dbfs") if kind in ("return", "unknown") else None
    return {
        "limit_kind": kind,
        "send_ceiling_dbfs": chain.get("send_ceiling_dbfs")
        if kind in ("send", "unknown")
        else None,
        "return_ceiling_dbfs": RETURN_FULL_SCALE_DBFS if ret is None else ret,
    }


def _onset_text(onset: dict | None) -> str:
    if onset is None:
        return "not reached"
    return f"{onset['send_dbfs']:.0f} dBFS sent, {onset['return_dbfs']:.0f} dBFS returned"


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
    low_starts = []
    for lvl in LOW_LEVELS_DBFS:
        tone = _tone(sr, lvl, TONE_S, LOW_TONE_HZ)
        low_starts.append(cursor)
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
    noise = rms_dbfs(take.rec[int(0.1 * sr) : pre, 0])
    # A tone counts as heard 30 dB above the return's own noise (noise then shifts its level by
    # 0.004 dB). An absolute threshold called a clean but quiet return "no return" (2026-10-20).
    heard = [st for st in steps if st["return_dbfs"] > noise + HEARD_ABOVE_NOISE_DB]
    gains = [st["gain_db"] for st in heard]
    # The chain gain is the quietest heard tone's: the one least likely to be compressed.
    gain = gains[0] if gains else -200.0
    # Linearity is judged over the levels a take really sends (its loudest peak plus a 3 dB
    # margin); louder tones only tell how much headroom is left.
    hottest = train_peak_dbfs + send_level_db
    used = [st for st in heard if st["send_dbfs"] <= hottest + LINEARITY_MARGIN_DB]
    spread = max(abs(st["gain_db"] - gain) for st in used) if used else 0.0
    predicted = hottest + gain
    # Clipping is judged on the tones a take could produce; the louder ones probe the ceiling.
    peak_seen = max((st["return_dbfs"] for st in used), default=peak_dbfs(take.rec[:, 0]))
    # Onset: the loudest tone before the first one compressed by more than CEILING_COMPRESSION_DB.
    # Above it the rig, not the pedal, shapes the signal.
    onset_1k = _onset(heard)
    low_steps = []
    for lvl, s in zip(LOW_LEVELS_DBFS, low_starts):
        send = take.play[pre + s + a : pre + s + b, 0]
        ret = take.rec[pre + s + a + lag : pre + s + b + lag, 0]
        low_steps.append(
            {
                "send_dbfs": lvl,
                "return_dbfs": rms_dbfs(ret) + 3.01,
                "gain_db": rms_dbfs(ret) - rms_dbfs(send),
            }
        )
    low_heard = [st for st in low_steps if st["return_dbfs"] > noise + HEARD_ABOVE_NOISE_DB]
    onset_low = _onset(low_heard)
    low_gain = low_heard[0]["gain_db"] if low_heard else gain
    kind = limit_kind(onset_1k, onset_low, gain - low_gain)
    onsets = [o for o in (onset_1k, onset_low) if o is not None]
    send_ceiling = min(o["send_dbfs"] for o in onsets) if onsets else None
    ceiling = min(o["return_dbfs"] for o in onsets) if onsets else None
    ceiling_1k = onset_1k["return_dbfs"] if onset_1k else None
    low_ceiling = onset_low["return_dbfs"] if onset_low else None
    result = {
        "lag_samples": int(lag),
        "gain_db": gain,
        "linearity_spread_db": spread,
        "return_noise_dbfs": noise,
        "limit_kind": kind,
        "send_ceiling_dbfs": send_ceiling,
        "return_ceiling_dbfs": ceiling if kind in ("return", "unknown") else None,
        "return_ceiling_1k_dbfs": ceiling_1k,
        "return_ceiling_60hz_dbfs": low_ceiling,
        "low_steps": low_steps,
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
        loudest = steps[-1]
        lines.append(
            advice(
                "fail",
                f"Nothing measurable reached In 1: the loudest tone "
                f"({loudest['send_dbfs']:.0f} dBFS sent) came back at "
                f"{loudest['return_dbfs']:.0f} dBFS, against a noise floor of {noise:.0f} dBFS "
                f"(it must be {HEARD_ABOVE_NOISE_DB:.0f} dB above it). Check, in "
                "order: (1) cable A from ProRMP OUTPUT into ProDI INPUT; (2) ProDI XLR OUT into "
                "the XLR of In 1; (3) Universal Control: 'Inputs 1-2 Line' OFF (with Line on the "
                "XLR is ignored), DAW 1 unmuted, Main not muted; (4) Main knob up, ProRMP level "
                "up, In 1 gain up.",
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
        # Where it saturates: if the return stops rising well below full scale, an analog stage
        # before the converter is clipping (2026-10-20: the In 1 preamp at maximum gain).
        # Flattening: the two loudest tones' returns differ by much less than their sends do.
        tops = [st["return_dbfs"] for st in heard[-2:]]
        sends = [st["send_dbfs"] for st in heard[-2:]]
        flat = (
            len(tops) == 2
            and (tops[1] - tops[0]) < (sends[1] - sends[0]) - 1.5
            and max(tops) < -3.0
        )
        where = (
            f" The return stops rising at about {max(tops):.0f} dBFS, below full scale: an analog "
            "stage before the converter is clipping, most likely the In 1 preamp near maximum "
            "gain. Lower In 1 gain and get the level back with the ProRMP level knob or an "
            "inline booster (e.g. a Cloudlifter) instead."
            if flat
            else " Lower the ProRMP level or the In 1 gain, then re-run."
        )
        lines.append(
            advice(
                "fail" if spread > LINEARITY_FAIL_DB else "warn",
                f"Gain changes by {spread:.2f} dB over the levels a take sends (limit "
                f"{LINEARITY_LIMIT_DB} dB): something saturates.{where}",
            )
        )
    else:
        lines.append(
            advice(
                "ok",
                f"Linear up to {hottest + LINEARITY_MARGIN_DB:.0f} dBFS sent (the loudest take "
                f"level + {LINEARITY_MARGIN_DB:.0f} dB): gain spread {spread:.2f} dB.",
            )
        )
    onset_text = (
        f"1 kHz: {_onset_text(onset_1k)}; {LOW_TONE_HZ:.0f} Hz: {_onset_text(onset_low)}; the "
        f"chain is {gain - low_gain:+.1f} dB louder at 1 kHz"
    )
    if kind == "send":
        lines.append(
            advice(
                "ok",
                f"Limit on the send side: the chain compresses above about {send_ceiling:.0f} dBFS "
                f"sent, at the same sent level at both frequencies ({onset_text}). Auto-level "
                f"keeps every take's sent peak {AUTO_LEVEL_HEADROOM_DB:.0f} dB under it; the "
                "pedal's gain does not change it.",
            )
        )
    elif kind is not None:
        where = "Limit on the return side" if kind == "return" else "Limit found (side unclear)"
        lines.append(
            advice(
                "ok",
                f"{where}: the return compresses above about {ceiling:.0f} dBFS ({onset_text}). "
                f"Auto-level keeps every take's return peak {AUTO_LEVEL_HEADROOM_DB:.0f} dB under "
                "it.",
            )
        )
    elif heard:
        lines.append(
            advice(
                "ok",
                f"No limit reached up to {heard[-1]['send_dbfs']:.0f} dBFS sent "
                f"({heard[-1]['return_dbfs']:.0f} dBFS returned).",
            )
        )
    above = [st for st in heard if st["send_dbfs"] > hottest + LINEARITY_MARGIN_DB]
    if above:
        st = above[-1]
        lines.append(
            advice(
                "ok",
                f"Headroom: at {st['send_dbfs']:.0f} dBFS sent ({st['send_dbfs'] - hottest:.0f} dB "
                f"above the loudest take level) the gain is {st['gain_db'] - gain:+.2f} dB off.",
            )
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
                "fail" if predicted < lo - TOO_QUIET_FAIL_DB else "warn",
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

    H comes from one FFT over the whole sweep plus its tail, H = FFT(return) / FFT(send): exact for
    a linear chain, with 0.2 Hz resolution. (Welch segments misread the bottom octave, which the
    sweep covers in its first 0.1 s under the window taper: they showed a false dip at 30 Hz and
    read 20 Hz 0.9 dB high, 2026-10-21.) Only frequencies the sweep excites are used. Welch is kept
    for the coherence, which only checks that the capture is clean.

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
    f_w, _, coh = frequency_response(sent, back, sr, nperseg=REFERENCE_NPERSEG)
    # One FFT over the sweep and the gap after it (the chain's response tail lands in the gap).
    m = min(n + int(GAP_S * sr), len(take.rec) - pre - lag)
    x = np.zeros(m)
    x[:n] = sent
    y = take.rec[pre + lag : pre + lag + m, 0].astype(np.float64)
    big_n = 1 << int(np.ceil(np.log2(2 * m)))
    f = np.fft.rfftfreq(big_n, 1 / sr)
    big_x, big_y = np.fft.rfft(x, big_n), np.fft.rfft(y, big_n)
    h = big_y / big_x
    sweep_top = min(20_000.0, 0.45 * sr)
    swept = (f >= REFERENCE_F_LO) & (f <= SWEPT_FRACTION * sweep_top)
    grid = log_grid(REFERENCE_F_LO, min(REFERENCE_F_HI, 0.45 * sr))
    mag = smooth_octave(f[swept], 20 * np.log10(np.abs(h[swept]) + 1e-12), grid)
    phase = np.interp(grid, f[swept], np.unwrap(np.angle(h[swept])))
    # Band coherence |Σ X*Y|² / (Σ|X|² Σ|Y|²) over 1/12 octave, with the chain's delay removed so
    # the phase does not turn across the band. A short disturbance during the sweep (a click, a
    # glitch) lowers it at the frequency the sweep was passing; Welch's averaged coherence hides it.
    s_xy = np.conj(big_x) * big_y * np.exp(2j * np.pi * f * (return_lag - lag) / sr)
    half = 2 ** (COHERENCE_BAND_OCT / 2)
    band_coh = np.ones(len(grid))
    for i, fc in enumerate(grid):
        lo, hi = np.searchsorted(f, [fc / half, fc * half])
        if hi > lo:
            num = np.abs(np.sum(s_xy[lo:hi])) ** 2
            den = np.sum(np.abs(big_x[lo:hi]) ** 2) * np.sum(np.abs(big_y[lo:hi]) ** 2)
            band_coh[i] = num / den if den > 0 else 0.0
    return {
        "name": name,
        "lag_samples": int(lag),
        "return_lag_samples": int(return_lag),
        "freqs_hz": [round(float(v), 5) for v in grid],
        "mag_db": [round(float(v), 5) for v in mag],
        "phase_rad": [round(float(v), 5) for v in phase],
        "coherence": [round(float(v), 5) for v in np.interp(grid, f_w, coh)],
        "band_coherence": [round(float(v), 5) for v in band_coh],
        "noise_dbfs": rms_dbfs(take.rec[int(0.1 * sr) : pre, 0]),
        "gain_1k_db": float(np.interp(np.log(1000.0), np.log(grid), mag)),
        "file": None,  # the raw capture, relative to the session folder, once saved
    }


AUTO_LEVEL_PROBE_DBFS = -45.0  # probe sweep peak at the send: quiet, so nothing clips
AUTO_LEVEL_HEADROOM_DB = 6.0  # a take's predicted return peak stays this far under the ceiling
AUTO_LEVEL_BAND_HZ = (30.0, 16_000.0)  # where the probe's largest gain is looked for


def plan_send_level(
    engine: CaptureEngine,
    sr: int,
    signal_peak_dbfs: float,
    default_send_db: float,
    chain: dict | None,
) -> dict:
    """Auto-level: choose a take's send level so the rig never limits it.

    Two limits (``rig_limits`` of the C2 result), each kept AUTO_LEVEL_HEADROOM_DB (H) away:

        send side:    signal peak + send                    <= S - H
        return side:  signal peak + send + G                <= R - H

    G is the largest gain from send to return (chain and pedal, 1/6-octave smoothed), measured with
    a quiet probe sweep. The send level is the larger one both allow, never above the profile's
    default. With the rig linear, engaged ÷ bypassed still cancels the DI boxes.

    Args:
        signal_peak_dbfs: Peak of the dry signal the take will play (before the send level).
        default_send_db: The profile's send level (the most a take may use).
        chain: The session's C2 result (or None: DEFAULT_CEILING_DBFS on the return).

    Returns:
        dict: send_level_db, default_send_db, probe_max_gain_db, probe_max_gain_hz,
        predicted_peak_dbfs (return), predicted_send_peak_dbfs, limit_kind,
        send_ceiling_dbfs, ceiling_dbfs (return), headroom_db, probe_return_peak_dbfs.
    """
    limits = rig_limits(chain)
    ceiling_dbfs = limits["return_ceiling_dbfs"]
    send_ceiling = limits["send_ceiling_dbfs"]
    probe_db = AUTO_LEVEL_PROBE_DBFS
    for _ in range(2):  # a second, quieter try if the probe itself reached the ceiling
        sweep = reference_signal(sr) * np.float32(10 ** ((probe_db - REFERENCE_DBFS) / 20))
        take = engine.run(np.concatenate([sweep, _gap(sr)]), sr, send_level_db=0.0, route="both")
        probe_peak = peak_dbfs(take.rec[:, 0])
        if probe_peak < ceiling_dbfs - 3.0:
            break
        probe_db -= 15.0
    ref = reference_from_take(take, "probe")
    f = np.asarray(ref["freqs_hz"])
    mag = np.asarray(ref["mag_db"])
    band = (f >= AUTO_LEVEL_BAND_HZ[0]) & (f <= AUTO_LEVEL_BAND_HZ[1])
    i = int(np.argmax(np.where(band, mag, -np.inf)))
    gain_max = float(mag[i])
    send = min(default_send_db, ceiling_dbfs - AUTO_LEVEL_HEADROOM_DB - signal_peak_dbfs - gain_max)
    if send_ceiling is not None:
        send = min(send, send_ceiling - AUTO_LEVEL_HEADROOM_DB - signal_peak_dbfs)
    return {
        "send_level_db": round(float(send), 2),
        "default_send_db": float(default_send_db),
        "probe_max_gain_db": round(gain_max, 2),
        "probe_max_gain_hz": round(float(f[i]), 1),
        "predicted_peak_dbfs": round(float(signal_peak_dbfs + send + gain_max), 2),
        "predicted_send_peak_dbfs": round(float(signal_peak_dbfs + send), 2),
        "limit_kind": limits["limit_kind"],
        "send_ceiling_dbfs": send_ceiling,
        "ceiling_dbfs": float(ceiling_dbfs),
        "headroom_db": AUTO_LEVEL_HEADROOM_DB,
        "probe_return_peak_dbfs": round(float(probe_peak), 2),
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
    if "band_coherence" in ref:
        band_coh = np.asarray(ref["band_coherence"])[sel]
        i = int(np.argmin(band_coh))
        if band_coh[i] < REFERENCE_COHERENCE_WARN:
            return [
                advice(
                    "warn",
                    f"Disturbed: coherence {band_coh[i]:.3f} at {f[sel][i]:.0f} Hz (a clean sweep "
                    f"reads ≥ {REFERENCE_COHERENCE_WARN}). Something happened while the sweep "
                    "passed that frequency (a click, a bump, a glitch): every take is divided by "
                    "this curve, so run this step again.",
                )
            ]
        return [
            advice("ok", f"Response measured; band coherence ≥ {band_coh[i]:.4f}, 50 Hz–10 kHz.")
        ]
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
