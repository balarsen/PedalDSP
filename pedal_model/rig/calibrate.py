"""Session calibration: loopback test (interface only) and chain test.

The chain is send → ProRMP → pedal → ProDI → In 1.

Both tests generate their own short signals, so they run before any capture signal is chosen. The
results
are stored in the session and used by QC to normalise every take.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import chirp

from pedal_model.rig.analysis import find_lag, peak_dbfs, rms_dbfs
from pedal_model.rig.engine import CaptureEngine

SWEEP_S = 1.0  # log sweep used for the lag measurement
SWEEP_F = (50.0, 15_000.0)
SWEEP_DBFS = -24.0  # quiet: keeps a nonlinear pedal near-linear so the correlation peak is sharp
TONE_HZ = 1000.0
TONE_S = 0.4
TONE_MEASURE_S = 0.2  # middle part of each tone, away from the on/off edges
GAP_S = 0.2
CHAIN_LEVELS_DBFS = (-36.0, -30.0, -24.0, -18.0, -12.0, -9.0)  # tone peaks at the send output
LOOP_TONE_DBFS = -20.0

LINEARITY_LIMIT_DB = 0.25  # docs/ge7_plan.md: flat2 showed <= 0.25 dB gain variation across levels
RETURN_PEAK_TARGET_DBFS = (-18.0, -6.0)  # where the loudest training section should land on In 1
NO_SIGNAL_DBFS = -70.0
CROSSTALK_LIMIT_DBFS = -50.0
LAG_SEARCH_S = 0.05


@dataclass
class Advice:
    """One line of guidance: status is "ok", "warn" or "fail"."""

    status: str
    text: str


@dataclass
class LoopbackResult:
    lag_samples: int
    gain_db: float
    loop_noise_dbfs: float
    return_dbfs_while_loop_only: float
    advice: list[Advice] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(a.status != "fail" for a in self.advice)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChainStep:
    send_dbfs: float
    return_dbfs: float
    gain_db: float


@dataclass
class ChainResult:
    lag_samples: int
    gain_db: float
    linearity_spread_db: float
    return_noise_dbfs: float
    predicted_train_peak_dbfs: float
    steps: list[ChainStep] = field(default_factory=list)
    advice: list[Advice] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(a.status != "fail" for a in self.advice)

    def to_dict(self) -> dict:
        return asdict(self)


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


def run_loopback_test(engine: CaptureEngine, sr: int) -> LoopbackResult:
    """Out 2 → In 2 only: interface latency and gain, and a check that In 1 stays quiet."""
    sweep, tone = _sweep(sr, LOOP_TONE_DBFS), _tone(sr, LOOP_TONE_DBFS, TONE_S)
    signal = np.concatenate([sweep, _gap(sr), tone, _gap(sr)])
    take = engine.run(signal, sr, send_level_db=0.0, route="loop")
    pre = take.pre_roll
    lag = find_lag(take.play[:, 1], take.rec[:, 1], sr, pre, pre + len(sweep), LAG_SEARCH_S)

    t0 = pre + len(sweep) + len(_gap(sr)) + lag
    mid = slice(
        t0 + int((TONE_S - TONE_MEASURE_S) / 2 * sr), t0 + int((TONE_S + TONE_MEASURE_S) / 2 * sr)
    )
    gain = rms_dbfs(take.rec[mid, 1]) - rms_dbfs(take.play[mid.start - lag : mid.stop - lag, 1])
    noise = rms_dbfs(take.rec[int(0.1 * sr) : pre, 1])
    crosstalk = rms_dbfs(take.rec[:, 0])

    advice: list[Advice] = []
    if take.xruns:
        advice.append(
            Advice(
                "fail",
                f"{take.xruns} dropouts. Raise the block size in rig.json, close other audio apps.",
            )
        )
    if rms_dbfs(take.rec[mid, 1]) < NO_SIGNAL_DBFS:
        advice.append(
            Advice(
                "fail",
                "No loopback signal on In 2. Check the Out 2 → In 2 cable, and that In 2 gain is "
                "not fully off.",
            )
        )
    else:
        advice.append(
            Advice(
                "ok",
                f"Loopback found: {lag} samples ({lag / sr * 1e3:.2f} ms), gain {gain:+.1f} dB.",
            )
        )
    if crosstalk > CROSSTALK_LIMIT_DBFS:
        advice.append(
            Advice(
                "fail",
                f"In 1 hears the loopback ({crosstalk:.0f} dBFS). Cables swapped? Out 1 and Out "
                "2 must go to different places.",
            )
        )
    if noise > -80:
        advice.append(
            Advice(
                "warn", f"In 2 noise is high ({noise:.0f} dBFS RMS). Check the cable and ground."
            )
        )
    return LoopbackResult(lag, gain, noise, crosstalk, advice)


def run_chain_test(
    engine: CaptureEngine, sr: int, send_level_db: float, train_peak_dbfs: float
) -> ChainResult:
    """Through the pedal: chain gain, latency and linearity, plus advice on the hardware knobs.

    Args:
        engine: Capture engine on the session's backend.
        sr: Session sample rate in Hz.
        send_level_db: The profile's digital send level (applied to every capture signal).
        train_peak_dbfs: Peak of the session's training signal before the send level.
    """
    sweep = _sweep(sr, SWEEP_DBFS)
    parts = [sweep, _gap(sr)]
    starts = []
    cursor = len(sweep) + len(_gap(sr))
    for lvl in CHAIN_LEVELS_DBFS:
        starts.append(cursor)
        tone = _tone(sr, lvl, TONE_S)
        parts += [tone, _gap(sr)]
        cursor += len(tone) + len(_gap(sr))
    signal = np.concatenate(parts)
    take = engine.run(signal, sr, send_level_db=0.0, route="both")
    pre = take.pre_roll
    lag = find_lag(take.play[:, 0], take.rec[:, 0], sr, pre, pre + len(sweep), LAG_SEARCH_S)

    steps: list[ChainStep] = []
    a = int((TONE_S - TONE_MEASURE_S) / 2 * sr)
    b = int((TONE_S + TONE_MEASURE_S) / 2 * sr)
    for lvl, s in zip(CHAIN_LEVELS_DBFS, starts):
        send = take.play[pre + s + a : pre + s + b, 0]
        ret = take.rec[pre + s + a + lag : pre + s + b + lag, 0]
        steps.append(ChainStep(lvl, rms_dbfs(ret) + 3.01, rms_dbfs(ret) - rms_dbfs(send)))

    audible = [st for st in steps if st.return_dbfs > NO_SIGNAL_DBFS + 20]
    gains = [st.gain_db for st in audible]
    gain = float(np.median(gains)) if gains else -200.0
    spread = float(max(gains) - min(gains)) if len(gains) > 1 else 0.0
    noise = rms_dbfs(take.rec[int(0.1 * sr) : pre, 0])
    predicted = train_peak_dbfs + send_level_db + gain
    peak_seen = peak_dbfs(take.rec[:, 0])

    advice: list[Advice] = []
    lo, hi = RETURN_PEAK_TARGET_DBFS
    if take.xruns:
        advice.append(
            Advice(
                "fail",
                f"{take.xruns} dropouts. Raise the block size in rig.json, close other audio apps.",
            )
        )
    if not audible:
        advice.append(
            Advice(
                "fail",
                "No return on In 1. Check: pedal powered and engaged, cables ProRMP → pedal → "
                "ProDI → In 1, "
                "ProRMP level not at zero, In 1 gain up.",
            )
        )
        return ChainResult(lag, gain, spread, noise, predicted, steps, advice)
    advice.append(
        Advice(
            "ok",
            f"Chain found: {lag} samples ({lag / sr * 1e3:.2f} ms), gain {gain:+.1f} dB at 1 kHz.",
        )
    )
    if peak_seen > -1.0:
        advice.append(
            Advice(
                "fail",
                f"The return clipped ({peak_seen:+.1f} dBFS). Lower the In 1 gain, or switch on "
                "the ProDI −15 dB pad.",
            )
        )
    if spread > LINEARITY_LIMIT_DB:
        advice.append(
            Advice(
                "warn",
                f"Gain changes by {spread:.2f} dB across levels (limit {LINEARITY_LIMIT_DB} dB): "
                "something is "
                "saturating. Lower the 1824c Main knob or the ProRMP level, then raise In 1 gain "
                "to compensate.",
            )
        )
    else:
        advice.append(
            Advice("ok", f"Linear: gain spread {spread:.2f} dB across {len(gains)} levels.")
        )
    if predicted > hi:
        advice.append(
            Advice(
                "warn",
                f"The loudest training section would reach {predicted:+.1f} dBFS on In 1 (target "
                f"{lo:.0f} to {hi:.0f}). "
                f"Lower the In 1 gain by about {predicted - (lo + hi) / 2:.0f} dB.",
            )
        )
    elif predicted < lo:
        advice.append(
            Advice(
                "warn",
                f"The loudest training section would only reach {predicted:+.1f} dBFS on In 1 "
                f"(target {lo:.0f} to {hi:.0f}). "
                f"Raise the In 1 gain by about {(lo + hi) / 2 - predicted:.0f} dB.",
            )
        )
    else:
        advice.append(
            Advice(
                "ok",
                f"Return level good: the training signal will peak near {predicted:+.1f} dBFS.",
            )
        )
    if noise > -70:
        advice.append(
            Advice(
                "warn",
                f"Return noise is high ({noise:.0f} dBFS RMS). Try the ProDI ground lift; check "
                "the pedal power supply.",
            )
        )
    return ChainResult(lag, gain, spread, noise, predicted, steps, advice)
