"""Session calibration: loopback test (interface only) and chain test.

The chain is send → ProRMP → pedal → ProDI → In 1.

Both tests generate their own short signals, so they run before any capture signal is chosen. The
results are stored in the session and used by QC to normalise every take.

The guided steps C1–C4 (plan R1) wrap them: C1 loopback, C2 chain alone (pedal out of the chain),
C3 pedal bypassed, C4 pedal engaged at its calibration preset. C2–C4 each store a reference
response curve, and QC divides every take by the profile's chosen one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import chirp

from pedal_model.rig.analysis import (
    MAX_LAG_S,
    find_lag,
    log_grid,
    peak_dbfs,
    rms_dbfs,
    smooth_octave,
    transfer_function_complex,
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

LINEARITY_LIMIT_DB = 0.25  # docs/ge7_plan.md: flat2 showed <= 0.25 dB gain variation across levels
RETURN_PEAK_TARGET_DBFS = (-18.0, -6.0)  # where the loudest training section should land on In 1
NO_SIGNAL_DBFS = -70.0
CROSSTALK_LIMIT_DBFS = -50.0
LAG_SEARCH_S = MAX_LAG_S
# A delay longer than the search still produces a partial-overlap peak just inside the edge, so
# only lags well inside the search are trusted.
LAG_TRUST_FRACTION = 0.8


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


def _lag_too_long(sr: int) -> Advice:
    return Advice(
        "fail",
        f"The delay is longer than the {LAG_SEARCH_S * 1e3:.0f} ms search: is the ASIO buffer in "
        "Universal Control set very large? Lower it and run again.",
    )


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
    if lag > LAG_TRUST_FRACTION * LAG_SEARCH_S * sr:
        advice.append(_lag_too_long(sr))
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
    if lag > LAG_TRUST_FRACTION * LAG_SEARCH_S * sr:
        advice.append(_lag_too_long(sr))
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


# ── Guided references C1–C4 (plan R1) ───────────────────────────────────── #
#
# Two guitar cables around the pedal get names, printed in the GUI: cable A runs ProRMP OUTPUT →
# pedal IN, cable B runs pedal OUT → ProDI INPUT. Each step says exactly what to move, then a test
# checks that the move happened; the next step stays locked until the test passes.

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


@dataclass(frozen=True)
class CalStep:
    """One guided calibration step.

    ``moves`` are the exact actions, in order; ``{pedal}`` is replaced by the pedal's name.
    ``pedal_state`` is what the simulated rig should do for this step (the demo's stand-in for the
    cable moves). ``cables`` are the diagram cables to highlight while the step is pending.
    """

    id: str
    title: str
    moves: tuple[str, ...]
    checks: str
    pedal_state: str | None
    cables: tuple[str, ...]
    reference: str | None  # name of the reference curve this step stores


CAL_STEPS: tuple[CalStep, ...] = (
    CalStep(
        "C1",
        "Loopback",
        ("No cable moves. Check the loopback cable: Out 2 (rear) → In 2 (front).",),
        "The interface's own latency and gain; In 1 must stay silent.",
        None,
        ("loop",),
        None,
    ),
    CalStep(
        "C2",
        "Chain reference (pedal out of the chain)",
        (
            "Pull cable A out of {pedal} IN (its other end stays in ProRMP OUTPUT).",
            "Pull cable B out of ProDI INPUT (leave it hanging from {pedal} OUT).",
            "Plug cable A's free end into ProDI INPUT. Now ProRMP OUTPUT → cable A → ProDI INPUT.",
        ),
        "Chain gain, latency, linearity and return level, then the chain's own frequency response "
        "H_chain(f), which every take is corrected by.",
        "cable",
        ("inst_in", "inst_out"),
        "chain",
    ),
    CalStep(
        "C3",
        "{pedal} bypassed",
        (
            "Pull cable A out of ProDI INPUT and plug it back into {pedal} IN.",
            "Plug cable B's free end back into ProDI INPUT.",
            "{pedal} powered, footswitch LED OFF (bypassed).",
        ),
        "The bypassed pedal ÷ the chain: its buffers, H_bypass(f). It must differ from the chain "
        "reference, or the pedal is not in the chain.",
        "bypass",
        ("inst_in", "inst_out"),
        "bypass",
    ),
    CalStep(
        "C4",
        "{pedal} engaged, calibration preset",
        (
            "Press the footswitch: LED ON.",
            "Set every control to the calibration preset shown.",
        ),
        "Engaged ÷ bypassed should be near 0 dB everywhere, and must differ from bypassed (or the "
        "footswitch is still off).",
        "engaged",
        ("inst_in", "inst_out"),
        "engaged_flat",
    ),
)
STEPS_BY_ID = {s.id: s for s in CAL_STEPS}


@dataclass
class Reference:
    """A measured response of the analog chain (return ÷ send).

    Aligned on the interface latency only, so the chain's and the pedal's own delays stay in the
    phase.
    """

    name: str
    lag_samples: int
    freqs_hz: list[float]
    mag_db: list[float]  # 1/6-octave smoothed
    phase_rad: list[float]  # unwrapped, unsmoothed
    coherence: list[float]
    noise_dbfs: float
    gain_1k_db: float
    file: str | None = None  # the raw capture, relative to the session folder
    # Lag of the return (In 1) in the same take. Minus ``lag_samples`` it is the analog chain's own
    # delay, free of the few-sample latency jitter between separate stream opens.
    return_lag_samples: int | None = None

    def db_at(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Magnitude in dB at other frequencies (interpolated in log frequency)."""
        return np.interp(np.log(freqs_hz), np.log(self.freqs_hz), self.mag_db)

    def complex(self) -> np.ndarray:
        return 10 ** (np.asarray(self.mag_db) / 20) * np.exp(1j * np.asarray(self.phase_rad))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("freqs_hz", "mag_db", "phase_rad", "coherence"):
            d[k] = [round(float(v), 5) for v in d[k]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Reference:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


def reference_signal(sr: int) -> np.ndarray:
    """A log sweep from 20 Hz to 20 kHz (or 0.45·sr), with fades."""
    f_hi = min(20_000.0, 0.45 * sr)
    t = np.arange(int(REFERENCE_SWEEP_S * sr)) / sr
    x = chirp(t, REFERENCE_F_LO, REFERENCE_SWEEP_S, f_hi, method="logarithmic")
    fade = np.minimum(1.0, np.minimum(t, REFERENCE_SWEEP_S - t) / 0.01)
    return (10 ** (REFERENCE_DBFS / 20) * x * fade).astype(np.float32)


def measure_reference(engine: CaptureEngine, sr: int, name: str) -> tuple[Reference, Take]:
    """Play the reference sweep on Out 1 and Out 2; H = return ÷ send, aligned on the loopback."""
    sweep = reference_signal(sr)
    take = engine.run(np.concatenate([sweep, _gap(sr)]), sr, send_level_db=0.0, route="both")
    return reference_from_take(take, name), take


def reference_from_take(take: Take, name: str) -> Reference:
    """The reference curve from a recorded reference-sweep take (live, or reloaded from FLAC)."""
    sr = take.sr
    pre, n = take.pre_roll, len(reference_signal(sr))
    lag = find_lag(take.play[:, 1], take.rec[:, 1], sr, pre, pre + n, LAG_SEARCH_S)
    return_lag = find_lag(take.play[:, 0], take.rec[:, 0], sr, pre, pre + n, LAG_SEARCH_S)
    x = take.play[pre : pre + n, 0]
    y = take.rec[pre + lag : pre + lag + n, 0]
    f, h, coh = transfer_function_complex(x, y, sr, nperseg=min(REFERENCE_NPERSEG, n))
    grid = log_grid(REFERENCE_F_LO, min(REFERENCE_F_HI, 0.45 * sr))
    keep = f > 0
    mag = smooth_octave(f[keep], 20 * np.log10(np.abs(h[keep]) + 1e-12), grid)
    phase = np.interp(grid, f[keep], np.unwrap(np.angle(h[keep])))
    ref = Reference(
        name=name,
        lag_samples=int(lag),
        freqs_hz=grid.tolist(),
        mag_db=mag.tolist(),
        phase_rad=phase.tolist(),
        coherence=np.interp(grid, f[keep], coh[keep]).tolist(),
        noise_dbfs=rms_dbfs(take.rec[int(0.1 * sr) : pre, 0]),
        gain_1k_db=float(np.interp(np.log(1000.0), np.log(grid), mag)),
        return_lag_samples=int(return_lag),
    )
    return ref


def _band_diff(a: Reference, b: Reference) -> np.ndarray:
    """a − b in dB over COMPARE_BAND_HZ, on a's grid."""
    f = np.asarray(a.freqs_hz)
    sel = (f >= COMPARE_BAND_HZ[0]) & (f <= COMPARE_BAND_HZ[1])
    return (np.asarray(a.mag_db) - b.db_at(f))[sel]


def check_reference(ref: Reference) -> list[Advice]:
    """Checks every reference must pass: a signal came back, and it is clean."""
    if ref.gain_1k_db < -60.0:
        return [
            Advice(
                "fail",
                "Nothing came back on In 1. Check the cables for this step, the pedal's power, and "
                "the ProRMP level.",
            )
        ]
    f = np.asarray(ref.freqs_hz)
    sel = (f >= COMPARE_BAND_HZ[0]) & (f <= COMPARE_BAND_HZ[1])
    coh = float(np.min(np.asarray(ref.coherence)[sel]))
    st = "ok" if coh > 0.98 else "warn"
    note = "" if st == "ok" else " Low: noise, hum or a loose cable."
    return [Advice(st, f"Response measured; coherence ≥ {coh:.3f} from 50 Hz to 10 kHz.{note}")]


def check_bypass(bypass: Reference, chain: Reference) -> list[Advice]:
    d = _band_diff(bypass, chain)
    med, span = float(np.median(d)), float(np.ptp(d))
    if abs(med) > BYPASS_LEVEL_FAIL_DB:
        return [
            Advice(
                "fail",
                f"Bypassed pedal is {med:+.1f} dB from the chain reference. Pedal powered? "
                "Cables A and B fully in?",
            )
        ]
    if np.max(np.abs(d)) < SAME_PATH_DB:
        return [
            Advice(
                "warn",
                "Identical to the chain reference: is the pedal really in the chain? Check that "
                "cable A goes into the pedal's IN and cable B from its OUT. (A true-bypass pedal "
                "would look like this; the GE-7 has buffers.)",
            )
        ]
    return [
        Advice(
            "ok",
            f"Bypassed pedal vs cable: {med:+.2f} dB, varying {span:.2f} dB from 50 Hz to 10 kHz.",
        )
    ]


def check_engaged(
    engaged: Reference, bypass: Reference, chain: Reference | None = None
) -> list[Advice]:
    if chain is not None and np.max(np.abs(_band_diff(engaged, chain))) < SAME_PATH_DB:
        return [
            Advice(
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
            Advice(
                "warn",
                "Identical to bypassed: is the footswitch LED on? Press it once and run again.",
            )
        ]
    if worst > FLAT_LIMIT_DB:
        return [
            Advice(
                "warn",
                f"Engaged ÷ bypassed reaches {worst:.1f} dB (limit {FLAT_LIMIT_DB:.0f} dB): are "
                "all controls at the calibration preset?",
            )
        ]
    return [
        Advice(
            "ok",
            f"Engaged ÷ bypassed: {med:+.2f} dB, within ±{worst:.2f} dB from 50 Hz to 10 kHz.",
        )
    ]


@dataclass
class StepResult:
    step: str
    advice: list[Advice]
    loopback: LoopbackResult | None = None
    chain: ChainResult | None = None
    reference: Reference | None = None
    take: Take | None = None  # the reference capture, for saving; not serialised

    @property
    def status(self) -> str:
        sts = {a.status for a in self.advice}
        return "fail" if "fail" in sts else ("warn" if "warn" in sts else "ok")

    @property
    def passed(self) -> bool:
        return self.status != "fail"


def run_step(
    step_id: str,
    engine: CaptureEngine,
    sr: int,
    *,
    send_level_db: float,
    train_peak_dbfs: float,
    references: dict[str, Reference],
) -> StepResult:
    """Run one guided step's test. ``references`` holds the curves of the steps already passed."""
    step = STEPS_BY_ID[step_id]
    if step_id == "C1":
        res = run_loopback_test(engine, sr)
        return StepResult(step_id, list(res.advice), loopback=res)
    if step_id == "C2":
        chain = run_chain_test(engine, sr, send_level_db, train_peak_dbfs)
        if not chain.ok:
            return StepResult(step_id, list(chain.advice), chain=chain)
        ref, take = measure_reference(engine, sr, step.reference)
        lo, hi = ref.db_at(np.array([100.0]))[0], ref.db_at(np.array([10_000.0]))[0]
        tilt = Advice("ok", f"Chain response: {hi - lo:+.2f} dB at 10 kHz relative to 100 Hz.")
        advice = list(chain.advice) + check_reference(ref) + [tilt]
        return StepResult(step_id, advice, chain=chain, reference=ref, take=take)
    ref, take = measure_reference(engine, sr, step.reference)
    advice = check_reference(ref)
    base, earlier = ("chain", "C2") if step_id == "C3" else ("bypass", "C3")
    if base not in references:
        advice.append(Advice("fail", f"No {base} reference yet: run {earlier} first."))
    elif all(a.status != "fail" for a in advice):
        if step_id == "C3":
            advice += check_bypass(ref, references["chain"])
        else:
            advice += check_engaged(ref, references["bypass"], references.get("chain"))
    return StepResult(step_id, advice, reference=ref, take=take)


def calibration_dict(results: list[StepResult]) -> dict:
    """The calibration as the session stores it, built from step results in order."""
    cal: dict = {"loopback": None, "chain": None, "references": {}, "steps": {}}
    for res in results:
        if res.loopback is not None:
            cal["loopback"] = res.loopback.to_dict()
        if res.chain is not None:
            cal["chain"] = res.chain.to_dict()
        if res.reference is not None and res.passed:
            cal["references"][res.reference.name] = res.reference.to_dict()
        cal["steps"][res.step] = {
            "status": res.status,
            "advice": [{"status": a.status, "text": a.text} for a in res.advice],
        }
    return cal


def simulate_guided_calibration(
    engine: CaptureEngine,
    sr: int,
    *,
    send_level_db: float,
    train_peak_dbfs: float,
    engaged_pedal=None,
) -> tuple[dict, list[StepResult]]:
    """Run C1-C4 on a simulated backend, doing each step's cable moves in software.

    ``engaged_pedal`` is the pedal model at its calibration preset (None = a wire).
    """
    backend = engine.backend
    backend.set_pedal(engaged_pedal)
    results: list[StepResult] = []
    refs: dict[str, Reference] = {}
    for step in CAL_STEPS:
        if step.pedal_state is not None:
            backend.set_pedal_state(step.pedal_state)
        res = run_step(
            step.id,
            engine,
            sr,
            send_level_db=send_level_db,
            train_peak_dbfs=train_peak_dbfs,
            references=refs,
        )
        results.append(res)
        if not res.passed:
            break
        if res.reference is not None:
            refs[res.reference.name] = res.reference
    return calibration_dict(results), results
