"""Quality checks on a finished take, and the expected-vs-measured frequency response for linear
pedals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pedal_model.rig.analysis import (
    aligned,
    find_lag,
    log_grid,
    peak_dbfs,
    rms_dbfs,
    smooth_octave,
    transfer_function,
)
from pedal_model.rig.calibrate import Reference
from pedal_model.rig.engine import Take
from pedal_model.signals.manifest import Manifest

# The 1824c latency moves a few samples between stream opens (3718-3727 seen, 2026-10-18). Each take
# aligns on its own loopback, so that is harmless; only a jump like a buffer-size change is flagged.
LAG_TOLERANCE_SAMPLES = 16
NOISE_OK_DBFS = -70.0  # docs/notaklon_capture_plan.md §8: noise floor < −70 dBFS in silent gaps
NOISE_FAIL_DBFS = -55.0
PRESENT_DBFS = -60.0
CLIP_WARN_DBFS = -1.0
CLIP_FAIL_DBFS = -0.1
LEVEL_OFFSET_WARN_DB = 1.0
LEVEL_OFFSET_FAIL_DB = 3.0
FR_TYPES = ("log_sine_sweep", "pink_noise")  # broadband sections used for the response estimate
DISPLAY_BAND_HZ = (20.0, 19_000.0)  # sweeps end at 20 kHz
STATUS_ORDER = {"ok": 0, "warn": 1, "fail": 2}
INTEGRITY_RANGE_DB = 40.0  # only blocks within this of the loudest send block are judged
INTEGRITY_DROP_DB = 6.0  # a loopback block this far below the typical ratio lost its audio
REFERENCE_NOTES = {
    "chain": "Take ÷ the chain alone (C2): the pedal's whole response, buffers included.",
    "bypass": "Take ÷ the bypassed pedal (C3): what the engaged circuit adds (plan decision D1).",
    "engaged_flat": "Take ÷ the engaged pedal at its calibration preset (C4).",
}


@dataclass
class QCItem:
    name: str
    status: str  # "ok", "warn", "fail"
    value: str
    detail: str = ""


@dataclass
class FrequencyResponse:
    """Pedal response on a log grid: measured (normalised by a calibration reference) and expected.

    ``reference`` names what the measurement was divided by: a reference curve ("chain", "bypass",
    "engaged_flat"), "chain_gain" (old scalar normalisation), or None.
    """

    freqs_hz: np.ndarray
    measured_db: np.ndarray
    expected_db: np.ndarray | None
    reference: str | None = None


@dataclass
class QCReport:
    """Result of check_take.

    Attributes:
        lag: Samples removed when aligning the return: the interface latency (this take's
            loopback) plus the chain's fixed analog offset from calibration. The pedal's own delay
            is NOT removed: it is part of what the model has to learn.
        loop_lag: This take's loopback (interface-only) latency.
        xcorr_lag: Lag at the peak of the send/return cross-correlation; includes the pedal's
            group delay.
    """

    items: list[QCItem] = field(default_factory=list)
    lag: int = 0
    loop_lag: int = 0
    xcorr_lag: int = 0
    response: FrequencyResponse | None = None

    @property
    def verdict(self) -> str:
        if not self.items:
            return "ok"
        return max((i.status for i in self.items), key=STATUS_ORDER.__getitem__)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "lag_samples": self.lag,
            "loop_lag_samples": self.loop_lag,
            "xcorr_lag_samples": self.xcorr_lag,
            "items": [vars(i) for i in self.items],
        }


def expected_response(
    pedal: str, controls: dict[str, float], sr: int, grid: np.ndarray
) -> np.ndarray | None:
    """Model response of the pedal at these settings, or None if there is no linear model."""
    if pedal != "boss_ge7":
        return None
    from pedal_model.synthetic.ge7 import Ge7Sim, Ge7Sliders

    f, db = Ge7Sim.from_sliders(Ge7Sliders.from_dict(controls)).frequency_response(sr, n_fft=32768)
    return np.interp(grid, f, db)


def loopback_integrity(
    take: Take, loop_lag: int, block: int = 1024
) -> tuple[int, int, float | None]:
    """Blocks where the loopback (In 2) lost the audio that was sent (Out 2).

    The driver does not always report a dropout: on the 1824c, output blocks went silent mid-take
    while the stream reported no under/overflow. The loopback is a plain cable, so its level must
    follow the send's level block by block; a block more than ``INTEGRITY_DROP_DB`` below the
    take's typical send-to-loopback ratio lost its audio. Blocks more than
    ``INTEGRITY_RANGE_DB`` below the loudest send block are not judged (the loopback is sent
    30 dB below the pedal send, so an absolute floor would skip most of it).

    Returns:
        (blocks judged, blocks lost, time of the first lost block in seconds or None).
    """
    sr, pre = take.sr, take.pre_roll
    # whole blocks of signal whose loopback still fits in the recording
    nb = min(take.signal_len, len(take.rec) - pre - loop_lag) // block
    if nb <= 0:
        return 0, 0, None
    send = take.play[pre : pre + nb * block, 1].astype(np.float64).reshape(nb, block)
    got = take.rec[pre + loop_lag : pre + loop_lag + nb * block, 1].astype(np.float64)
    got = got.reshape(nb, block)
    send_db = 10 * np.log10(np.mean(send**2, axis=1) + 1e-30)
    got_db = 10 * np.log10(np.mean(got**2, axis=1) + 1e-30)
    judged = send_db >= send_db.max() - INTEGRITY_RANGE_DB
    ratio = (got_db - send_db)[judged]
    if ratio.size == 0:
        return 0, 0, None
    lost = ratio < np.median(ratio) - INTEGRITY_DROP_DB
    first_block = np.flatnonzero(judged)[np.argmax(lost)] if lost.any() else None
    first = None if first_block is None else float(first_block * block) / sr
    return int(ratio.size), int(lost.sum()), first


def calibration_reference(calibration: dict | None, name: str) -> Reference | None:
    """The named reference curve of a calibration, else "chain", else None."""
    refs = (calibration or {}).get("references") or {}
    d = refs.get(name) or refs.get("chain")
    return Reference.from_dict(d) if d else None


def calibration_flat_level(calibration: dict | None, ref: Reference | None, band) -> float:
    """The pedal's own level at its calibration preset (C4) relative to the normalising reference,
    in dB (median over ``band``); 0 if there is no C4 reference.

    A model's expected curve is 0 dB at the flat setting, but a real pedal is not: the GE-7 loses
    about 1.4 dB when engaged at flat. "Level vs calibration" compares against this, not 0 dB.
    """
    refs = (calibration or {}).get("references") or {}
    if ref is None or "engaged_flat" not in refs:
        return 0.0
    flat = Reference.from_dict(refs["engaged_flat"])
    f = np.geomspace(band[0], band[1], 64)
    return float(np.median(flat.db_at(f) - ref.db_at(f)))


def measure_response(
    dry: np.ndarray, wet: np.ndarray, manifest: Manifest, sr: int, band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Response from the broadband sections only (sweeps + pink noise), 1/6-octave smoothed on a log
    grid.
    """
    secs = manifest.sections_of_type_multi(list(FR_TYPES)) or manifest.sections
    d = np.concatenate([dry[s.start_sample : s.end_sample] for s in secs])
    w = np.concatenate([wet[s.start_sample : s.end_sample] for s in secs])
    f, h_db, _ = transfer_function(d, w, sr, nperseg=min(16384, len(d)))
    grid = log_grid(band[0], band[1])
    return grid, smooth_octave(f, h_db, grid)


def check_take(
    take: Take,
    dry: np.ndarray,
    manifest: Manifest,
    *,
    pedal: str,
    controls: dict[str, float] | None,
    calibration: dict | None,
    qc_limits: dict,
    report: Callable[[str], None] | None = None,
) -> tuple[QCReport, np.ndarray]:
    """Run every check on one take and return the report and the aligned return (the "wet" WAV).

    Args:
        take: The recorded take.
        dry: The dry signal that was played (before the send level), shape (signal_len,).
        manifest: The dry signal's manifest (section positions).
        pedal: Profile pedal id, e.g. "boss_ge7".
        controls: Control values for this preset (for the expected response), or None.
        calibration: Session calibration dict ({"loopback": ..., "chain": ...}) or None.
        qc_limits: The profile's "qc" block.
    """
    sr = take.sr
    pre = take.pre_roll
    rep = QCReport()
    items = rep.items

    cal_chain = (calibration or {}).get("chain")
    cal_loop = (calibration or {}).get("loopback")

    first = (manifest.sections_of_type("log_sine_sweep") or manifest.sections)[0]
    w0 = pre + first.start_sample
    w1 = pre + min(first.end_sample, first.start_sample + 10 * sr)
    say = report or (lambda _msg: None)
    say("Aligning: measuring the loopback latency")
    rep.xcorr_lag = find_lag(take.play[:, 0], take.rec[:, 0], sr, w0, w1)
    rep.loop_lag = find_lag(take.play[:, 1], take.rec[:, 1], sr, w0, w1)
    # Align on the interface latency, not on the correlation peak: an EQ boost or a tone stack adds
    # group delay that belongs to the pedal, and the correlation peak would cut it out of the
    # training data. Calibration (pedal at its neutral preset) gives the chain's fixed offset
    # relative to the loopback.
    chain_ref = ((calibration or {}).get("references") or {}).get("chain") or {}
    if chain_ref.get("return_lag_samples") is not None:
        # Both inputs of the same C2 capture: the chain's own delay, without stream-open jitter.
        chain_offset = chain_ref["return_lag_samples"] - chain_ref["lag_samples"]
    elif cal_chain and cal_loop:
        chain_offset = cal_chain["lag_samples"] - cal_loop["lag_samples"]
    else:
        chain_offset = None
    rep.lag = rep.loop_lag + chain_offset if chain_offset is not None else rep.xcorr_lag
    wet = aligned(take.rec[:, 0], pre, rep.lag, take.signal_len)

    items.append(
        QCItem(
            "Dropouts",
            "ok" if take.xruns == 0 else "fail",
            str(take.xruns),
            ""
            if take.xruns == 0
            else "Audio blocks were lost; retake (raise block size if it repeats).",
        )
    )
    n_blocks, n_lost, first_lost = loopback_integrity(take, rep.loop_lag)
    items.append(
        QCItem(
            "Audio integrity (loopback)",
            "ok" if n_lost == 0 else "fail",
            f"{n_lost} of {n_blocks} blocks lost",
            ""
            if n_lost == 0
            else f"Audio went missing (first at {first_lost:.1f} s) although the driver reported "
            "no dropout: the stream did not keep up. Retake; if it repeats, report it.",
        )
    )
    if take.aborted:
        items.append(QCItem("Complete", "fail", "aborted", "The take was stopped early."))
    level = rms_dbfs(wet)
    items.append(
        QCItem(
            "Return present",
            "ok" if level > PRESENT_DBFS else "fail",
            f"{level:.1f} dBFS RMS",
            "" if level > PRESENT_DBFS else "Almost nothing came back on In 1: pedal on? cables?",
        )
    )
    pk = peak_dbfs(take.rec[:, 0])
    st = "ok" if pk < CLIP_WARN_DBFS else ("warn" if pk < CLIP_FAIL_DBFS else "fail")
    items.append(
        QCItem(
            "Return peak",
            st,
            f"{pk:+.1f} dBFS",
            "" if st == "ok" else "Close to clipping: lower In 1 gain and recalibrate.",
        )
    )

    ms = rep.lag / sr * 1e3
    if chain_offset is not None:
        items.append(
            QCItem(
                "Alignment",
                "ok",
                f"{rep.lag} samples ({ms:.2f} ms)",
                f"Loopback {rep.loop_lag} + chain offset {chain_offset:+d} from calibration.",
            )
        )
        d = rep.loop_lag - cal_loop["lag_samples"]
        st = "ok" if abs(d) <= LAG_TOLERANCE_SAMPLES else "warn"
        items.append(
            QCItem(
                "Interface latency",
                st,
                f"{rep.loop_lag} samples",
                ""
                if st == "ok"
                else f"{d:+d} samples vs calibration: the driver latency moved "
                "(buffer size changed?).",
            )
        )
        pedal_delay = rep.xcorr_lag - rep.lag
        items.append(
            QCItem(
                "Pedal delay",
                "ok",
                f"{pedal_delay:+d} samples ({pedal_delay / sr * 1e3:+.2f} ms)",
                "Correlation peak minus alignment: the pedal's own group delay, kept in the data.",
            )
        )
    else:
        items.append(
            QCItem(
                "Alignment",
                "warn",
                f"{rep.lag} samples ({ms:.2f} ms)",
                "No calibration: aligned on the correlation peak, which also removes the pedal's "
                "own "
                "delay. Calibrate for loopback-based alignment.",
            )
        )

    noise = rms_dbfs(take.rec[int(0.1 * sr) : pre, 0])
    st = "ok" if noise < NOISE_OK_DBFS else ("warn" if noise < NOISE_FAIL_DBFS else "fail")
    items.append(
        QCItem(
            "Noise floor",
            st,
            f"{noise:.1f} dBFS RMS",
            "" if st == "ok" else "Hum or hiss on the return: ground lift, power supply, cables.",
        )
    )

    band = tuple(qc_limits.get("fr_band_hz", (100.0, 8000.0)))
    # Measure over the whole audio band for display; score only inside the profile's band.
    say("Measuring the frequency response")
    grid, measured = measure_response(
        dry, wet, manifest, sr, (DISPLAY_BAND_HZ[0], min(DISPLAY_BAND_HZ[1], 0.95 * sr / 2))
    )
    measured = measured - take.send_level_db
    chain_gain = cal_chain["gain_db"] if cal_chain else None
    ref = calibration_reference(calibration, qc_limits.get("reference", "chain"))
    normalised = None
    if ref is not None:
        measured = measured - ref.db_at(grid)
        normalised = ref.name
        items.append(
            QCItem(
                "Normalised by",
                "ok",
                f"{ref.name} reference",
                REFERENCE_NOTES.get(ref.name, ""),
            )
        )
    elif chain_gain is not None:
        measured = measured - chain_gain
        normalised = "chain_gain"
        items.append(
            QCItem(
                "Normalised by",
                "warn",
                "chain gain at 1 kHz",
                "No reference curves, so the chain's own frequency response is still in this "
                "curve. Run the guided calibration (C1-C4).",
            )
        )
    expected = expected_response(pedal, controls or {}, sr, grid) if controls is not None else None
    if expected is not None:
        in_band = (grid >= band[0]) & (grid <= band[1])
        err = (measured - expected)[in_band]
        offset = float(np.median(err))
        shape = float(np.sqrt(np.mean((err - offset) ** 2)))
        ok_db, warn_db = qc_limits.get("fr_ok_db", 1.0), qc_limits.get("fr_warn_db", 2.0)
        st = "ok" if shape < ok_db else ("warn" if shape < warn_db else "fail")
        items.append(
            QCItem(
                "Response vs model",
                st,
                f"{shape:.2f} dB RMS",
                f"{band[0]:.0f}–{band[1]:.0f} Hz after removing a {offset:+.2f} dB offset."
                + ("" if st == "ok" else " Are the controls set as shown?"),
            )
        )
        if normalised is not None:
            flat_level = calibration_flat_level(calibration, ref, band)
            offset -= flat_level
            st = (
                "ok"
                if abs(offset) < LEVEL_OFFSET_WARN_DB
                else ("warn" if abs(offset) < LEVEL_OFFSET_FAIL_DB else "fail")
            )
            items.append(
                QCItem(
                    "Level vs calibration",
                    st,
                    f"{offset:+.2f} dB",
                    f"Relative to the pedal at its calibration preset (C4: {flat_level:+.2f} dB)."
                    + (
                        ""
                        if st == "ok"
                        else " The level moved since calibration (LEVEL slider? a knob bumped?)."
                    ),
                )
            )
        else:
            measured = measured - offset
    rep.response = FrequencyResponse(grid, measured, expected, normalised)
    return rep, wet
