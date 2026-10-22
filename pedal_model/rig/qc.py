"""Quality checks on a finished take, and the expected-vs-measured frequency response for linear
pedals.

A QC report is a plain dict, the same as the session stores it (plus the response arrays):

    {"verdict": "ok" | "warn" | "fail",
     "lag_samples": samples removed when aligning the return: this take's loopback (interface)
                    latency plus the chain's fixed analog offset from calibration. The pedal's own
                    delay is NOT removed: it is part of what the model has to learn.
     "loop_lag_samples": this take's loopback latency,
     "xcorr_lag_samples": lag at the send/return correlation peak (includes the pedal's delay),
     "items": [{"name", "status", "value", "detail"}, ...],
     "response": {"freqs_hz", "measured_db", "expected_db", "reference"} or None}
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.signal import spectrogram, welch

from pedal_model.rig.analysis import (
    aligned,
    find_lag,
    frequency_response,
    log_grid,
    peak_dbfs,
    rms_dbfs,
    smooth_octave,
)
from pedal_model.rig.calibrate import ref_db_at, worst_status
from pedal_model.rig.engine import CaptureEngine, Take
from pedal_model.signals.manifest import Manifest, Section

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
INTEGRITY_RANGE_DB = 40.0  # only blocks within this of the loudest send block are judged
# A loopback block this far below the take's typical ratio lost its audio. Real dropouts measured
# 20-48 dB deep (sessions 1238/1307); pink-noise blocks heavy in sub-audio energy dip up to ~7 dB
# on both inputs because the chain rolls off the lowest frequencies (2026-10-20).
INTEGRITY_DROP_DB = 15.0
# Linearity: the quietest and loudest sweep of a take must give the same response (dB difference,
# 1/6 octave, over the profile's band). 0.25 dB: flat2 varied that much across levels
# (docs/ge7_plan.md); frown's 2.4 dB at 800 Hz was clipping in the return path (2026-10-21).
LINEARITY_WARN_DB = 0.25
LINEARITY_FAIL_DB = 1.0
# Signal-to-noise of the quietest sweep (the return while the sweep passes each frequency over the
# pre-roll noise, 1/6 octave; see section_snr_db).
# First guesses (2026-10-22): revisit once C2 has measured the real return noise floor.
SNR_WARN_DB = 30.0
SNR_FAIL_DB = 15.0
NOISE_NPERSEG = 4096  # Welch segment for the noise floor: the pre-roll gives 0.4 s of silence
# A take that fails the linearity check is played again this much quieter, at most this often: it
# covers limits C2 cannot see (the pedal's own output stage on big boosts, the return input above
# what C2 reached).
AUTO_RETAKE_STEP_DB = 6.0
AUTO_RETAKES = 2
# Fitted GE-7 block models (plan: "Storage and use in the GUI"), one file per fit, never
# overwritten: ge7_pedalode_v<N>.json. The newest one gives the expected curve; without one,
# Ge7Sim does.
GE7_MODELS_DIR = Path(__file__).resolve().parent / "models" / "ge7"
REFERENCE_NOTES = {
    "chain": "Take ÷ the chain alone (C2): the pedal's whole response, buffers included.",
    "bypass": "Take ÷ the bypassed pedal (C3): what the engaged circuit adds (plan decision D1).",
    "engaged_flat": "Take ÷ the engaged pedal at its calibration preset (C4).",
}


def item(name: str, status: str, value: str, detail: str = "") -> dict:
    """One QC line: status is "ok", "warn" or "fail"."""
    return {"name": name, "status": status, "value": value, "detail": detail}


def expected_model(pedal: str) -> dict | None:
    """Which model gives a pedal's expected curve: the newest fitted parameter file, else the sim.

    Returns:
        dict: name, fitted (True for a parameter file fitted to this pedal), file (or None),
        params (the file's block parameters, or None); None if the pedal has no linear model.
    """
    if pedal != "boss_ge7":
        return None
    files = sorted(
        GE7_MODELS_DIR.glob("ge7_pedalode_v*.json"),
        key=lambda p: int(p.stem.rsplit("_v", 1)[1]),
    )
    if not files:
        return {"name": "Ge7Sim", "fitted": False, "file": None, "params": None}
    doc = json.loads(files[-1].read_text(encoding="utf-8"))
    return {"name": files[-1].stem, "fitted": True, "file": files[-1], "params": doc["params"]}


def expected_response(
    pedal: str, controls: dict[str, float], sr: int, grid: np.ndarray
) -> np.ndarray | None:
    """Model response of the pedal at these settings in dB on ``grid``, or None without a model.

    A fitted GE-7 block model gives engaged ÷ bypassed, 20·log10|H_ratio(f; sliders)|; Ge7Sim
    (``expected_model`` says which) gives its own digital response at ``sr``.
    """
    model = expected_model(pedal)
    if model is None:
        return None
    if model["fitted"]:
        import torch

        from pedal_model.ode import ge7_model

        p = ge7_model.params_from_json(model["params"])
        h = ge7_model.ratio_response(
            p, torch.as_tensor(grid, dtype=torch.float64), ge7_model.sliders_tensor([controls])
        )
        return 20 * np.log10(np.abs(h[0].numpy()) + 1e-12)
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


def calibration_reference(calibration: dict | None, name: str) -> dict | None:
    """The named reference curve of a calibration, else "chain", else None."""
    refs = (calibration or {}).get("references") or {}
    return refs.get(name) or refs.get("chain")


def calibration_flat_level(calibration: dict | None, reference: str | None, band) -> float:
    """The pedal's own level at its calibration preset (C4) relative to the normalising reference,
    in dB (median over ``band``); 0 if either curve is missing.

    A model's expected curve is 0 dB at the flat setting, but a real pedal is not: the GE-7 loses
    about 1.4 dB when engaged at flat. "Level vs calibration" compares against this, not 0 dB.
    """
    refs = (calibration or {}).get("references") or {}
    if reference not in refs or "engaged_flat" not in refs:
        return 0.0
    f = np.geomspace(band[0], band[1], 64)
    return float(np.median(ref_db_at(refs["engaged_flat"], f) - ref_db_at(refs[reference], f)))


def measure_response(
    dry: np.ndarray,
    wet: np.ndarray,
    manifest: Manifest,
    sr: int,
    band: tuple[float, float],
    send_level_db: float,
    calibration: dict | None,
    reference: str,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """The pedal's response from the broadband sections (sweeps + pink noise), normalised.

        measured_dB(f) = 20·log10|H(f)| − send level − reference_dB(f)

    H = wet ÷ dry (Welch H1, 1/6-octave smoothed on a log grid). The send level is the digital
    gain the take was played with (per take, since auto-level). The reference is the calibration
    curve named ``reference`` ("chain", "bypass", "engaged_flat"; falls back to "chain"); a
    calibration without curves only has the chain's 1 kHz gain, which leaves the chain's own
    tilt in the curve.

    Args:
        dry: The dry signal (before the send level), shape (n,).
        wet: The aligned return, shape (n,).
        band: Frequency range of the grid in Hz.
        send_level_db: The take's send level in dB.
        calibration: Session calibration dict, or None.
        reference: Name of the reference curve to divide by.

    Returns:
        (grid in Hz, measured dB, what it was divided by: a reference name, "chain_gain" or None).
    """
    secs = manifest.sections_of_type_multi(list(FR_TYPES)) or manifest.sections
    d = np.concatenate([dry[s.start_sample : s.end_sample] for s in secs])
    w = np.concatenate([wet[s.start_sample : s.end_sample] for s in secs])
    f, h, _ = frequency_response(d, w, sr)
    grid = log_grid(band[0], band[1])
    db = smooth_octave(f, 20 * np.log10(np.abs(h) + 1e-12), grid) - send_level_db
    ref = calibration_reference(calibration, reference)
    if ref is not None:
        return grid, db - ref_db_at(ref, grid), ref["name"]
    chain = (calibration or {}).get("chain")
    if chain and chain.get("gain_db") is not None:
        return grid, db - chain["gain_db"], "chain_gain"
    return grid, db, None


def sweep_section_db(
    dry: np.ndarray, wet: np.ndarray, sr: int, section: Section, grid: np.ndarray
) -> np.ndarray:
    """Gain of one sweep section, 20·log10|wet ÷ dry| (Welch H1), 1/6-octave smoothed on ``grid``.

    The send level and the chain are in it, but they cancel when two sections of one take are
    compared, which is all it is used for.
    """
    a, b = section.start_sample, section.end_sample
    f, h, _ = frequency_response(dry[a:b], wet[a:b], sr)
    k = f > 0
    return smooth_octave(f[k], 20 * np.log10(np.abs(h[k]) + 1e-12), grid)


def section_snr_db(
    wet: np.ndarray, noise: np.ndarray, sr: int, section: Section, grid: np.ndarray
) -> np.ndarray:
    """Signal-to-noise of a sweep section of the return, per frequency, in dB on ``grid``:

        SNR(f) = 10·log10( max_t P_return(f, t) / P_noise(f) ),   1/6-octave smoothed

    P_return(f, t) is the return's spectrogram and P_noise the pre-roll's Welch spectrum, with the
    same Hann window, so the ratio is the return over the noise while the sweep passes f. (A
    sweep is at each frequency only for a moment: averaging its power over the whole section
    would understate the SNR by ~10·log10(number of windows), ~25 dB for a 20 s sweep. The
    response estimate only weighs the windows where the sweep is, so it sees the peak SNR or
    better.)
    """
    seg = wet[section.start_sample : section.end_sample].astype(np.float64)
    n = min(NOISE_NPERSEG, len(noise), len(seg))
    f, _, s_sig = spectrogram(seg, sr, window="hann", nperseg=n)
    p_sig = s_sig.max(axis=1)
    _, p_noise = welch(noise.astype(np.float64), sr, window="hann", nperseg=n)
    k = f > 0
    sig_db = smooth_octave(f[k], 10 * np.log10(p_sig[k] + 1e-300), grid)
    noise_db = smooth_octave(f[k], 10 * np.log10(p_noise[k] + 1e-300), grid)
    return sig_db - noise_db


def capture_checked(
    engine: CaptureEngine,
    dry: np.ndarray,
    manifest: Manifest,
    sr: int,
    send_level_db: float,
    *,
    retake: bool,
    report: Callable[[str], None] | None = None,
    **check: object,
) -> tuple[Take, dict | None, np.ndarray | None, list[dict]]:
    """Play a take and check it; if ``retake`` and it is not linear ("Linearity (sweeps)" fails),
    play it again AUTO_RETAKE_STEP_DB quieter, at most AUTO_RETAKES times.

    Args:
        dry: The dry signal, shape (n,).
        send_level_db: The first attempt's send level (auto-level's plan).
        retake: Allow the quieter retakes (the profile's auto-level is on).
        check: The rest of ``check_take``'s keyword arguments.

    Returns:
        (the kept take, its report (None if aborted), its aligned return, the retakes made:
        [{send_level_db, linearity}] of each attempt that failed).
    """
    say = report or (lambda _msg: None)
    retakes: list[dict] = []
    send = send_level_db
    while True:
        say("capture")
        take = engine.run(dry, sr, send_level_db=send)
        if take.aborted:
            return take, None, None, retakes
        rep, wet = check_take(take, dry, manifest, report=report, **check)
        lin = next((i for i in rep["items"] if i["name"] == "Linearity (sweeps)"), None)
        if not retake or lin is None or lin["status"] != "fail" or len(retakes) >= AUTO_RETAKES:
            break
        retakes.append({"send_level_db": send, "linearity": lin["value"]})
        send -= AUTO_RETAKE_STEP_DB
        say(f"Not linear ({lin['value']}): playing it again {AUTO_RETAKE_STEP_DB:.0f} dB quieter")
    if retakes:
        tried = ", ".join(f"{r['send_level_db']:+.1f} dB ({r['linearity']})" for r in retakes)
        rep["items"].append(
            item(
                "Auto-level retakes",
                "warn",
                f"{len(retakes)}, kept {send:+.1f} dB",
                f"Not linear at {tried}. A limit C2 did not see (the pedal's own output on a big "
                "boost?).",
            )
        )
    return take, rep, wet, retakes


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
) -> tuple[dict, np.ndarray]:
    """Run every check on one take and return the report and the aligned return (the "wet" WAV).

    Args:
        take: The recorded take.
        dry: The dry signal that was played (before the send level), shape (signal_len,).
        manifest: The dry signal's manifest (section positions).
        pedal: Profile pedal id, e.g. "boss_ge7".
        controls: Control values for this preset (for the expected response), or None.
        calibration: Session calibration dict ({"loopback": ..., "chain": ...}) or None.
        qc_limits: The profile's "qc" block.

    Returns:
        (the report dict described in the module docstring, the aligned return).
    """
    sr = take.sr
    pre = take.pre_roll
    items: list[dict] = []

    cal_chain = (calibration or {}).get("chain")
    cal_loop = (calibration or {}).get("loopback")

    # The lags come from the whole first sweep. Its first 10 s (train_signal_v2: 20-630 Hz only)
    # gave a broad, low-frequency correlation peak: the loopback lag read 3 samples short on every
    # train take and the return lag wandered -7..+11 (session 2026-10-21_0237, found 2026-10-22).
    first = (manifest.sections_of_type("log_sine_sweep") or manifest.sections)[0]
    w0 = pre + first.start_sample
    w1 = pre + first.end_sample
    say = report or (lambda _msg: None)
    say("Aligning: measuring the loopback latency")
    xcorr_lag = find_lag(take.play[:, 0], take.rec[:, 0], sr, w0, w1)
    loop_lag = find_lag(take.play[:, 1], take.rec[:, 1], sr, w0, w1)
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
    lag = loop_lag + chain_offset if chain_offset is not None else xcorr_lag
    wet = aligned(take.rec[:, 0], pre, lag, take.signal_len)

    if take.xruns == 0:
        items.append(item("Dropouts", "ok", "0"))
    else:
        items.append(
            item(
                "Dropouts",
                "fail",
                str(take.xruns),
                "Audio blocks were lost; retake (raise block size if it repeats).",
            )
        )
    n_blocks, n_lost, first_lost = loopback_integrity(take, loop_lag)
    items.append(
        item(
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
        items.append(item("Complete", "fail", "aborted", "The take was stopped early."))
    level = rms_dbfs(wet)
    items.append(
        item(
            "Return present",
            "ok" if level > PRESENT_DBFS else "fail",
            f"{level:.1f} dBFS RMS",
            "" if level > PRESENT_DBFS else "Almost nothing came back on In 1: pedal on? cables?",
        )
    )
    pk = peak_dbfs(take.rec[:, 0])
    ceiling = (cal_chain or {}).get("return_ceiling_dbfs")
    if ceiling is not None and pk > ceiling - 1.0:
        items.append(
            item(
                "Return vs chain ceiling",
                "fail" if pk > ceiling + 1.0 else "warn",
                f"{pk:+.1f} dBFS (ceiling {ceiling:.0f})",
                "The return reached the level where the DI path compresses (C2): this take partly "
                "describes the DI boxes, not the pedal. Retake at a lower send level.",
            )
        )
    send_ceiling = (cal_chain or {}).get("send_ceiling_dbfs")
    if send_ceiling is not None and (cal_chain or {}).get("limit_kind") in ("send", "unknown"):
        sent = peak_dbfs(take.play[:, 0])
        if sent > send_ceiling - 1.0:
            items.append(
                item(
                    "Send vs chain ceiling",
                    "fail" if sent > send_ceiling + 1.0 else "warn",
                    f"{sent:+.1f} dBFS sent (ceiling {send_ceiling:.0f})",
                    "The send reached the level where the re-amp side compresses (C2): this take "
                    "partly describes the rig, not the pedal. Retake at a lower send level.",
                )
            )
    st = "ok" if pk < CLIP_WARN_DBFS else ("warn" if pk < CLIP_FAIL_DBFS else "fail")
    items.append(
        item(
            "Return peak",
            st,
            f"{pk:+.1f} dBFS",
            "" if st == "ok" else "Close to clipping: lower In 1 gain and recalibrate.",
        )
    )

    ms = lag / sr * 1e3
    if chain_offset is not None:
        items.append(
            item(
                "Alignment",
                "ok",
                f"{lag} samples ({ms:.2f} ms)",
                f"Loopback {loop_lag} + chain offset {chain_offset:+d} from calibration.",
            )
        )
        d = loop_lag - cal_loop["lag_samples"] if cal_loop else 0
        st = "ok" if abs(d) <= LAG_TOLERANCE_SAMPLES else "warn"
        items.append(
            item(
                "Interface latency",
                st,
                f"{loop_lag} samples",
                ""
                if st == "ok"
                else f"{d:+d} samples vs calibration: the driver latency moved "
                "(buffer size changed?).",
            )
        )
        pedal_delay = xcorr_lag - lag
        items.append(
            item(
                "Pedal delay",
                "ok",
                f"{pedal_delay:+d} samples ({pedal_delay / sr * 1e3:+.2f} ms)",
                "Correlation peak minus alignment: the pedal's own group delay, kept in the data.",
            )
        )
    else:
        items.append(
            item(
                "Alignment",
                "warn",
                f"{lag} samples ({ms:.2f} ms)",
                "No calibration: aligned on the correlation peak, which also removes the pedal's "
                "own delay. Calibrate for loopback-based alignment.",
            )
        )

    noise_window = take.rec[int(0.1 * sr) : pre, 0]
    noise = rms_dbfs(noise_window)
    st = "ok" if noise < NOISE_OK_DBFS else ("warn" if noise < NOISE_FAIL_DBFS else "fail")
    items.append(
        item(
            "Noise floor",
            st,
            f"{noise:.1f} dBFS RMS",
            "" if st == "ok" else "Hum or hiss on the return: ground lift, power supply, cables.",
        )
    )

    band = tuple(qc_limits.get("fr_band_hz", (100.0, 8000.0)))
    sweeps = sorted(manifest.sections_of_type("log_sine_sweep"), key=lambda s: s.amplitude_dbfs)
    score_grid = log_grid(band[0], band[1])
    if len(sweeps) >= 2 and sweeps[-1].amplitude_dbfs > sweeps[0].amplitude_dbfs:
        quiet, loud = sweeps[0], sweeps[-1]
        say("Checking linearity (quiet vs loud sweep)")
        diff = np.abs(
            sweep_section_db(dry, wet, sr, loud, score_grid)
            - sweep_section_db(dry, wet, sr, quiet, score_grid)
        )
        i = int(np.argmax(diff))
        lin = float(diff[i])
        graded = qc_limits.get("expect_linear", False)
        if not graded or lin < LINEARITY_WARN_DB:
            st = "ok"
        else:
            st = "warn" if lin < LINEARITY_FAIL_DB else "fail"
        detail = (
            f"Sweep at {loud.amplitude_dbfs:+.0f} vs {quiet.amplitude_dbfs:+.0f} dBFS, largest "
            f"difference {band[0]:.0f}–{band[1]:.0f} Hz."
        )
        if st != "ok":
            detail += (
                " Something compresses at the loud level: the return path (C2's ceiling) or the "
                "pedal. Retake at a lower send level (auto-level)."
            )
        elif not graded:
            detail += " Not graded: this pedal is not expected to be linear."
        value = f"{lin:.2f} dB at {score_grid[i]:.0f} Hz"
        items.append(item("Linearity (sweeps)", st, value, detail))
    if sweeps and len(noise_window) > 0:
        quiet = sweeps[0]
        snr = section_snr_db(wet, noise_window, sr, quiet, score_grid)
        i = int(np.argmin(snr))
        st = "ok" if snr[i] >= SNR_WARN_DB else ("warn" if snr[i] >= SNR_FAIL_DB else "fail")
        detail = (
            f"The {quiet.amplitude_dbfs:+.0f} dBFS sweep over the pre-roll noise, worst "
            f"frequency {band[0]:.0f}–{band[1]:.0f} Hz."
        )
        if st != "ok":
            detail += (
                " The quiet section is close to the noise: a low send level (auto-level) or low "
                "return gain. It adds little to the fit."
            )
        value = f"{snr[i]:.0f} dB at {score_grid[i]:.0f} Hz"
        items.append(item("Signal-to-noise (quietest sweep)", st, value, detail))

    # Measure over the whole audio band for display; score only inside the profile's band.
    say("Measuring the frequency response")
    grid, measured, normalised = measure_response(
        dry,
        wet,
        manifest,
        sr,
        (DISPLAY_BAND_HZ[0], min(DISPLAY_BAND_HZ[1], 0.95 * sr / 2)),
        take.send_level_db,
        calibration,
        qc_limits.get("reference", "chain"),
    )
    if normalised == "chain_gain":
        items.append(
            item(
                "Normalised by",
                "warn",
                "chain gain at 1 kHz",
                "No reference curves, so the chain's own frequency response is still in this "
                "curve. Run the guided calibration (C1-C4).",
            )
        )
    elif normalised is not None:
        items.append(
            item(
                "Normalised by",
                "ok",
                f"{normalised} reference",
                REFERENCE_NOTES.get(normalised, ""),
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
        model = expected_model(pedal)
        # A model not fitted to this pedal (Ge7Sim is off by 1-2 dB on real presets) can only
        # warn: the verdict must reflect the capture, not the model's error.
        capped = not model["fitted"] and st == "fail"
        if capped:
            st = "warn"
        detail = (
            f"Model {model['name']}, {band[0]:.0f}–{band[1]:.0f} Hz after removing a "
            f"{offset:+.2f} dB offset."
        )
        if st != "ok":
            detail += " Are the controls set as shown?"
        if not model["fitted"]:
            detail += " The model is not fitted to this pedal: graded warn at most."
        items.append(item("Response vs model", st, f"{shape:.2f} dB RMS", detail))
        if normalised is not None:
            flat_level = calibration_flat_level(calibration, normalised, band)
            offset -= flat_level
            st = (
                "ok"
                if abs(offset) < LEVEL_OFFSET_WARN_DB
                else ("warn" if abs(offset) < LEVEL_OFFSET_FAIL_DB else "fail")
            )
            if not model["fitted"] and st == "fail":
                st = "warn"  # Ge7Sim's LEVEL law is extrapolated, not measured
            items.append(
                item(
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
    rep = {
        "verdict": worst_status(items),
        "lag_samples": int(lag),
        "loop_lag_samples": int(loop_lag),
        "xcorr_lag_samples": int(xcorr_lag),
        "items": items,
        "response": {
            "freqs_hz": grid,
            "measured_db": measured,
            "expected_db": expected,
            "reference": normalised,
        },
    }
    return rep, wet
