"""Fitting data from a capture session: complex engaged ÷ bypassed responses, one per take.

Rung A of the GE-7 plan fits the block model to H(jω) measured at many slider settings. This module
turns a session's raw captures into those curves, recomputing every response from the raw FLACs
with one estimator, so takes and references are measured the same way:

- A take's played signal is rebuilt (dry × send level); its return is aligned on the take's own
  loopback lag, so the interface latency cancels per take and the chain's and pedal's delays stay
  in the phase.
- Each log sweep gets one FFT over the section plus a short tail (the response rings into the
  silence after it): S_xx = |X|², S_yy = |Y|², S_xy = X*·Y. For a linear system H = S_xy / S_xx is
  exact in every bin; averaging the bins' H over a 1/12-octave band then reduces the noise. A
  sweep only counts inside the range it sweeps (its first sixth of an octave and last 2.5 % are
  left out); grid points no sweep covers are None in the JSON. Pink noise is not used: on
  session 2026-10-21_0237 it read ±0.2 dB noisy below 200 Hz even at −24 dBFS, and the loud pink
  sections compressed the rig's low end (−1.4 dB at 30 Hz), while the three sweep levels agreed
  within 0.01 dB from 50 Hz up.
- Before averaging over a band, the bulk delay τ (return correlation lag minus loopback lag) is
  removed, S_xy·e^{+jωτ}, and put back at the band centre: a 5-sample delay turns the phase by
  ~0.7 rad across a 1/6-octave band at 10 kHz, which would cost ~0.2 dB of magnitude.
- A sweep whose response differs from the take's quietest sweep by more than LINEARITY_WARN_DB
  anywhere in CHECK_BAND_HZ is dropped and listed (clean sweeps agree within 0.01 dB).

Engaged ÷ bypassed = H_take / H_bypass divides out the chain (decision D1); bypassed ÷ chain,
from the C3 and C2 references, is kept for fitting the input and output blocks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from pedal_model.rig.analysis import find_lag
from pedal_model.rig.calibrate import (
    GAP_S,
    LAG_SEARCH_S,
    REFERENCE_F_LO,
    SWEPT_FRACTION,
    reference_signal,
)
from pedal_model.rig.config import code_version
from pedal_model.rig.engine import PRE_ROLL_S, Take
from pedal_model.rig.qc import LINEARITY_WARN_DB
from pedal_model.signals.manifest import Manifest

GRID_HZ = np.geomspace(20.0, 19_000.0, 200)  # the fitter's grid (ge7_fitter.log_grid)
BAND_FRACTION = 1 / 12  # octave width of the band averaged around each grid frequency
TAIL_S = 0.2  # silence after a section that still holds its response
CHECK_BAND_HZ = (50.0, 10_000.0)  # where sweeps are compared with the quietest sweep
BIN_HZ = 0.05  # finest FFT bin spacing: ~25 bins in the 1/12-octave band at 20 Hz
SWEEP_EDGE_OCT = 1 / 6  # the start of a sweep (under its fade) is left out by this much


def rebuild_take(raw_path: Path, signal: np.ndarray, send_level_db: float) -> Take:
    """A Take from a raw capture: the played signal is rebuilt (both outputs, as route "both").

    Args:
        raw_path: The raw FLAC (return, loopback), as recorded, unaligned.
        signal: The dry signal that was played, shape (n,).
        send_level_db: The take's send level in dB.
    """
    rec, sr = sf.read(str(raw_path), dtype="float32")
    pre = int(round(PRE_ROLL_S * sr))
    play = np.zeros_like(rec)
    gain = np.float32(10 ** (send_level_db / 20))
    play[pre : pre + len(signal), 0] = signal * gain
    play[pre : pre + len(signal), 1] = signal * gain
    return Take(sr, play, rec, pre, len(signal), send_level_db, 0, False, 0.0)


def take_lags(take: Take, start: int, stop: int) -> tuple[int, int]:
    """(loopback lag, return lag) in samples, from the window [start, stop) of the signal."""
    a, b = take.pre_roll + start, take.pre_roll + stop
    loop = find_lag(take.play[:, 1], take.rec[:, 1], take.sr, a, b, LAG_SEARCH_S)
    ret = find_lag(take.play[:, 0], take.rec[:, 0], take.sr, a, b, LAG_SEARCH_S)
    return loop, ret


def band_sums(
    take: Take,
    start: int,
    stop: int,
    loop_lag: int,
    delay: int,
    n_fft: int,
    swept_hz: tuple[float, float],
) -> np.ndarray:
    """One sweep's band sums on GRID_HZ, additive across sweeps. Rows:

        E = Σ S_xx,   Σ S_yy,   Σ S_xy·e^{+jωτ},   E · mean_k(H_k·e^{+jωτ}),   H_k = S_xy,k / S_xx,k

    The band's H is the plain mean of the bins' H_k, not Σ S_xy / Σ S_xx: that ratio weights each
    bin by |X_k|², and a sweep's spectrum ripples (most at low frequency, where it moves fast), so
    where H slopes it lands off the band centre (0.06 dB at 60 Hz on a 1 s test sweep through a
    bass boost). Sweeps combine weighted by their band energy E.

    x is the played signal and y the return (aligned on ``loop_lag``), both over [start, stop)
    plus TAIL_S. τ = ``delay`` samples is removed before averaging. Grid points outside
    ``swept_hz`` (the range the sweep covers) stay 0.

    Returns:
        Complex array, shape (4, len(GRID_HZ)).
    """
    sr, pre = take.sr, take.pre_roll
    tail = int(TAIL_S * sr)
    m = min(stop - start + tail, len(take.rec) - pre - loop_lag - start)
    # x covers the same window as y, tail included: back-to-back sections (the val sweeps) put
    # the next sweep's start in the tail, and leaving it out of x read as low-frequency noise
    # (coherence 0.09 at 36 Hz, 0.3-0.6 dB errors near 55 Hz).
    x = take.play[pre + start : pre + start + m, 0].astype(np.float64)
    y = take.rec[pre + loop_lag + start : pre + loop_lag + start + m, 0].astype(np.float64)
    big_x, big_y = np.fft.rfft(x, n_fft), np.fft.rfft(y, n_fft)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    s_xy = np.conj(big_x) * big_y * np.exp(2j * np.pi * f * delay / sr)
    half = 2 ** (BAND_FRACTION / 2)
    s_xx = np.abs(big_x) ** 2
    out = np.zeros((4, len(GRID_HZ)), dtype=complex)
    for i, fc in enumerate(GRID_HZ):
        if not swept_hz[0] <= fc <= swept_hz[1]:
            continue
        lo, hi = np.searchsorted(f, [fc / half, fc * half])
        if hi <= lo:  # band narrower than one bin (low frequencies): the nearest bin
            lo = int(np.argmin(np.abs(f - fc)))
            hi = lo + 1
        e = np.sum(s_xx[lo:hi])
        out[0, i] = e
        out[1, i] = np.sum(np.abs(big_y[lo:hi]) ** 2)
        out[2, i] = np.sum(s_xy[lo:hi])
        out[3, i] = e * np.mean(s_xy[lo:hi] / np.maximum(s_xx[lo:hi], 1e-30 * e))
    return out


def response_from_sums(sums: np.ndarray, delay: int, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """H (the energy-weighted mean of the sweeps' band means) with the delay put back
    (e^{-jω_c τ}), and the band coherence |Σ S_xy|² / (Σ S_xx · Σ S_yy); NaN where no sweep
    covered the band.
    """
    s_xx, s_yy, s_xy, e_h = sums
    covered = np.real(s_xx) > 0
    h = np.full(len(GRID_HZ), np.nan + 0j)
    coh = np.full(len(GRID_HZ), np.nan)
    h[covered] = (e_h / np.where(covered, s_xx, 1.0))[covered] * np.exp(
        -2j * np.pi * GRID_HZ[covered] * delay / sr
    )
    coh[covered] = np.abs(s_xy[covered]) ** 2 / np.real(s_xx[covered] * s_yy[covered])
    return h, np.clip(coh, 0.0, 1.0)


def n_fft_for(samples: int, sr: int) -> int:
    """FFT length: the window, zero-padded to at most BIN_HZ per bin.

    Zero padding only interpolates the spectrum (H per bin stays exact for a linear system), but
    a band needs many bins for its mean to sit at its centre frequency: with 0.73 Hz bins a
    1/12-octave band at 50 Hz holds 4 bins, off-centre by up to half a bin, which read 0.06 dB
    wrong on a bass boost's slope.
    """
    return 1 << int(np.ceil(np.log2(max(samples, sr / BIN_HZ))))


def swept_range(f_start_hz: float, f_end_hz: float) -> tuple[float, float]:
    """Frequencies a log sweep measures well: past its start (under the fade-in), below its top."""
    return f_start_hz * 2**SWEEP_EDGE_OCT, f_end_hz * SWEPT_FRACTION


def reference_response(take: Take) -> tuple[np.ndarray, np.ndarray, dict]:
    """H and coherence of a reference capture (the 3 s sweep plus its gap), loopback-aligned."""
    n = len(reference_signal(take.sr))
    loop, ret = take_lags(take, 0, n)
    swept = swept_range(REFERENCE_F_LO, min(20_000.0, 0.45 * take.sr))
    sums = band_sums(
        take, 0, n, loop, ret - loop, n_fft_for(n + int(GAP_S * take.sr), take.sr), swept
    )
    h, coh = response_from_sums(sums, ret - loop, take.sr)
    return h, coh, {"loop_lag_samples": int(loop), "delay_samples": int(ret - loop)}


def take_response(take: Take, manifest: Manifest) -> tuple[np.ndarray, np.ndarray, dict]:
    """H and coherence of one take over its linear sweeps, loopback-aligned.

    Returns:
        (H on GRID_HZ, coherence, info: loop_lag_samples, delay_samples, sections_used,
        sections_dropped [{label, deviation_db}]).
    """
    sr = take.sr
    sweeps = sorted(manifest.sections_of_type("log_sine_sweep"), key=lambda s: s.amplitude_dbfs)
    if not sweeps:
        raise ValueError(f"{manifest.path.name}: no log sweep to measure the response")
    first = sweeps[0]
    loop, ret = take_lags(take, first.start_sample, first.end_sample)  # whole sweep: see qc
    delay = ret - loop
    n_fft = n_fft_for(max(s.n_samples for s in sweeps) + int(TAIL_S * sr), sr)
    sums = {}
    for s in sweeps:
        swept = swept_range(s.params.get("f_start_hz", 20.0), s.params.get("f_end_hz", 20_000.0))
        sums[s.label] = band_sums(take, s.start_sample, s.end_sample, loop, delay, n_fft, swept)
    check = (GRID_HZ >= CHECK_BAND_HZ[0]) & (GRID_HZ <= CHECK_BAND_HZ[1])
    quiet_db = 20 * np.log10(np.abs(response_from_sums(sums[first.label], delay, sr)[0]))
    used, dropped = [], []
    for s in sweeps:
        db = 20 * np.log10(np.abs(response_from_sums(sums[s.label], delay, sr)[0]))
        dev = float(np.nanmax(np.abs(db - quiet_db)[check]))
        if dev > LINEARITY_WARN_DB:
            dropped.append({"label": s.label, "deviation_db": round(dev, 3)})
        else:
            used.append(s.label)
    h, coh = response_from_sums(sum(sums[label] for label in used), delay, sr)
    info = {
        "loop_lag_samples": int(loop),
        "delay_samples": int(delay),
        "sections_used": used,
        "sections_dropped": dropped,
    }
    return h, coh, info


def _values(x: np.ndarray, digits: int) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in x]


def _curve(h: np.ndarray, coh: np.ndarray) -> dict:
    return {
        "h_re": _values(h.real, 7),
        "h_im": _values(h.imag, 7),
        "coherence": _values(coh, 5),
    }


def session_dataset(
    root: Path, roles: tuple[str, ...] = ("train", "sweep"), cache: dict | None = None
) -> dict:
    """Every take of a session as engaged ÷ bypassed on GRID_HZ, plus bypassed ÷ chain.

    Args:
        root: The session folder.
        roles: Which takes to include ("train" for model presets, "sweep" for map steps; "val"
            is kept for the time-domain check).
        cache: A dict kept between calls (the analysis GUI's live mode): references and takes
            already computed are reused, so a new capture costs one take.

    Returns:
        dict, as written to JSON: session info, freqs_hz, bypass_over_chain, takes [{preset, kind,
        role, status, verdict, controls, send_level_db, loop_lag_samples, delay_samples,
        sections_used, sections_dropped, h_re, h_im, coherence}]. h is None where no sweep
        covered the frequency.
    """
    data = json.loads((root / "session.json").read_text(encoding="utf-8"))
    sr = int(data["sample_rate"])
    sweep = np.concatenate([reference_signal(sr), np.zeros(int(GAP_S * sr), np.float32)])
    cache = {} if cache is None else cache

    def references(cal_file: str | None) -> dict:
        """chain and bypass (H, coherence) of a calibration file (None: the session's latest).
        A take is divided by the calibration it was recorded with, so recalibrating in a later
        sitting never changes the earlier takes."""
        key = "ref:" + (cal_file or "")
        if key not in cache:
            cal = data.get("calibration") or {}
            if cal_file and (root / cal_file).is_file():
                cal = json.loads((root / cal_file).read_text(encoding="utf-8"))
            refs = cal.get("references") or {}
            if "bypass" not in refs or "chain" not in refs:
                raise ValueError(f"{root.name}: no chain and bypass references ({key or 'latest'})")
            cache[key] = {
                n: reference_response(rebuild_take(root / refs[n]["file"], sweep, 0.0))[:2]
                for n in ("chain", "bypass")
            }
        return cache[key]

    latest = references(None)
    h_bypass, coh_bypass = latest["bypass"]
    h_chain, coh_chain = latest["chain"]

    takes = []
    for name, p in data["presets"].items():
        for t in p.get("takes", []):
            if t["role"] not in roles:
                continue
            m = Manifest(root / data["signals"][t.get("signal", t["role"])])
            send = t.get("send_level_db", data["send_level_db"])
            key = f"take:{name}/{t['raw_wav']}"
            if key not in cache:
                take = rebuild_take(root / name / t["raw_wav"], m.load_all(), send)
                cache[key] = take_response(take, m)
            h, coh, info = cache[key]
            h_ref, coh_ref = references(t.get("calibration_file"))["bypass"]
            with np.errstate(invalid="ignore"):
                curve = _curve(h / h_ref, np.minimum(coh, coh_ref))
            takes.append(
                {
                    "preset": name,
                    "kind": p.get("kind"),
                    "role": t["role"],
                    "status": p.get("status"),
                    "verdict": t.get("verdict"),
                    "controls": p.get("controls"),
                    "send_level_db": send,
                    "calibration_file": t.get("calibration_file"),
                    **info,
                    **curve,
                }
            )
    with np.errstate(invalid="ignore"):  # NaN where no sweep covered a band
        bypass_over_chain = _curve(h_bypass / h_chain, np.minimum(coh_bypass, coh_chain))
    return {
        "dataset": "ge7_engaged_over_bypassed",
        "version": 1,
        "session": root.name,
        "sample_rate": sr,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_version": code_version(),
        "band_fraction_oct": BAND_FRACTION,
        "freqs_hz": [round(float(v), 4) for v in GRID_HZ],
        "bypass_over_chain": bypass_over_chain,
        "takes": takes,
    }
