"""Run page: show how to set the pedal, capture, check, accept or retake, then the next step."""

from __future__ import annotations

import bisect

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.analysis import log_bin_db, log_grid, peak_dbfs, spectrum_db
from pedal_model.rig.bandfit import fit_band
from pedal_model.rig.calibrate import DEFAULT_CEILING_DBFS, plan_send_level, ref_db_at
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.page_calibrate import confirm_playback, demo_pedal
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import (
    AutoY,
    CheckText,
    ControlPanel,
    LevelMeter,
    card,
    label,
    show_error,
)
from pedal_model.rig.gui.worker import run_async
from pedal_model.rig.profiles import KIND_MAP, Preset, Profile
from pedal_model.rig.qc import (
    FR_TYPES,
    calibration_reference,
    check_take,
    expected_response,
    item,
)
from pedal_model.signals.manifest import Manifest

LIVE_MS = 50  # GUI refresh period (20 Hz)
METER_SAMPLES = 2048
SPECTRUM_SAMPLES = 8192
LIVE_TF_MIN_SEND_DBFS = -50.0  # below this the live response estimate is mostly noise
LIVE_TF_FLOOR = (
    1e-7  # bins with less accumulated send power than this fraction of the max stay blank
)
ROLE_NAMES = {"train": "Train", "val": "Val", "sweep": "Sweep"}
SPECTRUM_Y = (-130.0, 0.0)  # fixed ranges used when "Auto scale" is off
RESPONSE_Y = (-30.0, 20.0)
# Smallest Y span autoscale may zoom to: a perfect (simulated) flat response would otherwise be
# zoomed to micro-dB noise, and the spectrum to the noise floor between takes.
SPECTRUM_MIN_SPAN_DB = 60.0
RESPONSE_MIN_SPAN_DB = 6.0
RATIO_Y = (-6.0, 6.0)
RATIO_MIN_SPAN_DB = 2.0


def band_fit_for(profile: Profile, preset: Preset, response: dict | None, sr: int) -> dict | None:
    """For a map step: fit the one moved control (a peak, a shelf for the top band, or a gain)."""
    moved = [c for c in profile.controls if abs(preset.controls.get(c["key"], 0.0)) > 1e-6]
    if preset.kind != KIND_MAP or len(moved) != 1 or response is None:
        return None
    c = moved[0]
    top_band = c.get("band_hz") == max(k.get("band_hz") or 0 for k in profile.controls)
    fit = fit_band(
        response["freqs_hz"],
        response["measured_db"],
        sr,
        c.get("band_hz"),
        preset.controls[c["key"]],
        try_shelf=top_band,
    )
    return {"control": c["key"], "setting": preset.controls[c["key"]], **fit}


def _capture_and_check(state: AppState, preset: Preset, role: str, report):
    """Worker body: play one role of a step, run QC (and the band fit), save. Returns (role,
    report).
    """
    session = state.session
    report("Loading the signal")
    sig = session.signal(role)
    manifest = Manifest(sig.manifest_path)
    dry = manifest.load_all()
    send = state.profile.send_level_db
    level_plan = None
    if state.profile.qc.get("auto_level", False):
        report("Measuring this step's gain (auto-level)")
        chain = (session.data.get("calibration") or {}).get("chain") or {}
        ceiling = chain.get("return_ceiling_dbfs") or DEFAULT_CEILING_DBFS
        level_plan = plan_send_level(state.engine, sig.sample_rate, peak_dbfs(dry), send, ceiling)
        send = level_plan["send_level_db"]
    report("capture")
    take = state.engine.run(dry, sig.sample_rate, send_level_db=send)
    if take.aborted:
        return role, None
    rep, wet = check_take(
        take,
        dry,
        manifest,
        pedal=state.profile.pedal,
        controls=preset.controls,
        calibration=session.data.get("calibration"),
        qc_limits=state.profile.qc,
        report=report,
    )
    fit = None
    if preset.kind == KIND_MAP:
        report("Fitting the band (centre, bandwidth, gain)")
        fit = band_fit_for(state.profile, preset, rep["response"], sig.sample_rate)
        if fit:
            rep["items"].append(
                item("Band fit", "ok", fit["summary"], f"RMS fit error {fit['rms_err_db']:.2f} dB")
            )
    report("Saving the WAV files and params.json")
    session.save_capture(preset, role, take, wet, rep, band_fit=fit, level_plan=level_plan)
    return role, rep


class RunPage(QWidget):
    def __init__(self, state: AppState, go_next) -> None:
        super().__init__()
        self.state = state
        self.index = 0
        self.phase = "idle"
        self.role = ""
        self.roles: tuple[str, ...] = ()
        self.reports: dict[str, object] = {}
        self.manifest: Manifest | None = None
        self.section_starts: list[int] = []
        self._capture_started = False  # set by the worker's "capture" stage message
        self.acc_dd = np.zeros(SPECTRUM_SAMPLES // 2 + 1)
        self.acc_dw = np.zeros(SPECTRUM_SAMPLES // 2 + 1, dtype=complex)

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # ── left: instructions + QC ─────────────────────────────────────── #
        left = QVBoxLayout()
        root.addLayout(left, 5)
        self.header = label("Run", "title")
        left.addWidget(self.header)
        f, lay = card()
        self.instruction = label("", "heading")
        self.desc = label("", "dim")
        lay.addWidget(self.instruction)
        lay.addWidget(self.desc)
        self.panel = ControlPanel(state.profile.controls)
        lay.addWidget(self.panel)
        self.forget = QCheckBox(
            "Demo: pretend I forgot to move the controls (leave them at the calibration preset)"
        )
        lay.addWidget(self.forget)
        row = QHBoxLayout()
        self.ready_btn = QPushButton("Ready ▶  (Space)")
        self.ready_btn.setObjectName("primary")
        self.ready_btn.clicked.connect(self._ready)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._skip)
        self.abort_btn = QPushButton("Abort take")
        self.abort_btn.setObjectName("danger")
        self.abort_btn.clicked.connect(lambda: self.state.engine.abort())
        self.review_btn = QPushButton("Go to Review →")
        self.review_btn.setObjectName("primary")
        self.review_btn.clicked.connect(go_next)
        for b in (self.ready_btn, self.skip_btn, self.abort_btn, self.review_btn):
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        lay.addWidget(self.progress)
        self.status = label("", "dim")
        lay.addWidget(self.status)
        left.addWidget(f)

        f, lay = card("Quality checks")
        # Scrolls instead of growing: two roles' checks are taller than a laptop screen.
        self.qc = CheckText()
        self.qc.setMinimumHeight(120)
        lay.addWidget(self.qc, 1)
        row = QHBoxLayout()
        self.accept_btn = QPushButton("Accept ✓  (next step)")
        self.accept_btn.setObjectName("primary")
        self.accept_btn.clicked.connect(self._accept)
        self.retake_btn = QPushButton("Retake ↺")
        self.retake_btn.clicked.connect(self._retake)
        row.addWidget(self.accept_btn)
        row.addWidget(self.retake_btn)
        row.addStretch()
        lay.addLayout(row)
        left.addWidget(f, 1)

        # ── right: live levels and spectrum, the pedal's response, its ratio to the model ── #
        right = QVBoxLayout()
        root.addLayout(right, 6)
        f, lay = card("Live")
        meters = QHBoxLayout()
        self.m_send, self.m_ret, self.m_loop = (
            LevelMeter("Send"),
            LevelMeter("Return"),
            LevelMeter("Loop"),
        )
        for m in (self.m_send, self.m_ret, self.m_loop):
            meters.addWidget(m)
        self.spec = theme.styled_plot(
            "Spectrum: send vs return", "Frequency (Hz)", "Level (dBFS)", log_x=True
        )
        self.spec.setXRange(np.log10(20), np.log10(20_000))
        # Live curves skip antialiasing: on noisy spectra it multiplied the repaint time ~20x.
        self.c_send = self.spec.plot(
            pen=pg.mkPen(theme.TEXT_DIM, width=1), name="send (Out 1)", antialias=False
        )
        self.c_ret = self.spec.plot(
            pen=pg.mkPen(theme.CYAN, width=1.5), name="return (In 1)", antialias=False
        )
        meters.addWidget(self.spec, 1)
        lay.addLayout(meters, 1)
        self.spec_y = AutoY(self.spec, SPECTRUM_Y, SPECTRUM_MIN_SPAN_DB, step_db=10.0)
        lay.addWidget(self.spec_y.box, 0, Qt.AlignRight)
        right.addWidget(f, 1)

        f, lay = card("Pedal response (÷ the calibration reference)")
        self.resp = theme.styled_plot("", "Frequency (Hz)", "Gain (dB)", log_x=True)
        self.resp.setXRange(np.log10(20), np.log10(20_000))
        self.c_expected = self.resp.plot(
            pen=pg.mkPen(theme.AMBER, width=2, style=Qt.DashLine), name="model (expected)"
        )
        self.c_live = self.resp.plot(
            pen=pg.mkPen(theme.TEXT_DIM, width=1.5), name="live estimate", antialias=False
        )
        self.c_measured = self.resp.plot(pen=pg.mkPen(theme.OK, width=2.5), name="measured")
        lay.addWidget(self.resp, 1)
        self.resp_y = AutoY(self.resp, RESPONSE_Y, RESPONSE_MIN_SPAN_DB, step_db=3.0)
        lay.addWidget(self.resp_y.box, 0, Qt.AlignRight)
        right.addWidget(f, 1)

        # Measured ÷ model, in dB: 0 where the pedal does what the model says.
        f, lay = card("Measured ÷ model (dB)")
        self.ratio = theme.styled_plot("", "Frequency (Hz)", "Difference (dB)", log_x=True)
        self.ratio.setXLink(self.resp)
        self.ratio.addLine(y=0.0, pen=pg.mkPen(theme.AMBER, width=1, style=Qt.DashLine))
        self.c_ratio_live = self.ratio.plot(
            pen=pg.mkPen(theme.TEXT_DIM, width=1.5), name="live ÷ model", antialias=False
        )
        self.c_ratio_measured = self.ratio.plot(
            pen=pg.mkPen(theme.OK, width=2.5), name="measured ÷ model"
        )
        lay.addWidget(self.ratio, 1)
        self.ratio_y = AutoY(self.ratio, RATIO_Y, RATIO_MIN_SPAN_DB, step_db=1.0)
        lay.addWidget(self.ratio_y.box, 0, Qt.AlignRight)
        right.addWidget(f, 1)
        # The page is sized by height-for-width (it has wrapped text), which uses each plot's
        # preferred height (480 px): three stacked plots then asked for more than a 1440p screen.
        # Ignoring the preferred height lets them share the window's height instead.
        for p in (self.spec, self.resp, self.ratio):
            p.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
            p.setMinimumHeight(220)
        self._exp_grid: np.ndarray | None = None
        self._exp_db: np.ndarray | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(LIVE_MS)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._space)
        state.session_changed.connect(self.reset)
        state.plan_changed.connect(self.reset)
        state.rig_changed.connect(self._update_mode)
        self.reset()

    # ── flow ────────────────────────────────────────────────────────────── #

    def _update_mode(self) -> None:
        self.forget.setVisible(self.state.simulated)

    def reset(self) -> None:
        if self.phase in ("capture", "analyse"):
            return  # never reset under a running take
        self._update_mode()
        s = self.state.session
        self.index = 0
        if s is not None:
            statuses = s.data["presets"]
            while self.index < len(self.state.plan) and statuses.get(
                self.state.plan[self.index].name, {}
            ).get("status") in ("accepted", "skipped"):
                self.index += 1
        self._show_current()

    def _current(self) -> Preset | None:
        return self.state.plan[self.index] if self.index < len(self.state.plan) else None

    def _show_current(self) -> None:
        self.reports = {}
        self.qc.clear()
        for c in (self.c_measured, self.c_live, self.c_ratio_live, self.c_ratio_measured):
            c.setData([], [])
        self._progress_busy(False)
        self.progress.setValue(0)
        p = self._current()
        n = len(self.state.plan)
        self.review_btn.setVisible(False)
        if self.state.session is None:
            self._set_phase("idle")
            self.header.setText("Run")
            self.instruction.setText("Start or resume a session first (Start page).")
            self.desc.setText("")
            return
        if p is None:
            self._set_phase("done")
            self.header.setText("Run: all done")
            self.instruction.setText("Every planned step is accepted or skipped.")
            self.desc.setText(f"Session folder: {self.state.session.root}")
            self.review_btn.setVisible(True)
            return
        kind = "map step" if p.kind == KIND_MAP else "preset"
        self.header.setText(f"Step {self.index + 1} / {n}: {p.name}  ({kind})")
        self.instruction.setText(
            f"Set the {self.state.profile.display_name} like this, then press Ready."
        )
        self.desc.setText(p.description + (f"<br>{p.notes}" if p.notes else ""))
        self.panel.set_values(p.controls)
        grid = log_grid(20, 20_000)
        exp = expected_response(self.state.profile.pedal, p.controls, self.state.sample_rate, grid)
        self.c_expected.setData(grid, exp if exp is not None else np.zeros_like(grid))
        self._exp_grid, self._exp_db = (grid, exp) if exp is not None else (None, None)
        self.resp_y.update(force=True)
        if not (self.state.calibration or {}).get("chain"):
            self.status.setText(
                f"<span style='color:{theme.AMBER}'>No calibration in this session yet: QC can't "
                "check latency or level. Run the Calibrate page first if you can.</span>"
            )
        else:
            self.status.setText("Ready when you are.")
        self._set_phase("ready")

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.ready_btn.setEnabled(phase == "ready")
        self.skip_btn.setEnabled(phase in ("ready", "verdict"))
        self.abort_btn.setEnabled(phase == "capture")
        self.accept_btn.setEnabled(phase == "verdict")
        self.retake_btn.setEnabled(phase == "verdict")
        self.forget.setEnabled(phase == "ready")

    def _progress_busy(self, on: bool, text: str = "") -> None:
        """Busy-indicator mode (moving bar) for the analysis stages, which have no percentage."""
        if on:
            self.progress.setRange(0, 0)
            self.progress.setFormat(text)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setFormat("%p%")

    def _space(self) -> None:
        if self.phase == "ready":
            self._ready()

    def _ready(self) -> None:
        if self.phase != "ready" or not confirm_playback(self, self.state):
            return
        p = self._current()
        cal_name = self.state.profile.raw.get("calibration_preset", "flat")
        controls = (
            self.state.profile.preset(cal_name).controls if self.forget.isChecked() else p.controls
        )
        demo_pedal(self.state, controls)
        self.roles = p.roles
        self.qc.clear()
        self._start_role(self.roles[0])

    def _start_role(self, role: str) -> None:
        self.role = role
        sig = self.state.session.signal(role)
        self.manifest = Manifest(sig.manifest_path)
        self.section_starts = [s.start_sample for s in self.manifest.sections]
        for m in (self.m_send, self.m_ret, self.m_loop):
            m.reset()
        self.acc_dd = np.zeros(SPECTRUM_SAMPLES // 2 + 1)
        self.acc_dw = np.zeros(SPECTRUM_SAMPLES // 2 + 1, dtype=complex)
        self._live_corr = None  # reference correction on the live grid, set on first use
        self.c_live.setData([], [])
        self.c_ratio_live.setData([], [])
        self._set_phase("capture")
        self._capture_started = False
        self._progress_busy(False)
        self.progress.setValue(0)
        self.status.setText(f"Recording the {role} signal ({sig.duration_s:.0f} s of audio)…")
        run_async(
            _capture_and_check,
            self.state,
            self._current(),
            role,
            on_done=self._role_done,
            on_fail=self._failed,
            on_progress=self._stage,
        )

    def _stage(self, text: str) -> None:
        """Stage messages from the worker: the capture itself shows % progress, analysis a moving
        bar.
        """
        if text == "capture":
            self._capture_started = True
            self._progress_busy(False)
            return
        # The engine still holds the previous take's position when a new role starts, so the stage
        # is tracked explicitly: anything reported after "capture" is analysis.
        if self._capture_started:
            self.phase = "analyse"
            self.status.setText(f"Checking the {self.role} take: {text.lower()}…")
        self._progress_busy(True, f"{ROLE_NAMES[self.role]}: {text}…")

    def _role_done(self, result) -> None:
        role, rep = result
        self._progress_busy(False)
        if rep is None:
            self.status.setText("Aborted. Press Ready to try again.")
            self._set_phase("ready")
            return
        self.reports[role] = rep
        self.qc.add_heading(ROLE_NAMES[role])
        for it in rep["items"]:
            self.qc.add(it["status"], f"{it['name']}: {it['value']}", it["detail"])
        if role == self.roles[0] and rep["response"] is not None:
            r = rep["response"]
            self.c_measured.setData(r["freqs_hz"], r["measured_db"])
            self.resp_y.update(force=True)
            if r["expected_db"] is not None:
                self.c_ratio_measured.setData(r["freqs_hz"], r["measured_db"] - r["expected_db"])
                self.ratio_y.update(force=True)
        i = self.roles.index(role)
        if i + 1 < len(self.roles):
            self._start_role(self.roles[i + 1])
            return
        worst = max(
            (r["verdict"] for r in self.reports.values()),
            key={"ok": 0, "warn": 1, "fail": 2}.__getitem__,
        )
        colour = theme.STATUS_COLOURS[worst]
        self.status.setText(
            f"<span style='color:{colour}'><b>QC: {worst.upper()}</b></span>. "
            "Accept to move on, or Retake."
        )
        self.progress.setValue(1000)
        self._set_phase("verdict")

    def _accept(self) -> None:
        self.state.session.set_status(self._current().name, "accepted")
        self.index += 1
        self._show_current()

    def _retake(self) -> None:
        self.state.session.set_status(self._current().name, "retake")
        self._show_current()

    def _skip(self) -> None:
        self.state.session.set_status(self._current().name, "skipped")
        self.index += 1
        self._show_current()

    def _failed(self, tb: str) -> None:
        self._progress_busy(False)
        self._set_phase("ready")
        self.status.setText("The take failed; see the message.")
        if self.state.session is not None:
            self.state.session.log("error", page="run", traceback=tb)
        show_error(self, "Capture failed", tb)

    # ── live views ──────────────────────────────────────────────────────── #

    def _tick(self) -> None:
        eng = self.state.engine
        if self.phase != "capture" or not self._capture_started or not eng.running:
            return
        pos, total = eng.position, eng.total
        self.progress.setValue(int(1000 * pos / max(total, 1)))
        if pos < METER_SAMPLES:
            return
        play, rec, sr = eng.play, eng.rec, eng.sr
        a = pos - METER_SAMPLES
        self.m_send.set_level(peak_dbfs(play[a:pos, 0]))
        self.m_ret.set_level(peak_dbfs(rec[a:pos, 0]))
        self.m_loop.set_level(peak_dbfs(rec[a:pos, 1]))
        if pos < SPECTRUM_SAMPLES:
            return
        # The return arrives one round trip after the send (about 3700 samples): compare it with
        # the send from that long ago, or a sweep shows the return at a lower frequency.
        cal = self.state.calibration or {}
        lag = (cal.get("loopback") or {}).get("lag_samples", 0)
        b = pos - SPECTRUM_SAMPLES
        if b - lag < 0:
            return
        f, s_db = spectrum_db(play[b - lag : pos - lag, 0], sr)
        _, r_db = spectrum_db(rec[b:pos, 0], sr)
        sel = slice(1, None)
        self.c_send.setData(*log_bin_db(f[sel], s_db[sel]))
        self.c_ret.setData(*log_bin_db(f[sel], r_db[sel]))
        self.spec_y.update()

        sec = self._section_at(pos - eng.pre_roll)
        if sec is not None:
            self.status.setText(
                f"Recording {self.role}: <b>{sec.label}</b> ({sec.type.replace('_', ' ')}), "
                f"{(pos - eng.pre_roll) / sr:.0f} / {(total - eng.pre_roll) / sr:.0f} s"
            )
            if sec.type in FR_TYPES:
                self._accumulate_response(play, rec, pos, sr)

    def _accumulate_response(self, play: np.ndarray, rec: np.ndarray, pos: int, sr: int) -> None:
        """Build the response up block by block (H1: |Σ D*·W| / Σ |D|²) as the sweep moves on."""
        cal = self.state.calibration or {}
        lag = cal["chain"]["lag_samples"] if cal.get("chain") else 0
        n = SPECTRUM_SAMPLES
        a = pos - n - lag
        if a < 0:
            return
        send = play[a : a + n, 0]
        if peak_dbfs(send) < LIVE_TF_MIN_SEND_DBFS:
            return
        w = np.hanning(n)
        d = np.fft.rfft(send * w)
        r = np.fft.rfft(rec[pos - n : pos, 0] * w)
        self.acc_dd += np.abs(d) ** 2
        self.acc_dw += np.conj(d) * r
        f = np.fft.rfftfreq(n, 1 / sr)
        valid = self.acc_dd > LIVE_TF_FLOOR * self.acc_dd.max()
        h_db = 20 * np.log10(np.abs(self.acc_dw) / (self.acc_dd + 1e-30) + 1e-12)
        grid = log_grid(30, min(16_000, 0.45 * sr), 12)
        live = np.interp(grid, f, h_db)
        live[np.interp(grid, f, valid.astype(float)) < 0.5] = np.nan
        # play[] already contains the send level, so only the reference is removed here (QC, which
        # divides by the dry signal before the send level, removes both).
        if self._live_corr is None:
            ref = calibration_reference(cal, self.state.profile.qc.get("reference", "chain"))
            if ref is not None:
                self._live_corr = ref_db_at(ref, grid)
            else:
                self._live_corr = cal["chain"]["gain_db"] if cal.get("chain") else 0.0
        live = live - self._live_corr
        self.c_live.setData(grid, live, connect="finite")
        self.resp_y.update()
        if self._exp_db is not None:
            exp = np.interp(np.log(grid), np.log(self._exp_grid), self._exp_db)
            self.c_ratio_live.setData(grid, live - exp, connect="finite")
            self.ratio_y.update()

    def _section_at(self, sample: int):
        if not self.manifest or sample < 0:
            return None
        i = bisect.bisect_right(self.section_starts, sample) - 1
        if i < 0:
            return None
        s = self.manifest.sections[i]
        return s if sample < s.end_sample else None
