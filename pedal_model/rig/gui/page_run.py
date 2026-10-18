"""Run page: tell the human how to set the pedal, capture train + val, check, accept or retake,
next.
"""

from __future__ import annotations

import bisect

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.analysis import log_grid, peak_dbfs, spectrum_db
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.page_calibrate import confirm_playback, demo_pedal
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import CheckList, ControlPanel, LevelMeter, card, label
from pedal_model.rig.gui.worker import run_async
from pedal_model.rig.qc import FR_TYPES, check_take, expected_response
from pedal_model.signals.manifest import Manifest

LIVE_MS = 50  # GUI refresh period (20 Hz)
METER_SAMPLES = 2048
SPECTRUM_SAMPLES = 8192
LIVE_TF_MIN_SEND_DBFS = -50.0  # below this the live response estimate is mostly noise
LIVE_TF_FLOOR = (
    1e-7  # bins whose accumulated send power is below this fraction of the maximum stay blank
)


def _capture_and_check(state: AppState, preset, role: str):
    """Worker body: play one role of a preset, run QC, save the files. Returns (role, report)."""
    session = state.session
    sig = session.signal(role)
    manifest = Manifest(sig.manifest_path)
    dry = manifest.load_all()
    take = state.engine.run(dry, sig.sample_rate, send_level_db=state.profile.send_level_db)
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
    )
    session.save_capture(preset, role, take, wet, rep.to_dict())
    return role, rep


class RunPage(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        self.index = 0
        self.phase = "idle"
        self.role = ""
        self.reports: dict[str, object] = {}
        self.manifest: Manifest | None = None
        self.section_starts: list[int] = []

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
        self.skip_btn = QPushButton("Skip preset")
        self.skip_btn.clicked.connect(self._skip)
        self.abort_btn = QPushButton("Abort take")
        self.abort_btn.setObjectName("danger")
        self.abort_btn.clicked.connect(lambda: self.state.engine.abort())
        for b in (self.ready_btn, self.skip_btn, self.abort_btn):
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
        self.qc_train = CheckList()
        self.qc_val = CheckList()
        lay.addWidget(label("<b>Train</b>"))
        lay.addWidget(self.qc_train)
        lay.addWidget(label("<b>Val</b>"))
        lay.addWidget(self.qc_val)
        lay.addStretch()
        row = QHBoxLayout()
        self.accept_btn = QPushButton("Accept ✓  (next preset)")
        self.accept_btn.setObjectName("primary")
        self.accept_btn.clicked.connect(self._accept)
        self.retake_btn = QPushButton("Retake ↺")
        self.retake_btn.clicked.connect(self._retake)
        row.addWidget(self.accept_btn)
        row.addWidget(self.retake_btn)
        row.addStretch()
        lay.addLayout(row)
        left.addWidget(f, 1)

        # ── right: live views ───────────────────────────────────────────── #
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
        self.spec.setYRange(-130, 0)
        self.spec.setXRange(np.log10(20), np.log10(20_000))
        self.c_send = self.spec.plot(pen=pg.mkPen(theme.TEXT_DIM, width=1), name="send (Out 1)")
        self.c_ret = self.spec.plot(pen=pg.mkPen(theme.CYAN, width=1.5), name="return (In 1)")
        meters.addWidget(self.spec, 1)
        lay.addLayout(meters)
        right.addWidget(f, 1)

        f, lay = card("Pedal response (normalised by the calibration chain gain)")
        self.resp = theme.styled_plot("", "Frequency (Hz)", "Gain (dB)", log_x=True)
        self.resp.setXRange(np.log10(20), np.log10(20_000))
        self.resp.setYRange(-18, 18)
        self.c_expected = self.resp.plot(
            pen=pg.mkPen(theme.AMBER, width=2, style=Qt.DashLine), name="model (expected)"
        )
        self.c_live = self.resp.plot(pen=pg.mkPen(theme.TEXT_DIM, width=1), name="live estimate")
        self.c_measured = self.resp.plot(pen=pg.mkPen(theme.OK, width=2.5), name="measured (train)")
        lay.addWidget(self.resp)
        right.addWidget(f, 1)

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
        if self.phase == "capture":
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

    def _current(self):
        return self.state.plan[self.index] if self.index < len(self.state.plan) else None

    def _show_current(self) -> None:
        self.reports = {}
        self.qc_train.clear()
        self.qc_val.clear()
        self.c_measured.setData([], [])
        self.c_live.setData([], [])
        self.progress.setValue(0)
        p = self._current()
        n = len(self.state.plan)
        if self.state.session is None:
            self._set_phase("idle")
            self.header.setText("Run")
            self.instruction.setText("Start or resume a session first (Start page).")
            self.desc.setText("")
            return
        if p is None:
            self._set_phase("done")
            self.header.setText("Run: all done")
            self.instruction.setText("Every planned preset is accepted or skipped.")
            self.desc.setText(f"Session folder: {self.state.session.root}. See the Review page.")
            return
        self.header.setText(f"Preset {self.index + 1} / {n}: {p.name}")
        self.instruction.setText(
            f"Set the {self.state.profile.display_name} like this, then press Ready."
        )
        self.desc.setText(p.description)
        self.panel.set_values(p.controls)
        grid = log_grid(20, 20_000)
        exp = expected_response(self.state.profile.pedal, p.controls, self.state.sample_rate, grid)
        self.c_expected.setData(grid, exp if exp is not None else np.zeros_like(grid))
        if self.state.calibration is None or not (self.state.calibration or {}).get("chain"):
            self.status.setText(
                "<span style='color:%s'>No calibration in this session yet: QC can't check "
                "latency or level. Run the Calibrate page first if you can.</span>" % theme.AMBER
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

    def _space(self) -> None:
        if self.phase == "ready":
            self._ready()

    def _ready(self) -> None:
        if self.phase != "ready" or not confirm_playback(self, self.state):
            return
        p = self._current()
        cal_name = self.state.profile.raw.get("calibration_preset", "flat")
        if self.forget.isChecked():
            demo_pedal(self.state, self.state.profile.preset(cal_name).controls)
        else:
            demo_pedal(self.state, p.controls)
        self._start_role("train")

    def _start_role(self, role: str) -> None:
        self.role = role
        sig = self.state.session.signal(role)
        self.manifest = Manifest(sig.manifest_path)
        self.section_starts = [s.start_sample for s in self.manifest.sections]
        self.m_send.reset()
        self.m_ret.reset()
        self.m_loop.reset()
        self.acc_dd = np.zeros(SPECTRUM_SAMPLES // 2 + 1)
        self.acc_dw = np.zeros(SPECTRUM_SAMPLES // 2 + 1, dtype=complex)
        self.c_live.setData([], [])
        self._set_phase("capture")
        self.status.setText(f"Recording the {role} signal ({sig.duration_s:.0f} s of audio)…")
        run_async(
            _capture_and_check,
            self.state,
            self._current(),
            role,
            on_done=self._role_done,
            on_fail=self._failed,
        )

    def _role_done(self, result) -> None:
        role, rep = result
        if rep is None:
            self.status.setText("Aborted. Press Ready to try again.")
            self._set_phase("ready")
            return
        self.reports[role] = rep
        lst = self.qc_train if role == "train" else self.qc_val
        lst.clear()
        for item in rep.items:
            lst.add(item.status, f"{item.name}: {item.value}", item.detail)
        if role == "train" and rep.response is not None:
            self.c_measured.setData(rep.response.freqs_hz, rep.response.measured_db)
        if role == "train":
            self._start_role("val")
            return
        worst = max(
            (r.verdict for r in self.reports.values()),
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
        self._set_phase("ready")
        self.status.setText("The take failed; see the message.")
        QMessageBox.critical(self, "Capture failed", tb[-2500:])

    # ── live views ──────────────────────────────────────────────────────── #

    def _tick(self) -> None:
        eng = self.state.engine
        if self.phase != "capture":
            return
        pos, total = eng.position, eng.total
        if not eng.running and pos == 0:
            return
        if pos >= total and total:
            self.status.setText(f"Checking the {self.role} take…")
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
        b = pos - SPECTRUM_SAMPLES
        f, s_db = spectrum_db(play[b:pos, 0], sr)
        _, r_db = spectrum_db(rec[b:pos, 0], sr)
        sel = slice(1, None)
        self.c_send.setData(f[sel], s_db[sel])
        self.c_ret.setData(f[sel], r_db[sel])

        sec = self._section_at(pos - eng.pre_roll)
        if sec is not None:
            self.status.setText(
                f"Recording {self.role}: <b>{sec.label}</b> ({sec.type.replace('_', ' ')}), "
                f"{(pos - eng.pre_roll) / sr:.0f} / {(total - eng.pre_roll) / sr:.0f} s"
            )
            if sec.type in FR_TYPES:
                self._accumulate_response(play, rec, pos, sr)

    def _accumulate_response(self, play: np.ndarray, rec: np.ndarray, pos: int, sr: int) -> None:
        """Build the response up block by block (H1: |sum D*·W| / sum |D|²), as the sweep moves
        through.
        """
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
        # play[] already contains the send level, so only the chain gain is removed here (QC, which
        # divides by the dry signal before the send level, removes both).
        cal_gain = cal["chain"]["gain_db"] if cal.get("chain") else 0.0
        self.c_live.setData(grid, live - cal_gain, connect="finite")

    def _section_at(self, sample: int):
        if not self.manifest or sample < 0:
            return None
        i = bisect.bisect_right(self.section_starts, sample) - 1
        if i < 0:
            return None
        s = self.manifest.sections[i]
        return s if sample < s.end_sample else None
