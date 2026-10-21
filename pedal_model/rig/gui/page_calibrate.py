"""Calibrate page: the guided steps C1–C4 (plan R1), each with exact cable moves and a test.

C1 loopback, C2 the chain alone (pedal out), C3 the pedal bypassed, C4 the pedal engaged at its
calibration preset. A step unlocks when the one before it has passed; running a step again clears
the steps after it, because their cables and references no longer match.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.analysis import peak_dbfs
from pedal_model.rig.calibrate import CAL_STEPS, ref_db_at, run_step
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.diagram import BlockDiagram
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import (
    AutoY,
    CheckList,
    ControlPanel,
    card,
    label,
    show_error,
)
from pedal_model.rig.gui.worker import run_async

STATUS_TEXT = {
    "locked": "locked: pass the step above first",
    "todo": "to do",
    "busy": "running…",
    "ok": "passed",
    "warn": "passed, with warnings",
    "fail": "failed: fix and run again",
}
STATUS_COLOUR = {**theme.STATUS_COLOURS, "locked": theme.TEXT_DIM, "todo": theme.CYAN}
REFERENCE_Y = (-10.0, 10.0)  # fixed range when "Auto scale" is off
REFERENCE_MIN_SPAN_DB = 6.0


def confirm_playback(parent: QWidget, state: AppState) -> bool:
    """On hardware, ask once per mode switch before the first sound goes out of Out 1/2."""
    if state.simulated or state.hardware_confirmed:
        return True
    ch = state.rig.channels
    ok = QMessageBox.question(
        parent,
        "Play test signals?",
        f"The app is about to play test signals out of Out {ch.send_out} and Out {ch.loop_out} "
        "of the 1824c (the Main outputs).\n\nMake sure studio monitors are not connected to them "
        "(or are turned down), the pedal chain is wired, and Studio One is closed.\n\nContinue?",
    )
    state.hardware_confirmed = ok == QMessageBox.Yes
    return state.hardware_confirmed


def demo_pedal(state: AppState, controls: dict[str, float] | None) -> None:
    """In demo mode, 'turn the knobs' of the simulated pedal to these settings (and engage it)."""
    if not state.simulated:
        return
    backend = state.engine.backend
    backend.set_pedal_state("engaged")
    if state.profile.pedal == "boss_ge7" and controls is not None:
        from pedal_model.synthetic.ge7 import Ge7Sim, Ge7Sliders

        sim = Ge7Sim.from_sliders(Ge7Sliders.from_dict(controls))
        backend.set_pedal(lambda x, sr: sim.process(x, sr))
    else:
        backend.set_pedal(None)


class StepCard(QFrame):
    """One guided step: what to move, what the test checks, a Run button and the results."""

    def __init__(self, step: dict, pedal: str, on_run) -> None:
        super().__init__()
        self.step = step
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        head = QHBoxLayout()
        sid = step["id"]
        head.addWidget(label(f"<b>{sid} · {step['title'].format(pedal=pedal)}</b>"), 1)
        self.chip = label("", wrap=False)
        head.addWidget(self.chip)
        lay.addLayout(head)
        moves = "".join(f"<li>{m.format(pedal=pedal)}</li>" for m in step["moves"])
        lay.addWidget(label(f"<ol style='margin-left:-20px'>{moves}</ol>"))
        lay.addWidget(label(f"Checks: {step['checks']}", "dim"))
        self.btn = QPushButton(f"Run {sid} test")
        self.btn.clicked.connect(lambda: on_run(sid))
        lay.addWidget(self.btn, 0, Qt.AlignLeft)
        self.results = CheckList()
        lay.addWidget(self.results)

    def set_status(self, status: str, unlocked: bool) -> None:
        colour = STATUS_COLOUR.get(status, theme.TEXT_DIM)
        self.chip.setText(f"<span style='color:{colour}'>{STATUS_TEXT[status]}</span>")
        self.btn.setEnabled(unlocked and status != "busy")
        self.setEnabled(unlocked or status in ("ok", "warn", "fail"))

    def show_advice(self, advice: list) -> None:
        self.results.clear()
        for a in advice:
            self.results.add(a["status"], a["text"])


class CalibratePage(QWidget):
    def __init__(self, state: AppState, go_next) -> None:
        super().__init__()
        self.state = state
        self.loopback: dict | None = None
        self.chain: dict | None = None
        self.references: dict[str, dict] = {}
        self.steps: dict[str, dict] = {}
        self.busy: str | None = None
        pedal = state.profile.display_name

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Calibrate", "title"))
        root.addWidget(
            label(
                "Four short steps at the start of every session. Each one says exactly which cable "
                "to move; a test after the move checks it, and the next step unlocks when it "
                "passes. <b>Cable A</b> runs ProRMP OUTPUT → pedal IN, <b>cable B</b> runs pedal "
                "OUT → ProDI INPUT. QC divides every take by the references measured here.",
                "dim",
            )
        )
        body = QHBoxLayout()
        root.addLayout(body, 1)

        steps_box = QWidget()
        steps_lay = QVBoxLayout(steps_box)
        steps_lay.setContentsMargins(0, 0, 8, 0)
        self.cards: dict[str, StepCard] = {}
        for step in CAL_STEPS:
            c = StepCard(step, pedal, self._run)
            self.cards[step["id"]] = c
            steps_lay.addWidget(c)
        steps_lay.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(steps_box)
        body.addWidget(scroll, 3)

        right = QVBoxLayout()
        body.addLayout(right, 2)
        f, lay = card("Cables to move now (orange)")
        self.diagram = BlockDiagram(lambda _box: None)
        self.diagram.set_pedal_name(pedal)
        self.diagram.setMinimumHeight(220)
        lay.addWidget(self.diagram)
        right.addWidget(f, 2)

        f, lay = card("Calibration preset (C4)")
        cal_name = state.profile.raw.get("calibration_preset", "flat")
        lay.addWidget(label(f"Set the {pedal} to <b>{cal_name}</b> for C4.", "dim"))
        self.panel = ControlPanel(state.profile.controls)
        try:
            self.panel.set_values(state.profile.preset(cal_name).controls)
        except KeyError:
            pass
        lay.addWidget(self.panel)
        right.addWidget(f, 2)

        f, lay = card("Reference responses")
        self.plot = theme.styled_plot("", "Frequency (Hz)", "dB", log_x=True)
        self.plot.setXRange(np.log10(20), np.log10(20_000))
        self.plot.setMinimumHeight(220)
        self.plot.addLegend(offset=(10, 10))
        self.c_chain = self.plot.plot(pen=pg.mkPen(theme.CYAN, width=2), name="chain (C2)")
        self.c_bypass = self.plot.plot(
            pen=pg.mkPen(theme.AMBER, width=2), name="bypassed ÷ chain (C3)"
        )
        self.c_engaged = self.plot.plot(
            pen=pg.mkPen(theme.OK, width=2), name="engaged ÷ bypassed (C4)"
        )
        lay.addWidget(self.plot)
        self.plot_y = AutoY(self.plot, REFERENCE_Y, REFERENCE_MIN_SPAN_DB, step_db=1.0)
        lay.addWidget(self.plot_y.box, 0, Qt.AlignRight)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["C2 send (dBFS)", "Return (dBFS)", "Gain (dB)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(150)
        lay.addWidget(self.table)
        right.addWidget(f, 3)

        self.save_note = label("", "dim")
        right.addWidget(self.save_note)
        nxt = QPushButton("Continue to Plan →")
        nxt.setObjectName("primary")
        nxt.clicked.connect(go_next)
        right.addWidget(nxt, 0, Qt.AlignRight)
        state.session_changed.connect(self._load_existing)
        self._load_existing()

    # ── step logic ──────────────────────────────────────────────────────── #

    def _status(self, step_id: str) -> str:
        if self.busy == step_id:
            return "busy"
        return self.steps.get(step_id, {}).get("status", "todo")

    def _refresh(self) -> None:
        """Lock/unlock the cards, colour the cables, redraw the reference plot."""
        unlocked = True
        pending = None
        for step in CAL_STEPS:
            st = self._status(step["id"])
            shown = st if unlocked or st != "todo" else "locked"
            self.cards[step["id"]].set_status(shown, unlocked and self.busy is None)
            if pending is None and unlocked and st in ("todo", "fail"):
                pending = step
            unlocked = unlocked and st in ("ok", "warn")
        cables = dict(self.state.cables)
        if pending is not None:
            cables.update({c: "move" for c in pending["cables"]})
        self.diagram.set_cable_states(cables)
        self._plot()

    def _plot(self) -> None:
        refs = self.references
        for curve in (self.c_chain, self.c_bypass, self.c_engaged):
            curve.setData([], [])
        if "chain" in refs:
            r = refs["chain"]
            self.c_chain.setData(r["freqs_hz"], np.asarray(r["mag_db"]) - r["gain_1k_db"])
        for name, base, curve in (
            ("bypass", "chain", self.c_bypass),
            ("engaged_flat", "bypass", self.c_engaged),
        ):
            if name in refs and base in refs:
                f = np.asarray(refs[name]["freqs_hz"])
                curve.setData(f, np.asarray(refs[name]["mag_db"]) - ref_db_at(refs[base], f))
        self.plot_y.update(force=True)

    def _run(self, step_id: str) -> None:
        st = self.state
        if st.session is None:
            QMessageBox.information(
                self, "No session", "Start or resume a session on the Start page first."
            )
            return
        if not confirm_playback(self, st):
            return
        step = next(s for s in CAL_STEPS if s["id"] == step_id)
        if st.simulated and step["pedal_state"] is not None:
            cal_name = st.profile.raw.get("calibration_preset", "flat")
            try:
                demo_pedal(st, st.profile.preset(cal_name).controls)
            except KeyError:
                demo_pedal(st, None)
            st.engine.backend.set_pedal_state(step["pedal_state"])
        train = st.session.signal("train")
        train_peak = peak_dbfs(sf.read(str(train.wav_path), dtype="float32")[0])
        refs = dict(self.references)
        self.busy = step_id
        self.cards[step_id].results.clear()
        self.cards[step_id].results.add("busy", "Running…")
        self._refresh()
        run_async(
            run_step,
            step_id,
            st.engine,
            st.sample_rate,
            send_level_db=st.profile.send_level_db,
            train_peak_dbfs=train_peak,
            references=refs,
            on_done=self._done,
            on_fail=self._failed,
        )

    def _done(self, res: dict) -> None:
        self.busy = None
        s = self.state.session
        ids = [x["id"] for x in CAL_STEPS]
        for later in CAL_STEPS[ids.index(res["step"]) + 1 :]:  # cables and references now stale
            self.steps.pop(later["id"], None)
            self.cards[later["id"]].results.clear()
            if later["reference"]:
                self.references.pop(later["reference"], None)
        if res["loopback"] is not None:
            self.loopback = res["loopback"]
        if res["chain"] is not None:
            self.chain = res["chain"]
            self._show_tones(self.chain["steps"])
        ref = res["reference"]
        if ref is not None and res["status"] != "fail":
            ref["file"] = s.save_reference(ref["name"], res["take"])
            self.references[ref["name"]] = ref
        self.steps[res["step"]] = {
            "status": res["status"],
            "advice": res["advice"],
            "time_local": datetime.now().isoformat(timespec="seconds"),
        }
        self.cards[res["step"]].show_advice(res["advice"])
        path = s.set_calibration(self.loopback, self.chain, self.references, self.steps)
        self.state.set_cables_from_calibration(s.data["calibration"])
        self.state.calibration_changed.emit()
        self.save_note.setText(f"Calibration saved: {s.root.name}/{path.name} (and session.json).")
        self._refresh()

    def _show_tones(self, steps: list[dict]) -> None:
        self.table.setRowCount(len(steps))
        for i, t in enumerate(steps):
            for j, k in enumerate(("send_dbfs", "return_dbfs", "gain_db")):
                self.table.setItem(i, j, QTableWidgetItem(f"{t[k]:+.2f}"))

    def _load_existing(self) -> None:
        """Resume a session's calibration (older sessions without guided steps restore C1 only)."""
        cal = self.state.calibration or {}
        self.loopback, self.chain = cal.get("loopback"), cal.get("chain")
        self.references = dict(cal.get("references") or {})
        self.steps = dict(cal.get("steps") or {})
        if not self.steps and self.loopback:
            ok = all(a["status"] != "fail" for a in self.loopback["advice"])
            self.steps["C1"] = {"status": "ok" if ok else "fail", "advice": self.loopback["advice"]}
        for step in CAL_STEPS:
            sid = step["id"]
            self.cards[sid].show_advice(self.steps.get(sid, {}).get("advice", []))
        self.table.setRowCount(0)
        if self.chain and "C2" in self.steps:
            self._show_tones(self.chain["steps"])
        self._refresh()

    def _failed(self, tb: str) -> None:
        step, self.busy = self.busy, None
        if self.state.session is not None:
            self.state.session.log("error", page="calibrate", step=step, traceback=tb)
        self._refresh()
        show_error(self, "Test failed", tb)
