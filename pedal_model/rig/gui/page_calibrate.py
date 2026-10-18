"""Calibrate page: loopback test, chain test, knob guidance; results saved into the session."""

from __future__ import annotations

import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.analysis import peak_dbfs
from pedal_model.rig.calibrate import run_chain_test, run_loopback_test
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import CheckList, ControlPanel, card, label
from pedal_model.rig.gui.worker import run_async


def confirm_playback(parent: QWidget, state: AppState) -> bool:
    """On hardware, ask once per mode switch before the first sound goes out of Out 1/2."""
    if state.simulated or state.hardware_confirmed:
        return True
    ch = state.rig.channels
    ok = QMessageBox.question(
        parent,
        "Play test signals?",
        f"The app is about to play test signals out of Out {ch.send_out} and Out {ch.loop_out} "
        "of the 1824c "
        "(the Main outputs).\n\nMake sure studio monitors are not connected to them (or are "
        "turned down), the "
        "pedal chain is wired, and Studio One is closed.\n\nContinue?",
    )
    state.hardware_confirmed = ok == QMessageBox.Yes
    return state.hardware_confirmed


def demo_pedal(state: AppState, controls: dict[str, float] | None) -> None:
    """In demo mode, 'turn the knobs' of the simulated pedal to these settings."""
    if not state.simulated:
        return
    if state.profile.pedal == "boss_ge7" and controls is not None:
        from pedal_model.synthetic.ge7 import Ge7Sim, Ge7Sliders

        sim = Ge7Sim.from_sliders(Ge7Sliders.from_dict(controls))
        state.engine.backend.set_pedal(lambda x, sr: sim.process(x, sr))
    else:
        state.engine.backend.set_pedal(None)


class CalibratePage(QWidget):
    def __init__(self, state: AppState, go_next) -> None:
        super().__init__()
        self.state = state
        self.loop_result = None
        self.chain_result = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Calibrate", "title"))
        root.addWidget(
            label(
                "Two short tests at the start of every session. They measure the interface and the "
                "whole chain, and tell you which knob to move. QC uses the results on every take.",
                "dim",
            )
        )
        body = QHBoxLayout()
        root.addLayout(body, 1)

        left = QVBoxLayout()
        body.addLayout(left, 1)
        f, lay = card("Before you start")
        cal_name = state.profile.raw.get("calibration_preset", "flat")
        lay.addWidget(
            label(
                f"Set the {state.profile.display_name} to <b>{cal_name}</b> and switch it on. "
                "Check the wiring against the Setup page."
            )
        )
        self.panel = ControlPanel(state.profile.controls)
        try:
            self.panel.set_values(state.profile.preset(cal_name).controls)
        except KeyError:
            pass
        lay.addWidget(self.panel)
        left.addWidget(f)

        f, lay = card("1. Loopback test (Out 2 → In 2)")
        lay.addWidget(
            label(
                "A 1 s sweep and a tone on Out 2 only. Measures the interface's own latency and "
                "gain, "
                "and checks that In 1 stays silent (cables not swapped).",
                "dim",
            )
        )
        self.loop_btn = QPushButton("Run loopback test")
        self.loop_btn.clicked.connect(self._run_loop)
        lay.addWidget(self.loop_btn)
        self.loop_list = CheckList()
        lay.addWidget(self.loop_list)
        left.addWidget(f)
        left.addStretch()

        right = QVBoxLayout()
        body.addLayout(right, 1)
        f, lay = card("2. Chain test (Out 1 → ProRMP → pedal → ProDI → In 1)")
        lay.addWidget(
            label(
                "A quiet sweep, then 1 kHz tones from −36 to −9 dBFS. Measures chain gain, "
                "latency and "
                "linearity, and predicts where the training signal will peak on In 1.",
                "dim",
            )
        )
        self.chain_btn = QPushButton("Run chain test")
        self.chain_btn.clicked.connect(self._run_chain)
        lay.addWidget(self.chain_btn)
        self.chain_list = CheckList()
        lay.addWidget(self.chain_list)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Send (dBFS peak)", "Return (dBFS peak)", "Gain (dB)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(200)
        lay.addWidget(self.table)
        right.addWidget(f)
        self.save_note = label("", "dim")
        right.addWidget(self.save_note)
        right.addStretch()
        nxt = QPushButton("Continue to Plan →")
        nxt.setObjectName("primary")
        nxt.clicked.connect(go_next)
        right.addWidget(nxt, 0, Qt.AlignRight)
        state.session_changed.connect(self._load_existing)
        self._load_existing()

    def _need_session(self) -> bool:
        if self.state.session is None:
            QMessageBox.information(
                self, "No session", "Start or resume a session on the Start page first."
            )
            return False
        return confirm_playback(self, self.state)

    def _busy(self, on: bool) -> None:
        self.loop_btn.setEnabled(not on)
        self.chain_btn.setEnabled(not on)

    def _run_loop(self) -> None:
        if not self._need_session():
            return
        self._busy(True)
        self.loop_list.clear()
        self.loop_list.add("busy", "Running…")
        run_async(
            run_loopback_test,
            self.state.engine,
            self.state.sample_rate,
            on_done=self._loop_done,
            on_fail=self._failed,
        )

    def _loop_done(self, res) -> None:
        self._busy(False)
        self.loop_result = res
        self._show_advice(self.loop_list, res.advice)
        self._store()

    def _run_chain(self) -> None:
        if not self._need_session():
            return
        st = self.state
        cal_name = st.profile.raw.get("calibration_preset", "flat")
        try:
            demo_pedal(st, st.profile.preset(cal_name).controls)
        except KeyError:
            demo_pedal(st, None)
        train = st.session.signal("train")
        train_peak = peak_dbfs(sf.read(str(train.wav_path), dtype="float32")[0])
        self._busy(True)
        self.chain_list.clear()
        self.chain_list.add("busy", "Running…")
        run_async(
            run_chain_test,
            st.engine,
            st.sample_rate,
            st.profile.send_level_db,
            train_peak,
            on_done=self._chain_done,
            on_fail=self._failed,
        )

    def _chain_done(self, res) -> None:
        self._busy(False)
        self.chain_result = res
        self._show_advice(self.chain_list, res.advice)
        self.table.setRowCount(len(res.steps))
        for i, s in enumerate(res.steps):
            for j, v in enumerate((s.send_dbfs, s.return_dbfs, s.gain_db)):
                self.table.setItem(i, j, QTableWidgetItem(f"{v:+.2f}"))
        self._store()

    def _show_advice(self, lst: CheckList, advice) -> None:
        lst.clear()
        for a in advice:
            status, text = (a.status, a.text) if hasattr(a, "status") else (a["status"], a["text"])
            lst.add(status, text)

    def _store(self) -> None:
        s = self.state.session
        old = s.data.get("calibration") or {}
        loop = self.loop_result.to_dict() if self.loop_result else old.get("loopback")
        chain = self.chain_result.to_dict() if self.chain_result else old.get("chain")
        path = s.set_calibration(loop, chain)
        self.state.set_cables_from_calibration(s.data["calibration"])
        self.state.calibration_changed.emit()
        self.save_note.setText(f"Calibration saved: {s.root.name}/{path.name} (and session.json).")

    def _load_existing(self) -> None:
        self.loop_result = self.chain_result = None
        self.loop_list.clear()
        self.chain_list.clear()
        self.table.setRowCount(0)
        cal = self.state.calibration or {}
        if cal.get("loopback"):
            self._show_advice(self.loop_list, cal["loopback"]["advice"])
        if cal.get("chain"):
            self._show_advice(self.chain_list, cal["chain"]["advice"])
            steps = cal["chain"]["steps"]
            self.table.setRowCount(len(steps))
            for i, s in enumerate(steps):
                for j, k in enumerate(("send_dbfs", "return_dbfs", "gain_db")):
                    self.table.setItem(i, j, QTableWidgetItem(f"{s[k]:+.2f}"))

    def _failed(self, tb: str) -> None:
        self._busy(False)
        QMessageBox.critical(self, "Test failed", tb[-2000:])
