"""Start page: demo or hardware, sample rate, pedal, signal; new session or resume one."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.backends import list_asio_devices
from pedal_model.rig.config import SAMPLE_RATES
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.state import DEMO_SPEEDS, AppState
from pedal_model.rig.gui.widgets import card, label
from pedal_model.rig.gui.worker import run_async
from pedal_model.rig.profiles import available_profiles
from pedal_model.rig.session import Session
from pedal_model.rig.signals import SIGNAL_KINDS


class StartPage(QWidget):
    def __init__(self, state: AppState, go_next) -> None:
        super().__init__()
        self.state = state
        self.go_next = go_next
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Start", "title"))
        root.addWidget(
            label(
                "Pick how to run, then start a new session or resume one. The sample rate is fixed "
                "for a session: changing it re-clocks the whole 1824c.",
                "dim",
            )
        )
        grid = QGridLayout()
        grid.setSpacing(14)
        root.addLayout(grid, 1)

        # Mode
        f, lay = card("1. Mode")
        self.demo = QRadioButton("Demo: simulated rig (nothing is played; safe anywhere)")
        self.hw = QRadioButton("Hardware: Studio 1824c over ASIO")
        self.demo.setChecked(True)
        lay.addWidget(self.demo)
        lay.addWidget(self.hw)
        self.hw_status = label("", "dim")
        lay.addWidget(self.hw_status)
        row = QHBoxLayout()
        row.addWidget(label("Demo speed:", wrap=False))
        self.speed = QComboBox()
        for s in DEMO_SPEEDS:
            self.speed.addItem(f"{s:g}× real time", s)
        self.speed.setCurrentIndex(DEMO_SPEEDS.index(state.demo_speed))
        row.addWidget(self.speed)
        row.addStretch()
        lay.addLayout(row)
        lay.addStretch()
        grid.addWidget(f, 0, 0)

        # Session settings
        f, lay = card("2. Session settings")
        lay.addWidget(label("Sample rate"))
        self.sr_group = QButtonGroup(self)
        r = QHBoxLayout()
        for sr in SAMPLE_RATES:
            b = QRadioButton(f"{sr // 1000} kHz")
            b.setChecked(sr == state.sample_rate)
            self.sr_group.addButton(b, sr)
            r.addWidget(b)
        r.addStretch()
        lay.addLayout(r)
        lay.addWidget(label("Pedal profile"))
        self.profile = QComboBox()
        for pid in available_profiles():
            self.profile.addItem(pid)
        lay.addWidget(self.profile)
        lay.addWidget(label("Signal"))
        self.kind_group = QButtonGroup(self)
        for k, text in SIGNAL_KINDS.items():
            b = QRadioButton(text)
            b.setChecked(k == state.signal_kind)
            self.kind_group.addButton(b)
            b.setProperty("kind", k)
            lay.addWidget(b)
        self.new_btn = QPushButton("New session")
        self.new_btn.setObjectName("primary")
        self.new_btn.clicked.connect(self._new_session)
        lay.addStretch()
        lay.addWidget(self.new_btn, 0, Qt.AlignLeft)
        grid.addWidget(f, 0, 1)

        # Resume
        f, lay = card("…or resume a session")
        self.sessions = QListWidget()
        lay.addWidget(self.sessions, 1)
        b = QHBoxLayout()
        self.open_btn = QPushButton("Resume selected")
        self.open_btn.clicked.connect(self._resume)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_sessions)
        b.addWidget(self.open_btn)
        b.addWidget(refresh)
        b.addStretch()
        lay.addLayout(b)
        grid.addWidget(f, 1, 0, 1, 2)

        self.status = label("", "dim")
        root.addWidget(self.status)
        self.demo.toggled.connect(self._mode_changed)
        self.speed.currentIndexChanged.connect(self._mode_changed)
        self._check_hardware()
        self.refresh_sessions()

    def _check_hardware(self) -> None:
        try:
            names = [d[1] for d in list_asio_devices()]
        except Exception as e:  # noqa: BLE001 — PortAudio missing or broken: show why
            names, err = [], str(e)
        else:
            err = ""
        want = self.state.rig.asio_device
        if want in names:
            self.hw_status.setText(
                f"<span style='color:{theme.OK}'>●</span> Found “{want}” (nothing has been played)."
            )
        else:
            self.hw_status.setText(
                f"<span style='color:{theme.ACCENT}'>●</span> “{want}” not found. {err} "
                "Is the 1824c on, and Studio One closed?"
            )
            self.hw.setEnabled(False)

    def _mode_changed(self) -> None:
        self.state.set_mode(self.demo.isChecked(), self.speed.currentData())
        self.refresh_sessions()

    def refresh_sessions(self) -> None:
        self.sessions.clear()
        for d in Session.list_sessions(self.profile.currentText(), simulated=self.demo.isChecked()):
            try:
                s = Session.load(d)
            except Exception:  # noqa: BLE001 — skip unreadable folders
                continue
            done = sum(p["status"] == "accepted" for p in s.data["presets"].values())
            item = QListWidgetItem(
                f"{d.parent.name}/{d.name}   ·   {s.data['signal_kind']}   ·   "
                f"{done}/{len(s.data['presets'])} presets accepted"
            )
            item.setData(Qt.UserRole, str(d))
            self.sessions.addItem(item)

    def _selected_kind(self) -> str:
        return self.kind_group.checkedButton().property("kind")

    def _new_session(self) -> None:
        st = self.state
        st.sample_rate = self.sr_group.checkedId()
        st.signal_kind = self._selected_kind()
        self.new_btn.setEnabled(False)
        self.status.setText("Preparing the session's signals (resampling the 96 kHz masters)…")
        run_async(
            Session.create,
            st.profile,
            st.sample_rate,
            st.signal_kind,
            st.rig,
            st.engine.backend.name,
            st.simulated,
            st.plan,
            on_done=self._session_ready,
            on_fail=self._failed,
        )

    def _session_ready(self, session: Session) -> None:
        self.new_btn.setEnabled(True)
        self.status.setText(f"Session created: {session.root}")
        self.state.set_session(session)
        self.refresh_sessions()
        self.go_next()

    def _resume(self) -> None:
        item = self.sessions.currentItem()
        if not item:
            return
        s = Session.load(Path(item.data(Qt.UserRole)))
        if bool(s.data["simulated"]) != self.state.simulated:
            self.state.set_mode(bool(s.data["simulated"]), self.state.demo_speed)
        self.state.set_session(s)
        self.status.setText(f"Resumed: {s.root}")
        self.go_next()

    def _failed(self, tb: str) -> None:
        self.new_btn.setEnabled(True)
        self.status.setText("Failed, see the message.")
        QMessageBox.critical(self, "Could not create the session", tb[-2000:])
