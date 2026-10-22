"""Main window: step navigation (Start → Setup → Calibrate → Plan → Run → Review), mode badge, tools
menu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.config import code_version
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.page_calibrate import CalibratePage
from pedal_model.rig.gui.page_plan import PlanPage
from pedal_model.rig.gui.page_review import ReviewPage
from pedal_model.rig.gui.page_run import RunPage
from pedal_model.rig.gui.page_setup import SetupPage
from pedal_model.rig.gui.page_start import StartPage
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import label

STEPS = ["1  Start", "2  Setup", "3  Calibrate", "4  Plan", "5  Run", "6  Review"]
BRIAN_APPS = Path(
    r"C:\Program Files\Brian Larsen"
)  # where the BrianAudioLab installers put the Standalones


class MainWindow(QMainWindow):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        self.setWindowTitle("PedalDSP Capture Rig")
        # Most of the screen: a 2560x1440 display shows every Run-page plot without scrolling;
        # on smaller screens the pages scroll.
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(2400, int(0.94 * screen.width())), min(1360, int(0.92 * screen.height())))

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(16, 10, 16, 10)
        top.addWidget(label("<b>PedalDSP</b> capture rig", "heading", wrap=False))
        top.addStretch()
        self.badge = label("", "badge", wrap=False)
        self.session_label = label("", "dim", wrap=False)
        top.addWidget(self.session_label)
        top.addWidget(self.badge)
        outer.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(170)
        self.nav.addItems(STEPS)
        self.stack = QStackedWidget()
        go = self.go
        for page in (
            StartPage(state, lambda: go(1)),
            SetupPage(state, lambda: go(2)),
            CalibratePage(state, lambda: go(3)),
            PlanPage(state, lambda: go(4)),
            RunPage(state, lambda: go(5)),
            ReviewPage(state),
        ):
            # Each page scrolls when the window is shorter than it, instead of forcing the window
            # taller than a laptop screen.
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        self.setCentralWidget(central)

        tools = self.menuBar().addMenu("Tools")
        for app in ("BrianEQ", "BrianLevel", "BrianTune"):
            act = QAction(f"Launch {app} (Standalone)", self)
            act.triggered.connect(lambda _=False, a=app: self.launch(a))
            tools.addAction(act)
        helpm = self.menuBar().addMenu("Help")
        about = QAction("Running BrianEQ alongside…", self)
        about.triggered.connect(self._brian_help)
        helpm.addAction(about)

        state.rig_changed.connect(self._update_header)
        state.session_changed.connect(self._update_header)
        self._update_header()
        self.nav.setCurrentRow(0)

    def go(self, index: int) -> None:
        self.nav.setCurrentRow(index)

    def _update_header(self) -> None:
        if self.state.simulated:
            self.badge.setText("DEMO · simulated rig")
            self.badge.setStyleSheet(f"background:{theme.AMBER}; color:#000;")
        else:
            self.badge.setText("HARDWARE · ASIO")
            self.badge.setStyleSheet(f"background:{theme.OK}; color:#000;")
        s = self.state.session
        where = f"{s.root.name} · {s.sample_rate // 1000} kHz" if s else "no session"
        self.session_label.setText(f"{where} · code {code_version()}   ")
        self.session_label.setToolTip(
            "Git commit of the running code (saved in every JSON and FLAC). "
            "'-dirty' means it has uncommitted edits."
        )

    def launch(self, app: str) -> None:
        exe = BRIAN_APPS / app / f"{app}.exe"
        if not exe.is_file():
            QMessageBox.warning(self, "Not installed", f"{exe} was not found.")
            return
        QProcess.startDetached(str(exe), [])

    def _brian_help(self) -> None:
        QMessageBox.information(
            self,
            "Running BrianEQ alongside",
            "This app holds the 1824c over ASIO. To watch the pedal return in BrianEQ / "
            "BrianLevel at the same time, "
            "open the Standalone's audio settings and pick the Windows Audio device "
            "“Mic/Inst/Line In 1/2 (Studio 1824c)” instead of ASIO.\n\n"
            "Whether the PreSonus driver allows both at once is the Phase 0b hardware test: not "
            "verified yet.",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pedal_model.rig", description="PedalDSP capture rig GUI"
    )
    parser.add_argument(
        "--hardware", action="store_true", help="start in hardware (ASIO) mode instead of demo"
    )
    args = parser.parse_args(argv)
    code_version()  # pin the version of the code as loaded, before anything is edited
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(theme.STYLESHEET)
    theme.setup_pyqtgraph()
    state = AppState()
    if args.hardware:
        state.set_mode(False, state.demo_speed)
    win = MainWindow(state)
    win.show()
    return app.exec()
