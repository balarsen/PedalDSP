"""Review page: the session's presets and QC, a response plot per preset, and a comparison with
another session.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.compare import agreement, preset_response
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import card, label
from pedal_model.rig.gui.worker import run_async
from pedal_model.rig.session import Session

COLUMNS = [
    "Preset",
    "Status",
    "Train QC",
    "Val QC",
    "Alignment",
    "Pedal delay",
    "Response vs model",
]


def _qc_value(take: dict | None, name: str) -> str:
    if not take:
        return ""
    for item in take["qc"]:
        if item["name"] == name:
            return item["value"]
    return ""


class ReviewPage(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Review", "title"))
        self.path = label("", "dim")
        root.addWidget(self.path)
        body = QHBoxLayout()
        root.addLayout(body, 1)

        f, lay = card("Presets in this session")
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._plot_selected)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        b1 = QPushButton("Open session folder")
        b1.clicked.connect(
            lambda: self._open(self.state.session.root if self.state.session else None)
        )
        b2 = QPushButton("Open preset folder")
        b2.clicked.connect(self._open_preset)
        b3 = QPushButton("Refresh")
        b3.clicked.connect(self.refresh)
        for b in (b1, b2, b3):
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        body.addWidget(f, 5)

        right = QVBoxLayout()
        body.addLayout(right, 6)
        f, lay = card("Response of the selected preset")
        crow = QHBoxLayout()
        crow.addWidget(label("Compare with:", wrap=False))
        self.other = QComboBox()
        self.other.currentIndexChanged.connect(self._plot_selected)
        crow.addWidget(self.other, 1)
        lay.addLayout(crow)
        self.plot = theme.styled_plot("", "Frequency (Hz)", "Gain (dB)", log_x=True)
        self.plot.setXRange(np.log10(20), np.log10(20_000))
        self.c_exp = self.plot.plot(
            pen=pg.mkPen(theme.AMBER, width=2, style=Qt.DashLine), name="model"
        )
        self.c_a = self.plot.plot(pen=pg.mkPen(theme.OK, width=2.5), name="this session")
        self.c_b = self.plot.plot(pen=pg.mkPen(theme.CYAN, width=2), name="compared session")
        lay.addWidget(self.plot, 1)
        self.agree = label("", "dim")
        lay.addWidget(self.agree)
        right.addWidget(f, 1)

        state.session_changed.connect(self.refresh)
        self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        s = self.state.session
        self.table.setRowCount(0)
        if s is None:
            self.path.setText("No session open.")
            return
        s.data = Session.load(s.root).data
        self.path.setText(
            f"{s.root}   ·   {s.sample_rate // 1000} kHz   ·   {s.data['signal_kind']} signal   "
            "·   "
            f"{s.data['backend']}"
        )
        presets = s.data["presets"]
        self.table.setRowCount(len(presets))
        for r, (name, p) in enumerate(presets.items()):
            takes = {t["role"]: t for t in p["takes"]}
            vals = [
                name,
                p["status"],
                takes.get("train", {}).get("verdict", ""),
                takes.get("val", {}).get("verdict", ""),
                _qc_value(takes.get("train"), "Alignment"),
                _qc_value(takes.get("train"), "Pedal delay"),
                _qc_value(takes.get("train"), "Response vs model"),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if v in theme.STATUS_COLOURS:
                    it.setForeground(pg.mkColor(theme.STATUS_COLOURS[v]))
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self.other.blockSignals(True)
        self.other.clear()
        self.other.addItem("(none)", None)
        for d in Session.list_sessions(s.data["profile"]):
            if d != s.root:
                self.other.addItem(f"{d.parent.name}/{d.name}", str(d))
        self.other.blockSignals(False)

    def _selected_preset(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.table.item(rows[0].row(), 0).text() if rows else None

    def _plot_selected(self) -> None:
        name = self._selected_preset()
        s = self.state.session
        if not name or s is None:
            return
        other = self.other.currentData()

        def work():
            a = preset_response(s, name)
            b = preset_response(Session.load(Path(other)), name) if other else None
            return a, b

        self.agree.setText("Computing…")
        run_async(
            work, on_done=self._show, on_fail=lambda tb: self.agree.setText(tb.splitlines()[-1])
        )

    def _show(self, result) -> None:
        a, b = result
        for c in (self.c_exp, self.c_a, self.c_b):
            c.setData([], [])
        if a is None:
            self.agree.setText("This preset has no capture yet.")
            return
        self.c_a.setData(a.freqs_hz, a.measured_db)
        if a.expected_db is not None:
            self.c_exp.setData(a.freqs_hz, a.expected_db)
        if b is None:
            self.agree.setText(
                "" if self.other.currentData() is None else "Not captured in the compared session."
            )
            return
        self.c_b.setData(b.freqs_hz, b.measured_db)
        ag = agreement(a, b)
        self.agree.setText(
            f"{a.sample_rate // 1000} kHz vs {b.sample_rate // 1000} kHz: RMS difference "
            f"{ag['rms_below_10k_db']:.2f} dB below 10 kHz, {ag['rms_above_10k_db']:.2f} dB above; "
            f"offset {ag['offset_db']:+.2f} dB."
        )

    def _open(self, path) -> None:
        if path is None:
            return
        if not os.path.isdir(path):
            QMessageBox.information(self, "Not there yet", f"{path} does not exist yet.")
            return
        os.startfile(str(path))  # noqa: S606 — opens Explorer on our own folder

    def _open_preset(self) -> None:
        name = self._selected_preset()
        if name and self.state.session:
            self._open(self.state.session.preset_dir(name))
