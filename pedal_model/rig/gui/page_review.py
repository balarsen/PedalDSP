"""Review page: presets and QC with a response plot (and session comparison), and the slider map."""

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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.analysis import log_grid
from pedal_model.rig.bandfit import fit_band
from pedal_model.rig.compare import agreement, preset_response
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.page_run import RESPONSE_MIN_SPAN_DB, RESPONSE_Y, _auto_box
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import card, label
from pedal_model.rig.gui.worker import run_async
from pedal_model.rig.profiles import KIND_MAP
from pedal_model.rig.qc import expected_response
from pedal_model.rig.session import Session

COLUMNS = ["Step", "Status", "QC", "Alignment", "Pedal delay", "Response vs model"]
MAP_COLUMNS = [
    "Slider",
    "Setting",
    "Fit",
    "Centre (Hz)",
    "Q",
    "BW (oct)",
    "Gain (dB)",
    "Slope",
    "Model centre",
    "Model Q",
    "Model gain",
]
POSITION_PENS = [theme.OK, theme.CYAN, theme.AMBER, theme.ACCENT, theme.CYAN_BRIGHT, theme.TEXT]


def _qc_value(take: dict | None, name: str) -> str:
    for item in (take or {}).get("qc", []):
        if item["name"] == name:
            return item["value"]
    return ""


def _fmt(v, spec: str) -> str:
    return "" if v is None else format(v, spec)


class ReviewPage(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.state = state
        self.map_rows: list = []
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Review", "title"))
        self.path = label("", "dim")
        root.addWidget(self.path)
        tabs = QTabWidget()
        root.addWidget(tabs, 1)
        tabs.addTab(self._presets_tab(), "Presets")
        tabs.addTab(self._map_tab(), "Slider map")
        state.session_changed.connect(self.refresh)
        self.refresh()

    # ── presets tab ─────────────────────────────────────────────────────── #

    def _presets_tab(self) -> QWidget:
        w = QWidget()
        body = QHBoxLayout(w)
        f, lay = card("Steps in this session (click one to plot it)")
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._plot_selected)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, fn in (
            ("Open session folder", self._open_session),
            ("Open step folder", self._open_preset),
            ("Refresh", self.refresh),
        ):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        body.addWidget(f, 5)

        f, lay = card("Response of the selected step")
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
        lay.addWidget(_auto_box(self.plot, RESPONSE_Y, RESPONSE_MIN_SPAN_DB), 0, Qt.AlignRight)
        self.agree = label("", "dim")
        lay.addWidget(self.agree)
        body.addWidget(f, 6)
        return w

    # ── slider map tab ──────────────────────────────────────────────────── #

    def _map_tab(self) -> QWidget:
        w = QWidget()
        body = QHBoxLayout(w)
        f, lay = card("Slider map: fitted centre, bandwidth and gain per slider position")
        lay.addWidget(
            label(
                "Measured by the map steps (Plan → Add slider map). The 'Model' columns are the "
                "same fit applied to Ge7Sim at that setting, so differences show where the model "
                "and the pedal disagree. Click a row to overlay every position of that slider.",
                "dim",
            )
        )
        self.map_table = QTableWidget(0, len(MAP_COLUMNS))
        self.map_table.setHorizontalHeaderLabels(MAP_COLUMNS)
        self.map_table.verticalHeader().setVisible(False)
        self.map_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.map_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.map_table.itemSelectionChanged.connect(self._plot_map)
        lay.addWidget(self.map_table, 1)
        body.addWidget(f, 6)
        f, lay = card("All positions of the selected slider")
        self.map_plot = theme.styled_plot("", "Frequency (Hz)", "Gain (dB)", log_x=True)
        self.map_plot.setXRange(np.log10(20), np.log10(20_000))
        lay.addWidget(self.map_plot, 1)
        lay.addWidget(_auto_box(self.map_plot, RESPONSE_Y, RESPONSE_MIN_SPAN_DB), 0, Qt.AlignRight)
        self.map_note = label("", "dim")
        lay.addWidget(self.map_note)
        body.addWidget(f, 5)
        return w

    # ── data ────────────────────────────────────────────────────────────── #

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        s = self.state.session
        self.table.setRowCount(0)
        self.map_table.setRowCount(0)
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
        self.table.blockSignals(True)
        self.table.setRowCount(len(presets))
        first_captured = None
        for r, (name, p) in enumerate(presets.items()):
            takes = {t["role"]: t for t in p["takes"]}
            main = takes.get("sweep") or takes.get("train")
            verdicts = "/".join(t["verdict"] for t in p["takes"])
            worst = "fail" if "fail" in verdicts else ("warn" if "warn" in verdicts else "ok")
            vals = [
                name,
                p["status"],
                verdicts,
                _qc_value(main, "Alignment"),
                _qc_value(main, "Pedal delay"),
                _qc_value(main, "Response vs model"),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c == 2 and verdicts:
                    it.setForeground(pg.mkColor(theme.STATUS_COLOURS[worst]))
                self.table.setItem(r, c, it)
            if first_captured is None and main:
                first_captured = r
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self.other.blockSignals(True)
        self.other.clear()
        self.other.addItem("(none)", None)
        for d in Session.list_sessions(s.data["profile"]):
            if d != s.root:
                self.other.addItem(f"{d.parent.name}/{d.name}", str(d))
        self.other.blockSignals(False)
        if first_captured is not None:
            self.table.selectRow(first_captured)  # selecting plots it
        else:
            self.agree.setText("Nothing captured yet in this session.")
        self._fill_map()

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

        self.agree.setText("Computing the response…")
        run_async(
            work, on_done=self._show, on_fail=lambda tb: self.agree.setText(tb.splitlines()[-1])
        )

    def _show(self, result) -> None:
        a, b = result
        for c in (self.c_exp, self.c_a, self.c_b):
            c.setData([], [])
        if a is None:
            self.agree.setText("This step has no capture yet.")
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

    def _fill_map(self) -> None:
        s = self.state.session
        prof = self.state.profile
        rows = []
        grid = log_grid(20, 20_000)
        top_hz = max(k.get("band_hz") or 0 for k in prof.controls)
        for name, p in s.data["presets"].items():
            if p.get("kind") != KIND_MAP:
                continue
            take = next((t for t in p["takes"] if t["role"] == "sweep"), None)
            fit = (take or {}).get("band_fit")
            if not fit:
                continue
            c = next(k for k in prof.controls if k["key"] == fit["control"])
            exp = expected_response(prof.pedal, p["controls"], s.sample_rate, grid)
            model = (
                fit_band(
                    grid,
                    exp,
                    s.sample_rate,
                    c.get("band_hz"),
                    fit["setting"],
                    try_shelf=c.get("band_hz") == top_hz,
                )
                if exp is not None
                else None
            )
            rows.append((name, c["label"], fit, model))
        order = {k["key"]: i for i, k in enumerate(prof.controls)}
        rows.sort(key=lambda r: (order[r[2]["control"]], -r[2]["setting"]))
        self.map_rows = rows
        self.map_table.blockSignals(True)
        self.map_table.setRowCount(len(rows))
        for i, (_name, lab, fit, model) in enumerate(rows):
            vals = [
                lab,
                f"{fit['setting']:+g} dB",
                fit["model"],
                _fmt(fit["fc_hz"], ".0f"),
                _fmt(fit["q"], ".2f"),
                _fmt(fit["bw_oct"], ".2f"),
                f"{fit['gain_db']:+.2f}",
                _fmt(fit.get("slope"), ".2f"),
                _fmt(model.fc_hz if model else None, ".0f"),
                _fmt(model.q if model else None, ".2f"),
                _fmt(model.gain_db if model else None, "+.2f"),
            ]
            for j, v in enumerate(vals):
                self.map_table.setItem(i, j, QTableWidgetItem(v))
        self.map_table.resizeColumnsToContents()
        self.map_table.blockSignals(False)
        self.map_note.setText("" if rows else "No slider-map steps captured in this session yet.")

    def _plot_map(self) -> None:
        rows = self.map_table.selectionModel().selectedRows()
        if not rows:
            return
        control = self.map_rows[rows[0].row()][2]["control"]
        names = [r[0] for r in self.map_rows if r[2]["control"] == control]
        s = self.state.session

        def work():
            return [(n, preset_response(s, n)) for n in names]

        self.map_note.setText("Computing…")
        run_async(
            work,
            on_done=self._show_map,
            on_fail=lambda tb: self.map_note.setText(tb.splitlines()[-1]),
        )

    def _show_map(self, results) -> None:
        self.map_plot.clear()  # also empties the legend
        for i, (name, r) in enumerate(results):
            if r is None:
                continue
            setting = self.state.session.data["presets"][name]["controls"]
            val = next(v for v in setting.values() if abs(v) > 1e-6)
            colour = POSITION_PENS[i % len(POSITION_PENS)]
            self.map_plot.plot(
                r.freqs_hz, r.measured_db, pen=pg.mkPen(colour, width=2), name=f"{val:+g} dB"
            )
            if r.expected_db is not None:
                self.map_plot.plot(
                    r.freqs_hz, r.expected_db, pen=pg.mkPen(colour, width=1, style=Qt.DashLine)
                )
        self.map_note.setText("Solid: measured. Dashed: Ge7Sim model at the same setting.")

    def _open_session(self) -> None:
        self._open(self.state.session.root if self.state.session else None)

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
