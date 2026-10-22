"""GE-7 analysis GUI: train the block model on a capture session, live, and watch it settle.

A separate program from the capture rig (its own process: a fit can never disturb the audio
stream). It follows a session folder: after each new capture it updates the dataset (only the new
take is computed), refits with a warm start, and adds a point to the training history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import torch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pedal_model.ode import ge7_model as M
from pedal_model.ode import ge7_train as T
from pedal_model.rig.config import code_version
from pedal_model.rig.dataset import session_dataset
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.widgets import CheckText, card, label, show_error
from pedal_model.rig.gui.worker import run_async
from pedal_model.rig.profiles import load_profile
from pedal_model.rig.session import Session

FOLLOW_MS = 5000  # how often "Follow live" looks for new captures
# Parameter groups for the "how well pinned" plot: median posterior 1σ (%) of each group.
GROUPS = {
    "band centre f": ("band_f",),
    "band Q": ("band_q",),
    "band strength σ": ("band_sigma",),
    "6.4k shelf": ("shelf_f", "shelf_sigma"),
    "EQ ρ in/fb": ("eq_rho_in", "eq_rho_fb"),
    "EQ pot law γ": ("eq_gamma",),
    "LEVEL stage": ("lvl_kappa", "lvl_f", "lvl_rho_in", "lvl_rho_fb"),
    "LEVEL pot law γ": ("lvl_gamma",),
    "overall gain": ("ratio_gain",),
}
GROUP_PENS = [
    theme.CYAN,
    theme.OK,
    theme.AMBER,
    theme.ACCENT,
    theme.CYAN_BRIGHT,
    "#c58cff",
    "#ff7ab8",
    "#9fd86b",
    theme.TEXT_DIM,
]


def _plot(title: str, xlabel: str, ylabel: str, log_x: bool = False) -> pg.PlotWidget:
    p = theme.styled_plot(title, xlabel, ylabel, log_x=log_x)
    p.addLegend(offset=(10, 10))
    # take the height there is (the window is sized to the screen), at least 220 px
    p.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
    p.setMinimumHeight(220)
    return p


class AnalysisWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"PedalDSP GE-7 analysis  ·  code {code_version()}")
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(2400, int(0.94 * screen.width())), min(1360, int(0.92 * screen.height())))
        self.profile = load_profile("ge7")
        self.root: Path | None = None
        self.cache: dict = {}
        self.dataset: dict | None = None
        self.fit: dict | None = None
        self.history: list[dict] = []
        self.busy = False
        self.seen = None  # (session.json mtime, number of takes) at the last update

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 10, 16, 10)
        top = QHBoxLayout()
        top.addWidget(label("<b>GE-7 block model</b> · training", "heading", wrap=False))
        top.addSpacing(20)
        self.sessions = QComboBox()
        self.sessions.setMinimumWidth(320)
        self.sessions.currentIndexChanged.connect(self._session_chosen)
        top.addWidget(self.sessions)
        refresh = QPushButton("↻ Sessions")
        refresh.clicked.connect(self._list_sessions)
        top.addWidget(refresh)
        self.follow = QCheckBox("Follow live (refit after each new capture)")
        self.follow.setChecked(True)
        top.addWidget(self.follow)
        self.fit_btn = QPushButton("Fit now")
        self.fit_btn.setObjectName("primary")
        self.fit_btn.clicked.connect(lambda: self._update(force=True))
        top.addWidget(self.fit_btn)
        self.write_btn = QPushButton("Write parameter file…")
        self.write_btn.clicked.connect(self._write)
        top.addWidget(self.write_btn)
        top.addStretch()
        self.status = label("", "dim", wrap=False)
        top.addWidget(self.status)
        outer.addLayout(top)

        body = QHBoxLayout()
        outer.addLayout(body, 1)
        left = QVBoxLayout()
        body.addLayout(left, 5)
        f, lay = card("Training progress: model error vs settings captured")
        self.p_rms = _plot("", "Settings in the fit", "RMS error (dB)")
        self.c_fit = self.p_rms.plot(
            pen=pg.mkPen(theme.CYAN, width=2), symbol="o", symbolSize=6, name="fit settings"
        )
        self.c_val = self.p_rms.plot(
            pen=pg.mkPen(theme.AMBER, width=2), symbol="s", symbolSize=6, name="validation"
        )
        floor = T.DATA_NOISE_NP * 20 / np.log(10)
        self.p_rms.addLine(
            y=floor, pen=pg.mkPen(theme.TEXT_DIM, width=1, style=Qt.DashLine)
        )  # the re-set floor
        lay.addWidget(self.p_rms, 1)
        lay.addWidget(
            label(
                f"Dashed: the re-set floor ({floor:.2f} dB), the most a model can be expected to "
                "reach while sliders are set by hand.",
                "dim",
            )
        )
        left.addWidget(f, 1)
        f, lay = card("How well each parameter is pinned (posterior 1σ; the schematic prior first)")
        self.p_std = _plot("", "Settings in the fit", "1σ (%)")
        self.p_std.setLogMode(x=False, y=True)
        self.c_std = {
            g: self.p_std.plot(pen=pg.mkPen(GROUP_PENS[i % len(GROUP_PENS)], width=2), name=g)
            for i, g in enumerate(GROUPS)
        }
        lay.addWidget(self.p_std, 1)
        left.addWidget(f, 1)

        right = QVBoxLayout()
        body.addLayout(right, 6)
        f, lay = card("A take, measured vs model (engaged ÷ bypassed)")
        row = QHBoxLayout()
        row.addWidget(label("Take:", "dim", wrap=False))
        self.takes = QComboBox()
        self.takes.setMinimumWidth(280)
        self.takes.currentIndexChanged.connect(self._show_take)
        row.addWidget(self.takes)
        row.addStretch()
        lay.addLayout(row)
        self.p_take = _plot("", "Frequency (Hz)", "Gain (dB)", log_x=True)
        self.p_take.setXRange(np.log10(20), np.log10(20_000))
        self.c_meas = self.p_take.plot(pen=pg.mkPen(theme.OK, width=2.5), name="measured")
        self.c_model = self.p_take.plot(
            pen=pg.mkPen(theme.AMBER, width=2, style=Qt.DashLine), name="trained model"
        )
        self.c_sch = self.p_take.plot(
            pen=pg.mkPen(theme.TEXT_DIM, width=1, style=Qt.DotLine), name="schematic only"
        )
        lay.addWidget(self.p_take, 1)
        right.addWidget(f, 1)
        f, lay = card("Parameters")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["parameter", "schematic", "trained", "±1σ", "prior σ", "trained / schematic"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)
        right.addWidget(f, 1)
        f, lay = card("Log")
        self.log = CheckText()
        self.log.setMinimumHeight(110)
        lay.addWidget(self.log)
        right.addWidget(f)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(FOLLOW_MS)
        self._list_sessions()

    # ── sessions ────────────────────────────────────────────────────────── #

    def _list_sessions(self) -> None:
        self.sessions.blockSignals(True)
        self.sessions.clear()
        for d in Session.list_sessions("ge7"):
            self.sessions.addItem(f"{d.parent.name}/{d.name}", str(d))
        self.sessions.blockSignals(False)
        self._session_chosen()

    def _session_chosen(self, *_) -> None:
        path = self.sessions.currentData()
        self.root = Path(path) if path else None
        # a fresh cache object: a fit still running for the previous session keeps its own
        self.cache, self.dataset, self.fit, self.seen = {}, None, None, None
        self.history = []
        if self.root is not None:
            hist = T.history_path(self.root.name)
            if hist.is_file():
                self.history = json.loads(hist.read_text(encoding="utf-8"))
            self.log.clear()
            self.log.add("ok", f"Session {self.root.name}", f"{len(self.history)} earlier fits.")
        self._draw_history()
        self._update(force=True)

    def _poll(self) -> None:
        if self.follow.isChecked() and not self.busy:
            self._update(force=False)

    # ── fit ─────────────────────────────────────────────────────────────── #

    def _update(self, force: bool) -> None:
        if self.root is None or self.busy:
            return
        sj = self.root / "session.json"
        if not sj.is_file():
            return
        data = json.loads(sj.read_text(encoding="utf-8"))
        n_takes = sum(len(p.get("takes", [])) for p in data["presets"].values())
        seen = (sj.stat().st_mtime, n_takes)
        if not force and seen == self.seen:
            return
        if not force and self.seen is not None and n_takes == self.seen[1]:
            self.seen = seen  # session.json changed without a new take (a status, a log line)
            return
        self.seen = seen
        self.busy = True
        self.fit_btn.setEnabled(False)
        self.status.setText(f"Updating: {n_takes} takes…")
        start = self.fit["params"] if self.fit else None
        run_async(
            self._work,
            self.root,
            start,
            on_done=self._done,
            on_fail=self._failed,
        )

    def _work(self, root: Path, start: dict | None) -> tuple[Path, dict, dict | None, str]:
        ds = session_dataset(root, cache=self.cache)
        try:
            fit = T.fit_dataset(ds, self.profile.raw, start=start)
        except ValueError as e:  # nothing to fit yet
            return root, ds, None, str(e)
        return root, ds, fit, ""

    def _done(self, result) -> None:
        self.busy = False
        self.fit_btn.setEnabled(True)
        root, ds, fit, why = result
        if root != self.root:  # the session was changed while this ran: start over for the new one
            self.seen = None
            self._update(force=True)
            return
        self.dataset = ds
        self._fill_takes()
        if fit is None:
            self.status.setText(why)
            return
        self.fit = fit
        self.history = T.append_history(self.root.name, fit)
        self._draw_history()
        self._fill_table()
        self._show_take()
        val = fit["rms_validation_db"]
        self.status.setText(
            f"{fit['n_fit']} fit settings: {fit['rms_fit_db']:.2f} dB RMS"
            + ("" if np.isnan(val) else f"; validation {val:.2f} dB")
        )
        self.log.add(
            "ok" if fit["converged"] else "warn",
            f"Fit {len(self.history)}: {fit['n_fit']} settings, {fit['rms_fit_db']:.2f} dB",
            ("" if np.isnan(val) else f"validation {val:.2f} dB; ")
            + f"{len(fit['groups']['test'])} test takes kept aside"
            + ("" if fit["converged"] else "; not fully converged (continues next update)"),
        )

    def _failed(self, tb: str) -> None:
        self.busy = False
        self.fit_btn.setEnabled(True)
        self.status.setText("Error (see the log)")
        self.log.add("fail", "Update failed", tb.splitlines()[-1])
        show_error(self, "Analysis error", tb)

    # ── drawing ─────────────────────────────────────────────────────────── #

    def _draw_history(self) -> None:
        h = self.history
        x = np.array([r["n_fit"] for r in h], dtype=float)
        self.c_fit.setData(x, [r["rms_fit_db"] for r in h])
        val = [
            (r["n_fit"], r["rms_validation_db"]) for r in h if r["rms_validation_db"] is not None
        ]
        self.c_val.setData([v[0] for v in val], [v[1] for v in val])
        for g, names in GROUPS.items():
            ys = []
            for r in h:
                vals = [s for lab, s in r["std"].items() if lab.split("[")[0] in names]
                ys.append(100 * float(np.median(vals)) if vals else np.nan)
            # the prior first, at x = 0: where every parameter starts before any data
            prior = 100 * float(np.median([M.PRIOR_SIGMA[n] for n in names]))
            self.c_std[g].setData(np.r_[0.0, x], np.r_[prior, ys], connect="finite")

    def _fill_takes(self) -> None:
        current = self.takes.currentText()
        self.takes.blockSignals(True)
        self.takes.clear()
        groups = (self.fit or {}).get("groups", {})
        where = {n: g for g, names in groups.items() for n in names}
        for t in (self.dataset or {}).get("takes", []):
            name = f"{t['preset']}/{t['role']}"
            self.takes.addItem(f"{name}   [{where.get(name, '…')}]", name)
        self.takes.blockSignals(False)
        i = self.takes.findText(current)
        self.takes.setCurrentIndex(i if i >= 0 else self.takes.count() - 1)

    def _show_take(self, *_) -> None:
        if not self.dataset or self.takes.currentIndex() < 0:
            return
        t = self.dataset["takes"][self.takes.currentIndex()]
        f = np.asarray(self.dataset["freqs_hz"])
        re = np.array([np.nan if v is None else v for v in t["h_re"]], dtype=float)
        im = np.array([np.nan if v is None else v for v in t["h_im"]], dtype=float)
        self.c_meas.setData(f, 20 * np.log10(np.abs(re + 1j * im)), connect="finite")
        freqs = torch.as_tensor(f, dtype=torch.float64)
        sliders = M.sliders_tensor([t.get("controls") or {}])
        sch = M.ratio_response(M.schematic_values(), freqs, sliders)[0].numpy()
        self.c_sch.setData(f, 20 * np.log10(np.abs(sch)))
        if self.fit is not None:
            mod = M.ratio_response(self.fit["params"], freqs, sliders)[0].numpy()
            self.c_model.setData(f, 20 * np.log10(np.abs(mod)))

    def _fill_table(self) -> None:
        fit, sch = self.fit, M.schematic_values()
        self.table.setRowCount(len(fit["labels"]))
        for i, (lab, th, sd, lg) in enumerate(
            zip(fit["labels"], fit["theta"], fit["std"], fit["is_log"])
        ):
            name, k = (lab.split("[")[0], int(lab.split("[")[1][:-1])) if "[" in lab else (lab, 0)
            s0 = float(sch[name].reshape(-1)[k])
            v = float(np.exp(th)) if lg else float(th)
            prior = M.PRIOR_SIGMA[name]
            cells = [
                lab,
                f"{s0:.4g}",
                f"{v:.4g}",
                f"{100 * sd:.2f} %" if lg else f"{sd:.3g}",
                f"{100 * prior:.0f} %" if lg else f"{prior:g}",
                f"{v / s0:.3f}" if s0 else "–",
            ]
            for j, c in enumerate(cells):
                item = QTableWidgetItem(c)
                if j == 3 and lg:  # colour: pinned well below its prior, or still at the prior
                    ratio = sd / prior
                    item.setForeground(
                        pg.mkColor(
                            theme.OK
                            if ratio < 0.2
                            else (theme.AMBER if ratio < 0.7 else theme.FAIL)
                        )
                    )
                self.table.setItem(i, j, item)

    # ── parameter file ──────────────────────────────────────────────────── #

    def _write(self) -> None:
        if self.fit is None:
            QMessageBox.information(self, "Parameter file", "No fit yet.")
            return
        ok = QMessageBox.question(
            self,
            "Write parameter file",
            "Write the trained model as the next ge7_pedalode_v<N>.json? The capture rig's QC "
            "and plots use the newest file as the expected curve from then on (and grade "
            "'Response vs model' fully). Best at the end of the record run.",
        )
        if ok != QMessageBox.Yes:
            return
        path = T.write_parameter_file(self.fit, [self.root.name])
        self.log.add("ok", f"Wrote {path.name}", str(path))


def main() -> int:
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(theme.STYLESHEET)
    theme.setup_pyqtgraph()
    win = AnalysisWindow()
    win.show()
    return app.exec()
