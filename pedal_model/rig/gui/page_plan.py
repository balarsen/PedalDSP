"""Plan page: which presets and measurement steps to capture, in what order, how long it takes."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.analysis import log_grid
from pedal_model.rig.gui import theme
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import ControlPanel, card, label
from pedal_model.rig.profiles import KIND_MAP, KIND_MODEL, NAME_PATTERN, Preset
from pedal_model.rig.qc import expected_response

GAP_PER_TAKE_S = 2.0  # pre-roll + tail per take


class CustomPresetDialog(QDialog):
    """Make a new preset: name, description, notes and every control's value."""

    def __init__(self, parent, controls: list[dict]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add a custom preset")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("lowercase_with_underscores")
        self.desc = QLineEdit()
        self.notes = QPlainTextEdit()
        self.notes.setFixedHeight(70)
        form.addRow("Name", self.name)
        form.addRow("One-line description", self.desc)
        form.addRow("What it does / why", self.notes)
        self.spins: dict[str, QDoubleSpinBox] = {}
        for c in controls:
            sp = QDoubleSpinBox()
            sp.setRange(c["min"], c["max"])
            sp.setSingleStep(0.5)
            sp.setSuffix(f" {c['unit']}")
            self.spins[c["key"]] = sp
            form.addRow(c["label"], sp)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._check)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _check(self) -> None:
        if not NAME_PATTERN.match(self.name.text().strip()):
            QMessageBox.warning(self, "Name", "Use lowercase letters, digits and underscores only.")
            return
        self.accept()

    def preset(self) -> Preset:
        return Preset(
            self.name.text().strip(),
            self.desc.text().strip(),
            {k: float(s.value()) for k, s in self.spins.items()},
            KIND_MODEL,
            self.notes.toPlainText().strip(),
        )


class PlanPage(QWidget):
    def __init__(self, state: AppState, go_next) -> None:
        super().__init__()
        self.state = state
        self._presets: dict[str, Preset] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Plan", "title"))
        root.addWidget(
            label(
                "Tick the steps to capture and put them in order. The baseline preset goes "
                "first so every session starts from the same reference. Add your own presets, or "
                "the slider map (a measurement of each slider's real centre and bandwidth).",
                "dim",
            )
        )
        body = QHBoxLayout()
        root.addLayout(body, 1)

        f, lay = card("Steps")
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._preview)
        self.list.itemChanged.connect(lambda *_: self._apply())
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (
            ("▲ Up", lambda: self._move(-1)),
            ("▼ Down", lambda: self._move(1)),
            ("All", lambda: self._check_all(True)),
            ("None", lambda: self._check_all(False)),
        ):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        row = QHBoxLayout()
        add = QPushButton("+ Add custom preset…")
        add.clicked.connect(self._add_custom)
        n_map = len(state.profile.map_presets())
        add_map = QPushButton(f"+ Add slider map ({n_map} steps)")
        add_map.setToolTip(state.profile.raw.get("map", {}).get("note", ""))
        add_map.clicked.connect(self._add_map)
        add_map.setEnabled(n_map > 0)
        row.addWidget(add)
        row.addWidget(add_map)
        row.addStretch()
        lay.addLayout(row)
        body.addWidget(f, 1)

        right = QVBoxLayout()
        f, lay = card("Preview")
        self.title = label("", "heading")
        self.desc = label("")
        self.notes = label("", "dim")
        lay.addWidget(self.title)
        lay.addWidget(self.desc)
        lay.addWidget(self.notes)
        self.panel = ControlPanel(state.profile.controls)
        lay.addWidget(self.panel)
        self.plot = theme.styled_plot(
            "Model response at these settings", "Frequency (Hz)", "Gain (dB)", log_x=True
        )
        self.plot.setMinimumHeight(170)
        self.plot.setXRange(np.log10(20), np.log10(20_000))
        self.curve = self.plot.plot(pen=pg.mkPen(theme.AMBER, width=2))
        lay.addWidget(self.plot)
        right.addWidget(f, 1)
        f, lay = card("Time")
        self.estimate = label("")
        lay.addWidget(self.estimate)
        right.addWidget(f)
        go = QPushButton("Use this plan → Run")
        go.setObjectName("primary")
        go.clicked.connect(lambda: (self._apply(), go_next()))
        right.addWidget(go, 0, Qt.AlignRight)
        body.addLayout(right, 1)

        state.session_changed.connect(self.reload)
        state.rig_changed.connect(self._update_estimate)
        self.reload()

    # ── list ────────────────────────────────────────────────────────────── #

    def _item(self, p: Preset, checked: bool, status: str = "") -> QListWidgetItem:
        tag = " · map" if p.kind == KIND_MAP else ""
        text = f"{p.name}{tag}" + (f"   [{status}]" if status and status != "pending" else "")
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, p.name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        item.setToolTip(p.notes or p.description)
        return item

    def reload(self) -> None:
        s = self.state.session
        folder = self.state.profile.presets()
        planned = list(self.state.plan)
        self._presets = {p.name: p for p in folder}
        self._presets.update({p.name: p for p in planned})
        status = {k: v["status"] for k, v in s.data["presets"].items()} if s else {}
        planned_names = [p.name for p in planned]
        self.list.blockSignals(True)
        self.list.clear()
        for name in planned_names + [p.name for p in folder if p.name not in planned_names]:
            self.list.addItem(
                self._item(self._presets[name], name in planned_names, status.get(name, ""))
            )
        self.list.blockSignals(False)
        self.list.setCurrentRow(0)
        self._update_estimate()

    def _names(self, checked_only: bool) -> list[str]:
        return [
            self.list.item(i).data(Qt.UserRole)
            for i in range(self.list.count())
            if not checked_only or self.list.item(i).checkState() == Qt.Checked
        ]

    def _move(self, d: int) -> None:
        r = self.list.currentRow()
        if 0 <= r + d < self.list.count():
            it = self.list.takeItem(r)
            self.list.insertItem(r + d, it)
            self.list.setCurrentRow(r + d)
            self._apply()

    def _check_all(self, on: bool) -> None:
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(False)
        self._apply()

    def _add_custom(self) -> None:
        dlg = CustomPresetDialog(self, self.state.profile.controls)
        if dlg.exec() != QDialog.Accepted:
            return
        p = dlg.preset()
        try:
            path = self.state.profile.save_custom_preset(p)
        except (ValueError, FileExistsError) as e:
            QMessageBox.warning(self, "Not saved", str(e))
            return
        self._presets[p.name] = p
        self.list.addItem(self._item(p, True))
        self.list.setCurrentRow(self.list.count() - 1)
        self._apply()
        self.estimate.setText(self.estimate.text() + f"<br>Saved {path.name}.")

    def _add_map(self) -> None:
        existing = set(self._names(False))
        self.list.blockSignals(True)
        for p in self.state.profile.map_presets():
            if p.name not in existing:
                self._presets[p.name] = p
                self.list.addItem(self._item(p, True))
        self.list.blockSignals(False)
        self._apply()

    # ── preview ─────────────────────────────────────────────────────────── #

    def _preview(self, item) -> None:
        if not item:
            return
        p = self._presets[item.data(Qt.UserRole)]
        kind = (
            "measurement (sweep only, ~1 min)"
            if p.kind == KIND_MAP
            else "model capture (train + val)"
        )
        self.title.setText(
            f"{p.name}  <span style='color:{theme.TEXT_DIM}; font-size:10pt'>· {kind}</span>"
        )
        self.desc.setText(p.description)
        self.notes.setText(p.notes or "No notes yet for this preset.")
        self.panel.set_values(p.controls)
        grid = log_grid(20, 20_000)
        exp = expected_response(self.state.profile.pedal, p.controls, self.state.sample_rate, grid)
        self.curve.setData(grid, exp if exp is not None else np.zeros_like(grid))

    # ── plan ────────────────────────────────────────────────────────────── #

    def _apply(self) -> None:
        plan = [self._presets[n] for n in self._names(True)]
        self.state.plan = plan
        s = self.state.session
        if s is not None:
            for p in plan:
                s.add_preset(p, save=False)
            # keep session.json in plan order (resume follows it)
            order = [p.name for p in plan] + [
                n for n in s.data["presets"] if n not in {p.name for p in plan}
            ]
            s.data["presets"] = {n: s.data["presets"][n] for n in order}
            s.save()
        self.state.plan_changed.emit()
        self._update_estimate()

    def _update_estimate(self, *_) -> None:
        names = self._names(True)
        s = self.state.session
        if s is None:
            self.estimate.setText(f"{len(names)} steps. Start a session to see the time per step.")
            return
        per_role = {r: s.signal(r).duration_s + GAP_PER_TAKE_S for r in ("train", "val", "sweep")}
        total = sum(sum(per_role[r] for r in self._presets[n].roles) for n in names)
        n_map = sum(self._presets[n].kind == KIND_MAP for n in names)
        speed = self.state.demo_speed if self.state.simulated else 1.0
        txt = (
            f"{len(names) - n_map} presets + {n_map} map steps = <b>{total / 60:.0f} min</b> of "
            "audio "
            f"(train {per_role['train'] / 60:.1f} min + val {per_role['val'] / 60:.1f} min per "
            "preset, "
            f"{per_role['sweep'] / 60:.1f} min per map step)"
        )
        if speed > 0 and speed != 1.0:
            txt += f"; demo at {speed:g}×: about {total / speed / 60:.1f} min"
        self.estimate.setText(txt + ", plus the time to set the controls.")
