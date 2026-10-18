"""Plan page: which presets to capture, in what order, and how long it will take."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import ControlPanel, card, label


class PlanPage(QWidget):
    def __init__(self, state: AppState, go_next) -> None:
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Plan", "title"))
        root.addWidget(
            label(
                "Tick the presets to capture and put them in order. The baseline preset goes first "
                "so every session starts from the same reference.",
                "dim",
            )
        )
        body = QHBoxLayout()
        root.addLayout(body, 1)

        f, lay = card("Presets")
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
        body.addWidget(f, 1)

        right = QVBoxLayout()
        f, lay = card("Preview")
        self.desc = label("", "dim")
        lay.addWidget(self.desc)
        self.panel = ControlPanel(state.profile.controls)
        lay.addWidget(self.panel)
        right.addWidget(f)
        f, lay = card("Time")
        self.estimate = label("")
        lay.addWidget(self.estimate)
        right.addWidget(f)
        go = QPushButton("Use this plan → Run")
        go.setObjectName("primary")
        go.clicked.connect(lambda: (self._apply(), go_next()))
        right.addWidget(go, 0, Qt.AlignLeft)
        right.addStretch()
        body.addLayout(right, 1)

        state.session_changed.connect(self.reload)
        state.rig_changed.connect(self._update_estimate)
        self.reload()

    def reload(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        presets = self.state.profile.presets()
        planned = {p.name for p in self.state.plan}
        order = [p.name for p in self.state.plan] + [
            p.name for p in presets if p.name not in planned
        ]
        by_name = {p.name: p for p in presets}
        status = {}
        if self.state.session:
            status = {k: v["status"] for k, v in self.state.session.data["presets"].items()}
        for n in order:
            p = by_name[n]
            st = status.get(n, "")
            item = QListWidgetItem(f"{n}" + (f"   [{st}]" if st and st != "pending" else ""))
            item.setData(Qt.UserRole, n)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if n in planned else Qt.Unchecked)
            item.setToolTip(p.description)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.list.setCurrentRow(0)
        self._update_estimate()

    def _move(self, d: int) -> None:
        r = self.list.currentRow()
        if 0 <= r + d < self.list.count():
            it = self.list.takeItem(r)
            self.list.insertItem(r + d, it)
            self.list.setCurrentRow(r + d)
            self._apply()

    def _check_all(self, on: bool) -> None:
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _preview(self, item) -> None:
        if not item:
            return
        p = self.state.profile.preset(item.data(Qt.UserRole))
        self.desc.setText(f"<b>{p.name}</b>: {p.description}")
        self.panel.set_values(p.controls)

    def _apply(self) -> None:
        names = [
            self.list.item(i).data(Qt.UserRole)
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.Checked
        ]
        by_name = {p.name: p for p in self.state.profile.presets()}
        self.state.plan = [by_name[n] for n in names]
        if self.state.session:
            for n in names:  # presets added to an existing session get an entry in session.json
                self.state.session.data["presets"].setdefault(
                    n,
                    {
                        "status": "pending",
                        "controls": by_name[n].controls,
                        "description": by_name[n].description,
                        "takes": [],
                    },
                )
            self.state.session.save()
        self.state.plan_changed.emit()
        self._update_estimate()

    def _update_estimate(self, *_) -> None:
        n = sum(self.list.item(i).checkState() == Qt.Checked for i in range(self.list.count()))
        s = self.state.session
        if s is None:
            self.estimate.setText(f"{n} presets. Start a session to see the time per preset.")
            return
        per = s.signal("train").duration_s + s.signal("val").duration_s + 2.0
        total = n * per
        speed = self.state.demo_speed if self.state.simulated else 1.0
        txt = f"{n} presets × {per / 60:.1f} min of audio = <b>{total / 60:.0f} min</b>"
        if speed != 1.0:
            txt += f" (demo at {speed:g}×: about {total / speed / 60:.1f} min)"
        txt += ", plus the time to set the controls."
        self.estimate.setText(txt)
