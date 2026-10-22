"""Setup page: the block diagram, each box's settings (saved to configs/rig.json), and the level
primer.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.config import DEFAULT_RIG_PATH
from pedal_model.rig.gui.diagram import BOXES, LEVEL_PRIMER_HTML, BlockDiagram
from pedal_model.rig.gui.state import AppState
from pedal_model.rig.gui.widgets import card, label

SETTING_LABELS = {
    "main_knob": "Main knob (Out 1/2)",
    "return_gain": "Return input gain (knob position)",
    "return_48v": "Return input 48 V",
    "loop_gain": "Loopback input gain",
    "loop_48v": "Loopback input 48 V",
    "level": "Level knob",
    "ground_lift": "Ground lift",
    "engaged": "Engaged",
    "power": "Power",
    "pad_15db": "−15 dB pad",
}


class SetupPage(QWidget):
    def __init__(self, state: AppState, go_next) -> None:
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Setup: wire it like this", "title"))
        root.addWidget(
            label(
                "Click a box to see what it does and how to set it. Cables turn green or red after "
                "the calibration tests. This is a first draft: we iterate on it against the real "
                "rig.",
                "dim",
            )
        )
        body = QHBoxLayout()
        root.addLayout(body, 1)
        self.diagram = BlockDiagram(self._show_box)
        self.diagram.set_pedal_name(state.profile.display_name)
        body.addWidget(self.diagram, 5)

        tabs = QTabWidget()
        tabs.setMinimumWidth(360)
        body.addWidget(tabs, 3)

        box_tab = QWidget()
        bl = QVBoxLayout(box_tab)
        f, self.box_lay = card()
        self.box_title = label("", "heading")
        self.box_about = label("")
        self.box_lay.addWidget(self.box_title)
        self.box_lay.addWidget(self.box_about)
        self.form = QFormLayout()
        self.box_lay.addLayout(self.form)
        self.save_btn = QPushButton("Save settings to configs/rig.json")
        self.save_btn.clicked.connect(self._save)
        self.saved = label("", "dim")
        self.box_lay.addWidget(self.save_btn)
        self.box_lay.addWidget(self.saved)
        self.box_lay.addStretch()
        bl.addWidget(f)
        tabs.addTab(box_tab, "Selected box")

        primer = QTextBrowser()
        primer.setHtml(LEVEL_PRIMER_HTML)
        tabs.addTab(primer, "Why these boxes? (levels)")

        self._edits: dict[str, QLineEdit] = {}
        self._channel_spins: dict[str, QSpinBox] = {}
        self.current = "interface"
        self.diagram.select(self.current)
        self._show_box(self.current)
        footer = QHBoxLayout()
        footer.addStretch()
        nxt = QPushButton("Continue to Calibrate →")
        nxt.setObjectName("primary")
        nxt.clicked.connect(go_next)
        footer.addWidget(nxt)
        root.addLayout(footer)
        state.cables_changed.connect(self.diagram.set_cable_states)
        state.rig_changed.connect(lambda: self._show_box(self.current))

    def _clear_form(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self._edits.clear()
        self._channel_spins.clear()

    def _show_box(self, box_id: str) -> None:
        self.current = box_id
        info = BOXES[box_id]
        title = self.state.profile.display_name if box_id == "pedal" else info["title"]
        self.box_title.setText(title)
        self.box_about.setText(info["about"])
        self._clear_form()
        for key, value in self.state.rig.boxes.get(box_id, {}).items():
            e = QLineEdit(value)
            self._edits[key] = e
            self.form.addRow(SETTING_LABELS.get(key, key), e)
        if box_id == "interface":
            ch = self.state.rig.channels
            for key, text in (
                ("send_out", "Send → pedal: Out"),
                ("loop_out", "Loopback: Out"),
                ("return_in", "Return from pedal: In"),
                ("loop_in", "Loopback: In"),
            ):
                s = QSpinBox()
                s.setRange(1, 18)
                s.setValue(getattr(ch, key))
                self._channel_spins[key] = s
                self.form.addRow(text, s)
        if box_id == "pc":
            e = QLineEdit(self.state.rig.asio_device)
            self._edits["__asio__"] = e
            self.form.addRow("ASIO device", e)
            b = QSpinBox()
            b.setRange(64, 8192)
            b.setValue(self.state.rig.block_size)
            self._channel_spins["__block__"] = b
            self.form.addRow("Block size (frames)", b)
        self.saved.setText("")

    def _save(self) -> None:
        rig = self.state.rig
        for key, e in self._edits.items():
            if key == "__asio__":
                rig.asio_device = e.text().strip()
            else:
                rig.boxes.setdefault(self.current, {})[key] = e.text()
        for key, s in self._channel_spins.items():
            if key == "__block__":
                rig.block_size = s.value()
            else:
                setattr(rig.channels, key, s.value())
        rig.save()
        snapshot = rig.save_snapshot()
        saved = [DEFAULT_RIG_PATH.name, snapshot.name]
        session = self.state.session
        if session is not None:  # the open session records the setup it was captured with
            session.data["rig"] = rig.to_dict()
            copy = rig.save(session.root / snapshot.name)
            session.log("rig_setup_saved", file=copy.name)
            saved.append(f"{session.root.name}/{copy.name}")
        self.state.set_mode(self.state.simulated, self.state.demo_speed)  # rebuild the backend
        self.saved.setText("Saved: " + ", ".join(saved) + ".")
