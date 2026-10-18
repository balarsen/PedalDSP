"""Signal-flow block diagram of the rig: boxes you can click, cables labelled with connector and
level type.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from pedal_model.rig.gui import theme

# What each box is and does. Text is shown in the Setup page's side panel.
BOXES: dict[str, dict] = {
    "pc": {
        "title": "PC (this app)",
        "sub": "ASIO · Studio USB ASIO Driver",
        "about": "Generates the test signal, sends it to Out 1 (pedal) and Out 2 (loopback), and "
        "records In 1 and "
        "In 2 in the same ASIO stream, so send and return share one sample clock. Studio One "
        "must be "
        "closed: ASIO allows one program at a time.",
    },
    "interface": {
        "title": "PreSonus Studio 1824c",
        "sub": "Out 1/2 rear (Main) · In 1/2 front",
        "about": "Out 1 and Out 2 are the Main outputs on the back: balanced line level. I "
        "believe the front Main "
        "knob sets their level (verify on the unit), so mark its position: it is part of the "
        "send level. "
        "In 1 and In 2 are the front combo inputs with preamps. In 1 takes the ProDI's XLR (mic "
        "level, "
        "gain set during calibration, 48 V off). In 2 takes the loopback TRS at line level, gain "
        "at minimum.",
    },
    "prormp": {
        "title": "Radial ProRMP",
        "sub": "re-amper: line → instrument",
        "about": "Passive re-amp box. Takes the interface's balanced line-level output and turns "
        "it into an "
        "unbalanced, guitar-like instrument-level signal for the pedal input. Its transformer "
        "isolates the "
        "grounds (use the ground lift if there is hum). Measured insertion loss on this rig: "
        "about −12.4 dB.",
    },
    "pedal": {
        "title": "Pedal under test",
        "sub": "device under test",
        "about": "The pedal being captured, engaged and powered. The app tells you where to set "
        "the controls "
        "before every take.",
    },
    "prodi": {
        "title": "Radial ProDI",
        "sub": "passive DI: instrument → mic",
        "about": "Passive direct box. Gives the pedal output a high-impedance load (like an amp "
        "input) and turns it into a balanced, low-impedance mic-level signal for the In 1 "
        "preamp. Two switches: the −15 dB pad on the input side (for hot pedals) and the "
        "ground lift on the XLR output side (if there is hum).",
    },
}

# Cable id -> (level type, connector and notes).
CABLES: dict[str, tuple[str, str]] = {
    "usb": ("USB", ""),
    "send": ("LINE level", "balanced TRS, ~+4 dBu"),
    "inst_in": ("INSTRUMENT level", 'TS 1/4", unbalanced'),
    "inst_out": ("INSTRUMENT level", 'TS 1/4", unbalanced'),
    "mic": ("MIC level", "balanced XLR → In 1 preamp"),
    "loop": ("LINE (loopback)", "TRS, In 2 gain minimum"),
}

LEVEL_PRIMER_HTML = f"""
<h3 style='color:{theme.CYAN}'>Three kinds of signal level</h3>
<p><b style='color:{theme.CYAN_BRIGHT}'>Line level</b>: what an interface output and studio gear
use. About +4 dBu
nominal (roughly 1.2 V RMS), balanced, low impedance. Much hotter than a guitar.</p>
<p><b style='color:{theme.CYAN_BRIGHT}'>Instrument level</b>: what a guitar pickup puts out. Very
roughly 0.1 to 1 V
peak, unbalanced, from a <i>high-impedance</i> source. Pedals are designed for exactly this, and for
a
high-impedance load on their output (an amp input, about 1 MΩ).</p>
<p><b style='color:{theme.CYAN_BRIGHT}'>Mic level</b>: tens of millivolts or less, balanced, low
impedance. It needs a
preamp, which is what the In 1 gain knob is.</p>

<h3 style='color:{theme.CYAN}'>Why each box is there</h3>
<p><b>ProRMP (re-amper)</b>: steps the interface's line level down to instrument level and gives the
pedal a
guitar-like source. Without it the pedal gets a signal that is far too hot, from the wrong source
impedance:
drive and fuzz pedals in particular sound different (they react to the source impedance). The
transformer
also breaks ground loops.</p>
<p><b>Pedal</b>: the device under test. It sees exactly what it would see from a guitar.</p>
<p><b>ProDI (direct box)</b>: gives the pedal a proper high-impedance load and converts its output
to balanced
mic level, which runs cleanly into the In 1 preamp. Without a DI, a line input loads the pedal too
heavily
and the low end and the tone change.</p>
<p><b>Loopback (Out 2 → In 2)</b>: records the signal that was sent, through the interface only. It
measures the
interface's own latency and level on every take, so a change in the pedal path is never confused
with a change
in the interface.</p>

<h3 style='color:{theme.CYAN}'>Where the level is set (all of them matter)</h3>
<ol>
<li>The app's digital send level (from the pedal profile; GE-7: −10 dB)</li>
<li>The 1824c Main knob (Out 1/2) <i>— verify that it controls these outputs on your unit</i></li>
<li>The ProRMP level knob</li>
<li>The ProDI −15 dB pad</li>
<li>The In 1 gain knob</li>
</ol>
<p>Calibration measures the total and tells you which knob to move. Mark the knob positions (tape or
a photo)
and type them into the boxes' settings: they are saved with every session.</p>

<h3 style='color:{theme.CYAN}'>A possible later comparison</h3>
<p>The 1824c front inputs 1/2 also have an instrument (Hi-Z) mode, so the pedal could plug straight
into In 1
without the ProDI. Not used now; worth one test capture to compare.</p>
"""

_STATE_COLOURS = {"ok": theme.OK, "fail": theme.FAIL, "idle": "#2a6f7c"}


class BoxItem(QGraphicsRectItem):
    def __init__(self, box_id: str, rect: QRectF, on_click: Callable[[str], None]) -> None:
        super().__init__(rect)
        self.box_id = box_id
        self.on_click = on_click
        self.selected = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.title = ""
        self.sub = ""

    def paint(self, p: QPainter, _opt, _w=None) -> None:
        r = self.rect()
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(QColor("#0d2530" if self.selected else theme.PANEL)))
        p.setPen(
            QPen(
                QColor(theme.CYAN_BRIGHT if self.selected else theme.CYAN),
                2.5 if self.selected else 1.5,
            )
        )
        p.drawRoundedRect(r, 10, 10)
        f = QFont("Segoe UI", 11)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(theme.CYAN_BRIGHT))
        p.drawText(
            QRectF(r.left() + 8, r.top() + 8, r.width() - 16, 22),
            Qt.AlignHCenter | Qt.AlignTop,
            self.title,
        )
        f.setBold(False)
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(
            QRectF(r.left() + 8, r.top() + 32, r.width() - 16, 40),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            self.sub,
        )

    def mousePressEvent(self, event) -> None:
        self.on_click(self.box_id)
        event.accept()


class BlockDiagram(QGraphicsView):
    """The rig as a drawing. ``box_clicked(box_id)`` is called when a box is clicked."""

    def __init__(self, on_box_clicked: Callable[[str], None]) -> None:
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet(f"background: {theme.BACKGROUND}; border: none;")
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.on_box_clicked = on_box_clicked
        self.boxes: dict[str, BoxItem] = {}
        self.cables: dict[
            str, tuple[QGraphicsPathItem, QGraphicsSimpleTextItem, QGraphicsSimpleTextItem]
        ] = {}
        self._build()

    def _box(self, box_id: str, x: float, y: float, w: float, h: float) -> BoxItem:
        item = BoxItem(box_id, QRectF(x, y, w, h), self._clicked)
        item.title, item.sub = BOXES[box_id]["title"], BOXES[box_id]["sub"]
        self.scene_.addItem(item)
        self.boxes[box_id] = item
        return item

    def _port(self, x: float, y: float, text: str, right: bool = True) -> None:
        self.scene_.addEllipse(
            x - 5, y - 5, 10, 10, QPen(QColor(theme.CYAN), 1.5), QBrush(QColor(theme.BACKGROUND))
        )
        t = self.scene_.addSimpleText(text, QFont("Segoe UI", 9))
        t.setBrush(QColor(theme.TEXT))
        t.setPos(x - t.boundingRect().width() - 9 if right else x + 9, y - 8)

    def _cable(
        self, cable_id: str, points: list[tuple[float, float]], label_at: tuple[float, float]
    ) -> None:
        path = QPainterPath(QPointF(*points[0]))
        for pt in points[1:]:
            path.lineTo(QPointF(*pt))
        item = self.scene_.addPath(
            path, QPen(QColor(_STATE_COLOURS["idle"]), 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        item.setZValue(-1)
        kind, note = CABLES[cable_id]
        f = QFont("Segoe UI", 9)
        f.setBold(True)
        t1 = self.scene_.addSimpleText(kind, f)
        t2 = self.scene_.addSimpleText(note, QFont("Segoe UI", 8))
        t1.setBrush(QColor(theme.CYAN_BRIGHT))
        t2.setBrush(QColor(theme.TEXT_DIM))
        t1.setPos(label_at[0], label_at[1])
        t2.setPos(label_at[0], label_at[1] + 14)
        self.cables[cable_id] = (item, t1, t2)

    def _build(self) -> None:
        # Interface ports sit on the right edge of its box; the analog chain runs left → right on
        # top and comes back along the bottom, so every cable label has its own clear space.
        self._box("pc", 0, 200, 140, 80)
        self._box("interface", 230, 40, 230, 380)
        self._box("prormp", 600, 75, 190, 80)
        self._box("pedal", 940, 75, 190, 80)
        self._box("prodi", 940, 315, 190, 80)
        ports = {"out1": (460, 115), "out2": (460, 205), "in2": (460, 265), "in1": (460, 355)}
        self._port(*ports["out1"], "Out 1 (rear)")
        self._port(*ports["out2"], "Out 2 (rear)")
        self._port(*ports["in2"], "In 2 (front)")
        self._port(*ports["in1"], "In 1 (front)")
        self._cable("usb", [(140, 240), (230, 240)], (165, 218))
        self._cable("send", [ports["out1"], (600, 115)], (472, 70))
        self._cable("inst_in", [(790, 115), (940, 115)], (800, 70))
        self._cable("inst_out", [(1035, 155), (1035, 315)], (1045, 220))
        self._cable("mic", [(940, 355), ports["in1"]], (640, 363))
        self._cable("loop", [ports["out2"], (520, 205), (520, 265), ports["in2"]], (530, 222))

    def _clicked(self, box_id: str) -> None:
        self.select(box_id)
        self.on_box_clicked(box_id)

    def select(self, box_id: str) -> None:
        for bid, item in self.boxes.items():
            item.selected = bid == box_id
            item.update()

    def set_pedal_name(self, name: str) -> None:
        self.boxes["pedal"].title = name
        self.boxes["pedal"].update()

    def set_cable_states(self, states: dict[str, str]) -> None:
        for cid, (item, _, _) in self.cables.items():
            colour = _STATE_COLOURS.get(states.get(cid, "idle"), _STATE_COLOURS["idle"])
            pen = item.pen()
            pen.setColor(QColor(colour))
            item.setPen(pen)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fitInView(
            self.scene_.itemsBoundingRect().adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio
        )
