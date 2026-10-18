"""Signal-flow block diagram of the rig: clickable boxes, named jacks, labelled cables."""

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

# Text sizes in scene pixels. The view scales the whole drawing to fit (typically 0.7-0.9x).
TITLE_PX, SUB_PX, PORT_PX, CABLE_PX, NOTE_PX = 24, 17, 17, 19, 17

# What each box is and does. Shown in the Setup page's side panel.
BOXES: dict[str, dict] = {
    "pc": {
        "title": "PC (this app)",
        "sub": "ASIO",
        "about": (
            "Generates the test signal, sends it to Out 1 (pedal) and Out 2 (loopback), and "
            "records In 1 and In 2 in the same ASIO stream, so send and return share one sample "
            "clock. Studio One must be closed: ASIO allows one program at a time."
        ),
    },
    "interface": {
        "title": "PreSonus Studio 1824c",
        "sub": "Out 1/2: rear, Main L/R · In 1/2: front combo jacks",
        "about": (
            "Out 1 and Out 2 are the Main outputs on the back (balanced TRS, line level). The "
            "front Main knob may set their level (to be tested: run the chain test, turn the "
            "knob, run it again; if the gain changes, mark the knob position). In 1 and In 2 are "
            "the front combo jacks (XLR or 1/4\"). In 1 takes the ProDI's XLR at mic level: gain "
            "set during calibration, 48 V off. In 2 takes the loopback TRS at line level: gain at "
            "minimum, 48 V off."
        ),
    },
    "prormp": {
        "title": "Radial ProRMP",
        "sub": "re-amper: line → instrument",
        "about": (
            "Passive re-amp box. INPUT (XLR) takes the interface's balanced line-level output. "
            'OUTPUT (1/4" TRS jack) gives an unbalanced, guitar-like instrument-level signal for '
            "the pedal; a plain guitar cable (TS) from it to the pedal is fine. Level knob on the "
            "box; its transformer isolates the grounds (ground lift if there is hum). Measured "
            "insertion loss on this rig: about −12.4 dB."
        ),
    },
    "pedal": {
        "title": "Pedal under test",
        "sub": "engaged and powered",
        "about": (
            'The pedal being captured: IN and OUT are 1/4" TS at instrument level, exactly as '
            "with a guitar. The app tells you where to set the controls before every take."
        ),
    },
    "prodi": {
        "title": "Radial ProDI",
        "sub": "passive DI: instrument → mic",
        "about": (
            'Passive direct box. INPUT (1/4" TS) gives the pedal output a high-impedance load, '
            "like an amp input; the −15 dB pad switch is on this input side (for hot pedals). "
            "XLR OUT sends a balanced, low-impedance mic-level signal to the In 1 preamp; the "
            "ground lift switch is on this output side (if there is hum). The THRU jack is not "
            "used."
        ),
    },
}

# Jacks: id -> (box, side of the box, position along that side 0..1, label lines).
PORTS: dict[str, tuple[str, str, float, str]] = {
    "pc_usb": ("pc", "bottom", 0.5, "USB"),
    "if_usb": ("interface", "top", 0.409, "USB"),
    "if_out1": ("interface", "right", 0.298, "Out 1 · TRS\n(rear, Main L)"),
    "if_out2": ("interface", "right", 0.47, "Out 2 · TRS\n(rear, Main R)"),
    "if_in2": ("interface", "right", 0.63, "In 2 · TRS\n(front, gain min)"),
    "if_in1": ("interface", "right", 0.86, "In 1 · XLR\n(front, 48 V off)"),
    "rmp_in": ("prormp", "left", 0.8, "INPUT\nXLR"),
    "rmp_out": ("prormp", "right", 0.8, 'OUTPUT\n1/4" TRS'),
    "pedal_in": ("pedal", "left", 0.8, 'IN\n1/4" TS'),
    "pedal_out": ("pedal", "bottom", 0.5, 'OUT · 1/4" TS'),
    "di_in": ("prodi", "top", 0.5, 'INPUT · 1/4" TS\n(−15 dB pad here)'),
    "di_out": ("prodi", "left", 0.627, "XLR OUT\n(lift here)"),
}

# Cables: id -> (from jack, to jack, level, connectors, label position, bend points).
CABLES: dict[str, tuple] = {
    "usb": ("pc_usb", "if_usb", "USB", "", (186, 92), []),
    "send": ("if_out1", "rmp_in", "LINE level", "TRS → XLR", (382, 280), []),
    "inst_in": ("rmp_out", "pedal_in", "INSTRUMENT", "guitar cable", (752, 104), []),
    "inst_out": ("pedal_out", "di_in", "INSTRUMENT level", "TS → TS", (1000, 345), []),
    "mic": ("di_out", "if_in1", "MIC level", "XLR → XLR, balanced", (470, 545), []),
    "loop": ("if_out2", "if_in2", "LINE (loopback)", "TRS → TRS", (405, 366), [(395, None)]),
}

LEVEL_PRIMER_HTML = f"""
<h3 style='color:{theme.CYAN}'>Three kinds of signal level</h3>
<p><b style='color:{theme.CYAN_BRIGHT}'>Line level</b>: what an interface output and studio gear
use. About +4 dBu nominal (roughly 1.2 V RMS), balanced, low impedance. Much hotter than a
guitar.</p>
<p><b style='color:{theme.CYAN_BRIGHT}'>Instrument level</b>: what a guitar pickup puts out. Very
roughly 0.1 to 1 V peak, unbalanced, from a <i>high-impedance</i> source. Pedals are designed for
exactly this, and for a high-impedance load on their output (an amp input, about 1 MΩ).</p>
<p><b style='color:{theme.CYAN_BRIGHT}'>Mic level</b>: tens of millivolts or less, balanced, low
impedance. It needs a preamp, which is what the In 1 gain knob is.</p>

<h3 style='color:{theme.CYAN}'>Connectors</h3>
<p><b>TRS</b> (tip-ring-sleeve 1/4"): carries a balanced line signal (the 1824c outputs, the
loopback). <b>TS</b> (tip-sleeve 1/4", the guitar cable): unbalanced, instrument level. <b>XLR</b>:
balanced; here the ProRMP input and the ProDI output. The ProRMP output is a TRS jack, but the
signal on it is unbalanced instrument level, so a guitar cable to the pedal is fine.</p>
<h3 style='color:{theme.CYAN}'>Why each box is there</h3>
<p><b>ProRMP (re-amper)</b>: steps the interface's line level down to instrument level and gives the
pedal a guitar-like source. Without it the pedal gets a signal that is far too hot, from the wrong
source impedance: drive and fuzz pedals in particular sound different (they react to the source
impedance). The transformer also breaks ground loops.</p>
<p><b>Pedal</b>: the device under test. It sees exactly what it would see from a guitar.</p>
<p><b>ProDI (direct box)</b>: gives the pedal a proper high-impedance load and converts its output
to balanced mic level, which runs cleanly into the In 1 preamp. Without a DI, a line input loads
the pedal too heavily and the low end and the tone change.</p>
<p><b>Loopback (Out 2 → In 2)</b>: records the signal that was sent, through the interface only. It
measures the interface's own latency and level on every take, so a change in the pedal path is never
confused with a change in the interface.</p>

<h3 style='color:{theme.CYAN}'>Where the level is set (all of them matter)</h3>
<ol>
<li>The app's digital send level (from the pedal profile; GE-7: −10 dB)</li>
<li>The 1824c Main knob, if it controls Out 1/2 (to be tested)</li>
<li>The ProRMP level knob</li>
<li>The ProDI −15 dB pad (input side)</li>
<li>The In 1 gain knob</li>
</ol>
<p>Calibration measures the total and tells you which knob to move. Mark the knob positions (tape or
a photo) and type them into the boxes' settings: they are saved with every session.</p>

<h3 style='color:{theme.CYAN}'>A possible later comparison</h3>
<p>The 1824c front inputs 1/2 also have an instrument (Hi-Z) mode, so the pedal could plug straight
into In 1 without the ProDI. Not used now; worth one test capture to compare.</p>
"""

_STATE_COLOURS = {"ok": theme.OK, "fail": theme.FAIL, "idle": "#2a8a99"}


def _font(px: int, bold: bool = False) -> QFont:
    f = QFont("Segoe UI")
    f.setPixelSize(px)
    f.setBold(bold)
    return f


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
                3 if self.selected else 2,
            )
        )
        p.drawRoundedRect(r, 12, 12)
        p.setFont(_font(TITLE_PX, bold=True))
        p.setPen(QColor(theme.CYAN_BRIGHT))
        p.drawText(
            QRectF(r.left() + 10, r.top() + 8, r.width() - 20, 28),
            Qt.AlignHCenter | Qt.AlignTop,
            self.title,
        )
        p.setFont(_font(SUB_PX))
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(
            QRectF(r.left() + 10, r.top() + 36, r.width() - 20, 44),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            self.sub,
        )

    def mousePressEvent(self, event) -> None:
        self.on_click(self.box_id)
        event.accept()


class BlockDiagram(QGraphicsView):
    """The rig as a drawing; ``on_box_clicked(box_id)`` is called when a box is clicked."""

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

    def _box(self, box_id: str, x: float, y: float, w: float, h: float) -> None:
        item = BoxItem(box_id, QRectF(x, y, w, h), self._clicked)
        item.title, item.sub = BOXES[box_id]["title"], BOXES[box_id]["sub"]
        self.scene_.addItem(item)
        self.boxes[box_id] = item

    def _jack_pos(self, jack: str) -> QPointF:
        box, side, t, _ = PORTS[jack]
        r = self.boxes[box].rect()
        return {
            "left": QPointF(r.left(), r.top() + t * r.height()),
            "right": QPointF(r.right(), r.top() + t * r.height()),
            "top": QPointF(r.left() + t * r.width(), r.top()),
            "bottom": QPointF(r.left() + t * r.width(), r.bottom()),
        }[side]

    def _jack(self, jack: str) -> None:
        box, side, _, text = PORTS[jack]
        p = self._jack_pos(jack)
        self.scene_.addEllipse(
            p.x() - 7,
            p.y() - 7,
            14,
            14,
            QPen(QColor(theme.CYAN), 2),
            QBrush(QColor(theme.BACKGROUND)),
        )
        if box == "pc" or jack == "if_usb":
            return
        t = self.scene_.addSimpleText(text, _font(PORT_PX))
        t.setBrush(QColor(theme.TEXT))
        w, h = t.boundingRect().width(), t.boundingRect().height()
        # Side jacks: label inside the box next to the jack. Top and bottom jacks: label outside the
        # box, left of the cable, so it never covers the box title.
        pos = {
            "left": (p.x() + 12, p.y() - h / 2),
            "right": (p.x() - w - 12, p.y() - h / 2),
            "top": (p.x() - w - 12, p.y() - h - 6),
            "bottom": (p.x() - w - 12, p.y() + 6),
        }[side]
        t.setPos(*pos)

    def _cable(self, cable_id: str) -> None:
        src, dst, kind, note, label_at, bends = CABLES[cable_id]
        a, b = self._jack_pos(src), self._jack_pos(dst)
        path = QPainterPath(a)
        if bends:  # the loopback: out to the right, down, and back in
            x = bends[0][0]
            path.lineTo(QPointF(x, a.y()))
            path.lineTo(QPointF(x, b.y()))
        elif abs(a.x() - b.x()) > 1 and abs(a.y() - b.y()) > 1:
            path.lineTo(QPointF(b.x(), a.y()))
        path.lineTo(b)
        item = self.scene_.addPath(
            path, QPen(QColor(_STATE_COLOURS["idle"]), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        item.setZValue(-1)
        t1 = self.scene_.addSimpleText(kind, _font(CABLE_PX, bold=True))
        t2 = self.scene_.addSimpleText(note, _font(NOTE_PX))
        t1.setBrush(QColor(theme.CYAN_BRIGHT))
        t2.setBrush(QColor(theme.TEXT_DIM))
        t1.setPos(*label_at)
        t2.setPos(label_at[0], label_at[1] + CABLE_PX + 4)
        self.cables[cable_id] = (item, t1, t2)

    def _build(self) -> None:
        # The interface on the left with its jacks down its right edge; the analog chain runs left
        # to right along the top and comes back along the bottom, so every label has clear space.
        self._box("pc", 60, 0, 230, 84)
        self._box("interface", 40, 130, 330, 470)
        self._box("prormp", 520, 150, 230, 150)
        self._box("pedal", 860, 150, 260, 150)
        self._box("prodi", 860, 440, 260, 150)
        for jack in PORTS:
            self._jack(jack)
        for cable in CABLES:
            self._cable(cable)

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
            self.scene_.itemsBoundingRect().adjusted(-16, -16, 16, 16), Qt.KeepAspectRatio
        )
