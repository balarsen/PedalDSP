"""Reusable widgets: card frame, level meter, pedal control panel, QC / advice checklist."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QLabel,
    QMessageBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pedal_model.rig.gui import theme


def card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    """A rounded panel with an optional heading; returns the frame and its layout."""
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(8)
    if title:
        h = QLabel(title)
        h.setObjectName("heading")
        lay.addWidget(h)
    return f, lay


def auto_scale_box(plot: pg.PlotWidget, fixed: tuple[float, float], min_span: float) -> QCheckBox:
    """An "Auto scale" checkbox for a plot (on by default): auto Y range, else a fixed range."""
    box = QCheckBox("Auto scale")
    box.setChecked(True)
    plot.setLimits(minYRange=min_span)

    def apply(on: bool) -> None:
        if on:
            plot.enableAutoRange(axis="y", enable=True)
        else:
            plot.setYRange(*fixed)

    box.toggled.connect(apply)
    apply(True)
    return box


def show_error(parent: QWidget, title: str, tb: str) -> None:
    """Error box: the last line of a traceback up front, the whole traceback under Show Details,
    and a Copy button that puts the whole traceback on the clipboard (for pasting into a report).
    """
    lines = [ln for ln in tb.strip().splitlines() if ln.strip()]
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title)
    box.setText(lines[-1] if lines else title)
    box.setDetailedText(tb)
    box.setTextInteractionFlags(Qt.TextSelectableByMouse)
    copy = box.addButton("Copy", QMessageBox.ActionRole)
    # The button box closes the dialog through the button's own clicked() connection; dropping it
    # keeps the box open after copying.
    copy.clicked.disconnect()
    copy.clicked.connect(lambda: QApplication.clipboard().setText(title + "\n" + tb))
    box.addButton(QMessageBox.Ok)
    box.exec()


class LiveAutoY:
    """Auto scale for plots redrawn many times a second, with its "Auto scale" checkbox (``box``).

    pyqtgraph's own auto range re-lays-out the axes on every update, which made one Run-page repaint
    cost about 0.9 s at 2400x1300 (2026-10-20). This recomputes the range at most every
    ``period_s``, snapped to ``step_db``, and only touches the axes when the snapped range changes.
    """

    def __init__(
        self,
        plot: pg.PlotWidget,
        fixed: tuple[float, float],
        min_span: float,
        step_db: float,
        period_s: float = 0.5,
    ) -> None:
        self.plot, self.fixed, self.min_span = plot, fixed, min_span
        self.step, self.period = step_db, period_s
        self._last = 0.0
        self._range: tuple[float, float] | None = None
        plot.disableAutoRange(axis="y")
        self._set(fixed)
        self.box = QCheckBox("Auto scale")
        self.box.setChecked(True)
        self.box.toggled.connect(lambda on: self.update(force=True) if on else self._set(fixed))

    def _set(self, rng: tuple[float, float]) -> None:
        if rng != self._range:
            self.plot.setYRange(*rng, padding=0)
            self._range = rng

    def update(self, force: bool = False) -> None:
        """Rescale to the plotted data if due (call after setData)."""
        if not self.box.isChecked():
            return
        now = time.monotonic()
        if not force and now - self._last < self.period:
            return
        self._last = now
        ys = [np.asarray(c.yData) for c in self.plot.listDataItems() if c.yData is not None]
        ys = np.concatenate([y[np.isfinite(y)] for y in ys]) if ys else np.empty(0)
        if ys.size == 0:
            return
        lo = np.floor(ys.min() / self.step) * self.step
        hi = np.ceil(ys.max() / self.step) * self.step
        if hi - lo < self.min_span:
            mid = (hi + lo) / 2
            lo, hi = mid - self.min_span / 2, mid + self.min_span / 2
        self._set((float(lo), float(hi)))


def label(text: str, kind: str = "", wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    if kind:
        lab.setObjectName(kind)
    lab.setWordWrap(wrap)
    lab.setTextFormat(Qt.RichText)
    return lab


class LevelMeter(QWidget):
    """Vertical dBFS meter with peak hold. Green below −12, amber to −3, orange above."""

    FLOOR_DB = -60.0

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.level = self.FLOOR_DB
        self.peak = self.FLOOR_DB
        self.setMinimumSize(54, 180)

    def set_level(self, db: float) -> None:
        self.level = max(self.FLOOR_DB, db)
        self.peak = max(
            self.level, self.peak - 0.6
        )  # peak hold falls about 12 dB/s at 20 updates/s
        self.update()

    def reset(self) -> None:
        self.level = self.peak = self.FLOOR_DB
        self.update()

    def _y(self, db: float, top: float, h: float) -> float:
        return top + h * (db / self.FLOOR_DB)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        top, bottom = 8.0, h - 34.0
        bar = QRectF(w / 2 - 9, top, 18, bottom - top)
        p.fillRect(bar, QColor(theme.PANEL_DARK))
        p.setPen(QPen(QColor(theme.GRID), 1))
        p.drawRect(bar)
        for db in (-48, -36, -24, -12, -6, 0):
            y = self._y(db, top, bottom - top)
            p.drawLine(int(bar.right() + 2), int(y), int(bar.right() + 6), int(y))
        lvl = self.level
        colour = theme.OK if lvl < -12 else (theme.AMBER if lvl < -3 else theme.ACCENT)
        y = self._y(lvl, top, bottom - top)
        p.fillRect(QRectF(bar.left() + 2, y, bar.width() - 4, bar.bottom() - y), QColor(colour))
        py = self._y(self.peak, top, bottom - top)
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        p.drawLine(int(bar.left()), int(py), int(bar.right()), int(py))
        p.setPen(QColor(theme.TEXT))
        f = QFont(self.font())
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(QRectF(0, bottom + 2, w, 14), Qt.AlignCenter, self.name)
        p.setPen(QColor(theme.TEXT_DIM))
        txt = "—" if lvl <= self.FLOOR_DB else f"{lvl:.0f}"
        p.drawText(QRectF(0, bottom + 16, w, 14), Qt.AlignCenter, txt)


class ControlPanel(QWidget):
    """Drawn pedal controls showing where to set them. GE-7: eight sliders, ±15 dB."""

    def __init__(self, controls: list[dict]) -> None:
        super().__init__()
        self.controls = controls
        self.values: dict[str, float] = {c["key"]: 0.0 for c in controls}
        self.setMinimumSize(420, 230)

    def set_values(self, values: dict[str, float]) -> None:
        self.values = {c["key"]: float(values.get(c["key"], 0.0)) for c in self.controls}
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        n = len(self.controls)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#d9dde0"))  # the GE-7's grey-white face
        top, bottom = 26.0, h - 44.0
        col = w / n
        f = QFont(self.font())
        for i, c in enumerate(self.controls):
            x = col * (i + 0.5)
            lo, hi = c["min"], c["max"]
            v = self.values[c["key"]]
            p.setPen(QPen(QColor("#222"), 3))
            p.drawLine(int(x), int(top), int(x), int(bottom))
            for db in (hi, hi / 2, 0, lo / 2, lo):
                y = top + (hi - db) / (hi - lo) * (bottom - top)
                p.setPen(QPen(QColor("#555"), 1 if db else 2))
                p.drawLine(int(x - 12), int(y), int(x - 6), int(y))
            y = top + (hi - v) / (hi - lo) * (bottom - top)
            knob = QRectF(x - 14, y - 9, 28, 18)
            moved = abs(v) > 0.05
            p.setPen(QPen(QColor("#111"), 1))
            p.setBrush(QColor(theme.ACCENT if moved else "#3a3f44"))
            p.drawRoundedRect(knob, 3, 3)
            p.setPen(QPen(QColor("#fff"), 2))
            p.drawLine(int(x - 10), int(y), int(x + 10), int(y))
            f.setPointSize(9)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("#111"))
            p.drawText(QRectF(x - col / 2, 4, col, 18), Qt.AlignCenter, c["label"])
            f.setPointSize(10)
            p.setFont(f)
            p.setPen(QColor("#b4471a" if moved else "#333"))
            p.drawText(
                QRectF(x - col / 2, bottom + 10, col, 18),
                Qt.AlignCenter,
                f"{v:+.0f}" if moved else "0",
            )
        f.setPointSize(8)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor("#555"))
        p.drawText(
            QRectF(0, h - 18, w, 16),
            Qt.AlignCenter,
            "+15 dB at the top, 0 at the centre detent, −15 dB at the bottom",
        )


class CheckList(QWidget):
    """Coloured status dots with text: for calibration advice and QC items."""

    def __init__(self) -> None:
        super().__init__()
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(4)

    def clear(self) -> None:
        while self.lay.count():
            item = self.lay.takeAt(0)
            w = item.widget()
            if w:
                w.hide()  # deleteLater alone leaves it painted until the next event-loop pass
                w.setParent(None)
                w.deleteLater()

    def add_heading(self, text: str) -> None:
        self.lay.addWidget(label(f"<b style='color:{theme.CYAN_BRIGHT}'>{text}</b>"))

    def add(self, status: str, text: str, detail: str = "") -> None:
        colour = theme.STATUS_COLOURS.get(status, theme.TEXT_DIM)
        sym = {"ok": "●", "warn": "▲", "fail": "✖", "busy": "◌"}.get(status, "○")
        body = f"<span style='color:{colour}; font-size:12pt'>{sym}</span>&nbsp; {text}"
        if detail:
            body += (
                f"<br><span style='color:{theme.TEXT_DIM}'>&nbsp;&nbsp;&nbsp;&nbsp;{detail}</span>"
            )
        self.lay.addWidget(label(body))


class CheckText(QTextBrowser):
    """CheckList's look in one scrollable rich-text view, for long lists (a whole step's QC).

    Separate labels inside a scroll area clip their wrapped second lines; a text view wraps and
    scrolls by itself.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"background: transparent; color: {theme.TEXT};")
        self._parts: list[str] = []

    def clear(self) -> None:
        self._parts = []
        super().clear()

    def add_heading(self, text: str) -> None:
        self._parts.append(
            f"<p style='margin:8px 0 2px 0'><b style='color:{theme.CYAN_BRIGHT}'>{text}</b></p>"
        )
        self.setHtml("".join(self._parts))

    def add(self, status: str, text: str, detail: str = "") -> None:
        colour = theme.STATUS_COLOURS.get(status, theme.TEXT_DIM)
        sym = {"ok": "●", "warn": "▲", "fail": "✖", "busy": "◌"}.get(status, "○")
        body = f"<span style='color:{colour}'>{sym}</span>&nbsp; {text}"
        if detail:
            body += f"<br><span style='color:{theme.TEXT_DIM}'>{detail}</span>"
        self._parts.append(f"<p style='margin:2px 0 4px 18px; text-indent:-18px'>{body}</p>")
        self.setHtml("".join(self._parts))
