"""Dark HUD theme, colours copied from BrianEQ's shared/ui/HudTheme.h so the two apps look like one
family.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

BACKGROUND = "#05080c"
PANEL = "#0a141c"
PANEL_DARK = "#060c12"
CYAN = "#22d3e6"
CYAN_BRIGHT = "#8ff4ff"
GRID = "#11313b"
TEXT = "#c4f2f8"
TEXT_DIM = "#86b3bc"
ACCENT = "#ff8a3d"  # orange: warnings, peak hold
OK = "#3dffa8"
AMBER = "#eda100"
FAIL = "#ff5a4d"

STATUS_COLOURS = {"ok": OK, "warn": AMBER, "fail": FAIL, "idle": TEXT_DIM, "busy": CYAN}

STYLESHEET = f"""
QWidget {{ background: {BACKGROUND}; color: {TEXT}; font-family: "Segoe UI"; font-size: 10pt; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 8px; }}
QLabel#title {{ color: {CYAN_BRIGHT}; font-size: 16pt; font-weight: 600; background: transparent; }}
QLabel#heading {{ color: {CYAN}; font-size: 12pt; font-weight: 600; background: transparent; }}
QLabel#dim {{ color: {TEXT_DIM}; background: transparent; }}
QLabel#badge {{ border-radius: 9px; padding: 2px 10px; font-weight: 600; }}
QLabel {{ background: transparent; }}
QPushButton {{ background: {PANEL_DARK}; border: 1px solid {CYAN}; border-radius: 6px; padding: 6px
14px; color: {TEXT}; }}
QPushButton:hover {{ background: #0e2430; }}
QPushButton:disabled {{ border-color: {GRID}; color: #3d5a62; }}
QPushButton#primary {{ background: #0b3a44; border: 2px solid {CYAN_BRIGHT}; font-weight: 600;
font-size: 11pt; padding: 8px 18px; }}
QPushButton#danger {{ border-color: {ACCENT}; color: {ACCENT}; }}
QListWidget, QTableWidget, QTextBrowser, QLineEdit, QComboBox, QSpinBox {{
    background: {PANEL_DARK}; border: 1px solid {GRID}; border-radius: 6px;
    selection-background-color: #0e3a46; }}
QListWidget#nav {{ font-size: 11pt; padding: 6px; }}
QListWidget#nav::item {{ padding: 10px 8px; border-radius: 6px; }}
QListWidget#nav::item:selected {{ background: #0e3a46; color: {CYAN_BRIGHT}; }}
QHeaderView::section {{ background: {PANEL}; color: {TEXT_DIM}; border: none; padding: 4px; }}
QProgressBar {{ background: {PANEL_DARK}; border: 1px solid {GRID}; border-radius: 6px; text-align:
center; height: 18px; }}
QProgressBar::chunk {{ background: {CYAN}; border-radius: 5px; }}
QRadioButton, QCheckBox {{ background: transparent; spacing: 8px; }}
QTabWidget::pane {{ border: 1px solid {GRID}; border-radius: 6px; }}
QTabBar::tab {{ background: {PANEL_DARK}; color: {TEXT_DIM}; padding: 6px 14px;
border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {CYAN_BRIGHT}; }}
QScrollArea {{ border: none; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {CYAN}; }}
"""


def setup_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=True, background=PANEL_DARK, foreground=TEXT_DIM)


def styled_plot(title: str, xlabel: str, ylabel: str, log_x: bool = False) -> pg.PlotWidget:
    """A plot with HUD colours, labelled axes and a faint grid."""
    w = pg.PlotWidget()
    w.setTitle(title, color=TEXT, size="10pt")
    w.setLabel("bottom", xlabel)
    w.setLabel("left", ylabel)
    w.showGrid(x=True, y=True, alpha=0.15)
    if log_x:
        w.setLogMode(x=True, y=False)
        # Fixed audio ticks: pyqtgraph's default log ticks label every minor line (2·10¹, 3·10¹,
        # ...).
        ticks = [
            (np.log10(f), txt)
            for f, txt in (
                (20, "20"),
                (50, "50"),
                (100, "100"),
                (200, "200"),
                (500, "500"),
                (1000, "1k"),
                (2000, "2k"),
                (5000, "5k"),
                (10_000, "10k"),
                (20_000, "20k"),
            )
        ]
        w.getAxis("bottom").setTicks([ticks, []])
    w.addLegend(offset=(10, 10), labelTextColor=TEXT)
    return w
