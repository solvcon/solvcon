# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The Canvas page of the Painter inspector: how the canvas shows the world.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from ._sections import Placeholder, Section
from ._style import blend, mono_font, rule, shade, PaletteStyled

__all__ = [
    'CanvasPage',
]


def _format_zoom(zoom):
    """Return the zoom as the readout shows it, screen pixels per world unit
    read as a percentage."""
    percent = 100.0 * zoom
    # Whole percents while the view is anywhere near its own scale, which is
    # what the design shows; a far zoom out needs the digits below one.
    return f"{round(percent):g}%" if percent >= 1.0 else f"{percent:.3g}%"


class _Readout(QtWidgets.QFrame):
    """One boxed reading: what it measures, then the number."""

    _HEIGHT = 28
    _MARGINS = (8, 0, 8, 0)
    _GAP = 8

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setObjectName("box")
        self.setFixedHeight(self._HEIGHT)
        name = QtWidgets.QLabel(label, self)
        name.setObjectName("name")
        self._value = QtWidgets.QLabel(self)
        self._value.setObjectName("value")
        self._value.setFont(mono_font())
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*self._MARGINS)
        layout.setSpacing(self._GAP)
        layout.addWidget(name)
        layout.addStretch(1)
        layout.addWidget(self._value)

    @property
    def value(self):
        """The number the box reads, as it is written there."""
        return self._value.text()

    def set_value(self, text):
        """Show ``text``, which is empty for a box reading nothing."""
        self._value.setText(text)


class CanvasPage(PaletteStyled):
    """The inspector's Canvas page: the view over the bound canvas.

    Like the other pages, this one reads the canvas it is bound to on a timer,
    because the view carries no change signal. What the poll compares is the
    zoom, which is what the View section reads; the pan is left out, since it
    moves the view without changing that.

    The buttons that act on the view, and the sections the model cannot back
    yet, arrive with later steps of the redesign; they stand here as their
    designed titles, greyed out.
    """

    # Poll period in milliseconds, the Design page's rate and for its reason:
    # the reading is cheap enough to take this often, and a slower one reads as
    # lag while the wheel moves the zoom under it.
    _POLL_MS = 60

    #: The greyed-out sections, as ``(title, what the section waits for)``.
    PLACEHOLDERS = (
        ("Grid", "grid and snap options"),
        ("Axes & origin", "axes and origin styling"),
        ("Background", "background presets"),
        ("Units", "display units and precision"),
    )

    _LABEL_PX = 10
    _VALUE_PX = 11
    _RADIUS = 5

    # How far each of these sits from the panel color.
    _MUTED_MIX = 0.35
    _GREYED_MIX = 0.6
    _BORDER_MIX = 0.25

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._key = None
        self.placeholders = {}
        self._build()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self.refresh)
        self._apply_style()
        self._render()

    @property
    def zoom(self):
        """The zoom the readout shows, as it is written there."""
        return self._readout.value

    def set_canvas_source(self, source):
        """Bind the page to ``source``, a callable handing back the canvas.

        The callable returns the 2D canvas to read, or ``None`` when none is
        active. The canvas is asked for on every read rather than held,
        because it is a C++ widget owned by its sub-window: a reference kept
        across the window's close outlives the object behind it, and the next
        read walks freed memory.
        """
        self._source = source
        self._key = self._state_key()
        self._render()

    def showEvent(self, event):
        """Poll the bound canvas only while the page is on screen."""
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event):
        """Stop polling once the page leaves the screen."""
        super().hideEvent(event)
        self._timer.stop()

    def refresh(self):
        """Redraw the page when the canvas it is bound to has changed."""
        key = self._state_key()
        if key == self._key:
            return
        self._key = key
        self._render()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_view())
        for title, waits_for in self.PLACEHOLDERS:
            layout.addWidget(rule(QtWidgets.QFrame.HLine))
            self.placeholders[title] = Placeholder(
                title, waits_for, parent=self)
            layout.addWidget(self.placeholders[title])

        layout.addStretch(1)

    def _build_view(self):
        """Build the View section, for now the zoom readout alone."""
        self._view = Section("View", parent=self)
        self._readout = _Readout("Zoom", self._view)
        self._view.body.addWidget(self._readout)
        return self._view

    def _canvas(self):
        """The 2D canvas to read, or ``None`` when none is active."""
        return None if self._source is None else self._source()

    def _state_key(self):
        """What the page shows, as a value the poll can compare."""
        widget = self._canvas()
        return None if widget is None else widget.viewTransform.zoom

    def _render(self):
        """Show the zoom of the canvas the page is bound to."""
        self._readout.set_value(
            "" if self._key is None else _format_zoom(self._key))
        self._view.setEnabled(self._key is not None)

    def _apply_style(self):
        """Color the section heads and the readout from the palette."""
        palette = self.palette()
        text = palette.color(QtGui.QPalette.WindowText)
        panel = palette.color(QtGui.QPalette.Window)
        muted = blend(text, panel, self._MUTED_MIX)
        self.setStyleSheet(f"""
            QLabel#section {{
                color: {muted.name()};
            }}
            QLabel#section:disabled {{
                color: {blend(text, panel, self._GREYED_MIX).name()};
            }}
            QFrame#box {{
                border: 1px solid {shade(self, self._BORDER_MIX).name()};
                border-radius: {self._RADIUS}px;
                background: {palette.color(QtGui.QPalette.Base).name()};
            }}
            QLabel#name {{
                font-size: {self._LABEL_PX}px;
                color: {muted.name()};
            }}
            QLabel#value {{
                font-size: {self._VALUE_PX}px;
            }}
            """)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
