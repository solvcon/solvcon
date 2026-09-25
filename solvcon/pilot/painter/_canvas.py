# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The Canvas page of the Painter inspector: how the canvas shows the world.
"""

import json
import math

from PySide6 import QtCore, QtWidgets

from .._style import PaletteStyled
from ._sections import Placeholder, Section
from ._style import Parts, Rules
from ..panel._tree_panel import EntityTreeWidget

__all__ = [
    'CanvasPage',
]


def _format_zoom(zoom):
    """Return the zoom as the readout shows it, screen pixels per world unit
    read as a percentage."""
    percent = 100.0 * zoom
    return f"{round(percent):g}%" if percent >= 1.0 else f"{percent:.3g}%"


def _box_center(bounds):
    """The middle of ``bounds`` as ``(min_x, min_y, max_x, max_y)``.

    Halved before summed: two bounds far enough out add up past the double
    range, and the middle between them would come back infinite though it lies
    well inside.
    """
    min_x, min_y, max_x, max_y = bounds
    return 0.5 * min_x + 0.5 * max_x, 0.5 * min_y + 0.5 * max_y


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
        name.setObjectName("label")
        self._value = QtWidgets.QLabel(self)
        self._value.setObjectName("value")
        self._value.setFont(Parts.mono_font())
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
    because neither the world nor the view carries a change signal. What the
    poll compares is the zoom, which the readout shows, and counts of what the
    world holds, which say whether there is anything to fit. The pan is left
    out: it moves the view without changing either.
    """

    # Poll period in milliseconds, the Design page's rate and for its reason:
    # a slower one reads as lag while the wheel moves the zoom under it.
    _POLL_MS = 60

    #: The greyed-out sections, as ``(title, what the section waits for)``.
    PLACEHOLDERS = (
        ("Grid", "grid and snap options"),
        ("Axes & origin", "axes and origin styling"),
        ("Background", "background presets"),
        ("Units", "display units and precision"),
    )

    _FIT_MARGIN = 0.9
    # Stand-in span when content has no extent on either axis.
    _MIN_SPAN = 1.0

    _ACTION_HEIGHT = 26
    _ACTION_GAP = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._key = (None, None)
        self.buttons = {}
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
        # Rendered rather than refreshed: two canvases can share a zoom and the
        # same empty-world counts, so a refresh would take the page for
        # unchanged.
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
            layout.addWidget(Parts.rule(QtWidgets.QFrame.HLine))
            self.placeholders[title] = Placeholder(
                title, waits_for, parent=self)
            layout.addWidget(self.placeholders[title])

        layout.addStretch(1)

    def _build_view(self):
        """Build the View section: the zoom readout over the view buttons."""
        self._view = Section("View", parent=self)
        self._readout = _Readout("Zoom", self._view)
        self._view.body.addWidget(self._readout)
        self._view.body.addLayout(self._build_actions())
        return self._view

    def _build_actions(self):
        """Build the view buttons, in design order."""
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(self._ACTION_GAP)
        for name, handler in (("Fit", self._on_fit),
                              ("100%", self._on_actual),
                              ("Center", self._on_center)):
            button = QtWidgets.QPushButton(name, self._view)
            button.setObjectName("action")
            button.setFixedHeight(self._ACTION_HEIGHT)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.clicked.connect(handler)
            layout.addWidget(button)
            self.buttons[name] = button
        return layout

    def _canvas(self):
        """The 2D canvas to read, or ``None`` when none is active."""
        return None if self._source is None else self._source()

    def _state_key(self):
        """What the page reads off the canvas, as a value the poll can compare:
        the zoom it shows, and counts of what the world holds.

        The counts stand in for the world itself, which would cost a full
        serialization every tick. They do not move when a shape is dragged, and
        they need not: all the page shows of the world is whether it holds
        anything, and Fit measures the geometry when it is pressed.
        """
        widget = self._canvas()
        if widget is None:
            return (None, None)
        world = widget.world
        return (widget.viewTransform.zoom,
                None if world is None else (world.nshape, world.npoint,
                                            world.nsegment, world.nbezier))

    def _render(self):
        """Show the bound canvas's zoom and whether its world can be fitted."""
        zoom, counts = self._key
        self._readout.set_value("" if zoom is None else _format_zoom(zoom))
        self._view.setEnabled(zoom is not None)
        self.buttons["Fit"].setEnabled(self._can_fit(counts))

    def _can_fit(self, counts):
        """Whether the world reports anything Fit could frame.

        Pad leftovers from a removed shape keep ``nsegment`` / ``nbezier``
        non-zero, so Fit may stay enabled and no-op when bounds are empty.
        """
        return counts is not None and any(counts)

    @staticmethod
    def _content_bounds(widget):
        """The extent of what ``widget`` draws, or ``None`` for nothing."""
        world = widget.world
        if world is None:
            return None
        return EntityTreeWidget.world_bounds(
            json.loads(world.describe_state()))

    def _on_fit(self):
        """Frame everything the world draws, with a margin around it."""
        widget = self._canvas()
        if widget is None:
            return
        bounds = self._content_bounds(widget)
        if bounds is None:
            return
        min_x, min_y, max_x, max_y = bounds
        width, height = widget.viewportSize
        zoom = self._fit_zoom(width, max_x - min_x, height, max_y - min_y)
        self._show(widget, zoom, _box_center(bounds))

    @classmethod
    def _fit_zoom(cls, size_x, span_x, size_y, span_y):
        """The zoom that frames ``span_x`` by ``span_y`` in a canvas ``size_x``
        by ``size_y``.

        Use the tightest positive-span axis; skip a flat axis so ``_MIN_SPAN``
        does not shrink the other, and fall back to that span when both are
        flat.
        """
        room = [size / span
                for size, span in ((size_x, span_x), (size_y, span_y))
                if span > 0.0]
        return cls._FIT_MARGIN * min(
            room or [min(size_x, size_y) / cls._MIN_SPAN])

    def _on_actual(self):
        """Zoom to one pixel per world unit, about the view's own middle."""
        widget = self._canvas()
        if widget is None:
            return
        width, height = widget.viewportSize
        # Set zoom to 1; scaling by 1/zoom need not land exactly there.
        self._show(widget, 1.0, widget.viewTransform.world_from_screen(
            0.5 * width, 0.5 * height))

    def _on_center(self):
        """Put the content, or the origin, in the middle of the canvas."""
        widget = self._canvas()
        if widget is None:
            return
        bounds = self._content_bounds(widget)
        self._show(widget, widget.viewTransform.zoom,
                   (0.0, 0.0) if bounds is None else _box_center(bounds))

    def _show(self, widget, zoom, center):
        """Zoom ``widget`` to ``zoom`` with ``center`` in the middle of it.

        Apply zoom first so the pan uses the clamped transform; on a
        non-finite pan, restore the prior view so zoom is not left alone.
        """
        before = widget.viewTransform
        # A second copy of the same transform: the first is the way back.
        view = widget.viewTransform
        view.zoom = zoom
        widget.setViewTransform(view)
        view = widget.viewTransform
        width, height = widget.viewportSize
        center_x, center_y = center
        view.pan_x = 0.5 * width - view.zoom * center_x
        view.pan_y = 0.5 * height + view.zoom * center_y
        if not (math.isfinite(view.pan_x) and math.isfinite(view.pan_y)):
            widget.setViewTransform(before)
            return
        widget.setViewTransform(view)
        widget.requestRepaint()
        self.refresh()

    def _apply_style(self):
        """Color the section heads, the readout, and the buttons from the
        palette."""
        self.setStyleSheet(
            Rules.sheet(self, "section", "box", "readout", "action"))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
