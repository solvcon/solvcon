# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The Layers page of the Painter inspector: every object the world holds.
"""

import json

from PySide6 import QtCore, QtGui, QtWidgets

from . import _painter_icons
from . import _painter_rows
from ._painter_style import blend, PaletteStyled

__all__ = [
    'LayersPage',
]


class LayersPage(PaletteStyled):
    """The inspector's Layers page: one row per object the world holds.

    Like the Design page, the page reads the canvas it is bound to on a timer,
    because the world changes in C++ without a change signal. What the poll
    compares is the row content itself, which is both what the page shows and
    the only fingerprint that cannot go stale as the rows grow what they say.

    The rows are read-only: what a row shows comes from the world, and picking
    an object stays the canvas's job until the list grows selection of its own.
    """

    # Poll period in milliseconds. Between the entity tree's half second and
    # the Design page's 60ms: the whole world is serialized per tick, which is
    # too much to pay at the faster rate, while the half second reads as lag
    # against a canvas the user is drawing on.
    _POLL_MS = 250

    #: What the list says while the world holds nothing.
    EMPTY_TEXT = "No objects"

    _LIST_MARGINS = (6, 8, 6, 8)
    _EMPTY_MARGINS = (12, 4, 12, 4)
    _EMPTY_PX = 12

    # How far the empty note sits from the panel color.
    _GREYED_MIX = 0.6

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._key = None
        self._pixmaps = {}
        self.rows = []
        self._build()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self.refresh)
        self._apply_style()
        self._render(())

    @property
    def shown_rows(self):
        """The rows standing for an object right now.

        A row outlives the object it showed: the list keeps the rows it has
        built and hides the surplus, so a world that shrinks and grows again
        costs no widgets.
        """
        return [row for row in self.rows if not row.isHidden()]

    def set_canvas_source(self, source):
        """Bind the page to ``source``, a callable handing back the canvas.

        The callable returns the 2D canvas to read, or ``None`` when none is
        active. The canvas is asked for on every read rather than held,
        because it is a C++ widget owned by its sub-window: a reference kept
        across the window's close outlives the object behind it, and the next
        read walks freed memory.
        """
        self._source = source
        # Rendered rather than refreshed: two blank canvases show the same
        # nothing, so a refresh would take the page for unchanged and leave
        # the previous canvas's objects on screen.
        self._key = self._content()
        self._render(self._key)

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
        """Redraw the list when the canvas it is bound to has changed."""
        content = self._content()
        if content == self._key:
            return
        self._key = content
        self._render(content)

    def _build(self):
        self._empty = QtWidgets.QLabel(self.EMPTY_TEXT, self)
        self._empty.setObjectName("empty")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_list(), 1)

    def _build_list(self):
        """The scrolling column of rows, with the empty note above it.

        A world of many objects would otherwise grow the page past the dock,
        so the column scrolls inside the page rather than stretching it.
        """
        body = QtWidgets.QWidget()
        self._rows_layout = QtWidgets.QVBoxLayout(body)
        self._rows_layout.setContentsMargins(*self._LIST_MARGINS)
        self._rows_layout.setSpacing(0)
        self._empty.setContentsMargins(*self._EMPTY_MARGINS)
        self._rows_layout.addWidget(self._empty)
        self._rows_layout.addStretch(1)
        area = QtWidgets.QScrollArea(self)
        area.setWidget(body)
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        return area

    def _canvas(self):
        """The 2D canvas to read, or ``None`` when none is active."""
        return None if self._source is None else self._source()

    def _world(self):
        """The active canvas's world, or ``None`` when it has none."""
        widget = self._canvas()
        return None if widget is None else widget.world

    @staticmethod
    def _objects(world):
        """Every live shape the world holds, as the world describes it.

        Newest first, which is the order the design lists: the world draws in
        registry order, so the last shape added is the one on top of the
        canvas and the one the list names first.
        """
        return json.loads(world.describe_state())["shapes"][::-1]

    def _content(self):
        """One entry per row, in list order, as the rows will show it.

        This is what the poll compares and what :meth:`_render` draws, so the
        world is read once per tick rather than once to decide and again to
        draw.
        """
        world = self._world()
        if world is None:
            return ()
        return tuple((shape["type"], self._name_of(shape))
                     for shape in self._objects(world))

    def _render(self, content):
        for index, (kind, name) in enumerate(content):
            self._row(index).show_object(name, self._icon_of(kind))
        for row in self.rows[len(content):]:
            row.hide()
        self._empty.setVisible(not content)

    def _row(self, index):
        """The row at ``index``, built and shown if it is a new one."""
        while len(self.rows) <= index:
            row = _painter_rows.ObjectRow(self)
            # Above the trailing stretch, so the rows stay packed at the top.
            self._rows_layout.insertWidget(len(self.rows) + 1, row)
            self.rows.append(row)
        self.rows[index].show()
        return self.rows[index]

    @staticmethod
    def _name_of(shape):
        """The row label for ``shape``, its type and its id.

        The world has no names yet, so the type carries the row until it does.
        """
        return f"{shape['type'].title()} {shape['id']}"

    def _icon_of(self, name):
        """The icon for the shape type ``name``, or ``None`` for a type the
        icon set does not draw yet.

        The pixmaps are cached per type: a world of many objects would
        otherwise rasterize the same handful of icons on every change. The
        cache is dropped when the palette changes, and keyed by the scale it
        rasterized for, because the move to a screen of another scale is not
        reported on every Qt the pilot builds against.
        """
        if name not in _painter_icons.ICONS:
            return None
        ratio = self.devicePixelRatioF()
        key = (name, ratio)
        if key not in self._pixmaps:
            self._pixmaps[key] = _painter_icons.render(
                name, _painter_rows.icon_color(self), _painter_rows.ICON_PX,
                ratio)
        return self._pixmaps[key]

    def _apply_style(self):
        """Color the rows and the empty note from the palette."""
        palette = self.palette()
        text = palette.color(QtGui.QPalette.WindowText)
        panel = palette.color(QtGui.QPalette.Window)
        self.setStyleSheet(_painter_rows.rules() + f"""
            QLabel#empty {{
                font-size: {self._EMPTY_PX}px;
                color: {blend(text, panel, self._GREYED_MIX).name()};
            }}
            """)
        self._pixmaps.clear()
        # The rows hold the icons the cleared cache handed out, and a stale
        # one keeps the color it was rendered in through the theme switch.
        self._key = None
        self.refresh()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
