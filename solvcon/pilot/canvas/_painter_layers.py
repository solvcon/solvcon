# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The Layers page of the Painter inspector: every object the world holds.
"""

import json

from PySide6 import QtCore, QtGui, QtWidgets

from . import _painter_icons
from . import _painter_rows
from ._painter_style import (blend, mono_font, obb_metrics, rule, shade,
                             PaletteStyled)

__all__ = [
    'LayersPage',
]


class _Filters(QtWidgets.QWidget):
    """The greyed-out search field over the filter chips.

    Both wait on the same follow-ups, object names and guides, and the design
    draws them as one block above the list, so they are built together.
    """

    _SEARCH_MARGINS = (10, 10, 10, 8)
    _SEARCH_HEIGHT = 28
    _SEARCH_PADDING = (8, 0, 8, 0)
    _SEARCH_GAP = 8
    _CHIP_MARGINS = (10, 0, 10, 8)
    _CHIP_GAP = 6
    _ICON_PX = 12

    #: The filter chips, in design order; the first is the one shown active.
    CHIPS = ("All", "Shapes", "Guides")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chips = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(self._build_search())
        layout.addLayout(self._build_chips())
        self.setEnabled(False)

    def set_icon(self, icon):
        """Draw the search glass in ``icon``."""
        self._glass.setPixmap(icon)

    def _build_search(self):
        self.search = QtWidgets.QFrame(self)
        self.search.setObjectName("search")
        self.search.setFixedHeight(self._SEARCH_HEIGHT)
        self.search.setToolTip("Search objects  (needs object names)")
        self._glass = QtWidgets.QLabel(self.search)
        self._glass.setFixedWidth(self._ICON_PX)
        edit = QtWidgets.QLineEdit(self.search)
        edit.setObjectName("query")
        edit.setPlaceholderText("Search objects")
        inside = QtWidgets.QHBoxLayout(self.search)
        inside.setContentsMargins(*self._SEARCH_PADDING)
        inside.setSpacing(self._SEARCH_GAP)
        inside.addWidget(self._glass)
        inside.addWidget(edit, 1)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(*self._SEARCH_MARGINS)
        layout.addWidget(self.search)
        return layout

    def _build_chips(self):
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(*self._CHIP_MARGINS)
        layout.setSpacing(self._CHIP_GAP)
        for index, name in enumerate(self.CHIPS):
            chip = QtWidgets.QPushButton(name, self)
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setChecked(0 == index)
            chip.setToolTip(f"{name}  (needs guides and object names)")
            layout.addWidget(chip)
            self.chips[name] = chip
        layout.addStretch(1)
        return layout


class _Footer(QtWidgets.QWidget):
    """The list footer: the greyed-out add and remove buttons, then a count."""

    _MARGINS = (10, 8, 12, 8)
    _GAP = 6
    _BUTTON_SIZE = (26, 24)
    _ICON_PX = 12

    #: The greyed-out buttons, as ``(icon name, what the button does)``.
    BUTTONS = (("plus", "Add object"), ("minus", "Remove object"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.buttons = {}
        self.count = QtWidgets.QLabel(self)
        self.count.setObjectName("count")
        self.count.setFont(mono_font())
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*self._MARGINS)
        layout.setSpacing(self._GAP)
        for name, what in self.BUTTONS:
            layout.addWidget(self._build_button(name, what))
        layout.addStretch(1)
        layout.addWidget(self.count)

    def set_icons(self, ratio, color):
        """Draw each button in ``color``, rasterized for ``ratio``.

        The buttons are greyed out for good, so the icon is offered for the
        disabled state alone and carries the greyed color the page mixed,
        rather than whatever fade the style would apply to a normal one.
        """
        for name, button in self.buttons.items():
            button.setIcon(_painter_icons.placeholder_icon(
                name, self._ICON_PX, color, ratio))

    def _build_button(self, name, what):
        button = QtWidgets.QToolButton(self)
        button.setObjectName("footer")
        button.setFixedSize(*self._BUTTON_SIZE)
        button.setIconSize(QtCore.QSize(self._ICON_PX, self._ICON_PX))
        button.setToolTip(f"{what}  (needs editing from the list)")
        button.setEnabled(False)
        self.buttons[name] = button
        return button


class LayersPage(PaletteStyled):
    """The inspector's Layers page: one row per object the world holds.

    Like the Design page, the page reads the canvas it is bound to on a timer,
    because the world changes in C++ without a change signal. What the poll
    compares is the row content itself, which is both what the page shows and
    the only fingerprint that cannot go stale: a coarser one, such as the
    world's serialized boxes, reads the same for a shape lying along the axes
    and one of the other proportions turned onto them.

    The rows are read-only: what a row shows comes from the world, and picking
    an object stays the canvas's job until the list grows selection of its own.
    """

    # Poll period in milliseconds. Between the entity tree's half second and
    # the Design page's 60ms: the whole world is serialized per tick, which is
    # too much to pay at the faster rate, while the half second reads as lag
    # when a click moves the highlight from row to row.
    _POLL_MS = 250

    #: What the list says while the world holds nothing.
    EMPTY_TEXT = "No objects"

    _LIST_MARGINS = (6, 0, 6, 0)
    _EMPTY_MARGINS = (12, 4, 12, 4)
    _EMPTY_PX = 12
    _COUNT_PX = 10
    _CHIP_PX = 10
    _SEARCH_PX = 11
    _RADIUS = 5
    _CHIP_RADIUS = 11
    _ICON_PX = 12

    # How far each of these sits from the panel color.
    _MUTED_MIX = 0.35
    _GREYED_MIX = 0.6
    _CHIP_MIX = 0.15
    _BORDER_MIX = 0.25

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
    def placeholders(self):
        """The greyed-out controls as ``{name: widget}``."""
        entries = {"Search": self._filters.search}
        entries.update(self._filters.chips)
        entries.update({what: self._footer.buttons[name]
                        for name, what in _Footer.BUTTONS})
        return entries

    @property
    def count(self):
        """The footer's object count, as the footer words it."""
        return self._footer.count.text()

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
        self._filters = _Filters(self)
        self._empty = QtWidgets.QLabel(self.EMPTY_TEXT, self)
        self._empty.setObjectName("empty")
        self._footer = _Footer(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._filters)
        layout.addWidget(self._build_list(), 1)
        layout.addWidget(rule(QtWidgets.QFrame.HLine))
        layout.addWidget(self._footer)

    def _build_list(self):
        """The scrolling column of rows, with the empty note above it.

        A world of many objects would otherwise push the footer off the dock,
        so the column scrolls inside the page rather than growing it.
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

    def _selection(self):
        """The active canvas's world and its live selection.

        ``selectedShape`` hands back the stored id without checking it, so a
        shape that was removed or undone away leaves a stale id behind and the
        queries the page runs on it would throw. Such a selection reads here
        as none at all.
        """
        widget = self._canvas()
        world = None if widget is None else widget.world
        if world is None:
            return None, -1
        selected = widget.selectedShape
        return world, selected if world.shape_is_live(selected) else -1

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
        world, selected = self._selection()
        if world is None:
            return ()
        return tuple((shape["type"], self._name_of(shape),
                      self._metric_of(world, shape),
                      shape["id"] == selected)
                     for shape in self._objects(world))

    def _render(self, content):
        for index, (kind, name, metric, picked) in enumerate(content):
            self._row(index).show_object(
                name, metric, self._icon_of(kind, picked), picked)
        for row in self.rows[len(content):]:
            row.hide()
        self._empty.setVisible(not content)
        self._footer.count.setText(
            "1 shape" if 1 == len(content) else f"{len(content)} shapes")

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

    @staticmethod
    def _metric_of(world, shape):
        """The one measurement the row shows: a circle's radius, or the
        width and height of anything else.

        Both come from the shape's own oriented box, because the serialized
        box is axis-aligned and reads wider than a rotated shape really is.
        """
        width, height = obb_metrics(world.shape_obb(shape["id"]))[2:]
        if "circle" == shape["type"]:
            return f"r {0.5 * width:g}"
        return f"{width:g} x {height:g}"

    def _icon_of(self, name, selected):
        """The icon for the shape type ``name``, or ``None`` for a type the
        icon set does not draw yet.

        The pixmaps are cached per type and appearance: a world of many
        objects would otherwise rasterize the same handful of icons on every
        change. The cache is dropped when the palette changes, and keyed by
        the scale it rasterized for, because the move to a screen of another
        scale is not reported on every Qt the pilot builds against.
        """
        if name not in _painter_icons.ICONS:
            return None
        ratio = self.devicePixelRatioF()
        key = (name, selected, ratio)
        if key not in self._pixmaps:
            self._pixmaps[key] = _painter_icons.render(
                name, _painter_rows.icon_color(self, selected),
                _painter_rows.ICON_PX, ratio)
        return self._pixmaps[key]

    def _apply_style(self):
        """Color the rows, the placeholders, and the footer from the
        palette."""
        palette = self.palette()
        text = palette.color(QtGui.QPalette.WindowText)
        panel = palette.color(QtGui.QPalette.Window)
        greyed = blend(text, panel, self._GREYED_MIX)
        self.setStyleSheet(_painter_rows.rules(self) + f"""
            QLabel#empty {{
                font-size: {self._EMPTY_PX}px;
                color: {greyed.name()};
            }}
            QLabel#count {{
                font-size: {self._COUNT_PX}px;
                color: {blend(text, panel, self._MUTED_MIX).name()};
            }}
            QFrame#search {{
                border: 1px solid {shade(self, self._BORDER_MIX).name()};
                border-radius: {self._RADIUS}px;
                background: {palette.color(QtGui.QPalette.Base).name()};
            }}
            QLineEdit#query {{
                border: none;
                background: transparent;
                font-size: {self._SEARCH_PX}px;
            }}
            QPushButton#chip {{
                border: none;
                border-radius: {self._CHIP_RADIUS}px;
                padding: 3px 8px;
                font-size: {self._CHIP_PX}px;
                background: transparent;
                color: {greyed.name()};
            }}
            QPushButton#chip:checked {{
                background: {shade(self, self._CHIP_MIX).name()};
            }}
            QToolButton#footer {{
                border: 1px solid {shade(self, self._BORDER_MIX).name()};
                border-radius: {self._RADIUS}px;
                background: {palette.color(QtGui.QPalette.Base).name()};
            }}
            """)
        ratio = self.devicePixelRatioF()
        self._filters.set_icon(
            _painter_icons.render("search", greyed, self._ICON_PX, ratio))
        self._footer.set_icons(ratio, greyed)
        self._pixmaps.clear()
        # The rows hold the icons the cleared cache handed out, and a stale
        # one keeps the color it was rendered in through the theme switch.
        self._key = None
        self.refresh()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
