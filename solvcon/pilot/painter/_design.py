# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The Design page of the Painter inspector: what is selected and where it sits.
"""

import math

from PySide6 import QtCore, QtGui, QtWidgets

from . import _icons
from ._sections import Placeholder
from ._style import (blend, mono_font, obb_metrics, rule, shade,
                     PaletteStyled)

__all__ = [
    'DesignPage',
]


def _format(value):
    """Return a world coordinate or length as the fields show it."""
    return f"{value:g}"


def _lands_finite(obb, delta):
    """Return whether moving ``obb`` by ``delta`` keeps its corners finite.

    The shape lies inside its own box, so corners that survive the step bound
    the geometry that follows them.
    """
    dx, dy = delta
    return all(math.isfinite(x + dx) and math.isfinite(y + dy)
               for x, y in zip(obb[0::2], obb[1::2]))


class _Field(QtWidgets.QFrame):
    """One box of the position grid: an axis letter, then its value.

    An editable field commits on ``editingFinished``, which is Enter or a
    focus change, and only when the text parses to another number; text that
    does not parse falls back to the value the field was showing.
    """

    #: The edited value, once it parses to a number the shape does not hold.
    committed = QtCore.Signal(float)

    _HEIGHT = 28
    _MARGINS = (8, 0, 8, 0)
    _GAP = 6
    _LETTER_WIDTH = 10

    def __init__(self, letter, editable, parent=None):
        super().__init__(parent)
        self._value = None
        self._typed = False
        self.setObjectName("field")
        self.setFixedHeight(self._HEIGHT)
        label = QtWidgets.QLabel(letter, self)
        label.setObjectName("axis")
        label.setFixedWidth(self._LETTER_WIDTH)
        self._edit = QtWidgets.QLineEdit(self)
        self._edit.setFont(mono_font())
        self._edit.setReadOnly(not editable)
        # An editor asks for room for a line of text, and four of them would
        # open the dock wider than the inspector the design draws. The grid
        # column decides the width here.
        self._edit.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                 QtWidgets.QSizePolicy.Fixed)
        self._edit.textEdited.connect(self._on_typed)
        self._edit.editingFinished.connect(self._on_edited)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*self._MARGINS)
        layout.setSpacing(self._GAP)
        layout.addWidget(label)
        layout.addWidget(self._edit, 1)

    @property
    def edit(self):
        """The value editor, for a caller that drives or reads its text."""
        return self._edit

    def value(self):
        """Return the value the field stands for, or ``None`` for none."""
        return self._value

    def set_value(self, value):
        """Show ``value``, which is ``None`` for an empty field."""
        self._value = value
        # A field the user is typing in keeps its text: the page refreshes on
        # a poll, which would otherwise overwrite the entry mid-edit.
        if not self._edit.hasFocus():
            self._show()

    def revert(self):
        """Show the value again, dropping whatever the field holds."""
        self._show()

    def _show(self):
        self._typed = False
        self._edit.blockSignals(True)
        try:
            self._edit.setText(
                "" if self._value is None else _format(self._value))
        finally:
            self._edit.blockSignals(False)

    def _on_typed(self, _text):
        self._typed = True

    def _on_edited(self):
        # editingFinished fires on a focus change as well, and the text is a
        # rounded view of the value, so a field nobody typed in would commit
        # its own rounding error as a move.
        if not self._typed:
            return
        # One finished edit consumes the typing. Enter leaves the field
        # focused with its text as typed, and a poll may move the value on
        # underneath it; the focus change that follows must not write the old
        # text back over that.
        self._typed = False
        try:
            value = float(self._edit.text())
        except ValueError:
            self._show()
            return
        # "nan" and "inf" parse, and a shape moved to either is not brought
        # back by undo: the inverse translation stays non-finite too.
        if not math.isfinite(value):
            self._show()
            return
        if self._value is not None and value != self._value:
            self.committed.emit(value)


class DesignPage(PaletteStyled):
    """The inspector's Design page: the selected shape and its position.

    The page reads the canvas it is bound to on a timer, because the world
    changes in C++ without a change signal. What the poll compares is the
    selection itself, id and type and box, which is both what the page shows
    and cheap enough to read every tick.

    X and Y are the selection's center and are editable; W and H come from the
    same oriented box and stay read-only until the world grows a scale
    operation.
    """

    # Poll period in milliseconds. The fields are read while a drag moves the
    # shape under them, so the entity tree's half second reads as lag, and one
    # shape's box is cheap enough to re-read this often.
    _POLL_MS = 60

    #: The greyed-out sections, as ``(title, what the section waits for)``.
    PLACEHOLDERS = (
        ("Stroke", "per-shape stroke style"),
        ("Fill", "per-shape fill color"),
        ("Grid & snap", "grid and snap options"),
    )

    #: The header text while nothing is selected.
    EMPTY_TEXT = "Nothing selected"

    _POSITION = (("X", True), ("Y", True), ("W", False), ("H", False))

    _ICON_PX = 13
    _MARGINS = (12, 11, 12, 11)
    _GAP = 10
    _HEADER_GAP = 8
    _GRID_GAP = 6
    _NAME_PX = 12
    _SMALL_PX = 10
    _VALUE_PX = 11
    _RADIUS = 5

    # How far each of these sits from the panel color.
    _MUTED_MIX = 0.35
    _GREYED_MIX = 0.6
    _BADGE_MIX = 0.15
    _BORDER_MIX = 0.25

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._key = None
        self._icon_name = None
        self.fields = {}
        self.placeholders = {}
        self._build()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self.refresh)
        self._apply_style()
        self._render()

    def set_canvas_source(self, source):
        """Bind the page to ``source``, a callable handing back the canvas.

        The callable returns the 2D canvas to read, or ``None`` when none is
        active. The canvas is asked for on every read rather than held,
        because it is a C++ widget owned by its sub-window: a reference kept
        across the window's close outlives the object behind it, and the next
        read walks freed memory.
        """
        self._source = source
        # Rendered rather than refreshed: a canvas with nothing selected and
        # no canvas at all share a poll key, so a refresh would take the page
        # for unchanged and leave the previous canvas's selection on screen.
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
        layout.addWidget(self._build_selection())
        for title, waits_for in self.PLACEHOLDERS:
            layout.addWidget(rule(QtWidgets.QFrame.HLine))
            layout.addWidget(self._add_placeholder(title, waits_for))

        layout.addWidget(rule(QtWidgets.QFrame.HLine))
        layout.addWidget(
            self._add_placeholder("Layers", "shape names", folded=False))
        layout.addStretch(1)

    def _add_placeholder(self, title, waits_for, folded=True):
        self.placeholders[title] = Placeholder(title, waits_for, folded, self)
        return self.placeholders[title]

    def _build_selection(self):
        """Build the selection header over the position grid."""
        block = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(block)
        layout.setContentsMargins(*self._MARGINS)
        layout.setSpacing(self._GAP)
        layout.addLayout(self._build_header(block))
        layout.addLayout(self._build_grid(block))
        return block

    def _build_header(self, block):
        self._icon = QtWidgets.QLabel(block)
        self._name = QtWidgets.QLabel(block)
        self._name.setObjectName("name")
        self._badge = QtWidgets.QLabel("1 selected", block)
        self._badge.setObjectName("badge")
        self._badge.setFont(mono_font())
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(self._HEADER_GAP)
        layout.addWidget(self._icon)
        layout.addWidget(self._name)
        layout.addStretch(1)
        layout.addWidget(self._badge)
        return layout

    def _build_grid(self, block):
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(self._GRID_GAP)
        for index, (letter, editable) in enumerate(self._POSITION):
            field = _Field(letter, editable, block)
            if editable:
                field.committed.connect(
                    lambda value, at=letter: self._on_committed(at, value))
            else:
                field.setToolTip(f"{letter}  (needs the scale operation)")
            layout.addWidget(field, index // 2, index % 2)
            self.fields[letter] = field
        return layout

    def _canvas(self):
        """The 2D canvas to read, or ``None`` when none is active."""
        return None if self._source is None else self._source()

    def _state_key(self):
        """What the page shows, as a value the poll can compare.

        The key is read from the selection alone, which is all the page
        shows. A world-wide fingerprint would cost a full serialization every
        tick and still miss a plain selection click, which moves no geometry.
        """
        _widget, world, shape = self._selection()
        if world is None:
            return None
        return (shape, world.shape_type_of(shape),
                tuple(world.shape_obb(shape)))

    def _selection(self):
        """The active canvas, its world, and its live selection.

        ``selectedShape`` hands back the stored id without checking it, so a
        shape that was removed or undone away leaves a stale id behind and the
        queries the page runs on it would throw. Such a selection reads here
        as none at all.
        """
        widget = self._canvas()
        world = None if widget is None else widget.world
        if world is None or not world.shape_is_live(widget.selectedShape):
            return widget, None, -1
        return widget, world, widget.selectedShape

    def _render(self):
        _widget, world, shape = self._selection()
        if world is None:
            self._icon_name = None
            self._name.setText(self.EMPTY_TEXT)
            self._badge.setVisible(False)
            for field in self.fields.values():
                field.setEnabled(False)
                field.set_value(None)
        else:
            kind = world.shape_type_of(shape)
            self._icon_name = kind if kind in _icons.ICONS else None
            self._name.setText(f"{kind.title()} {shape}")
            self._badge.setVisible(True)
            metrics = obb_metrics(world.shape_obb(shape))
            for (letter, _editable), value in zip(self._POSITION, metrics):
                self.fields[letter].setEnabled(True)
                self.fields[letter].set_value(value)
        self._show_icon()

    def _show_icon(self):
        """Draw the selection's shape icon in the accent color."""
        if self._icon_name is None:
            self._icon.clear()
            self._icon.setVisible(False)
            return
        self._icon.setPixmap(_icons.render(
            self._icon_name, self.palette().color(QtGui.QPalette.Highlight),
            self._ICON_PX, self.devicePixelRatioF()))
        self._icon.setVisible(True)

    def _on_committed(self, axis, value):
        """Move the selection so its center reads ``value`` on ``axis``."""
        widget, world, shape = self._selection()
        if world is None:
            return
        obb = world.shape_obb(shape)
        center_x, center_y = obb_metrics(obb)[:2]
        if "X" == axis:
            delta = (value - center_x, 0.0)
        else:
            delta = (0.0, value - center_y)
        # The step between two finite coordinates can overflow, and so can a
        # wide shape's far corner once a finite step lands on it, so the box
        # is checked where it would end up rather than the step alone.
        if not _lands_finite(obb, delta):
            self.fields[axis].revert()
            return

        world.begin_operation()
        try:
            world.translate_shape(shape, *delta)
        finally:
            world.end_operation()

        widget.requestRepaint()
        self.refresh()

    def _apply_style(self):
        """Color the header, the badge, and the fields from the palette."""
        palette = self.palette()
        text = palette.color(QtGui.QPalette.WindowText)
        panel = palette.color(QtGui.QPalette.Window)
        muted = blend(text, panel, self._MUTED_MIX)
        greyed = blend(text, panel, self._GREYED_MIX)
        self.setStyleSheet(f"""
            QLabel#name {{
                font-size: {self._NAME_PX}px;
            }}
            QLabel#axis {{
                font-size: {self._SMALL_PX}px;
                color: {muted.name()};
            }}
            QLabel#axis:disabled {{
                color: {greyed.name()};
            }}
            QLabel#badge {{
                border-radius: 4px;
                padding: 2px 6px;
                font-size: {self._SMALL_PX}px;
                background: {shade(self, self._BADGE_MIX).name()};
                color: {muted.name()};
            }}
            QFrame#field {{
                border: 1px solid {shade(self, self._BORDER_MIX).name()};
                border-radius: {self._RADIUS}px;
                background: {palette.color(QtGui.QPalette.Base).name()};
            }}
            QLineEdit {{
                border: none;
                background: transparent;
                font-size: {self._VALUE_PX}px;
            }}
            QLineEdit:read-only {{
                color: {muted.name()};
            }}
            """)
        self._show_icon()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
