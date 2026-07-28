# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The Design page of the Painter inspector: what is selected and where it sits.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from . import _painter_icons
from ._painter_style import blend, shade, PaletteStyled

__all__ = [
    'DesignPage',
]


def _mono_font():
    """The stand-in for the mono font the design gives every number."""
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)


class DesignPage(PaletteStyled):
    """The inspector's Design page: what the canvas has selected.

    The page reads the canvas it is bound to on a timer, because the world
    changes in C++ without a change signal. What the poll compares is the
    selection itself, which is both what the page shows and cheap enough to
    read every tick.
    """

    # Poll period in milliseconds, the cadence the entity tree already uses.
    # The timer runs only while the page is on screen, so a hidden dock or
    # another inspector tab costs nothing.
    _POLL_MS = 500

    #: The header text while nothing is selected.
    EMPTY_TEXT = "Nothing selected"

    _ICON_PX = 13
    _MARGINS = (12, 11, 12, 11)
    _GAP = 10
    _HEADER_GAP = 8
    _NAME_PX = 12
    _SMALL_PX = 10

    # How far the muted text and the badge sit from the panel color.
    _MUTED_MIX = 0.35
    _BADGE_MIX = 0.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._key = None
        self._icon_name = None
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
        layout.addStretch(1)

    def _build_selection(self):
        """The selection header: the shape's icon, name, and count badge."""
        block = QtWidgets.QWidget(self)
        self._icon = QtWidgets.QLabel(block)
        self._name = QtWidgets.QLabel(block)
        self._name.setObjectName("name")
        self._badge = QtWidgets.QLabel("1 selected", block)
        self._badge.setObjectName("badge")
        self._badge.setFont(_mono_font())
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(self._HEADER_GAP)
        header.addWidget(self._icon)
        header.addWidget(self._name)
        header.addStretch(1)
        header.addWidget(self._badge)
        layout = QtWidgets.QVBoxLayout(block)
        layout.setContentsMargins(*self._MARGINS)
        layout.setSpacing(self._GAP)
        layout.addLayout(header)
        return block

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
        return None if world is None else (shape, world.shape_type_of(shape))

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
        else:
            kind = world.shape_type_of(shape)
            self._icon_name = kind if kind in _painter_icons.ICONS else None
            self._name.setText(f"{kind.title()} {shape}")
            self._badge.setVisible(True)
        self._show_icon()

    def _show_icon(self):
        """Draw the selection's shape icon in the accent color."""
        if self._icon_name is None:
            self._icon.clear()
            self._icon.setVisible(False)
            return
        self._icon.setPixmap(_painter_icons.render(
            self._icon_name, self.palette().color(QtGui.QPalette.Highlight),
            self._ICON_PX, self.devicePixelRatioF()))
        self._icon.setVisible(True)

    def _apply_style(self):
        """Color the header and the badge from the palette."""
        palette = self.palette()
        muted = blend(palette.color(QtGui.QPalette.WindowText),
                      palette.color(QtGui.QPalette.Window), self._MUTED_MIX)
        self.setStyleSheet(f"""
            QLabel#name {{
                font-size: {self._NAME_PX}px;
            }}
            QLabel#badge {{
                border-radius: 4px;
                padding: 2px 6px;
                font-size: {self._SMALL_PX}px;
                background: {shade(self, self._BADGE_MIX).name()};
                color: {muted.name()};
            }}
            """)
        self._show_icon()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
