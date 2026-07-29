# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The object row the Painter inspector lists.

The design draws the same row twice, filling the Layers page and the short
list at the foot of the Design page, so the widget and the rules that color it
live together here rather than in either page.
"""

from PySide6 import QtGui, QtWidgets

from ._painter_style import blend

__all__ = [
    'ICON_PX',
    'HEIGHT',
    'ObjectRow',
    'icon_color',
    'rules',
]

#: The type icon's size and the row's own, in device-independent pixels.
ICON_PX = 13
HEIGHT = 30

_MARGINS = (8, 0, 8, 0)
_GAP = 8
_NAME_PX = 12
_RADIUS = 5

# How far the type icon sits from the panel color.
_MUTED_MIX = 0.35


class ObjectRow(QtWidgets.QFrame):
    """One object row: a type icon, then the object's name."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("row")
        self.setFixedHeight(HEIGHT)
        self._icon = QtWidgets.QLabel(self)
        self._icon.setFixedWidth(ICON_PX)
        self._name = QtWidgets.QLabel(self)
        self._name.setObjectName("name")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*_MARGINS)
        layout.setSpacing(_GAP)
        layout.addWidget(self._icon)
        layout.addWidget(self._name)
        layout.addStretch(1)

    @property
    def name(self):
        """The object name the row shows."""
        return self._name.text()

    @property
    def icon(self):
        """The type icon the row shows."""
        return self._icon.pixmap()

    def show_object(self, name, icon):
        """Show one object, ``icon`` being its pixmap or ``None`` for a type
        the icon set does not draw."""
        self._name.setText(name)
        if icon is None:
            self._icon.clear()
        else:
            self._icon.setPixmap(icon)


def icon_color(widget):
    """The color a row on ``widget`` strokes its type icon in."""
    palette = widget.palette()
    return blend(palette.color(QtGui.QPalette.WindowText),
                 palette.color(QtGui.QPalette.Window), _MUTED_MIX)


def rules():
    """The style rules a page carries for the rows it holds.

    They travel with the row rather than sitting in each page's own sheet, so
    the two lists the design draws cannot drift apart. A page appends them to
    its sheet, which keeps one sheet covering every row it holds.
    """
    return f"""
        QFrame#row {{
            border-radius: {_RADIUS}px;
        }}
        QLabel#name {{
            font-size: {_NAME_PX}px;
        }}
        """

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
