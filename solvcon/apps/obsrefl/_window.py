# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The window of the reflection app: the run strip over the figure."""

from PySide6.QtCore import QSize

from ...pilot import panel

from . import _column

__all__ = [
    'ReflectionWindow',
]


class ReflectionWindow(panel.RunWindow):
    """The reflection window: the run strip over the figure it is handed.
    The control column docks beside it as a panel of its own.

    :param figure: The widget to show under the strip, any :class:`QWidget`.
    """

    #: The problem name opens with the Mach number.
    SHORTEST = "Mach 3..."

    def __init__(self, figure, parent=None):
        super().__init__(figure, parent)
        self.setObjectName("obsrefl")

    def sizeHint(self):
        return QSize(1000 - _column.ControlColumn.WIDTH, 560)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
