# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The window of the shock-tube app: the run strip over the figure."""

from PySide6.QtCore import QSize

from ...pilot import panel

from . import _column

__all__ = [
    'ShockTubeWindow',
]


class ShockTubeWindow(panel.RunWindow):
    """The shock-tube window: the run strip over the figure it is handed.
    The control column docks beside it as a panel of its own.

    :param figure: The widget to show under the strip, any :class:`QWidget`.
    """

    #: The problem name opens with the problem and its points.
    SHORTEST = "Sod, 2..."

    def __init__(self, figure, parent=None):
        super().__init__(figure, parent)
        self.setObjectName("shocktube")

    def sizeHint(self):
        return QSize(1000 - _column.ControlColumn.WIDTH, 560)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
