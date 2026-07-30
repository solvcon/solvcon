# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The titled sections the pages of the Painter inspector are built from.

The design heads every section the same way, a small uppercase title with a
fold arrow, and leaves several of them to follow-ups. Both the open section and
the greyed-out stand-in live here, so a page that fills one and a page that
still waits on the model draw the same head.
"""

from PySide6 import QtGui, QtWidgets

__all__ = [
    'ARROW_OPEN',
    'ARROW_FOLDED',
    'Section',
    'Placeholder',
]

#: The fold arrow of an open section and of a folded one.
ARROW_OPEN = chr(0x25be)
ARROW_FOLDED = chr(0x25b8)

_MARGINS = (12, 11, 12, 11)
_GAP = 9
_FONT_PX = 10
_LETTER_SPACING = 110


class Section(QtWidgets.QWidget):
    """One section of a page: the design's head over the controls it holds.

    A page adds its controls to :attr:`body`, the layout the head already sits
    in, so they land under the title with the design's own spacing.
    """

    def __init__(self, title, arrow=ARROW_OPEN, parent=None):
        """
        :param arrow: The fold arrow to head the title with, or an empty
            string for a section that folds nothing.
        :type arrow: str
        """
        super().__init__(parent)
        self.body = QtWidgets.QVBoxLayout(self)
        self.body.setContentsMargins(*_MARGINS)
        self.body.setSpacing(_GAP)
        self.body.addWidget(self._build_title(title, arrow))

    def _build_title(self, title, arrow):
        label = QtWidgets.QLabel(
            f"{arrow}  {title.upper()}" if arrow else title.upper(), self)
        label.setObjectName("section")
        font = label.font()
        font.setPixelSize(_FONT_PX)
        font.setLetterSpacing(QtGui.QFont.PercentageSpacing, _LETTER_SPACING)
        label.setFont(font)
        return label


class Placeholder(Section):
    """A greyed-out stand-in for a section the model cannot fill yet.

    It carries the design's own title, so the page keeps its designed shape.
    """

    def __init__(self, title, waits_for, folded=True, parent=None):
        super().__init__(title, ARROW_FOLDED if folded else "", parent)
        self.setToolTip(f"{title}  (needs {waits_for})")
        self.setEnabled(False)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
