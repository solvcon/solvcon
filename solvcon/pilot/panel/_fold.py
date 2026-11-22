# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
A group of a config column that folds behind its head.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QGridLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

__all__ = [
    'FoldGroup',
]


ARROW_OPEN = chr(0x25be)
ARROW_FOLDED = chr(0x25b8)


class FoldGroup(QWidget):
    """A group of a config column that folds behind its head.  The head
    carries a summary on its right, so a folded group still says what it
    holds.

    The head and the summary are named ``foldHead`` and ``foldSummary`` for
    a style sheet.

    :ivar is_open: Whether the body shows.
    :ivar toggled: Owner-supplied callback fired with the group when its
        head is clicked.
    """

    #: The style of the head; a subclass styled by a sheet of its own
    #: clears it.
    HEAD_SHEET = "text-align: left; padding: 4px 2px;"

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_open = True
        self.toggled = None
        self._head = QPushButton()
        self._head.setObjectName("foldHead")
        self._head.setFlat(True)
        self._head.setStyleSheet(self.HEAD_SHEET)
        self._head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._head.setCursor(Qt.PointingHandCursor)
        font = QFont(self.font())
        font.setWeight(QFont.DemiBold)
        self._head.setFont(font)
        self._head.clicked.connect(self._on_head)
        self._summary = QLabel("")
        self._summary.setObjectName("foldSummary")
        # The summary lets the mouse through to the head under it.
        self._summary.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._summary.setEnabled(False)
        self.body = QWidget()
        head = QGridLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._head, 0, 0)
        head.addWidget(self._summary, 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.body)
        self._sync()

    def _sync(self):
        arrow = ARROW_OPEN if self.is_open else ARROW_FOLDED
        self._head.setText(f"{arrow}  {self.title.upper()}")
        self.body.setVisible(self.is_open)

    def set_open(self, open_):
        """Show the body, or fold it behind the head."""
        self.is_open = bool(open_)
        self._sync()

    def summary(self):
        """Return the text on the right of the head."""
        return self._summary.text()

    def set_summary(self, text):
        """Say ``text`` on the right of the head."""
        self._summary.setText(text)
        self._head.setToolTip(text)

    def _on_head(self):
        if self.toggled is not None:
            self.toggled(self)
        else:
            self.set_open(not self.is_open)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
