# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
A group of a config column that folds behind its head.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QGridLayout, QLabel, QPushButton, QSizePolicy,
                               QStyle, QStyleOptionButton, QVBoxLayout,
                               QWidget)

from .. import _style

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
    a style sheet, and the summary carries a ``dirty`` property while edits
    wait.

    :ivar is_open: Whether the body shows.
    :ivar toggled: Owner-supplied callback fired with the group when its
        head is clicked.
    """

    #: The style of the head; a subclass styled by a sheet of its own
    #: clears it.
    HEAD_SHEET = "text-align: left; padding: 4px 2px;"
    #: The least room between the title and the summary, which elides
    #: where it would run over; None lets the summary run.
    GAP = None

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_open = True
        self.toggled = None
        self._text = ""
        self._head = QPushButton()
        self._head.setObjectName("foldHead")
        self._head.setFlat(True)
        self._head.setStyleSheet(self.HEAD_SHEET)
        self._head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._head.setCursor(Qt.PointingHandCursor)
        font = QFont(self.font())
        font.setWeight(QFont.DemiBold)
        # Uppercase by the font, so the head reads its title as written.
        font.setCapitalization(QFont.AllUppercase)
        self._head.setFont(font)
        self._head.setAccessibleName(title)
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
        self._head.setText(f"{arrow}  {self.title}")
        self.body.setVisible(self.is_open)

    def set_open(self, open_):
        """Show the body, or fold it behind the head."""
        self.is_open = bool(open_)
        self._sync()

    def summary(self):
        """Return the text on the right of the head."""
        return self._text

    def set_summary(self, text, dirty=False):
        """Say ``text`` on the right of the head, marked ``dirty`` while
        edits wait."""
        self._text = text
        self._head.setToolTip(text)
        _style.set_style_property(self._summary, "dirty",
                                  "true" if dirty else "false")
        self._fit_summary()

    def _fit_summary(self):
        text = self._text
        if self.GAP is not None:
            # The padding before the title is read off the style rather
            # than repeated from the sheet.
            option = QStyleOptionButton()
            option.initFrom(self._head)
            left = self._head.style().subElementRect(
                QStyle.SE_PushButtonContents, option, self._head).left()
            margins = self._summary.contentsMargins()
            room = (self.width() - left - self.GAP - margins.left()
                    - margins.right() - self._head.fontMetrics()
                    .horizontalAdvance(self._head.text()))
            text = self._summary.fontMetrics().elidedText(
                text, Qt.ElideRight, max(0, room))
        self._summary.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.GAP is not None:
            self._fit_summary()

    def _on_head(self):
        if self.toggled is not None:
            self.toggled(self)
        else:
            self.set_open(not self.is_open)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
