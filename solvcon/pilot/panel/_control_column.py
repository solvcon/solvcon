# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The control column an app docks beside its window."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget

from . import _app_widgets

__all__ = [
    'ControlColumn',
]


class ControlColumn(_app_widgets.Styled, QScrollArea):
    """The control column: the results card over the folded groups, one
    open at a time.

    An app docks it as a panel beside its window.  A subclass builds its
    card and its groups, keeps them as attributes, and hands them here.

    :param card: The :class:`~._results_card.ResultsCard` at the head.
    :param groups: The :class:`~._app_widgets.ColumnGroup` instances under
        it, in order.
    :param opened: The group open at start, or None for all folded.
    """

    #: The fixed width of the column, in device-independent pixels.
    WIDTH = 300
    ROLES = ("buttons", "card", "fold", "form")

    def __init__(self, card, groups, opened=None, parent=None):
        super().__init__(parent)
        self.setObjectName("controlColumn")
        #: Owner-supplied callback offered every key the controls of the column
        #: leave; it returns whether it took the key.
        self.key_pressed = None
        #: The results card.
        self.card = card
        #: The folded groups.
        self.groups = tuple(groups)
        for group in self.groups:
            group.toggled = self._on_group
            group.set_open(group is opened)

        inner = QWidget()
        inner.setObjectName("column")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)
        layout.addWidget(self.card)
        layout.addSpacing(8)
        for group in self.groups:
            layout.addWidget(group)
        layout.addStretch(1)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedWidth(self.WIDTH)
        self._styling = False
        self._apply_style()

    def _on_group(self, group):
        """Open the clicked group and fold the others, or fold it.

        :param group: The group whose head was clicked.
        """
        opening = not group.is_open
        for other in self.groups:
            other.set_open(opening and other is group)

    def restyle(self):
        """Set the fonts of the card and of every group."""
        self.card.restyle()
        for group in self.groups:
            group.restyle()

    def keyPressEvent(self, event):
        if self.key_pressed is None or not self.key_pressed(event):
            super().keyPressEvent(event)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
