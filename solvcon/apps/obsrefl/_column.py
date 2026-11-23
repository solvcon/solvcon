# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The control column of the reflection app."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget

from ...pilot import panel

from . import _forms

__all__ = [
    'ControlColumn',
    'ResultsCard',
]


class ResultsCard(panel.ResultsCard):
    """The verdict of the run: the field computed in each zone against the
    analytic value and the error between them, the fitted reflected shock
    against the exact one, and the figures of the run, such as the mass
    drift and the CFL range.
    """

    SECTIONS = (('zones', "ZONES"), ('fits', "SHOCK FIT"))

    def head_text(self, section, title, results):
        if 'zones' == section:
            return f"{title}  {results.field.upper()}"
        return title

    def empty_text(self, section, results):
        if 'zones' == section:
            return f"No exact {results.field} to compare with"
        return "Not formed yet"


class ControlColumn(panel.Styled, QScrollArea):
    """The control column: the results card over the folded groups.

    The column is a plain widget, which the app docks as a panel beside
    the window.  Edits to the free stream and the numerics wait for Reset,
    which the run strip marks while they wait.

    The column and its children are passive: they show what they are handed
    and call back into their owner.
    """

    WIDTH = 300
    ROLES = ("buttons", "card", "fold", "form")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlColumn")
        #: Owner-supplied callback offered every key the controls of the column
        #: leave; it returns whether it took the key.
        self.key_pressed = None
        self.card = ResultsCard()
        self.freestream = _forms.FreeStreamGroup()
        self.numerics = _forms.NumericsGroup()
        self.display = _forms.DisplayGroup()
        self.groups = (self.freestream, self.numerics, self.display)
        for group in self.groups:
            group.toggled = self._on_group
            group.set_open(group is self.display)

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
