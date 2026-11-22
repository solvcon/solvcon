# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The control column of the reflection app."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout

from ... import pilot

from . import _viewmodel
from . import _widgets

__all__ = [
    'ResultsCard',
]


class ResultsCard(QFrame):
    """The verdict of the run: the field computed in each zone against the
    analytic value and the error between them, and the fitted reflected
    shock against the exact one.

    An error is toned by how far it is off, so a glance tells a good match
    from a poor one; the tip of a cell says how it was taken.
    """

    #: The heads of the value columns.
    COLUMNS = ("computed", "exact", "error")
    #: The sections of value rows, in the order the card reads them, and
    #: their heads.
    SECTIONS = (('zones', "ZONES"), ('fits', "SHOCK FIT"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(2)
        self._note = QLabel("")
        self._note.setObjectName("caption")
        self._note.setWordWrap(True)
        self._shape = None
        self._labels = []
        self._heads = []
        self._values = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(3)
        layout.addLayout(self._grid)
        layout.addWidget(self._note)
        self.restyle()
        self.show_results(None)

    def restyle(self):
        """Set the fonts of the heads, the note, and the numbers."""
        small = QFont(self.font())
        small.setPixelSize(pilot.SMALL_PX)
        self._note.setFont(small)
        number = pilot.number_font(self.font(), 12)
        plain = QFont(self.font())
        plain.setPixelSize(12)
        for label in self._labels:
            label.setFont(number if label.property("numeric") else
                          small if label.property("small") else plain)
        for head in self._heads:
            head.setFont(_widgets.head_font(self.font()))

    def _label(self, text="", name=None, numeric=False, align=Qt.AlignLeft):
        label = QLabel(text)
        if name:
            label.setObjectName(name)
        label.setAlignment(align | Qt.AlignVCenter)
        label.setProperty("numeric", numeric)
        self._labels.append(label)
        return label

    def _head(self, text):
        head = QLabel(text)
        head.setObjectName("cardHead")
        self._heads.append(head)
        return head

    def _clear(self):
        while self._grid.count():
            widget = self._grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self._labels, self._heads = [], []
        self._values = {}
        self._shape = None

    def _rebuild(self, results):
        self._clear()
        right = Qt.AlignRight
        row = 0
        for section, title in self.SECTIONS:
            records = getattr(results, section)
            head = self._head(title)
            self._grid.addWidget(head, row, 0, 1, 4)
            row += 1
            if 'zones' == section:
                self._field = head
                for col, text in enumerate(self.COLUMNS, 1):
                    caption = self._label(text, "caption", align=right)
                    caption.setProperty("small", True)
                    self._grid.addWidget(caption, row, col)
                row += 1
            if not records:
                empty = self._label("", "caption")
                self._grid.addWidget(empty, row, 0, 1, 4)
                self._values[(section, 'empty')] = empty
                row += 1
            for it in range(len(records)):
                name = self._label(name="metric")
                self._grid.addWidget(name, row, 0)
                cells = []
                for col in range(1, 4):
                    cell = self._label(numeric=True, align=right)
                    self._grid.addWidget(cell, row, col)
                    cells.append(cell)
                self._values[(section, it)] = (name, cells)
                row += 1
        self._grid.setColumnStretch(0, 1)
        self._shape = self._shape_of(results)
        self.restyle()

    @staticmethod
    def _shape_of(results):
        return (len(results.zones), len(results.fits))

    def show_results(self, results):
        """Fill the card, or blank it.

        :param results: The :class:`~._viewmodel.Results` to show, or None
            to blank the card.
        """
        if results is None:
            self._clear()
            _widgets.say(self._note, "No run")
            pilot.set_style_property(self._note, "tone", "muted")
            return
        if self._shape != self._shape_of(results):
            self._rebuild(results)
        self._field.setText(f"ZONES  {results.field.upper()}")
        self._field.setToolTip(results.caption)
        for section, _ in self.SECTIONS:
            records = getattr(results, section)
            if not records:
                self._values[(section, 'empty')].setText(
                    f"No exact {results.field} to compare with"
                    if 'zones' == section else "Not formed yet")
            for it, record in enumerate(records):
                name, cells = self._values[(section, it)]
                name.setText(record.label)
                for label, cell in zip(cells, record[1:]):
                    label.setText(cell.text)
                    label.setToolTip(cell.tip)
                    pilot.set_style_property(label, "tone", cell.tone)
        text, tone = results.note if results.note else ("", "")
        _widgets.say(self._note, text)
        pilot.set_style_property(self._note, "tone", tone or "muted")

    def cell(self, section, row, col):
        """Return the :class:`~._viewmodel.Cell` shown in a table of the
        card, or None where there is none.

        :param section: The table, 'zones' or 'fits'.
        :param row: The row in the table.
        :param col: The column, 0 for computed to 2 for the error.
        """
        found = self._values.get((section, row))
        if found is None:
            return None
        label = found[1][col]
        return _viewmodel.Cell(label.text(), label.property("tone") or '',
                               label.toolTip())

    def note(self):
        """Return what the note under the tables says."""
        return "" if self._note.isHidden() else self._note.text()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
