# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The results card at the head of the control column of an app."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout

from .. import _style
from . import _app_widgets
from . import _readout

__all__ = [
    'ResultsCard',
]


class ResultsCard(QFrame):
    """The verdict of a run: tables of values computed against the exact
    ones with the error between them, over the figures of the run, such as
    a drift or a range.

    An app subclass names its tables in :attr:`SECTIONS`, and words their
    heads in :meth:`head_text` and an empty table in :meth:`empty_text`.
    The record :meth:`show_results` takes holds a tuple of
    :class:`~._readout.ResultRow` under the name of each table, ``rows`` of
    ``(label, text)`` for the figures, ``caption`` for the tip of the first
    head, and ``note``, ``(text, tone)`` or None, for the line under the
    tables.
    """

    #: The heads of the value columns.
    COLUMNS = ("computed", "exact", "error")
    #: The tables of value rows, in the order the card reads them: the name
    #: of the tuple of rows in the record and the head of the table.
    SECTIONS = ()

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
        self._section_heads = {}
        self._values = {}
        self._rows = {}

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
        small.setPixelSize(_style.SMALL_PX)
        self._note.setFont(small)
        number = _style.number_font(self.font(), 12)
        plain = QFont(self.font())
        plain.setPixelSize(12)
        for label in self._labels:
            label.setFont(number if label.property("numeric") else
                          small if label.property("small") else plain)
        for head in self._heads:
            head.setFont(_app_widgets.head_font(self.font()))

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
        self._section_heads, self._values, self._rows = {}, {}, {}
        self._shape = None

    def _rebuild(self, results):
        self._clear()
        right = Qt.AlignRight
        row = 0
        for section, title in self.SECTIONS:
            records = getattr(results, section)
            head = self._head(title)
            self._section_heads[section] = head
            self._grid.addWidget(head, row, 0, 1, 4)
            row += 1
            # The heads of the value columns go under the first table only.
            if section == self.SECTIONS[0][0]:
                for col, text in enumerate(self.COLUMNS, 1):
                    caption = self._label(text, "caption", align=right)
                    caption.setProperty("small", True)
                    self._grid.addWidget(caption, row, col)
                row += 1
            # An empty table keeps a line to say why it is empty.
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
        rule = QFrame()
        rule.setObjectName("cardRule")
        self._grid.addWidget(rule, row, 0, 1, 4)
        row += 1
        for label, _ in results.rows:
            self._grid.addWidget(self._label(label, "metric"), row, 0)
            value = self._label(numeric=True, align=right)
            self._grid.addWidget(value, row, 1, 1, 3)
            self._rows[label] = value
            row += 1
        self._grid.setColumnStretch(0, 1)
        self._shape = self._shape_of(results)
        self.restyle()

    def _shape_of(self, results):
        return (tuple(len(getattr(results, section))
                      for section, _ in self.SECTIONS),
                tuple(label for label, _ in results.rows))

    def head_text(self, section, title, results):
        """Return what the head of a table reads for a record.

        :param section: The name of the table, a key of :attr:`SECTIONS`.
        :param title: The head :attr:`SECTIONS` gives it.
        :param results: The record shown.
        """
        return title

    def empty_text(self, section, results):
        """Return what a table without rows reads.

        :param section: The name of the table, a key of :attr:`SECTIONS`.
        :param results: The record shown.
        """
        return "None"

    def show_results(self, results):
        """Fill the card, or blank it.

        :param results: The record to show, or None to blank the card.
        """
        if results is None:
            self._clear()
            _app_widgets.say(self._note, "No run")
            _style.set_style_property(self._note, "tone", "muted")
            return
        # A march only changes the texts, so the grid is rebuilt only when
        # its rows change.
        if self._shape != self._shape_of(results):
            self._rebuild(results)
        for section, title in self.SECTIONS:
            self._section_heads[section].setText(
                self.head_text(section, title, results))
            records = getattr(results, section)
            if not records:
                self._values[(section, 'empty')].setText(
                    self.empty_text(section, results))
        self._section_heads[self.SECTIONS[0][0]].setToolTip(results.caption)
        for section, _ in self.SECTIONS:
            records = getattr(results, section)
            for it, record in enumerate(records):
                name, cells = self._values[(section, it)]
                name.setText(record.label)
                for label, cell in zip(cells, record[1:]):
                    label.setText(cell.text)
                    label.setToolTip(cell.tip)
                    _style.set_style_property(label, "tone", cell.tone)
        for label, text in results.rows:
            self._rows[label].setText(text)
        text, tone = results.note if results.note else ("", "")
        _app_widgets.say(self._note, text)
        _style.set_style_property(self._note, "tone", tone or "muted")

    def cell(self, section, row, col):
        """Return the :class:`~._readout.Cell` shown in a table of the
        card, or None where there is none.

        :param section: The table, a key of :attr:`SECTIONS`.
        :param row: The row in the table.
        :param col: The column, 0 for computed to 2 for the error.
        """
        found = self._values.get((section, row))
        if found is None:
            return None
        label = found[1][col]
        return _readout.Cell(label.text(), label.property("tone") or '',
                             label.toolTip())

    def row_text(self, label):
        """Return the text of a run figure, or None.

        :param label: The label of the figure, such as 'mass drift'.
        """
        found = self._rows.get(label)
        return None if found is None else found.text()

    def note(self):
        """Return what the note under the tables says."""
        return "" if self._note.isHidden() else self._note.text()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
