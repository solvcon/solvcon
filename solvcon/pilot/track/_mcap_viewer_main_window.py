# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The main window of the open recording: one topic at a time, every scalar
leaf decoded once and paged through a table.
"""

import os
import datetime

from PySide6.QtCore import (Qt, Signal, QAbstractTableModel, QModelIndex,
                            QRect, QSize)
from PySide6.QtGui import QFontMetrics, QIntValidator
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QLineEdit, QTableView,
                               QHeaderView, QStackedWidget, QAbstractItemView)

from ...track import mcap
from .._style import PaletteStyled
from ._style import (Rules, font, header_colors, row_colors, ROW_PAD,
                     ROW_GAP, CAPTION_TEXT_PIXEL_SIZE, TOPIC_NAME_PIXEL_SIZE,
                     TOPIC_TYPE_PIXEL_SIZE)

__all__ = [
    "McapMainWindow",
    "TopicTableModel",
    "format_time",
    "PAGE_SIZE",
    "WINDOW_SIZE",
]

PAGE_SIZE = 50
ROW_HEIGHT = 22
WINDOW_SIZE = (900, 580)

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
_INDEX_ALIGN = int(Qt.AlignRight | Qt.AlignVCenter)
_CELL_ALIGN = int(Qt.AlignLeft | Qt.AlignVCenter)


def format_time(ns):
    """Return ``ns`` since the epoch as UTC ``YYYY-MM-DD HH:MM:SS.ffffff``."""
    seconds, rest = divmod(int(ns), 1_000_000_000)
    when = _EPOCH + datetime.timedelta(seconds=seconds,
                                       microseconds=rest // 1000)
    return when.strftime("%Y-%m-%d %H:%M:%S.%f")


def _format_bool(value):
    return "true" if value else "false"


class TopicTableModel(QAbstractTableModel):
    """
    The extracted leaves of one topic, ``PAGE_SIZE`` messages at a time.

    Column 0 is the message index and column 1 the log time; the rest are
    the leaves in plan order, each typed by the plan and named in the
    header with the type as its tool tip. Pages count from 1, the way the
    footer shows them, and the cells of a page are formatted once when it
    is turned to.
    """

    def __init__(self, extraction, plan, parent=None):
        super().__init__(parent)
        self._time = extraction.time
        self._columns = [
            (extraction.columns[field], _format_bool if t == "bool" else str)
            for field, t in zip(plan.fields, plan.types)]
        self._headers = [("#", ""), ("log_time", "")] + \
            list(zip(plan.fields, plan.types))
        self._page = 1
        self._fill()

    @property
    def message_count(self):
        return len(self._time)

    @property
    def page_count(self):
        return max(1, -(-self.message_count // PAGE_SIZE))

    @property
    def page(self):
        return self._page

    @property
    def start(self):
        """The message index of the first row on the page."""
        return (self._page - 1) * PAGE_SIZE

    def set_page(self, page):
        """Turn to ``page``, clamped to the pages there are; return it."""
        page = min(max(1, int(page)), self.page_count)
        if page != self._page:
            self.beginResetModel()
            self._page = page
            self._fill()
            self.endResetModel()
        return page

    def _fill(self):
        stop = min(self.start + PAGE_SIZE, self.message_count)
        self._cells = [
            [str(i), format_time(self._time[i])] +
            [fmt(column[i]) for column, fmt in self._columns]
            for i in range(self.start, stop)]

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._cells)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return None
        if role == Qt.DisplayRole:
            return self._headers[section][0]
        if role == Qt.ToolTipRole:
            return self._headers[section][1]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._cells[index.row()][index.column()]
        if role == Qt.TextAlignmentRole:
            return _INDEX_ALIGN if index.column() == 0 else _CELL_ALIGN
        return None


class _LeafHeader(QHeaderView):
    """Paint a column header: the dotted path over its type."""

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._path_font = font(TOPIC_NAME_PIXEL_SIZE, mono=True, bold=True)
        self._type_font = font(TOPIC_TYPE_PIXEL_SIZE)
        self._path_metrics = QFontMetrics(self._path_font)
        self._type_metrics = QFontMetrics(self._type_font)
        self.setHighlightSections(False)
        self.setSectionsClickable(False)

    def _height(self):
        return (self._path_metrics.height() + self._type_metrics.height()
                + 2 * ROW_PAD)

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self._height())

    def _texts(self, section):
        """The ``(path, type)`` of ``section``, blank without a model."""
        model = self.model()
        if model is None:
            return "", ""
        return (model.headerData(section, Qt.Horizontal, Qt.DisplayRole),
                model.headerData(section, Qt.Horizontal, Qt.ToolTipRole))

    def sectionSizeFromContents(self, section):
        path, type_ = self._texts(section)
        width = max(self._path_metrics.horizontalAdvance(path),
                    self._type_metrics.horizontalAdvance(type_))
        return QSize(width + 2 * ROW_GAP, self._height())

    def paintSection(self, painter, rect, index):
        surface, line = header_colors(self)
        path_color, type_color = row_colors(self, False)
        path, type_ = self._texts(index)
        inner = rect.adjusted(ROW_GAP, ROW_PAD, -ROW_GAP, -ROW_PAD)

        painter.save()
        painter.fillRect(rect, surface)
        painter.setPen(line)
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.drawLine(rect.topRight(), rect.bottomRight())
        top = QRect(inner.left(), inner.top(), inner.width(),
                    self._path_metrics.height())
        painter.setFont(self._path_font)
        painter.setPen(path_color)
        painter.drawText(top, Qt.AlignLeft | Qt.AlignVCenter, path)
        bottom = QRect(inner.left(), top.bottom() + 1, inner.width(),
                       self._type_metrics.height())
        painter.setFont(self._type_font)
        painter.setPen(type_color)
        painter.drawText(bottom, Qt.AlignLeft | Qt.AlignVCenter, type_)
        painter.restore()


class McapMainWindow(PaletteStyled):
    """One topic of the open recording, decoded into a paged table.

    The sub-window forwards its close here, so ``closed`` is the one
    signal the owner needs to learn the window went away.
    """

    closed = Signal()

    def __init__(self, reader, parent=None):
        super().__init__(parent)
        self._reader = reader
        self._topic = None
        self._model = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._build_toolbar(layout)
        self._pages = QStackedWidget()
        self._notice = QLabel("Select a topic in the MCAP dock")
        self._notice.setAlignment(Qt.AlignCenter)
        self._notice.setWordWrap(True)
        self._notice.setFont(font(CAPTION_TEXT_PIXEL_SIZE))
        self._pages.addWidget(self._notice)
        self._pages.addWidget(self._build_table())
        layout.addWidget(self._pages, 1)
        self._apply_style()

    @property
    def topic(self):
        """The topic shown, or ``None`` before the first ``table()``."""
        return self._topic

    @property
    def model(self):
        """The table model, or ``None`` until ``table()`` decodes a topic."""
        return self._model

    @property
    def title(self):
        return "MCAP viewer - {}".format(os.path.basename(self._reader.path))

    def _build_toolbar(self, layout):
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(12, 6, 12, 6)
        bar.setSpacing(10)
        self._name = QLabel()
        self._name.setFont(font(TOPIC_NAME_PIXEL_SIZE, mono=True, bold=True))
        self._type = QLabel()
        self._type.setFont(font(CAPTION_TEXT_PIXEL_SIZE))
        self._type.hide()
        self._summary = QLabel()
        self._summary.setFont(font(CAPTION_TEXT_PIXEL_SIZE))
        bar.addWidget(self._name)
        bar.addWidget(self._type)
        bar.addStretch(1)
        bar.addWidget(self._summary)
        layout.addWidget(toolbar)

    def _build_table(self):
        self._table_page = QWidget()
        page = QVBoxLayout(self._table_page)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)
        self._table = QTableView()
        self._table.setHorizontalHeader(_LeafHeader(self._table))
        self._table.verticalHeader().hide()
        self._table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self._table.setFont(font(CAPTION_TEXT_PIXEL_SIZE, mono=True))
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        page.addWidget(self._table, 1)

        footer = QWidget()
        footer.setObjectName("footer")
        strip = QHBoxLayout(footer)
        strip.setContentsMargins(12, 4, 12, 4)
        strip.setSpacing(8)
        self._range = QLabel()
        self._prev = QPushButton("<")
        self._prev.clicked.connect(self._on_prev)
        self._page_word = QLabel("Page")
        self._page_input = QLineEdit()
        self._page_input.setValidator(QIntValidator(1, 10 ** 9))
        self._page_input.setFixedWidth(52)
        self._page_input.setAlignment(Qt.AlignRight)
        self._page_input.setToolTip("Page number, Enter to jump")
        self._page_input.setFont(font(CAPTION_TEXT_PIXEL_SIZE, mono=True))
        self._page_input.returnPressed.connect(self._on_jump)
        self._page_total = QLabel()
        self._next = QPushButton(">")
        self._next.clicked.connect(self._on_next)
        for widget in (self._range, self._prev, self._page_word,
                       self._page_total, self._next):
            widget.setFont(font(CAPTION_TEXT_PIXEL_SIZE))
        strip.addWidget(self._range)
        strip.addStretch(1)
        for widget in (self._prev, self._page_word, self._page_input,
                       self._page_total, self._next):
            strip.addWidget(widget)
        page.addWidget(footer)
        return self._table_page

    def table(self, topic):
        """Show every scalar leaf of ``topic`` from its first page.

        Return the table model, or ``None`` when the decoder cannot read
        the topic; the body then says why.
        """
        self._topic = topic
        # The view does not own its model; dropping the reference after
        # detaching it is what frees the previous topic's columns.
        self._table.setModel(None)
        self._model = None
        self._name.setText(topic)
        self._type.hide()
        self._summary.setText("")
        try:
            schema = self._reader.schema(topic)
            if schema is not None:
                self._type.setText(schema.name)
                self._type.show()
            # Without a schema there is no plan, and the reader says so.
            plan = None if schema is None else mcap.DecodePlan(schema)
            # TODO: the extraction runs on the GUI thread and is redone on
            # every selection; move it to a worker and cache it per topic.
            extraction = self._reader.extract(topic, plan)
        except mcap.McapError as error:
            self._notice.setText("Cannot decode {}: {}".format(topic, error))
            self._pages.setCurrentWidget(self._notice)
            return None
        self._model = TopicTableModel(extraction, plan)
        self._table.setModel(self._model)
        self._table.resizeColumnsToContents()
        # The first page sizes the columns, so make room for the last index.
        widest = self._table.fontMetrics().horizontalAdvance(
            str(self._model.message_count)) + 2 * ROW_GAP
        self._table.setColumnWidth(0, max(widest, self._table.columnWidth(0)))
        self._summary.setText("{:,} messages \u00b7 {} columns".format(
            self._model.message_count, len(plan.fields)))
        self._pages.setCurrentWidget(self._table_page)
        self.page(1)
        return self._model

    def page(self, number):
        """Turn to page ``number``; out of range clamps to the ends.

        Return the page shown, or ``None`` until ``table()`` decodes a topic.
        """
        if self._model is None:
            return None
        number = self._model.set_page(number)
        self._table.scrollToTop()
        rows = self._model.rowCount()
        first = self._model.start + 1 if rows else 0
        self._range.setText("Rows {:,}\u2013{:,} of {:,}".format(
            first, self._model.start + rows, self._model.message_count))
        self._page_input.setText(str(number))
        self._page_total.setText("of {:,}".format(self._model.page_count))
        self._prev.setEnabled(number > 1)
        self._next.setEnabled(number < self._model.page_count)
        return number

    def _on_prev(self):
        self.page(self._model.page - 1)

    def _on_next(self):
        self.page(self._model.page + 1)

    def _on_jump(self):
        self.page(int(self._page_input.text() or 1))

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def _apply_style(self):
        self.setStyleSheet(Rules.sheet(self, "bar", "table", "button",
                                       "field"))
        self._type.setStyleSheet(Rules.sheet(self, "badge"))
        self._notice.setStyleSheet(Rules.sheet(self, "faint"))
        sheet = Rules.sheet(self, "label")
        for label in (self._summary, self._range, self._page_word,
                      self._page_total):
            label.setStyleSheet(sheet)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
