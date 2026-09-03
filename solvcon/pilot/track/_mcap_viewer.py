# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Open an MCAP recording in the pilot: the Track menu opens a file, and the
dock on the right shows what the reader gets without decoding a message.
"""

import os

from PySide6.QtCore import Qt, Signal, QSize, QRect
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QPushButton, QListWidget,
                               QListWidgetItem, QStyledItemDelegate, QStyle,
                               QDockWidget, QFileDialog, QFrame)

from ...track import mcap
from .._style import PaletteStyled
from ..base import _gui_common
from ._style import (Rules, font, row_colors, BODY_TEXT_PIXEL_SIZE,
                     CAPTION_TEXT_PIXEL_SIZE, TOPIC_NAME_PIXEL_SIZE,
                     TOPIC_TYPE_PIXEL_SIZE)

__all__ = [
    "McapDock",
    "McapPanel",
]

_TOPIC_ROLE = Qt.UserRole
_COUNT_ROLE = Qt.UserRole + 1
_TYPE_ROLE = Qt.UserRole + 2
_ROW_PAD = 4
_ROW_GAP = 8


def format_size(nbytes):
    """Return ``nbytes`` in the largest unit that keeps it under 1000."""
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            break
        size /= 1000
    return "{:.0f} {}".format(size, unit) if unit == "B" else \
        "{:.2f} {}".format(size, unit)


def format_duration(ns):
    """Return ``ns`` as ``1h 02m 03s`` without the leading zero units."""
    total = round(ns / 1e9)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return "{}h {:02d}m {:02d}s".format(hours, minutes, seconds)
    if minutes:
        return "{}m {:02d}s".format(minutes, seconds)
    return "{}s".format(seconds)


def summarize(reader):
    """Return the ``(size, duration, messages)`` rows of ``reader``."""
    size = format_size(reader.size)
    time_range = reader.time_range()
    duration = "unknown" if time_range is None else \
        format_duration(time_range[1] - time_range[0])
    count = reader.message_count()
    messages = "unknown" if count is None else "{:,}".format(count)
    return size, duration, messages


def topic_rows(reader):
    """Return the ``(topic, count, type)`` rows in file order.

    Channels that share a topic make one row, because the count already
    covers the topic as a whole. The row takes the type of the first such
    channel.
    """
    counts = reader.message_counts()
    rows = {}
    for channel in reader.channels():
        if channel.topic not in rows:
            schema = reader.schema_of(channel)
            rows[channel.topic] = "" if schema is None else schema.name
    return [(topic, None if counts is None else counts[topic], type_)
            for topic, type_ in rows.items()]


class _TopicDelegate(QStyledItemDelegate):
    """Paint a topic row: the name and the count, then the type below."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name_height = QFontMetrics(
            font(TOPIC_NAME_PIXEL_SIZE, mono=True)).height()
        self._type_height = QFontMetrics(font(TOPIC_TYPE_PIXEL_SIZE)).height()

    def sizeHint(self, option, index):
        height = self._name_height + self._type_height + 2 * _ROW_PAD
        return QSize(option.rect.width(), height)

    def paint(self, painter, option, index):
        self.initStyleOption(option, index)
        option.text = ""
        option.widget.style().drawControl(QStyle.CE_ItemViewItem, option,
                                          painter, option.widget)
        text, sub = row_colors(self.parent(),
                               option.state & QStyle.State_Selected)
        rect = option.rect.adjusted(_ROW_GAP, _ROW_PAD, -_ROW_GAP, -_ROW_PAD)

        painter.save()
        painter.setPen(text)
        top = QRect(rect.left(), rect.top(), rect.width(), self._name_height)
        painter.setFont(font(CAPTION_TEXT_PIXEL_SIZE, mono=True))
        count = index.data(_COUNT_ROLE)
        width = painter.fontMetrics().horizontalAdvance(count) + _ROW_GAP
        painter.drawText(top, Qt.AlignRight | Qt.AlignVCenter, count)
        painter.setFont(font(TOPIC_NAME_PIXEL_SIZE, mono=True))
        name = painter.fontMetrics().elidedText(
            index.data(_TOPIC_ROLE), Qt.ElideRight, top.width() - width)
        painter.drawText(top, Qt.AlignLeft | Qt.AlignVCenter, name)
        painter.setFont(font(TOPIC_TYPE_PIXEL_SIZE))
        painter.setPen(sub)
        bottom = QRect(rect.left(), top.bottom() + 1, rect.width(),
                       self._type_height)
        type_ = painter.fontMetrics().elidedText(
            index.data(_TYPE_ROLE), Qt.ElideRight, bottom.width())
        painter.drawText(bottom, Qt.AlignLeft | Qt.AlignVCenter, type_)
        painter.restore()


class McapDock(PaletteStyled):
    """The file summary and the topic list of the open recording."""

    open_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)
        self._build_summary(layout)
        self._caption = QLabel("TOPICS")
        self._caption.setFont(font(CAPTION_TEXT_PIXEL_SIZE))
        layout.addWidget(self._caption)
        self._build_topics(layout)
        self._apply_style()

    def _build_summary(self, layout):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        grid.setColumnMinimumWidth(0, 64)
        grid.setColumnStretch(1, 1)
        self._keys = [QLabel(key) for key in
                      ("File", "Size", "Duration", "Messages")]
        self._values = [QLabel("-") for _ in self._keys]
        self._file, self._size, self._duration, self._messages = self._values
        for irow, (key, value) in enumerate(zip(self._keys, self._values)):
            key.setFont(font(BODY_TEXT_PIXEL_SIZE))
            value.setFont(font(BODY_TEXT_PIXEL_SIZE))
            grid.addWidget(key, irow, 0)
            grid.addWidget(value, irow, 1)
        self._file.setFont(font(CAPTION_TEXT_PIXEL_SIZE, mono=True))
        layout.addLayout(grid)
        row = QHBoxLayout()
        self._open = QPushButton("Open MCAP...")
        self._open.setFont(font(BODY_TEXT_PIXEL_SIZE))
        self._open.clicked.connect(self.open_requested)
        row.addWidget(self._open)
        row.addStretch(1)
        layout.addLayout(row)
        self._error = QLabel()
        self._error.setFont(font(CAPTION_TEXT_PIXEL_SIZE))
        self._error.setWordWrap(True)
        self._error.hide()
        layout.addWidget(self._error)
        self._rule = QFrame()
        self._rule.setFixedHeight(1)
        layout.addWidget(self._rule)
        layout.addSpacing(4)

    def _build_topics(self, layout):
        self._box = QFrame()
        self._box.setObjectName("topics")
        box = QVBoxLayout(self._box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        head = QWidget()
        head.setObjectName("header")
        header = QHBoxLayout(head)
        header.setContentsMargins(_ROW_GAP, 4, _ROW_GAP, 4)
        self._columns = [QLabel("Topic"), QLabel("Count")]
        for label in self._columns:
            label.setFont(font(CAPTION_TEXT_PIXEL_SIZE, bold=True))
        header.addWidget(self._columns[0], 1)
        header.addWidget(self._columns[1])
        box.addWidget(head)
        self._topics = QListWidget()
        self._topics.setUniformItemSizes(True)
        self._topics.setItemDelegate(_TopicDelegate(self))
        self._topics.setFrameShape(QFrame.NoFrame)
        box.addWidget(self._topics, 1)
        layout.addWidget(self._box, 1)

    def set_reader(self, reader):
        """Fill the dock from ``reader``; ``None`` clears it."""
        self._topics.clear()
        self._error.hide()
        if reader is None:
            for label in self._values:
                label.setText("-")
            self._file.setToolTip("")
            self._caption.setText("TOPICS")
            return
        self._file.setText(os.path.basename(reader.path))
        self._file.setToolTip(reader.path)
        size, duration, messages = summarize(reader)
        self._size.setText(size)
        self._duration.setText(duration)
        self._messages.setText(messages)
        rows = topic_rows(reader)
        self._caption.setText("TOPICS \u00b7 {}".format(len(rows)))
        for topic, count, type_ in rows:
            item = QListWidgetItem()
            item.setData(_TOPIC_ROLE, topic)
            item.setData(_COUNT_ROLE, "?" if count is None else
                         "{:,}".format(count))
            item.setData(_TYPE_ROLE, type_)
            item.setToolTip(topic)
            self._topics.addItem(item)

    def set_error(self, path, message):
        """Report that ``path`` would not open, and clear the summary."""
        self.set_reader(None)
        self._file.setText(os.path.basename(path))
        self._file.setToolTip(path)
        self._error.setText(message)
        self._error.show()

    def _apply_style(self):
        sheet = Rules.sheet(self, "label")
        for label in self._keys + self._columns:
            label.setStyleSheet(sheet)
        self._caption.setStyleSheet(Rules.sheet(self, "caption"))
        self._rule.setStyleSheet(Rules.sheet(self, "rule"))
        self._open.setStyleSheet(Rules.sheet(self, "button"))
        self._box.setStyleSheet(Rules.sheet(self, "topics"))


class McapPanel(_gui_common.PilotFeature):
    """Open MCAP files from the Track menu into the dock."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._reader = None
        self._diag = QFileDialog()
        self._diag.setFileMode(QFileDialog.ExistingFile)
        self._diag.setWindowTitle("Open MCAP file")
        self._diag.setNameFilter("MCAP recording (*.mcap);;All files (*)")
        self._diag.fileSelected.connect(self._on_selected)

    @property
    def reader(self):
        return self._reader

    @property
    def panel(self):
        return None if self._dock is None else self._dock.widget()

    def populate_menu(self):
        self.add_action(
            "Track", "Open MCAP", "Open an MCAP recording",
            self._diag.open, id="track.open_mcap", weight=10)
        self._action = self.add_action(
            "View/Panels", "MCAP", "Toggle the MCAP panel",
            None, id="panel.mcap", weight=25, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    def open_file(self, path):
        """Read the summary of ``path`` into the dock."""
        reader = mcap.Reader(path)
        if self._reader is not None:
            self._reader.close()
        self._reader = reader
        self._shown_panel().set_reader(reader)
        return reader

    def _on_selected(self, path):
        """Open what the dialog chose, and report a file that will not open.

        An exception raised in a slot leaves Qt with nothing to say to the
        user, so the dock carries the failure instead.
        """
        try:
            self.open_file(path)
        except (OSError, mcap.McapError) as error:
            self._shown_panel().set_error(path, str(error))

    def _shown_panel(self):
        """Return the dock, brought up if the user had it away."""
        self._ensure_panel()
        self._action.setChecked(True)
        return self.panel

    def _on_toggled(self, checked):
        if checked:
            self._ensure_panel()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        if self._dock is not None:
            return
        panel = McapDock()
        panel.open_requested.connect(self._diag.open)
        self._dock = QDockWidget("MCAP")
        self._dock.setWidget(panel)
        self._mainWindow.addDockWidget(Qt.RightDockWidgetArea, self._dock)
        self._dock.visibilityChanged.connect(self._action.setChecked)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
