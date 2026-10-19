# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Plot tab of the MCAP main window: one scalar field of the shown topic
drawn as a line against the log time since the file started.

The page owns the header row (topic name and the field dropdown).  The
widget below it paints the axes and the series with a ``QPainter``.
"""

import math

import numpy as np

from PySide6.QtCore import Qt, Signal, QRect, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QComboBox, QSizePolicy)

from ...track.mcap._decodeplan import COLUMN_TYPES
from .._style import PaletteStyled, Shades
from ..visual._plot import _nice_ticks, _tick_label
from ._style import (Rules, font, control_colors, CAPTION_TEXT_PIXEL_SIZE,
                     TOPIC_TYPE_PIXEL_SIZE)

__all__ = [
    "McapPlotPage",
]

DEFAULT_SPAN = 30.0
FIELD_WIDTH = 190
FIELD_HEIGHT = 22
CHEVRON = 4
SERIES_COLOR = QColor("#0f766e")
SERIES_WIDTH = 1.5
MARGINS = (64, 28, 64, 40)
TICK_GAP = 6
MAX_TIME_TICKS = 8
Y_TICKS = 8
Y_PAD = 0.05
SECOND_NS = 1_000_000_000
NO_FIELD = "This topic has no plottable field"

_TIME_STEPS = (1, 2, 5, 10, 15, 30, 60, 120)


def plottable_fields(plan):
    """Return the ``(path, dtype)`` of every scalar leaf a line can draw.

    The leaves come in plan order; an enum reads as its ordinal and stays
    out.
    """
    return [(path, dtype) for path, dtype in zip(plan.fields, plan.types)
            if dtype in COLUMN_TYPES and path not in plan.enums]


def format_mss(seconds):
    """Return ``seconds`` as ``m:ss``, the way the time axis is labelled."""
    whole = int(round(seconds))
    return "{}:{:02d}".format(whole // 60, whole % 60)


def time_ticks(t0, t1):
    """Return the tick times between ``t0`` and ``t1`` in seconds.

    The step is the smallest of 1/2/5/10/15/30/60/120 s that fits at most
    ``MAX_TIME_TICKS`` ticks in the span; a span too long for 120 s keeps
    that step and shows more.
    """
    span = t1 - t0
    if not span > 0 or not math.isfinite(span):
        return []
    for step in _TIME_STEPS:
        first, last = math.ceil(t0 / step), math.floor(t1 / step)
        if last - first < MAX_TIME_TICKS:
            break
    return [it * step for it in range(first, last + 1)]


def value_ticks(lo, hi):
    """Return the ``(ymin, ymax, ticks)`` the ordinate is drawn with.

    The limits pad the data by ``Y_PAD`` of its span.  A flat series is
    padded by half its magnitude on each side, or by 1 at zero, so a
    constant still draws mid-frame.
    """
    if hi == lo:
        pad = abs(lo) * 0.5 or 1.0
    else:
        pad = (hi - lo) * Y_PAD
    ymin, ymax = lo - pad, hi + pad
    return ymin, ymax, _nice_ticks(ymin, ymax, want=Y_TICKS)


class _FieldCombo(QComboBox):
    """Paint the dropdown as a bordered box, the entry in bold and a
    chevron at the right.

    The popup list stays the platform's.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(FIELD_WIDTH, FIELD_HEIGHT)
        self.setFont(font(CAPTION_TEXT_PIXEL_SIZE, mono=True, bold=True))

    def paintEvent(self, event):
        surface, border, text = control_colors(self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(border)
        painter.setBrush(surface)
        box = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(text if self.isEnabled() else border)
        inner = self.rect().adjusted(8, 0, -(8 + 3 * CHEVRON), 0)
        label = painter.fontMetrics().elidedText(
            self.currentText(), Qt.ElideRight, inner.width())
        painter.drawText(inner, Qt.AlignLeft | Qt.AlignVCenter, label)
        x = self.width() - 8 - CHEVRON
        y = self.height() / 2 - CHEVRON / 2
        pen = QPen(painter.pen().color())
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF([
            QPointF(x - CHEVRON, y), QPointF(x, y + CHEVRON),
            QPointF(x + CHEVRON, y)]))


class McapPlotWidget(QWidget):
    """Paint one field against time in seconds.

    :meth:`set_series` hands the widget the times and the values it draws;
    :meth:`set_range` the seconds it draws between.  The widget draws a
    ``bool`` field as steps between ``false`` and ``true``, not as a line.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._times = np.empty(0, dtype="float64")
        self._values = np.empty(0, dtype="float64")
        self._title = ""
        self._boolean = False
        self._range = (0.0, DEFAULT_SPAN)
        self._tick_font = font(TOPIC_TYPE_PIXEL_SIZE, mono=True)
        self._title_font = font(CAPTION_TEXT_PIXEL_SIZE, mono=True)
        self._empty_font = font(CAPTION_TEXT_PIXEL_SIZE)
        self._select()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 160)

    @property
    def title(self):
        return self._title

    @property
    def range(self):
        """Return the ``(t0, t1)`` range drawn, in seconds."""
        return self._range

    def set_series(self, title, boolean, times, values):
        """Draw ``values`` at ``times``, in seconds and sorted ascending.

        ``title`` heads the ordinate; ``boolean`` picks the step drawing.
        A sample whose value is not finite has nowhere to go and is left
        out.
        """
        times = np.asarray(times, dtype="float64")
        values = np.asarray(values, dtype="float64")
        keep = np.isfinite(values)
        self._times, self._values = times[keep], values[keep]
        self._title = title
        self._boolean = boolean
        self._select()

    def clear(self):
        """Show the empty state instead of a series."""
        self.set_series("", False, (), ())

    def set_range(self, t0, t1):
        self._range = (float(t0), float(t1))
        self._select()

    def _select(self):
        """Cut the samples in range and scale the ordinate once per change,
        not per paint."""
        lo = np.searchsorted(self._times, self._range[0], side="left")
        hi = np.searchsorted(self._times, self._range[1], side="right")
        self._visible = self._times[lo:hi], self._values[lo:hi]
        values = self._visible[1]
        if self._boolean:
            self._limits = (-0.25, 1.25, [0.0, 1.0])
        elif len(values) == 0:
            self._limits = value_ticks(0.0, 1.0)
        else:
            self._limits = value_ticks(float(values.min()),
                                       float(values.max()))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        shades = Shades(self)
        left, top, right, bottom = MARGINS
        frame = QRect(left, top, max(1, self.width() - left - right),
                      max(1, self.height() - top - bottom))
        painter.fillRect(self.rect(), shades.base)
        if not self._title:
            self._draw_empty(painter, shades, frame)
            return
        times, values = self._visible
        ymin, ymax, ticks = self._limits
        t0, t1 = self._range

        def to_x(t):
            return frame.left() + (t - t0) / (t1 - t0) * frame.width()

        def to_y(y):
            return frame.bottom() - (y - ymin) / (ymax - ymin) * frame.height()

        self._draw_grid(painter, shades, frame, ticks, to_x, to_y)
        self._draw_axes(painter, shades, frame)
        self._draw_series(painter, frame, to_x(times), to_y(values))

    def _draw_empty(self, painter, shades, frame):
        pen = QPen(shades.border)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(frame)
        painter.setPen(shades.greyed)
        painter.setFont(self._empty_font)
        painter.drawText(frame, Qt.AlignCenter, NO_FIELD)

    def _draw_grid(self, painter, shades, frame, ticks, to_x, to_y):
        grid = shades.raised(0.08)
        painter.setFont(self._tick_font)
        height = painter.fontMetrics().height()
        t0, t1 = self._range
        for value in ticks:
            at = round(to_y(value))
            painter.setPen(grid)
            painter.drawLine(frame.left(), at, frame.right(), at)
            painter.setPen(shades.muted)
            if self._boolean:
                text = "true" if value else "false"
            else:
                text = _tick_label(value)
            painter.drawText(
                QRect(0, at - height // 2, frame.left() - TICK_GAP, height),
                Qt.AlignRight | Qt.AlignVCenter, text)
        for second in time_ticks(t0, t1):
            at = round(to_x(second))
            painter.setPen(grid)
            painter.drawLine(at, frame.top(), at, frame.bottom())
            painter.setPen(shades.muted)
            painter.drawText(QRect(at - 40, frame.bottom() + 4, 80, height),
                             Qt.AlignCenter, format_mss(second))

    def _draw_axes(self, painter, shades, frame):
        painter.setPen(shades.border)
        painter.drawLine(frame.bottomLeft(), frame.bottomRight())
        painter.drawLine(frame.topLeft(), frame.bottomLeft())
        painter.setPen(shades.muted)
        painter.setFont(self._title_font)
        height = painter.fontMetrics().height()
        painter.drawText(QRect(frame.left(), 0, frame.width(), MARGINS[1]),
                         Qt.AlignLeft | Qt.AlignVCenter, self._title)
        painter.drawText(
            QRect(frame.left(), self.height() - height - 2, frame.width(),
                  height),
            Qt.AlignCenter, "log time  (s since file start)")

    def _draw_series(self, painter, frame, xs, ys):
        if len(xs) == 0:
            return
        if self._boolean:
            xs, ys = np.repeat(xs, 2)[1:], np.repeat(ys, 2)[:-1]
        points = [QPointF(x, y) for x, y in zip(xs.tolist(), ys.tolist())]
        pen = QPen(SERIES_COLOR)
        pen.setWidthF(SERIES_WIDTH)
        painter.save()
        painter.setClipRect(frame.adjusted(0, -2, 0, 2))
        painter.setPen(pen)
        if len(points) == 1:
            painter.drawPoint(points[0])
        else:
            painter.drawPolyline(QPolygonF(points))
        painter.restore()


class McapPlotPage(PaletteStyled):
    """The Plot tab: the header row over the plot widget.

    ``field_changed`` carries the path the user picked in the dropdown;
    a field set by :meth:`set_field` or chosen for a new topic is silent,
    so the owner echoes only what was clicked.
    """

    field_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._topic = None
        self._extraction = None
        self._seconds = None
        self._fields = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._build_header(layout)
        self._plot = McapPlotWidget()
        layout.addWidget(self._plot, 1)
        self._apply_style()

    def _build_header(self, layout):
        header = QWidget()
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(10)
        self._name = QLabel()
        self._name.setFont(font(CAPTION_TEXT_PIXEL_SIZE, mono=True))
        self._field_word = QLabel("Field")
        self._field_word.setFont(font(CAPTION_TEXT_PIXEL_SIZE))
        self._field = _FieldCombo()
        self._field.currentIndexChanged.connect(self._redraw)
        self._field.activated.connect(self._on_field_chosen)
        row.addWidget(self._name)
        row.addWidget(self._field_word)
        row.addWidget(self._field)
        row.addStretch(1)
        layout.addWidget(header)

    @property
    def topic(self):
        return self._topic

    @property
    def field(self):
        """The path drawn, or ``None`` without a plottable field."""
        index = self._field.currentIndex()
        return None if index < 0 else list(self._fields)[index]

    @property
    def plot(self):
        return self._plot

    def show_topic(self, topic, extraction, plan, span):
        """Offer the plottable fields of ``topic`` and draw the first.

        ``span`` is the ``(start_ns, end_ns)`` of the file: the abscissa
        counts from its start, and the range opens on its first
        ``DEFAULT_SPAN`` seconds, the whole file when shorter, or 1 s when
        it has no duration.  A topic the decoder cannot read (``None`` for
        ``plan``) or one with no plottable field clears the page.
        """
        self._topic = topic
        self._extraction = extraction
        start, end = span
        # Subtract in uint64: float64 loses the nanoseconds of an epoch, and
        # int64 wraps a log time past 2**63.  The difference reads signed.
        self._seconds = None if extraction is None else (
            extraction.time.ndarray - np.uint64(start)).view(
                "int64") / SECOND_NS
        duration = (end - start) / SECOND_NS
        self._plot.set_range(0.0, min(DEFAULT_SPAN, duration) or 1.0)
        self._fields = dict([] if plan is None else plottable_fields(plan))
        self._name.setText(topic)
        self._field.clear()
        for path, dtype in self._fields.items():
            self._field.addItem("{} \u00b7 {}".format(path, dtype))
        self._field.setEnabled(bool(self._fields))
        if not self._fields:
            self._plot.clear()

    def set_field(self, path):
        """Draw ``path``; raise ``ValueError`` when the topic lacks it."""
        if path not in self._fields:
            raise ValueError("{!r} is not a plottable field of {!r}".format(
                path, self._topic))
        self._field.setCurrentIndex(list(self._fields).index(path))

    def _on_field_chosen(self, index):
        self.field_changed.emit(list(self._fields)[index])

    def _redraw(self, index):
        if index < 0:
            return
        path = list(self._fields)[index]
        self._plot.set_series(path, self._fields[path] == "bool",
                              self._seconds,
                              self._extraction.columns[path].ndarray)

    def _apply_style(self):
        self._field_word.setStyleSheet(Rules.sheet(self, "label"))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
