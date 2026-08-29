# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Draw a :class:`~solvcon.pilot.RPlotModel` into a Qt widget.

The plot model is Qt-free by design and paints nothing, and its
``view()`` carries one zoom, which suits a spatial plot and flattens an xy
one whose axes share no scale.  :class:`LinePlotWidget` is the front end it
lacks: it stretches each axis onto the frame and draws the axes, the
ticks, and the legend around the curves.  It is a widget and not a canvas,
that name being the 2D drawing surface of :mod:`solvcon.pilot.canvas`.

The ordinate can go on a log scale, for a quantity spanning decades.  The
series keep their true values; only the mapping to the screen is
logarithmic.
"""

import math

import numpy as np

from PySide6.QtCore import Qt, QRect, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF
from PySide6.QtWidgets import QWidget, QSizePolicy

from ... import core
from .. import _pilot_core as _pcore

__all__ = [  # noqa: F822
    'LinePlotWidget',
]


def _nice_ticks(lo, hi, want=5):
    """Round tick positions covering ``lo`` to ``hi``, about ``want`` of
    them.

    The step is the 1, 2, or 5 times a power of ten nearest below the even
    spacing, which is what keeps tick labels short enough to read.
    """
    span = hi - lo
    if not span > 0 or not math.isfinite(span):
        return []
    raw = span / max(1, want)
    power = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = mult * power
        if raw <= step:
            break
    # Count the steps rather than accumulate them: where the span is small
    # beside the offset, `first + step` rounds back to `first` and never
    # ends, inside the paint that called it.  The slack keeps a tick that
    # sits on the end and the clamp keeps rounding from carrying one past
    # it; zero is snapped off the residue that would print as an exponent.
    first = math.ceil(lo / step)
    last = math.floor((hi + 1e-9 * step) / step)
    return [0.0 if 0 == it else min(max(it * step, lo), hi)
            for it in range(first, last + 1)]


def _decade_ticks(lo, hi):
    """Powers of ten covering the decades from ``lo`` to ``hi``.

    Both bounds are already logarithms, so the ticks are the integers in
    between; a range inside one decade still gets its two ends marked.
    """
    low, high = math.ceil(lo), math.floor(hi)
    if high < low:
        return [lo, hi]
    return [float(it) for it in range(int(low), int(high) + 1)]


def _decade_label(value):
    """Format a log-axis tick, which is drawn at its exponent.

    A range inside one decade is ticked at its own ends, which are not
    whole exponents and do not read as powers of ten.
    """
    if float(value).is_integer():
        return f"1e{value:.0f}"
    return _tick_label(10.0 ** value)


def _tick_label(value):
    """Format a tick to something short enough to sit under an axis."""
    if 0.0 == value:
        return "0"
    if 1e-3 <= abs(value) < 1e5:
        return f"{value:.4g}"
    return f"{value:.0e}"


class LinePlotWidget(QWidget):
    """Paint one plot model, with axes, ticks, and a legend.

    The model is the widget's own; a caller fills its series and calls
    :meth:`refresh`.  Limits are taken from the data every refresh, so a
    plot that is appended to while a run marches keeps following it.

    :ivar model: The :class:`~solvcon.pilot.RPlotModel` being drawn.
    :ivar log_y: Whether the ordinate is mapped logarithmically.
    """

    #: Room for the axis labels, as (left, top, right, bottom) pixels.  The
    #: left holds a tick label, the bottom a label and the axis title, and
    #: the top the plot title.
    MARGINS = (58, 22, 12, 34)
    #: Smallest ordinate a log plot draws; zero and below have no
    #: logarithm to place.
    LOG_FLOOR = 1e-16

    def __init__(self, title="", xlabel="", ylabel="", log_y=False,
                 parent=None):
        super().__init__(parent)
        self.model = _pcore.RPlotModel()
        self.log_y = log_y
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._limits = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 160)

    def add_series(self, label, color=None):
        """Add one named curve; returns the new series."""
        series = self.model.add_series()
        series.label = label
        if color is not None:
            series.color = color
        return series

    def set_ylabel(self, text):
        """Rename the ordinate, for a plot whose quantity is selectable."""
        self._ylabel = text
        self.update()

    def refresh(self):
        """Recompute the limits from the series and repaint."""
        self._limits = self._calc_limits()
        self.update()

    def limits(self):
        """The drawn limits as ``(xmin, xmax, ymin, ymax)``, or None while
        there is nothing to draw.

        The ordinate is in the mapped space, so a log plot reports the
        exponents it drew between.
        """
        return self._limits

    def _points(self, series):
        """The series data as ``(x, y)`` pairs, mapped for the axis.

        A log plot drops what it cannot place, a non-positive ordinate
        having no logarithm, rather than breaking the curve around it.
        """
        out = []
        for it in range(series.size):
            x, y = series.x(it), series.y(it)
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if self.log_y:
                if y <= self.LOG_FLOOR:
                    continue
                y = math.log10(y)
            out.append((x, y))
        return out

    def _calc_limits(self):
        """The box the curves are drawn in, or None with nothing to draw.

        The model does the scaling, margin and zero-span guard included.
        Only the mapping onto the screen is the widget's.
        """
        if self.log_y:
            return self._log_limits()
        if None is self.model.data_limits():
            return None
        self.model.autoscale()
        return self.model.view_limits()

    def _log_limits(self):
        """Scale the mapped points through a model of their own.

        The exponents a log plot draws are not in the series, so they are
        scaled by the same rule rather than by a second one written here.
        """
        xs, ys = [], []
        for it in range(self.model.size):
            for x, y in self._points(self.model.series(it)):
                xs.append(x)
                ys.append(y)
        if not xs:
            return None
        mapped = _pcore.RPlotModel()
        mapped.margin = self.model.margin
        mapped.add_series().set_data(_array(xs), _array(ys))
        mapped.autoscale()
        return mapped.view_limits()

    def _lineplot_rect(self):
        left, top, right, bottom = self.MARGINS
        return QRect(left, top, max(1, self.width() - left - right),
                     max(1, self.height() - top - bottom))

    def _mapper(self, rect):
        xmin, xmax, ymin, ymax = self._limits
        xspan = xmax - xmin
        yspan = ymax - ymin

        def to_screen(x, y):
            return QPointF(
                rect.left() + (x - xmin) / xspan * rect.width(),
                rect.bottom() - (y - ymin) / yspan * rect.height())

        return to_screen

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        ink = self.palette().windowText().color()
        rect = self._lineplot_rect()
        self._draw_titles(painter, ink, rect)
        if None is self._limits:
            painter.setPen(QPen(ink))
            painter.drawRect(rect)
            return
        to_screen = self._mapper(rect)
        self._draw_grid(painter, ink, rect, to_screen)
        painter.setPen(QPen(ink))
        painter.drawRect(rect)
        self._draw_series(painter, rect, to_screen)
        self._draw_legend(painter, rect)

    def _draw_titles(self, painter, ink, rect):
        painter.setPen(QPen(ink))
        height = self.fontMetrics().height()
        painter.drawText(QRect(0, 0, self.width(), self.MARGINS[1]),
                         Qt.AlignCenter, self._title)
        painter.drawText(
            QRect(rect.left(), self.height() - height, rect.width(), height),
            Qt.AlignCenter, self._xlabel)
        painter.drawText(QRect(0, 0, self.width(), height),
                         Qt.AlignLeft | Qt.AlignVCenter, self._ylabel)

    def _draw_grid(self, painter, ink, rect, to_screen):
        xmin, xmax, ymin, ymax = self._limits
        faint = QColor(ink)
        faint.setAlpha(48)
        metrics = self.fontMetrics()
        for value in _nice_ticks(xmin, xmax):
            at = to_screen(value, ymin).x()
            painter.setPen(QPen(faint))
            painter.drawLine(at, rect.top(), at, rect.bottom())
            painter.setPen(QPen(ink))
            painter.drawText(
                QRect(round(at) - 40, rect.bottom() + 2, 80,
                      metrics.height()),
                Qt.AlignCenter, _tick_label(value))
        ticks = (_decade_ticks(ymin, ymax) if self.log_y
                 else _nice_ticks(ymin, ymax, want=4))
        for value in ticks:
            at = to_screen(xmin, value).y()
            painter.setPen(QPen(faint))
            painter.drawLine(rect.left(), at, rect.right(), at)
            painter.setPen(QPen(ink))
            text = (_decade_label(value) if self.log_y
                    else _tick_label(value))
            painter.drawText(
                QRect(0, round(at) - metrics.height() // 2,
                      self.MARGINS[0] - 4, metrics.height()),
                Qt.AlignRight | Qt.AlignVCenter, text)

    def _draw_series(self, painter, rect, to_screen):
        painter.save()
        painter.setClipRect(rect)
        for it in range(self.model.size):
            series = self.model.series(it)
            points = self._points(series)
            if len(points) < 2:
                continue
            pen = QPen(_qcolor(series.color))
            pen.setWidthF(series.line_width)
            painter.setPen(pen)
            painter.drawPolyline(
                QPolygonF([to_screen(x, y) for x, y in points]))
        painter.restore()

    def _draw_legend(self, painter, rect):
        """Name each curve in its own color, inside the top right.

        The names are backed by the window color, since a curve running
        through the top right would otherwise be read as part of the text.
        """
        metrics = self.fontMetrics()
        labels = [self.model.series(it).label
                  for it in range(self.model.size)]
        labels = [label for label in labels if label]
        if not labels:
            return
        width = max(metrics.horizontalAdvance(text) for text in labels) + 8
        box = QRect(rect.right() - width - 4, rect.top() + 4, width,
                    metrics.height() * len(labels))
        backing = QColor(self.palette().window().color())
        backing.setAlpha(216)
        painter.fillRect(box, backing)
        top = box.top()
        for it in range(self.model.size):
            series = self.model.series(it)
            if not series.label:
                continue
            painter.setPen(QPen(_qcolor(series.color)))
            painter.drawText(QRect(box.left(), top, width,
                                   metrics.height()),
                             Qt.AlignRight | Qt.AlignVCenter, series.label)
            top += metrics.height()


def _array(values):
    return core.SimpleArrayFloat64(array=np.asarray(values, dtype='float64'))


def _qcolor(color):
    return QColor(color.r, color.g, color.b, color.a)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
