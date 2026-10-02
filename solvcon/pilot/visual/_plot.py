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

The mouse navigates the view the model holds.  The wheel zooms about the
cursor and a left drag pans, both axes over the frame and one axis over its
tick labels.  A double-click returns to following the data.
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
    'plot_array',
]


class LinePlotWidget(QWidget):
    """Paint one plot model, with axes, ticks, and a legend.

    The model is the widget's own; a caller fills its series and calls
    :meth:`refresh`.  Limits are taken from the data every refresh, so a
    plot that is appended to while a run marches keeps following it, until
    the view is zoomed or panned.  A navigated view holds through refreshes
    until :meth:`reset_view`.

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
    #: Wheel zoom per degree of rotation, as on the 2D canvas: one full
    #: revolution doubles the zoom.
    ZOOM_STEP_PER_DEGREE = 1.0 / 360.0

    def __init__(self, title="", xlabel="", ylabel="", log_y=False,
                 parent=None):
        super().__init__(parent)
        self.model = _pcore.RPlotModel()
        # A log plot draws exponents the series do not hold, so they are
        # scaled and navigated in a model of their own.
        self._log_model = _pcore.RPlotModel()
        self._log_series = self._log_model.add_series()
        self._xticker = _pcore.RPlotTicker(5)
        self._yticker = _pcore.RPlotTicker(4)
        self.log_y = log_y
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._limits = None
        # The axes a left drag moves and the data point it holds under the
        # cursor, or None between drags.
        self._drag = None
        self._decimator = _pcore.RPlotDecimator()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 160)

    def add_series(self, label, color=None):
        """Add one named curve; returns the new series."""
        series = _pcore.RPlotSeries()
        series.label = label
        if color is not None:
            series.color = color
        return self.model.add_series(series)

    def set_ylabel(self, text):
        """Rename the ordinate, for a plot whose quantity is selectable."""
        self._ylabel = text
        self.update()

    def refresh(self):
        """Recompute the limits from the series and repaint."""
        self._limits = self._calc_limits()
        self.update()

    def reset_view(self):
        """Drop a zoomed or panned view and follow the data again."""
        self.model.reset_view()
        self._log_model.reset_view()
        self.refresh()

    def _view_model(self):
        """The model holding the view in the space that is drawn."""
        return self._log_model if self.log_y else self.model

    def limits(self):
        """The drawn limits as a ``PlotLimits2d``, or None while there
        is nothing to draw.

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

    def _drawn_points(self, series, rect):
        """Return the points of ``series`` worth drawing in ``rect``, mapped
        for the axis.

        A series with more than four samples per device pixel column is
        cut to the first, lowest, highest, and last sample of each column,
        which keeps every column's extent at a cost bounded by the width.
        A sparser series keeps every point :meth:`_points` gives.
        """
        columns = math.ceil(rect.width() * self.devicePixelRatioF())
        floor = self.LOG_FLOOR if self.log_y else -math.inf
        kept = self._decimator.select(series, self._limits.xmin,
                                      self._limits.xmax, columns, floor)
        out = []
        for it in kept:
            x, y = series.x(it), series.y(it)
            out.append((x, math.log10(y) if self.log_y else y))
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
        if self.model.auto_view:
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
        mapped = self._log_model
        mapped.margin = self.model.margin
        self._log_series.set_data(plot_array(xs), plot_array(ys))
        if mapped.auto_view:
            mapped.autoscale()
        return mapped.view_limits()

    def _lineplot_rect(self):
        left, top, right, bottom = self.MARGINS
        return QRect(left, top, max(1, self.width() - left - right),
                     max(1, self.height() - top - bottom))

    def _mapper(self, rect):
        limits = self._limits
        xspan = limits.xmax - limits.xmin
        yspan = limits.ymax - limits.ymin

        def to_screen(x, y):
            return QPointF(
                rect.left() + (x - limits.xmin) / xspan * rect.width(),
                rect.bottom() - (y - limits.ymin) / yspan * rect.height())

        return to_screen

    def _to_data(self, pos):
        """The inverse of :meth:`_mapper`: the drawn-space point at the
        widget position ``pos``."""
        rect = self._lineplot_rect()
        limits = self._limits
        return (
            limits.xmin + (pos.x() - rect.left()) / rect.width()
            * (limits.xmax - limits.xmin),
            limits.ymin + (rect.bottom() - pos.y()) / rect.height()
            * (limits.ymax - limits.ymin))

    def _axes_at(self, pos):
        """Which axes a gesture at ``pos`` navigates, as ``(x, y)`` flags.

        The frame takes both axes, the strip of tick labels under it the
        abscissa, and the one beside it the ordinate.  Elsewhere, over the
        titles or the corner between the strips, nothing is navigated and
        the answer is None.
        """
        if None is self._limits:
            return None
        rect = self._lineplot_rect()
        x, y = pos.x(), pos.y()
        across = rect.left() <= x <= rect.right()
        along = rect.top() <= y <= rect.bottom()
        if across and along:
            return (True, True)
        if across and y > rect.bottom():
            return (True, False)
        if along and x < rect.left():
            return (False, True)
        return None

    def wheelEvent(self, event):
        pos = event.position()
        axes = self._axes_at(pos)
        degrees = event.angleDelta().y() / 8.0
        if None is axes or 0 == degrees:
            event.ignore()
            return
        factor = 2.0 ** (degrees * self.ZOOM_STEP_PER_DEGREE)
        anchor_x, anchor_y = self._to_data(pos)
        self._view_model().zoom_view(factor if axes[0] else 1.0,
                                     factor if axes[1] else 1.0,
                                     anchor_x, anchor_y)
        self.refresh()
        event.accept()

    def mousePressEvent(self, event):
        pos = event.position()
        axes = self._axes_at(pos)
        if Qt.LeftButton != event.button() or None is axes:
            super().mousePressEvent(event)
            return
        self._drag = (axes, self._to_data(pos))
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if None is self._drag or None is self._limits:
            super().mouseMoveEvent(event)
            return
        # Pan by the gap between the held point and the one now under the
        # cursor, which puts the held point back under it.
        (move_x, move_y), (held_x, held_y) = self._drag
        at_x, at_y = self._to_data(event.position())
        self._view_model().pan_view(held_x - at_x if move_x else 0.0,
                                    held_y - at_y if move_y else 0.0)
        self.refresh()
        event.accept()

    def mouseReleaseEvent(self, event):
        if Qt.LeftButton != event.button() or None is self._drag:
            super().mouseReleaseEvent(event)
            return
        self._drag = None
        self.unsetCursor()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        axes = self._axes_at(event.position())
        if Qt.LeftButton != event.button() or None is axes:
            super().mouseDoubleClickEvent(event)
            return
        self.reset_view()
        event.accept()

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
        limits = self._limits
        faint = QColor(ink)
        faint.setAlpha(48)
        metrics = self.fontMetrics()
        xticks = self._xticker.locate(limits.xmin, limits.xmax)
        for value, text in zip(xticks, self._xticker.labels(xticks)):
            at = to_screen(value, limits.ymin).x()
            painter.setPen(QPen(faint))
            painter.drawLine(at, rect.top(), at, rect.bottom())
            painter.setPen(QPen(ink))
            painter.drawText(
                QRect(round(at) - 40, rect.bottom() + 2, 80,
                      metrics.height()),
                Qt.AlignCenter, text)
        if self.log_y:
            ticks = self._yticker.locate_decades(limits.ymin, limits.ymax)
            texts = self._yticker.decade_labels(ticks)
        else:
            ticks = self._yticker.locate(limits.ymin, limits.ymax)
            texts = self._yticker.labels(ticks)
        for value, text in zip(ticks, texts):
            at = to_screen(limits.xmin, value).y()
            painter.setPen(QPen(faint))
            painter.drawLine(rect.left(), at, rect.right(), at)
            painter.setPen(QPen(ink))
            painter.drawText(
                QRect(0, round(at) - metrics.height() // 2,
                      self.MARGINS[0] - 4, metrics.height()),
                Qt.AlignRight | Qt.AlignVCenter, text)

    def _draw_series(self, painter, rect, to_screen):
        painter.save()
        painter.setClipRect(rect)
        for it in range(self.model.size):
            series = self.model.series(it)
            points = self._drawn_points(series, rect)
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


def plot_array(values):
    """Copy values into the float64 array a plot series takes."""
    return core.SimpleArrayFloat64(array=np.asarray(values, dtype='float64'))


def _qcolor(color):
    return QColor(color.r, color.g, color.b, color.a)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
