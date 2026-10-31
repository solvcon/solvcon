# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The window of the shock-tube app, painted in the colors of the palette so
it follows the theme.
"""

import math

import numpy as np

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QPainter,
                           QPalette, QPen, QPolygonF)
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from ... import pilot
from ...pilot import visual

from . import _plotmath

__all__ = [
    'Legend',
    'ProfilePlot',
    'XtDiagram',
    'ShockTubeWindow',
]

#: The name and the chip text of each quantity, named as the fields of
#: :class:`~._shocktube.ShockTube` and the solver.
QUANTITIES = (('density', "density"), ('velocity', "velocity"),
              ('pressure', "pressure"), ('temperature', "temperature"),
              ('internal_energy', "internal energy"),
              ('entropy', "entropy"))
#: The quantities shown at start.
SHOWN = ('density', 'velocity', 'pressure')


def _blend(color, other, ratio):
    """Mix ``ratio`` of ``other`` into ``color``."""
    keep = 1.0 - ratio
    return QColor(round(color.red() * keep + other.red() * ratio),
                  round(color.green() * keep + other.green() * ratio),
                  round(color.blue() * keep + other.blue() * ratio))


class Rules(pilot.RuleCatalog):
    """The style of the toggles under the panels: the chips that each turn
    one thing on or off, and the two-way switch whose halves join into one
    pill.  They sit on the base ground of the window, so every shade steps
    off the base: an outline that shows a toggle is there, and the accent
    tint that shows it is on."""

    _SMALL_PX = 11
    _EDGE_MIX = 0.45
    _MUTED_MIX = 0.35
    _HOVER_MIX = 0.07
    _TINT_MIX = 0.16

    @classmethod
    def toggles(cls, shades):
        base, ink = shades.base, shades.on_base
        return """
            QToolButton#chip, QToolButton#switch {{
                border: 1px solid {edge};
                padding: 1px 9px;
                font-size: {small}px;
                color: {muted};
                background: transparent;
            }}
            QToolButton#chip {{ border-radius: 10px; }}
            QToolButton#switch[edge="left"] {{
                border-top-left-radius: 10px;
                border-bottom-left-radius: 10px;
                border-right: none;
            }}
            QToolButton#switch[edge="right"] {{
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }}
            QToolButton#chip:hover, QToolButton#switch:hover {{
                background: {hover};
                color: {ink};
            }}
            QToolButton#chip:checked, QToolButton#switch:checked {{
                background: {tint};
                color: {ink};
            }}
            QToolButton#chip:checked {{ border-color: {accent}; }}
            QLabel#switchLabel {{ font-size: {small}px; color: {muted}; }}
            """.format(
            edge=shades.blend(base, ink, cls._EDGE_MIX).name(),
            muted=shades.blend(ink, base, cls._MUTED_MIX).name(),
            hover=shades.blend(base, ink, cls._HOVER_MIX).name(),
            tint=shades.blend(base, shades.accent, cls._TINT_MIX).name(),
            accent=shades.accent.name(), ink=ink.name(),
            small=cls._SMALL_PX)


class Legend(QWidget):
    """The key of the curves in the panels: each name after a swatch drawn
    as the panels draw its curve, a line for the analytic solution and a
    marker for the numerical one, in the same colors of the palette.

    :ivar entries: ``(kind, label)`` of each curve, ``kind`` ``'line'`` or
        ``'marker'``.
    """

    SWATCH = 18
    GAP = 14

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries = []
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_entries(self, entries):
        """Name ``entries``, ``(kind, label)``, in order."""
        self.entries = list(entries)
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        metrics = self.fontMetrics()
        width = sum(self.SWATCH + 5 + metrics.horizontalAdvance(label)
                    for _, label in self.entries)
        width += self.GAP * max(0, len(self.entries) - 1)
        return QSize(width, metrics.height() + 2)

    def paintEvent(self, event):
        palette = self.palette()
        ink = palette.color(QPalette.Text)
        mark = palette.color(QPalette.Highlight)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        metrics = self.fontMetrics()
        middle = self.height() / 2
        left = 0.0
        for kind, label in self.entries:
            if 'line' == kind:
                painter.setPen(QPen(ink, 1.25))
                painter.drawLine(QPointF(left + 1, middle),
                                 QPointF(left + self.SWATCH - 1, middle))
            else:
                half = ProfilePlot.MARKER / 2
                painter.setPen(QPen(mark, 1.3))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(left + self.SWATCH / 2, middle),
                                    half, half)
            left += self.SWATCH + 5
            width = metrics.horizontalAdvance(label)
            painter.setPen(ink)
            painter.drawText(QRectF(left, 0, width + 1, self.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, label)
            left += width + self.GAP
        painter.end()


class ProfilePlot(QWidget):
    """The profile panels, one per shown quantity: the analytic solution as
    a line, the numerical one as markers, and the wave fronts of the
    analytic solution as dotted guides.

    A band between the titles and the frames of the top row names the
    guides once: each front over its guide, and above them the region
    between two fronts where it has room.

    The wheel zooms the x range about the cursor and a double-click shows
    the whole tube again.  Over a frame, the cursor draws a vertical line
    through every panel and reads out the point of the march nearest
    it.

    Two options change the axes.  On a log y axis a panel whose values are
    all positive spans whole decades; the others stay linear.  A residual
    strip under each frame draws the numerical solution minus the exact
    one averaged over the same cells, on an axis symmetric about zero.

    :ivar quantities: The names of the shown quantities.
    :ivar fronts: ``(label, x)`` of the wave fronts, left first.
    :ivar x0: Where the diaphragm stood at t = 0, or None.
    :ivar show_fronts: Whether to draw the guides.
    :ivar options: Whether each of ``'log'`` and ``'residual'`` is on.
    :ivar reference: The exact cell means at the numerical points by
        quantity, which the residual is taken against, or None.
    :ivar view: The x range zoomed into, or None for the whole tube.
    :ivar cursor: ``(name, x)`` of the panel and the position under the
        cursor, or None.
    :ivar view_changed: Owner-supplied callback fired with :attr:`view`.
    """

    TITLE = 18
    GAP = 10
    LEFT = 52
    BOTTOM = 20
    MARKER = 5.0
    #: The height of a row of the band.
    BAND_ROW = 13
    #: The share of a panel the residual strip under its frame takes, and
    #: the least height it keeps.
    STRIP = 0.26
    STRIP_MIN = 18
    #: The least height of a strip that also names its scale.
    STRIP_NAMED = 24
    #: The narrowest view, as a share of the tube.
    MIN_VIEW = 0.005
    #: The view a wheel notch keeps.
    WHEEL_ZOOM = 0.8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.quantities = list(SHOWN)
        self.xlim = (0.0, 1.0)
        self.exact = None
        self.points = None
        self.fronts = []
        self.x0 = None
        self.show_fronts = True
        self.options = {'log': False, 'residual': False}
        self.reference = None
        self.view = None
        self.cursor = None
        self.view_changed = None
        self._tickers = {}
        self.setMouseTracking(True)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_quantities(self, names):
        """Draw a panel for each of ``names``."""
        self.quantities = list(names)
        self.update()

    def set_data(self, xlim, exact, points=None):
        """Draw the curves over ``xlim``.

        :param exact: ``(x, values)`` of the analytic solution, ``values``
            mapping a quantity name to an array over ``x``.
        :param points: ``(x, values)`` of the numerical solution, or None.

        Another tube shows whole again.
        """
        if tuple(xlim) != self.xlim:
            self.set_view(None)
        self.xlim = tuple(xlim)
        self.exact = exact
        self.points = points
        self.update()

    def set_fronts(self, fronts, x0):
        """Guide the wave ``fronts``, ``(label, x)`` left first, and the
        diaphragm at ``x0``."""
        self.fronts = list(fronts)
        self.x0 = x0
        self.update()

    def set_show_fronts(self, shown):
        """Draw or hide the guides."""
        self.show_fronts = shown
        self.update()

    def set_option(self, name, on):
        """Switch the option ``name``, ``'log'`` or ``'residual'``."""
        self.options[name] = bool(on)
        self.update()

    def xrange(self):
        """Return the x range shown: the view, or the whole tube."""
        return self.view if self.view is not None else self.xlim

    def set_view(self, view):
        """Show the x range ``view``, or the whole tube with None."""
        if view != self.view:
            self.view = view
            self.update()
            if self.view_changed is not None:
                self.view_changed(view)

    def zoom(self, factor, x):
        """Scale the x range shown by ``factor`` about ``x``, inside the
        tube; a range as wide as the tube shows it whole."""
        lo, hi = self.xlim
        left, right = self.xrange()
        width = max((right - left) * factor, self.MIN_VIEW * (hi - lo))
        if width >= hi - lo:
            self.set_view(None)
            return
        left = min(max(x - (x - left) * factor, lo), hi - width)
        self.set_view((left, left + width))

    def readout(self, name, x):
        """Return the lines that read out the panel of ``name`` at ``x``:
        the point of the march nearest it, the exact solution there, and,
        with a reference, the march minus the exact cell mean."""
        if self.points is None:
            return []
        xs, values = self.points
        it = int(np.argmin(np.abs(xs - x)))
        lines = [f"x = {xs[it]:.4g}", f"march {values[name][it]:.5g}"]
        if self.exact is not None:
            exact = np.interp(xs[it], self.exact[0], self.exact[1][name])
            lines.append(f"exact {exact:.5g}")
        residual = self.residual(name)
        if residual is not None:
            lines.append(f"march - mean {residual[it]:+.2e}")
        return lines

    def _hit(self, pos):
        """Return ``(name, x)`` of the frame or strip under ``pos``, or
        None."""
        lo, hi = self.xrange()
        for name, frame, strip, _ in self._layout():
            area = frame if strip is None else frame.united(strip)
            if area.contains(pos):
                return name, lo + (pos.x() - frame.left()) \
                    / frame.width() * (hi - lo)
        return None

    def wheelEvent(self, event):
        hit = self._hit(event.position())
        if hit is None:
            event.ignore()
            return
        notches = event.angleDelta().y() / 120
        self.zoom(self.WHEEL_ZOOM ** notches, hit[1])
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.set_view(None)

    def mouseMoveEvent(self, event):
        self.cursor = self._hit(event.position())
        self.update()

    def leaveEvent(self, event):
        self.cursor = None
        self.update()
        super().leaveEvent(event)

    def set_reference(self, reference):
        """Take the residual against ``reference``, the exact cell means at
        the numerical points by quantity, or against nothing with None."""
        self.reference = reference
        self.update()

    def guides(self):
        """Return ``(x, label, dashed)`` of each guide drawn: the fronts
        dotted, then the diaphragm dashed.  At t = 0 every front stands on
        the diaphragm, which alone is guided."""
        if not self.show_fronts or self.x0 is None:
            return []
        out = [(x, label, False) for label, x in self.fronts
               if x != self.x0]
        return out + [(self.x0, "x0", True)]

    def panels(self):
        """Return ``(name, rect, bottom)`` of the plot frame of each panel,
        ``bottom`` whether it is the lowest of its column."""
        return [(name, rect, bottom)
                for name, rect, _, bottom in self._layout()]

    def strips(self):
        """Return ``(name, rect)`` of the residual strip under each frame,
        or nothing while the residual is off."""
        return [(name, strip) for name, _, strip, _ in self._layout()
                if strip is not None]

    def _layout(self):
        """Return ``(name, frame, strip, bottom)`` of each panel, ``strip``
        None while the residual is off."""
        count = max(1, len(self.quantities))
        columns = 1 if count <= 3 else 2
        rows = math.ceil(count / columns)
        band = self.BAND_ROW * self.band_rows()
        width = (self.width() - (columns - 1) * self.GAP) / columns
        height = (self.height() - self.BOTTOM - band) / rows
        total = height - self.TITLE - self.GAP
        # The strip keeps a readable least height, but never more than the
        # frame above it.
        strip = (min(max(self.STRIP_MIN, self.STRIP * total), total / 2 - 2)
                 if self.options['residual'] else 0.0)
        out = []
        for it, name in enumerate(self.quantities):
            row, column = divmod(it, columns)
            left = column * (width + self.GAP) + self.LEFT
            top = row * height + self.TITLE + band
            frame = QRectF(left, top, width - self.LEFT - 2,
                           total - (strip + 4 if strip else 0))
            below = (QRectF(left, frame.bottom() + 4, frame.width(), strip)
                     if strip else None)
            out.append((name, frame, below, it + columns >= count))
        return out

    def log_axis(self, name):
        """Return whether the panel of ``name`` draws on a log axis: with
        the option on, when every value it draws is positive."""
        if not self.options['log']:
            return False
        values = self._values(name)
        return bool(values.size) and bool(np.all(values > 0))

    def decades(self, name):
        """Return the whole decades ``(lo, hi)`` the log axis of ``name``
        spans, at least one apart."""
        values = self._values(name)
        lo = math.floor(math.log10(float(values.min())))
        hi = math.ceil(math.log10(float(values.max())))
        return lo, max(hi, lo + 1)

    def residual(self, name):
        """Return the numerical ``name`` minus its exact cell means at each
        numerical point, or None without both."""
        if self.points is None or self.reference is None:
            return None
        return self.points[1][name] - self.reference[name]

    def _values(self, name):
        """Return the finite values of ``name`` the panel draws."""
        parts = [data[1][name] for data in (self.exact, self.points)
                 if data is not None]
        values = (np.concatenate(parts) if parts
                  else np.empty(0, dtype='float64'))
        return values[np.isfinite(values)]

    def band_rows(self):
        """Return the rows of the band: the fronts, and the regions once the
        waves have left the diaphragm; none without guides."""
        guides = self.guides()
        if not guides:
            return 0
        return 2 if len(guides) > 1 else 1

    def band_labels(self, rect):
        """Return ``(text, left, row)`` of the names over ``rect``, a frame
        of the top row, row 0 the nearer one.

        Each front is named over its guide, in the upper row when a name
        before it fills the lower one, and each region between the fronts
        in the upper row where it has room.  A name that would cross the
        frame, or overlap one placed before it, is left out.
        """
        lo, hi = self.xrange()
        metrics = QFontMetrics(self._small_font())

        def pixel(x):
            return rect.left() + (x - lo) / (hi - lo) * rect.width()

        rows = tuple(range(self.band_rows()))
        texts, boxes = [], []
        for pos, label, _ in self.guides():
            texts.append(label)
            boxes.append((pixel(pos), metrics.horizontalAdvance(label) + 6,
                          rows))
        if self.band_rows() > 1:
            for name, left, right in _plotmath.region_spans(self.fronts, lo,
                                                            hi):
                width = metrics.horizontalAdvance(name) + 6
                if pixel(right) - pixel(left) >= width:
                    texts.append(name)
                    boxes.append((pixel(0.5 * (left + right)), width, (1,)))
        placed = _plotmath.place(boxes, rect.left() - 2, rect.right() + 2)
        return [(texts[index], left, row) for index, row, left in placed]

    def _small_font(self):
        small = QFont(self.font())
        small.setPixelSize(10)
        return small

    def yrange(self, name):
        """Return the ``(lo, hi)`` the panel of ``name`` spans."""
        values = self._values(name)
        if 0 == values.size:
            return 0.0, 1.0
        lo, hi = float(values.min()), float(values.max())
        pad = 0.07 * (hi - lo) if hi > lo else max(0.1, 0.05 * abs(hi))
        return lo - pad, hi + pad

    def paintEvent(self, event):
        palette = self.palette()
        ground = palette.color(QPalette.Base)
        ink = palette.color(QPalette.Text)
        grid, frame = _blend(ground, ink, 0.1), _blend(ground, ink, 0.35)
        muted = _blend(ink, ground, 0.35)
        mark = palette.color(QPalette.Highlight)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), ground)
        small = self._small_font()
        band = self.BAND_ROW * self.band_rows()
        bold = QFont(self.font())
        bold.setPixelSize(12)
        bold.setWeight(QFont.DemiBold)
        lo, hi = self.xrange()
        layout = self._layout()
        for name, rect, strip, bottom in layout:
            log = self.log_axis(name)
            ylo, yhi = self.decades(name) if log else self.yrange(name)
            yaxis = _plotmath.Axis(ylo, yhi, rect.bottom(), rect.top(), log)
            top = rect.top() <= layout[0][1].top()

            def at(x, y, rect=rect, yaxis=yaxis):
                return (rect.left() + (x - lo) / (hi - lo) * rect.width(),
                        yaxis(y))

            painter.setFont(bold)
            painter.setPen(ink)
            title = rect.top() - self.TITLE - (band if top else 0)
            painter.drawText(QRectF(rect.left(), title, rect.width(),
                                    self.TITLE - 3),
                             Qt.AlignLeft | Qt.AlignBottom,
                             dict(QUANTITIES)[name])
            painter.setFont(small)
            if top:
                painter.setPen(muted)
                for text, left, row in self.band_labels(rect):
                    y = rect.top() - (row + 1) * self.BAND_ROW
                    painter.drawText(QRectF(left, y, rect.width(),
                                            self.BAND_ROW),
                                     Qt.AlignLeft | Qt.AlignVCenter,
                                     f" {text}")
            yticks = ([(10.0 ** k, f"1e{k}") for k in range(ylo, yhi + 1)]
                      if log else self._ticks(ylo, yhi, rect.height(), 28))
            for value, text in yticks:
                y = at(lo, value)[1]
                painter.setPen(QPen(grid, 1))
                painter.drawLine(QPointF(rect.left(), y),
                                 QPointF(rect.right(), y))
                painter.setPen(muted)
                painter.drawText(QRectF(0, y - 7, rect.left() - 5, 14),
                                 Qt.AlignRight | Qt.AlignVCenter, text)
            for value, text in self._ticks(lo, hi, rect.width(), 60):
                x = at(value, ylo)[0]
                painter.setPen(QPen(grid, 1))
                painter.drawLine(QPointF(x, rect.top()),
                                 QPointF(x, rect.bottom()))
                if bottom:
                    # Labels at the ends stay inside the widget, under the
                    # strip of a panel that has one.
                    left = min(max(x - 30, 0), self.width() - 60)
                    under = (strip or rect).bottom()
                    painter.setPen(muted)
                    painter.drawText(QRectF(left, under + 2, 60, 14),
                                     Qt.AlignHCenter | Qt.AlignTop, text)
            painter.save()
            painter.setClipRect(rect.adjusted(-4, -4, 4, 4))
            for pos, _, dashed in self.guides():
                x = at(pos, ylo)[0]
                painter.setPen(QPen(muted, 1, Qt.DashLine if dashed
                                    else Qt.DotLine))
                painter.drawLine(QPointF(x, rect.top()),
                                 QPointF(x, rect.bottom()))
            if self.exact is not None:
                xs, values = self.exact
                painter.setPen(QPen(ink, 1.25))
                painter.drawPolyline(QPolygonF([
                    QPointF(*at(x, y)) for x, y in zip(xs, values[name])
                    if math.isfinite(y)]))
            if self.points is not None:
                xs, values = self.points
                painter.setPen(QPen(mark, 1.3))
                painter.setBrush(Qt.NoBrush)
                half = self.MARKER / 2
                for x, y in zip(xs, values[name]):
                    px, py = at(x, y)
                    if math.isfinite(py):
                        painter.drawEllipse(QPointF(px, py), half, half)
            painter.restore()
            painter.setPen(QPen(frame, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            if strip is not None:
                self._draw_strip(painter, name, strip, (grid, frame, muted,
                                                        mark))
            if self.cursor is not None:
                self._draw_cursor(painter, name, rect, strip, (ground, ink,
                                                               frame))
        painter.end()

    def _draw_cursor(self, painter, name, rect, strip, colors):
        """Draw the line of the cursor through the frame and the strip of
        ``name``, and the read-out in the frame under the cursor."""
        ground, ink, frame = colors
        lo, hi = self.xrange()
        hovered, pos = self.cursor
        x = rect.left() + (pos - lo) / (hi - lo) * rect.width()
        bottom = (strip or rect).bottom()
        painter.setPen(QPen(frame, 1))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, bottom))
        if name != hovered:
            return
        lines = self.readout(name, pos)
        if not lines:
            return
        painter.setFont(self._small_font())
        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line) for line in lines) + 10
        height = len(lines) * metrics.height() + 6
        # The box stays clear of the cursor, on the side with more room.
        left = (x + 8 if x + 8 + width <= rect.right()
                else x - 8 - width)
        box = QRectF(left, rect.top() + 4, width, height)
        painter.setPen(QPen(frame, 1))
        painter.setBrush(ground)
        painter.drawRect(box)
        painter.setPen(ink)
        for it, line in enumerate(lines):
            painter.drawText(QRectF(box.left() + 5,
                                    box.top() + 3 + it * metrics.height(),
                                    width, metrics.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, line)
        painter.setBrush(Qt.NoBrush)

    def _draw_strip(self, painter, name, strip, colors):
        """Draw the residual of ``name`` in ``strip``: a marker per point
        on an axis symmetric about the zero line, the largest magnitude
        named at its top."""
        grid, frame, muted, mark = colors
        lo, hi = self.xrange()
        zero = strip.center().y()
        painter.setPen(QPen(grid, 1))
        painter.drawLine(QPointF(strip.left(), zero),
                         QPointF(strip.right(), zero))
        residual = self.residual(name)
        finite = (residual[np.isfinite(residual)] if residual is not None
                  else np.empty(0, dtype='float64'))
        span = float(np.abs(finite).max()) if finite.size else 0.0
        painter.setPen(muted)
        painter.drawText(QRectF(0, zero - 7, strip.left() - 5, 14),
                         Qt.AlignRight | Qt.AlignVCenter, "0")
        if strip.height() >= self.STRIP_NAMED:
            painter.drawText(QRectF(strip.left() + 3, strip.top(), 120, 12),
                             Qt.AlignLeft | Qt.AlignTop,
                             f"residual \u00b1{span:.2g}" if span
                             else "residual")
        if span:
            painter.save()
            painter.setClipRect(strip)
            painter.setPen(QPen(mark, 1.1))
            painter.setBrush(Qt.NoBrush)
            scale = 0.45 * strip.height() / span
            for x, value in zip(self.points[0], residual):
                if math.isfinite(value):
                    painter.drawEllipse(QPointF(
                        strip.left() + (x - lo) / (hi - lo) * strip.width(),
                        zero - value * scale), 1.6, 1.6)
            painter.restore()
        painter.setPen(QPen(frame, 1))
        painter.drawRect(strip)

    def _ticks(self, lo, hi, pixels, spacing):
        return _ticks(self._tickers, lo, hi, pixels, spacing)


class XtDiagram(QWidget):
    """The x-t diagram: the history of the numerical density colored in the
    x-t plane, time running up, under the wave fronts of the analytic
    solution drawn from the diaphragm, fan edges dashed, the contact dotted,
    and the shock solid.  A line marks the time of the last step.

    :ivar times: The time of each row of :attr:`rows`, ascending.
    :ivar rows: ``[len(times), len(x)]`` density history.
    :ivar waves: ``(kind, speeds)`` of each wave of the analytic solution.
    :ivar label: What the colors are of, named after the title.
    """

    TITLE = ProfilePlot.TITLE
    #: The width of the colorbar at the right of the head.
    BAR = 120
    LEFT = ProfilePlot.LEFT
    BOTTOM = ProfilePlot.BOTTOM
    #: The dash pattern of each kind of wave, None for a solid line.
    DASHES = {'rarefaction': [4, 3], 'contact': [1, 2.5]}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.xlim = (0.0, 1.0)
        self.x = np.empty(0, dtype='float64')
        self.times = np.empty(0, dtype='float64')
        self.rows = np.empty((0, 0), dtype='float64')
        self.waves = []
        self.x0 = 0.0
        self.t_end = 1.0
        self.view = None
        self.label = "density"
        self._tickers = {}
        self.setMinimumSize(320, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_view(self, view):
        """Show the x range ``view`` of the panels, or the whole tube with
        None."""
        self.view = view
        self.update()

    def set_waves(self, waves, x0, t_end):
        """Draw ``waves``, :class:`~._riemann.Wave`, from ``x0`` up to
        ``t_end``, the time of the last step."""
        self.waves = [(wave.kind, wave.speeds) for wave in waves
                      if 'vacuum' != wave.kind]
        self.x0 = x0
        self.t_end = max(float(t_end), 1.e-30)
        self.update()

    def set_label(self, label):
        """Name what the colors are of after the title."""
        self.label = label
        self.update()

    def colorbar_rect(self):
        """Return the colorbar at the right of the head, between the ends
        of the range it keys."""
        rect = self.plot_rect()
        width = QFontMetrics(self._small_font()).horizontalAdvance(
            self._ends()[1]) + 6
        return QRectF(rect.right() - width - self.BAR, 4, self.BAR,
                      self.TITLE - 9)

    def _ends(self):
        """Return the texts of the ends of the range of the colors."""
        lo, hi = self.vrange()
        return f"{lo:.3g}", f"{hi:.3g}"

    def _small_font(self):
        small = QFont(self.font())
        small.setPixelSize(10)
        return small

    def set_history(self, xlim, x, times, rows):
        """Color ``rows``, a density over ``x`` at each of ``times``."""
        self.xlim = tuple(xlim)
        self.x = np.asarray(x, dtype='float64')
        self.times = np.asarray(times, dtype='float64')
        self.rows = np.asarray(rows, dtype='float64')
        self.update()

    def vrange(self):
        """Return the ``(lo, hi)`` the colors span."""
        values = self.rows[np.isfinite(self.rows)]
        if 0 == values.size:
            return 0.0, 1.0
        lo, hi = float(values.min()), float(values.max())
        return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)

    def colors(self):
        """Return the rows as ARGB words, the first row at t = 0; a value
        that is not finite stays clear."""
        lo, hi = self.vrange()
        rgb = np.rint(visual.colormap((self.rows - lo) / (hi - lo))
                      * 255).astype('uint32')
        words = (0xff000000 | rgb[..., 0] << 16 | rgb[..., 1] << 8
                 | rgb[..., 2]).astype('uint32')
        words[~np.isfinite(self.rows)] = 0
        return np.ascontiguousarray(words)

    def plot_rect(self):
        """Return the frame of the diagram."""
        return QRectF(self.LEFT, self.TITLE, self.width() - self.LEFT - 2,
                      self.height() - self.TITLE - self.BOTTOM)

    def paintEvent(self, event):
        palette = self.palette()
        ground = palette.color(QPalette.Base)
        ink = palette.color(QPalette.Text)
        frame = _blend(ground, ink, 0.35)
        muted = _blend(ink, ground, 0.35)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), ground)
        rect = self.plot_rect()
        lo, hi = self.view if self.view is not None else self.xlim

        def at(x, t):
            return QPointF(rect.left() + (x - lo) / (hi - lo) * rect.width(),
                           rect.bottom() - t / self.t_end * rect.height())

        bold = QFont(self.font())
        bold.setPixelSize(12)
        bold.setWeight(QFont.DemiBold)
        painter.setFont(bold)
        painter.setPen(ink)
        painter.drawText(QRectF(rect.left(), 0, rect.width(), self.TITLE - 3),
                         Qt.AlignLeft | Qt.AlignBottom, "x-t diagram")
        painter.setFont(self._small_font())
        painter.setPen(muted)
        head = painter.fontMetrics()
        painter.drawText(QRectF(rect.left() + QFontMetrics(bold)
                                .horizontalAdvance("x-t diagram") + 10, 0,
                                rect.width(), self.TITLE - 3),
                         Qt.AlignLeft | Qt.AlignBottom, self.label)
        if self.rows.size:
            bar = self.colorbar_rect()
            low, high = self._ends()
            steps = np.linspace(0.0, 1.0, int(bar.width()),
                                dtype='float64')
            rgb = np.rint(visual.colormap(steps) * 255).astype('int32')
            for it, (red, green, blue) in enumerate(rgb[:, :3]):
                painter.fillRect(QRectF(bar.left() + it, bar.top(), 1,
                                        bar.height()),
                                 QColor(int(red), int(green), int(blue)))
            painter.drawText(QRectF(0, bar.top() - 3, bar.left() - 4,
                                    bar.height() + 6),
                             Qt.AlignRight | Qt.AlignVCenter, low)
            painter.drawText(QRectF(bar.right() + 4, bar.top() - 3,
                                    head.horizontalAdvance(high) + 2,
                                    bar.height() + 6),
                             Qt.AlignLeft | Qt.AlignVCenter, high)
        painter.setFont(bold)
        painter.save()
        painter.setClipRect(rect)
        if self.rows.size and len(self.x) > 1:
            words = self.colors()
            image = QImage(words.data, words.shape[1], words.shape[0],
                           words.shape[1] * 4, QImage.Format_ARGB32)
            half = (self.x[-1] - self.x[0]) / (len(self.x) - 1) / 2
            top = at(self.x[-1] + half, self.times[-1])
            bottom = at(self.x[0] - half, 0.0)
            # The rows run up in time and down in the image, so the image is
            # drawn mirrored about the middle of its target.
            painter.translate(0, top.y() + bottom.y())
            painter.scale(1, -1)
            painter.drawImage(QRectF(bottom.x(), top.y(),
                                     top.x() - bottom.x(),
                                     bottom.y() - top.y()), image)
            painter.resetTransform()
        for kind, speeds in self.waves:
            for speed in speeds:
                pen = QPen(ink, 1.3)
                pen.setCapStyle(Qt.FlatCap)
                if kind in self.DASHES:
                    pen.setDashPattern(self.DASHES[kind])
                painter.setPen(pen)
                painter.drawLine(at(self.x0, 0.0),
                                 at(self.x0 + speed * self.t_end, self.t_end))
        if self.times.size:
            now = at(lo, self.times[-1]).y()
            painter.setPen(QPen(palette.color(QPalette.Highlight), 1.4))
            painter.drawLine(QPointF(rect.left(), now),
                             QPointF(rect.right(), now))
        painter.restore()
        painter.setPen(QPen(frame, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        small = QFont(self.font())
        small.setPixelSize(10)
        painter.setFont(small)
        painter.setPen(muted)
        for value, text in _ticks(self._tickers, 0.0, self.t_end,
                                  rect.height(), 28):
            y = at(lo, value).y()
            painter.drawText(QRectF(0, y - 7, rect.left() - 5, 14),
                             Qt.AlignRight | Qt.AlignVCenter, text)
        for value, text in _ticks(self._tickers, lo, hi, rect.width(), 60):
            left = min(max(at(value, 0.0).x() - 30, 0), self.width() - 60)
            painter.drawText(QRectF(left, rect.bottom() + 2, 60, 14),
                             Qt.AlignHCenter | Qt.AlignTop, text)
        painter.drawText(QRectF(4, rect.top(), rect.left() - 10, 14),
                         Qt.AlignLeft | Qt.AlignTop, "t")
        painter.end()


def _ticks(tickers, lo, hi, pixels, spacing):
    """Return ``(value, text)`` of the ticks over ``lo`` to ``hi``, about one
    per ``spacing`` of the ``pixels`` they span, with a ticker of ``tickers``
    kept by its target count."""
    target = max(2, min(9, int(pixels // spacing)))
    ticker = tickers.get(target)
    if ticker is None:
        ticker = tickers[target] = pilot.RPlotTicker(target)
    ticks = list(ticker.locate(lo, hi))
    return [(value, text) for value, text
            in zip(ticks, ticker.labels(ticks)) if lo <= value <= hi]


class ShockTubeWindow(QWidget):
    """The run controls and the title over the panels, and the chips that
    pick the quantities.

    :ivar plot: The :class:`ProfilePlot`.
    :ivar legend: The :class:`Legend` of its curves, right of the title.
    :ivar chips: The checkable button of each quantity, by name.
    :ivar exact_buttons: The ``'points'`` and ``'cells'`` halves of the
        switch that draws the analytic solution at points or averaged over
        the cells of the run.
    :ivar fronts_chip: The checkable button that shows the wave fronts.
    :ivar option_chips: The ``'log'`` and ``'residual'`` chips of the
        options of :attr:`plot`.
    :ivar xt: The :class:`XtDiagram` under the panels.
    :ivar xt_chip: The checkable button that shows :attr:`xt`.
    :ivar buttons: The ``'run'``, ``'step'``, and ``'reset'`` buttons.
    :ivar fit_button: The button that shows the whole tube again, enabled
        while the panels are zoomed in.
    :ivar run_clicked: Owner-supplied callback for the button that runs,
        pauses, or runs again.
    :ivar step_clicked: Owner-supplied callback for Step.
    :ivar reset_clicked: Owner-supplied callback for Reset.
    :ivar exact_clicked: Owner-supplied callback for
        :attr:`exact_buttons`.
    """

    #: What the run button reads by the state of the run.
    RUN_TEXT = {'ready': "Run", 'running': "Pause", 'paused': "Resume",
                'finished': "Run again"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)
        self.plot = ProfilePlot()
        self.run_clicked = self.step_clicked = self.reset_clicked = None
        self.exact_clicked = None
        self.buttons = {}
        controls = QHBoxLayout()
        controls.setSpacing(8)
        for key, text in (('run', "Run"), ('step', "Step"),
                          ('reset', "Reset")):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, name=key: self._on(name))
            self.buttons[key] = button
            controls.addWidget(button)
        self._state = QLabel("")
        controls.addSpacing(10)
        controls.addWidget(self._state)
        controls.addStretch(1)
        self.fit_button = QPushButton("Reset zoom")
        self.fit_button.setToolTip("Show the whole tube again (a "
                                   "double-click on the panels does the "
                                   "same)")
        self.fit_button.setEnabled(False)
        self.fit_button.clicked.connect(lambda: self.plot.set_view(None))
        controls.addWidget(self.fit_button)
        self._title = QLabel("")
        font = QFont(self.font())
        font.setPixelSize(14)
        font.setWeight(QFont.DemiBold)
        self._title.setFont(font)
        self.legend = Legend()
        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(self._title)
        head.addStretch(1)
        head.addWidget(self.legend)
        self._waves = QLabel("")
        self._waves.setToolTip("The waves of the exact solution, left to "
                               "right (S shock, R rarefaction, C contact), "
                               "and the star state between them")
        self.chips = {}
        chips = QHBoxLayout()
        chips.setSpacing(5)
        for name, text in QUANTITIES:
            chip = self._chip(text, name in SHOWN)
            chip.toggled.connect(lambda on, key=name: self._on_chip(key))
            self.chips[name] = chip
            chips.addWidget(chip)
        chips.addStretch(1)
        label = QLabel("exact:")
        label.setObjectName("switchLabel")
        chips.addWidget(label)
        self.exact_buttons = {}
        group = QButtonGroup(self)
        switch = QHBoxLayout()
        switch.setSpacing(0)
        for key, edge, tip in (
                ('points', 'left', "Draw the exact solution at points"),
                ('cells', 'right', "Draw the exact solution averaged over "
                 "the cells of the run, as a conservative scheme holds "
                 "it")):
            button = QToolButton()
            button.setObjectName("switch")
            button.setProperty('edge', edge)
            button.setText(key)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setChecked('points' == key)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _=False: self._on('exact'))
            group.addButton(button)
            self.exact_buttons[key] = button
            switch.addWidget(button)
        chips.addLayout(switch)
        chips.addSpacing(10)
        self.fronts_chip = self._chip("wave fronts", True)
        self.fronts_chip.toggled.connect(self.plot.set_show_fronts)
        chips.addWidget(self.fronts_chip)
        self.option_chips = {}
        for key, text, tip in (
                ('log', "log y", "A log axis on the panels whose values are "
                                 "all positive"),
                ('residual', "residual", "A strip under each panel with the "
                                         "march minus the exact cell means")):
            chip = self._chip(text, False)
            chip.setToolTip(tip)
            chip.toggled.connect(
                lambda on, name=key: self.plot.set_option(name, on))
            self.option_chips[key] = chip
            chips.addWidget(chip)
        self.xt = XtDiagram()
        self.plot.view_changed = self._on_view
        self.xt_chip = self._chip("x-t diagram", True)
        self.xt_chip.toggled.connect(self.xt.setVisible)
        chips.addWidget(self.xt_chip)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 10, 14, 8)
        self._layout.setSpacing(6)
        self._layout.addLayout(controls)
        self._layout.addLayout(head)
        self._layout.addWidget(self._waves)
        self._layout.addWidget(self.plot, 2)
        self._layout.addWidget(self.xt, 1)
        self._layout.addLayout(chips)
        self._apply_style()

    def sizeHint(self):
        return QSize(1000, 560)

    def event(self, event):
        # A theme switch changes the palette, and the toggles' shades with
        # it.
        if event.type() in (QEvent.PaletteChange,
                            QEvent.ApplicationPaletteChange):
            self._apply_style()
        return super().event(event)

    def exact_mode(self):
        """Return how the analytic solution is drawn: ``'points'`` or
        ``'cells'``."""
        return next(key for key, button in self.exact_buttons.items()
                    if button.isChecked())

    def set_title(self, title, time):
        """Head the panels with ``title`` at ``time``."""
        self._title.setText(f"{title}    t = {time:.4g}")

    def title(self):
        """Return the text heading the panels."""
        return self._title.text()

    def show_waves(self, exact):
        """Say the wave pattern and the star state of ``exact``, an
        :class:`~._riemann.ExactRiemann`."""
        self._waves.setText(
            f"{exact.pattern}    p* = {exact.p_star:.5g}    "
            f"u* = {exact.u_star:.5g}    "
            f"rho*L = {exact.rho_star_left:.5g}    "
            f"rho*R = {exact.rho_star_right:.5g}")

    def waves_text(self):
        """Return what the window says of the waves."""
        return self._waves.text()

    def show_state(self, state, step, nstep):
        """Show the run state (``'ready'``, ``'running'``, ``'paused'``, or
        ``'finished'``) and its step on the controls."""
        self.buttons['run'].setText(self.RUN_TEXT[state])
        self.buttons['step'].setEnabled(state in ('ready', 'paused'))
        self.buttons['reset'].setEnabled('ready' != state)
        self._state.setText(f"{state}, step {step} / {nstep}")

    def state_text(self):
        """Return what the controls say of the run."""
        return self._state.text()

    def _on(self, name):
        callback = getattr(self, f"{name}_clicked")
        if callback is not None:
            callback()

    def _on_view(self, view):
        self.fit_button.setEnabled(view is not None)
        self.xt.set_view(view)

    def _chip(self, text, checked):
        chip = QToolButton()
        chip.setObjectName("chip")
        chip.setText(text)
        chip.setCheckable(True)
        chip.setChecked(checked)
        chip.setFocusPolicy(Qt.NoFocus)
        return chip

    def _apply_style(self):
        self.setStyleSheet(Rules.sheet(self, "toggles"))

    def _on_chip(self, name):
        names = [key for key, _ in QUANTITIES if self.chips[key].isChecked()]
        if not names:
            # Keep the last panel so the window never goes blank.
            self.chips[name].blockSignals(True)
            self.chips[name].setChecked(True)
            self.chips[name].blockSignals(False)
            return
        self.plot.set_quantities(names)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
