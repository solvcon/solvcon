# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The colormap legend that says what the colors in the viewer mean.

:class:`ColorBar` draws the ramp of :func:`~._field_render.colormap` over
the range the viewer pins its colors to, with its two ends labelled, so
reading the field does not depend on remembering which end of the ramp is
high.

The bar says what the colors mean and nothing else.  Marking the analytic
zone values on it would put a second reading on a scale, where the eye
takes it for part of the scale itself; how far a run stands from those
states is the zone table's to say, in numbers that need no reading off a
strip of color.

The bar lies along either axis, since it is meant to sit against an edge of
the view it explains and the two pairs of edges want opposite runs.  It
grounds itself in the same color the viewer clears its frame to, so the
strip it takes reads as part of the view rather than as a panel laid over
it, and it inks itself to match: a ground that does not follow the theme
cannot be written on in a color that does.

The widget holds no run.  Its owner pushes in a range whenever a frame is
drawn, and it paints what it was last given.
"""

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QLinearGradient, QColor, QPen
from PySide6.QtWidgets import QWidget, QSizePolicy

from ._field_render import colormap

__all__ = [  # noqa: F822
    'ColorBar',
]


class ColorBar(QWidget):
    """A slim colormap ramp with its two ends labelled.

    :ivar lo: Low end of the pinned range, or None with nothing to show.
    :ivar hi: High end of the pinned range.
    :ivar vertical: Whether the ramp runs up the widget instead of across.
    """

    #: Colors sampled across the ramp to build the gradient.  The map is
    #: piecewise linear in four stops, so this is far more than it takes to
    #: draw it smoothly and still cheap to rebuild every frame.
    STOPS = 32
    #: Thickness of the ramp itself, in pixels, across its run.
    BAR_THICKNESS = 12
    #: Room kept at each end of the run for half an end label, so the text
    #: of a label centered on the extreme does not run off the widget.
    END_PAD = 24
    #: Width the end-label column takes beside a vertical ramp.
    LABEL_WIDTH = 52
    #: Room kept around the ramp, across its run.
    PAD = 4
    #: The ground under the bar, which is what the domain viewer clears its
    #: frame to, so the strip does not read as a panel over the view.
    BACKGROUND = (1.0, 1.0, 1.0)
    #: Ink for the frame and the labels.  Fixed rather than
    #: taken from the palette, because the ground it is written on is: a
    #: dark theme's near-white text would vanish on it.
    INK = (0.1, 0.1, 0.1)

    def __init__(self, vertical=False, parent=None):
        super().__init__(parent)
        self.lo = None
        self.hi = None
        self.vertical = vertical
        if vertical:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.setMinimumWidth(self.thickness())
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.setMinimumHeight(self.thickness())

    def thickness(self):
        """Room the bar needs across its run, labels included.

        One label line, not two: the ends are labelled on one side of the
        ramp and nothing is written on the other.
        """
        room = self.BAR_THICKNESS + 2 * self.PAD
        if self.vertical:
            return room + self.LABEL_WIDTH
        return room + self.fontMetrics().height()

    def show_scale(self, lo, hi):
        """Draw the ramp over ``lo`` to ``hi``.

        A missing or degenerate range blanks the widget rather than
        dividing by a zero span.
        """
        if None is lo or None is hi or not hi > lo:
            self.lo = self.hi = None
        else:
            self.lo, self.hi = float(lo), float(hi)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if None is self.lo:
            return
        painter.fillRect(self.rect(), QColor.fromRgbF(*self.BACKGROUND))
        bar = self._bar_rect()
        painter.fillRect(bar, self._ramp(bar))
        painter.setPen(QPen(QColor.fromRgbF(*self.INK)))
        painter.drawRect(bar)
        self._draw_ends(painter, bar)

    def _bar_rect(self):
        """Where the ramp is drawn, with room beside it for the labels."""
        if self.vertical:
            return QRect(self.LABEL_WIDTH + self.PAD, self.END_PAD,
                         self.BAR_THICKNESS,
                         max(1, self.height() - 2 * self.END_PAD))
        return QRect(self.END_PAD, self.PAD,
                     max(1, self.width() - 2 * self.END_PAD),
                     self.BAR_THICKNESS)

    def _ramp(self, bar):
        """The colormap as a gradient along ``bar``.

        A vertical ramp runs low at the bottom, the way an axis does, so
        the gradient is laid out bottom to top.
        """
        if self.vertical:
            gradient = QLinearGradient(0, bar.bottom(), 0, bar.top())
        else:
            gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
        for it in range(self.STOPS + 1):
            t = it / self.STOPS
            red, green, blue = colormap(t)
            gradient.setColorAt(t, QColor.fromRgbF(red, green, blue))
        return gradient

    def _at(self, value, bar):
        """Where ``value`` falls along the ramp, in widget pixels."""
        t = (value - self.lo) / (self.hi - self.lo)
        if self.vertical:
            return bar.bottom() - t * (bar.height() - 1)
        return bar.left() + t * (bar.width() - 1)

    def _label_rect(self, at, bar, height):
        """Room for a label beside the ramp at ``at`` along its run."""
        if self.vertical:
            return QRect(0, round(at) - height // 2, self.LABEL_WIDTH, height)
        return QRect(round(at) - self.END_PAD, bar.bottom() + 1,
                     2 * self.END_PAD, height)

    def _draw_ends(self, painter, bar):
        """Label the two ends of the range against the ramp."""
        height = self.fontMetrics().height()
        align = (Qt.AlignRight if self.vertical else Qt.AlignHCenter)
        for value in (self.lo, self.hi):
            painter.drawText(self._label_rect(self._at(value, bar), bar,
                                              height),
                             align | Qt.AlignVCenter, _number(value))


def _number(value):
    """Format an end label to three significant digits."""
    return f"{value:#.3g}"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
