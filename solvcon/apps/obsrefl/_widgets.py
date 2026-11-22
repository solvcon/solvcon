# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The small widgets and helpers the reflection window and its control
column share.

None of it knows the reflection.
"""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QGuiApplication, QIcon, QPainter, QPainterPath,
                           QPen, QPixmap)

__all__ = [
    'GLYPHS',
    'icon',
]

#: The glyphs of the run buttons on a 12 by 12 grid: a filled polygon, a
#: rounded box (corner radius, two corners), or a stroked arc (pen width,
#: two corners, start and span in degrees).
GLYPHS = {
    'play': (('fill', (3, 1.5), (10.5, 6), (3, 10.5)),),
    'pause': (('box', 0.6, (2.5, 1.5), (5, 10.5)),
              ('box', 0.6, (7, 1.5), (9.5, 10.5))),
    'step': (('fill', (2, 1.5), (8, 6), (2, 10.5)),
             ('box', 0.5, (8.5, 1.5), (10.5, 10.5))),
    'reset': (('arc', 1.5, (2, 2), (10, 10), 100, 280),
              ('fill', (4.2, 0.2), (8.2, 2.1), (4.6, 4.6))),
}


def icon(name, color, size=14, ratio=None):
    """Return a :class:`QIcon` of a glyph of :data:`GLYPHS`.

    The glyphs are drawn rather than taken from a font, so they keep one
    weight on every platform and follow the button text color.

    :param name: The glyph, a key of :data:`GLYPHS`.
    :param color: The color it is drawn in.
    :param size: Its side, in device-independent pixels.
    :param ratio: The device pixel ratio it is drawn at, or None for the
        highest of any screen, which no screen then scales up.
    """
    if ratio is None:
        ratio = QGuiApplication.instance().devicePixelRatio()
    pixmap = QPixmap(round(size * ratio), round(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    unit = size / 12.0

    def at(across, down):
        return QPointF(across * unit, down * unit)

    for kind, *args in GLYPHS[name]:
        if kind in ('fill', 'box'):
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
        else:
            pen = QPen(color, unit * args.pop(0))
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
        if 'fill' == kind:
            path = QPainterPath(at(*args[0]))
            for corner in args[1:]:
                path.lineTo(at(*corner))
            path.closeSubpath()
            painter.drawPath(path)
        elif 'box' == kind:
            radius, one, other = args
            painter.drawRoundedRect(QRectF(at(*one), at(*other)),
                                    unit * radius, unit * radius)
        else:
            one, other, start, span = args
            painter.drawArc(QRectF(at(*one), at(*other)), start * 16,
                            span * 16)
    painter.end()
    return QIcon(pixmap)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
