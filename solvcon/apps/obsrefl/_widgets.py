# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The small widgets and helpers the reflection window and its control
column share.

None of it knows the reflection; the style sheet comes from the rules of
:mod:`._style`.
"""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QFont, QGuiApplication, QIcon, QPainter,
                           QPainterPath, QPen, QPixmap)
from PySide6.QtWidgets import QLabel

from ... import pilot

from . import _style

__all__ = [
    'GLYPHS',
    'Styled',
    'form_label',
    'head_font',
    'icon',
    'message',
    'reserve_width',
    'say',
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


def reserve_width(widget, texts):
    """Hold a widget at the width of the widest text it swaps through, so
    the swap neither shifts the row around it nor moves its least width.

    :param widget: A widget with ``text()`` and ``setText()``.
    :param texts: The texts it shows.
    """
    keep = widget.text()
    width = 0
    for text in texts:
        widget.setText(text)
        width = max(width, widget.sizeHint().width())
    widget.setText(keep)
    widget.setMinimumWidth(width)


def form_label(text, tip=""):
    """Return the muted label of a row of a form.

    :param text: What the label reads.
    :param tip: Its tooltip, or empty for none.
    """
    label = QLabel(text)
    label.setObjectName("formLabel")
    if tip:
        label.setToolTip(tip)
    return label


def message(role):
    """Return a hidden, wrapping label for a message.

    :param role: The object name the sheet styles it by: ``'note'``,
        ``'hint'``, or ``'fieldError'`` for what is wrong.
    """
    label = QLabel("")
    label.setObjectName(role)
    label.setWordWrap(True)
    label.hide()
    return label


def say(label, text):
    """Show a text in a label, or hide the label for an empty one.

    :param label: The label to show the text in.
    :param text: The text, or empty to hide the label.
    """
    label.setText(text)
    label.setVisible(bool(text))


def head_font(font):
    """Return a copy of a font as small, bold, and spaced out as a section
    head names its section.

    :param font: The font to copy.
    """
    head = QFont(font)
    head.setPixelSize(pilot.SMALL_PX)
    head.setWeight(QFont.DemiBold)
    head.setLetterSpacing(QFont.PercentageSpacing, 110)
    return head


class Styled(pilot.PaletteStyle):
    """Carry the style sheet of the roles a widget and its children draw.

    The subclass names the roles in :attr:`ROLES` and sets its fonts and
    icons in ``restyle()``; the sheet and ``restyle()`` run again on a
    theme switch, once its children exist and it has turned
    :attr:`_styling` off.
    """

    #: The roles of :class:`._style.Rules` the sheet carries.
    ROLES = ()
    _styling = True

    def restyle(self):
        """Set the fonts and the icons the sheet cannot."""

    def _apply_style(self):
        # Setting the sheet re-polishes the children, whose palette events
        # come back here; one pass is all a palette change needs.
        if self._styling:
            return
        self._styling = True
        try:
            sheet = _style.Rules.sheet(self, *self.ROLES)
            if sheet != self.styleSheet():
                self.setStyleSheet(sheet)
            self.restyle()
        finally:
            self._styling = False

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
