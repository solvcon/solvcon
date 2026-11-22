# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The small widgets and helpers the reflection window and its control
column share.

None of it knows the reflection; the style sheet comes from the rules of
:mod:`._style`.
"""

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PySide6.QtGui import (QFont, QGuiApplication, QIcon, QPainter,
                           QPainterPath, QPen, QPixmap)
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy

from ... import pilot
from ...pilot import panel

from . import _style

__all__ = [
    'ElidedLabel',
    'Fitted',
    'FoldGroup',
    'GLYPHS',
    'Styled',
    'form_label',
    'guard_wheel',
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


class _WheelGuard(QObject):
    """Hand a wheel over an unfocused field on to its parent."""

    def eventFilter(self, watched, event):
        if QEvent.Wheel == event.type() and not watched.hasFocus():
            event.ignore()
            return True
        return False


def guard_wheel(*widgets):
    """Let the wheel scroll past the widgets until one is focused, so
    scrolling the column never changes a value under the pointer.

    :param widgets: The spin boxes and combo boxes to guard.
    """
    for widget in widgets:
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.installEventFilter(_WheelGuard(widget))


def message(role):
    """Return a hidden, wrapping label for a message.

    :param role: The object name the sheet styles it by: ``'hint'``, or
        ``'fieldError'`` for what is wrong.
    """
    label = QLabel("")
    label.setObjectName(role)
    label.setWordWrap(True)
    label.hide()
    return label


def _room(label, text):
    """Return the width a label takes to show a text whole.

    :param label: The label whose font and margins count.
    :param text: The text to measure.
    """
    margins = label.contentsMargins()
    return (label.fontMetrics().horizontalAdvance(text) + margins.left()
            + margins.right())


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


class ElidedLabel(QLabel):
    """A one-line label that elides its text to the room it gets, and
    keeps the whole of it in the tip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        #: The text before eliding.
        self.full_text = ""
        #: A text as wide as the least the label shows of its text.
        self.shortest = "M..."
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def set_full_text(self, text):
        """Show a text, elided to the width the label gets.

        :param text: The whole text, which the tooltip carries.
        """
        if text == self.full_text:
            return
        # The width it asks for is all that can move the layout, and a
        # relayout every frame would cost more than the frame's text.
        refit = (self.fontMetrics().horizontalAdvance(text)
                 != self.fontMetrics().horizontalAdvance(self.full_text))
        self.full_text = text
        self.setToolTip(text)
        if refit:
            self.updateGeometry()
        self._elide()

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(_room(self, self.full_text) + 4)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(min(self.sizeHint().width(),
                          _room(self, self.shortest)))
        return hint

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(
            self.full_text, Qt.ElideRight,
            max(0, self.contentsRect().width())))


class Fitted(QLabel):
    """A one-line label that shows the longest of its forms the room it
    gets holds, so a readout gives up words before a digit is cut."""

    #: The longest form, which the tip carries.
    full_text = ""
    _forms = _widest = ("",)

    def set_forms(self, forms, widest):
        """Show the longest of the forms that fits, and size the label by
        the widest of them, so the strip holds still while the numbers run.

        :param forms: The forms of the readout, longest first.
        :param widest: The same forms at their widest numbers.
        """
        if tuple(widest) != self._widest:
            self._widest = tuple(widest)
            self.updateGeometry()
        self._forms = tuple(forms)
        self.full_text = forms[0]
        self.setToolTip(forms[0])
        self._fit()

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(_room(self, self._widest[0]))
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(_room(self, self._widest[-1]))
        return hint

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        self.setText(next((form for form in self._forms
                           if _room(self, form) <= self.width()),
                          self._forms[-1]))


class FoldGroup(panel.FoldGroup):
    """A group of the column that folds behind its head, styled by the
    sheet of the column, whose summary elides short of the title."""

    #: No sheet of its own: the sheet of the column styles the head.
    HEAD_SHEET = ""
    #: The least room, in pixels, between the title and the summary.
    GAP = 28
    #: The room around the body of every group, left, top, right, bottom.
    MARGINS = (6, 2, 4, 10)

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self._head.setFocusPolicy(Qt.TabFocus)
        self._summary.setContentsMargins(0, 2, 6, 0)
        # Not restyle(): a group extends it to the body it builds after
        # this returns.
        self._head.setFont(head_font(self.font()))

    def restyle(self):
        """Set the font of the head; the sheet sizes the summary."""
        self._head.setFont(head_font(self.font()))
        self._fit_summary()

    def _body_grid(self, across=8, down=4, labels=0):
        """Return the grid of the body, inside the margins every group
        keeps.

        :param across: The pixels between the columns.
        :param down: The pixels between the rows.
        :param labels: The least width of the first column, of labels.
        """
        grid = QGridLayout(self.body)
        grid.setContentsMargins(*self.MARGINS)
        grid.setHorizontalSpacing(across)
        grid.setVerticalSpacing(down)
        grid.setColumnMinimumWidth(0, labels)
        return grid

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
