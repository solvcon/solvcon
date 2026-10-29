# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The window of the shock-tube app, painted in the colors of the palette so
it follows the theme.
"""

import math

import numpy as np

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QToolButton, QVBoxLayout,
                               QWidget)

from ... import pilot

__all__ = [
    'ProfilePlot',
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


class ProfilePlot(QWidget):
    """The profile panels, one per shown quantity: the analytic solution as
    a line, the numerical one as markers, and the wave fronts of the
    analytic solution as dotted guides.

    :ivar quantities: The names of the shown quantities.
    :ivar fronts: ``(label, x)`` of the wave fronts, left first.
    :ivar x0: Where the diaphragm stood at t = 0, or None.
    :ivar show_fronts: Whether to draw the guides.
    """

    TITLE = 18
    GAP = 10
    LEFT = 52
    BOTTOM = 20
    MARKER = 5.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.quantities = list(SHOWN)
        self.xlim = (0.0, 1.0)
        self.exact = None
        self.points = None
        self.fronts = []
        self.x0 = None
        self.show_fronts = True
        self._tickers = {}
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
        """
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
        count = max(1, len(self.quantities))
        columns = 1 if count <= 3 else 2
        rows = math.ceil(count / columns)
        width = (self.width() - (columns - 1) * self.GAP) / columns
        height = (self.height() - self.BOTTOM) / rows
        out = []
        for it, name in enumerate(self.quantities):
            row, column = divmod(it, columns)
            out.append((name, QRectF(
                column * (width + self.GAP) + self.LEFT,
                row * height + self.TITLE, width - self.LEFT - 2,
                height - self.TITLE - self.GAP), it + columns >= count))
        return out

    def yrange(self, name):
        """Return the ``(lo, hi)`` the panel of ``name`` spans."""
        parts = [data[1][name] for data in (self.exact, self.points)
                 if data is not None]
        values = (np.concatenate(parts) if parts
                  else np.empty(0, dtype='float64'))
        values = values[np.isfinite(values)]
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
        small = QFont(self.font())
        small.setPixelSize(10)
        bold = QFont(self.font())
        bold.setPixelSize(12)
        bold.setWeight(QFont.DemiBold)
        lo, hi = self.xlim
        for name, rect, bottom in self.panels():
            ylo, yhi = self.yrange(name)

            def at(x, y, rect=rect, ylo=ylo, yhi=yhi):
                return (rect.left() + (x - lo) / (hi - lo) * rect.width(),
                        rect.bottom() - (y - ylo) / (yhi - ylo)
                        * rect.height())

            painter.setFont(bold)
            painter.setPen(ink)
            painter.drawText(QRectF(rect.left(), rect.top() - self.TITLE,
                                    rect.width(), self.TITLE - 3),
                             Qt.AlignLeft | Qt.AlignBottom,
                             dict(QUANTITIES)[name])
            painter.setFont(small)
            for value, text in self._ticks(ylo, yhi, rect.height(), 28):
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
                    # Labels at the ends stay inside the widget.
                    left = min(max(x - 30, 0), self.width() - 60)
                    painter.setPen(muted)
                    painter.drawText(QRectF(left, rect.bottom() + 2, 60, 14),
                                     Qt.AlignHCenter | Qt.AlignTop, text)
            painter.save()
            painter.setClipRect(rect.adjusted(-4, -4, 4, 4))
            for it, (pos, label, dashed) in enumerate(self.guides()):
                x = at(pos, ylo)[0]
                painter.setPen(QPen(muted, 1, Qt.DashLine if dashed
                                    else Qt.DotLine))
                painter.drawLine(QPointF(x, rect.top()),
                                 QPointF(x, rect.bottom()))
                # Staggered, so the labels of close fronts stay apart.
                painter.drawText(QRectF(x + 3, rect.top() + 2 + it % 2 * 12,
                                        60, 12),
                                 Qt.AlignLeft | Qt.AlignTop, label)
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
                    if math.isfinite(y):
                        painter.drawEllipse(QPointF(*at(x, y)), half, half)
            painter.restore()
            painter.setPen(QPen(frame, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
        painter.end()

    def _ticks(self, lo, hi, pixels, spacing):
        """Return ``(value, text)`` of the ticks over ``lo`` to ``hi``,
        about one per ``spacing`` of the ``pixels`` they span."""
        target = max(2, min(9, int(pixels // spacing)))
        ticker = self._tickers.get(target)
        if ticker is None:
            ticker = self._tickers[target] = pilot.RPlotTicker(target)
        ticks = list(ticker.locate(lo, hi))
        return [(value, text) for value, text
                in zip(ticks, ticker.labels(ticks)) if lo <= value <= hi]


class ShockTubeWindow(QWidget):
    """The run controls and the title over the panels, and the chips that
    pick the quantities.

    :ivar plot: The :class:`ProfilePlot`.
    :ivar chips: The checkable button of each quantity, by name.
    :ivar fronts_chip: The checkable button that shows the wave fronts.
    :ivar buttons: The ``'run'``, ``'step'``, and ``'reset'`` buttons.
    :ivar run_clicked: Owner-supplied callback for the button that runs,
        pauses, or runs again.
    :ivar step_clicked: Owner-supplied callback for Step.
    :ivar reset_clicked: Owner-supplied callback for Reset.
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
        self._title = QLabel("")
        font = QFont(self.font())
        font.setPixelSize(14)
        font.setWeight(QFont.DemiBold)
        self._title.setFont(font)
        self._waves = QLabel("")
        self._waves.setToolTip("The waves of the exact solution, left to "
                               "right (S shock, R rarefaction, C contact), "
                               "and the star state between them")
        self.chips = {}
        chips = QHBoxLayout()
        chips.setSpacing(5)
        for name, text in QUANTITIES:
            chip = QToolButton()
            chip.setText(text)
            chip.setCheckable(True)
            chip.setChecked(name in SHOWN)
            chip.setFocusPolicy(Qt.NoFocus)
            chip.toggled.connect(lambda on, key=name: self._on_chip(key))
            self.chips[name] = chip
            chips.addWidget(chip)
        chips.addStretch(1)
        self.fronts_chip = QToolButton()
        self.fronts_chip.setText("wave fronts")
        self.fronts_chip.setCheckable(True)
        self.fronts_chip.setChecked(True)
        self.fronts_chip.setFocusPolicy(Qt.NoFocus)
        self.fronts_chip.toggled.connect(self.plot.set_show_fronts)
        chips.addWidget(self.fronts_chip)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 10, 14, 8)
        self._layout.setSpacing(6)
        self._layout.addLayout(controls)
        self._layout.addWidget(self._title)
        self._layout.addWidget(self._waves)
        self._layout.addWidget(self.plot, 1)
        self._layout.addLayout(chips)

    def sizeHint(self):
        return QSize(1000, 560)

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
