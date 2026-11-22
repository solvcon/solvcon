# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The theme-aware ink of the reflection app."""

from PySide6.QtGui import QColor

from ... import pilot

__all__ = [
    'Ink',
]


class Ink(pilot.Ink):
    """The shades of the base ground the figure and the results card sit
    on: the steps of :class:`pilot.Shades` taken from the base toward the
    ink on it.

    Every color is read off the palette of the widget that draws it, or
    picked from a pair (light, dark) by the lightness of that palette, so a
    theme switch recolors the app without a hex value surviving the switch.
    Each painted role holds a floor against the base and the panel, the
    ground a plot outside the figure shows through: ``ink``, ``muted``,
    ``bad``, and ``good`` read as text, at 4.5:1, and the colors of
    :meth:`series` show as marks, at the 3:1 that WCAG 2.1 asks of graphics
    (success criterion 1.4.11,
    https://www.w3.org/TR/WCAG21/#non-text-contrast).  The computed series
    takes a color of the Okabe-Ito palette (M. Okabe and K. Ito, Color
    Universal Design, 2008, https://jfly.uni-koeln.de/color/); the exact
    solution takes the ink.

    :param widget: The widget whose palette the shades are read off.
    """

    def __init__(self, widget):
        super().__init__(widget)
        grounds = self.grounds()
        #: The text color of what is wrong.
        self.bad = pilot.legible(self, self.bright_text, *grounds)
        #: The text color of what is good.
        self.good = pilot.legible(self, pilot.AppRules.tone(self, 'good'),
                                  *grounds)
        self._computed = pilot.legible(
            self, QColor(pilot.DATA[1 if self.dark else 0]), *grounds,
            floor=pilot.GRAPHIC_CONTRAST)

    def grounds(self):
        return (self.base, self.panel)

    def dimmed(self, ratio):
        """Return the ink moved back toward the base.

        :param ratio: How far to move, from 0 for the ink to 1 for the
            base.
        """
        return self.blend(self.on_base, self.base, ratio)

    def raised(self, ratio):
        """Return the base moved toward the ink.

        :param ratio: How far to move, from 0 for the base to 1 for the
            ink.
        """
        return self.blend(self.base, self.on_base, ratio)

    def series(self, key):
        """Return the data color of a series, a mark at 3:1 against the
        grounds.

        :param key: The series, ``'computed'`` for the computed values; the
            exact solution and any key without a color of its own take the
            ink.
        """
        return QColor(self._computed if 'computed' == key else self.ink)

    def tone(self, name):
        """Return the text color of a tone, at 4.5:1 against the grounds.

        :param name: 'good', 'bad', or 'muted'; any other reads in the ink.
        """
        return QColor({'good': self.good, 'bad': self.bad,
                       'muted': self.muted}.get(name, self.ink))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
