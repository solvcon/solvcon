# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
What the pilot's Python widgets build their style sheets out of.

The C++ theme manager owns the curated palette and the application-wide sheet
it derives from it; a Python widget that wants more than the palette gives
mixes its own shades here instead of naming a color. Every shade is a step
between two palette colors, so a widget styled from :class:`Shades` follows a
light/dark switch without a hex value anywhere in its code.
"""

import functools

from PySide6 import QtCore, QtGui, QtWidgets

__all__ = [
    'Shades',
    'RuleCatalog',
    'PaletteStyle',
    'PaletteStyled',
]


# Qt 6.6 added this event, and the theme backends still guard for older Qt.
# Naming it in a class body would fail the import of the whole pilot there.
_DPR_CHANGE_EVENT = getattr(QtCore.QEvent, 'DevicePixelRatioChange', None)

# How far the shades every page shares sit from the color they mix against.
_MUTED_MIX = 0.35
_GREYED_MIX = 0.6
_BORDER_MIX = 0.25


class Shades:
    """The colors a widget styles itself in, read off its own palette.

    A mix off them is asked for by how far it moves rather than by name, so
    a rule states the step it wants and gets it in whatever theme is current.
    """

    @staticmethod
    def blend(color, other, ratio):
        """Mix ``ratio`` of ``other`` into ``color``."""
        keep = 1.0 - ratio
        return QtGui.QColor(
            round(color.red() * keep + other.red() * ratio),
            round(color.green() * keep + other.green() * ratio),
            round(color.blue() * keep + other.blue() * ratio))

    def __init__(self, widget):
        palette = widget.palette()
        self.text = palette.color(QtGui.QPalette.WindowText)
        self.panel = palette.color(QtGui.QPalette.Window)
        self.base = palette.color(QtGui.QPalette.Base)
        self.accent = palette.color(QtGui.QPalette.Highlight)
        self.on_accent = palette.color(QtGui.QPalette.HighlightedText)
        self.on_base = palette.color(QtGui.QPalette.Text)

    # Mixed on demand: a paint that wants one of the three should not pay
    # for the other two.
    @functools.cached_property
    def muted(self):
        """The shade a secondary label reads in."""
        return self.dimmed(_MUTED_MIX)

    @functools.cached_property
    def greyed(self):
        """The shade a control the model cannot back yet reads in."""
        return self.dimmed(_GREYED_MIX)

    @functools.cached_property
    def border(self):
        """The hairline a box is drawn with."""
        return self.raised(_BORDER_MIX)

    def dimmed(self, ratio):
        """The text color moved ``ratio`` back toward the panel."""
        return self.blend(self.text, self.panel, ratio)

    def raised(self, ratio):
        """The panel color moved ``ratio`` toward the text, which steps a
        surface darker under a light palette and lighter under a dark one."""
        return self.blend(self.panel, self.text, ratio)

    def tinted(self, ratio):
        """The panel color moved ``ratio`` toward the accent."""
        return self.blend(self.panel, self.accent, ratio)


class RuleCatalog:
    """A catalog of style rules, one method per role it styles.

    A neighbourhood subclasses this and adds a static method per role, named
    for the thing it styles and returning the rules that color it. A widget
    asks :meth:`sheet` for the roles it draws, so a role defined once covers
    every page that draws it.
    """

    @classmethod
    def sheet(cls, widget, *roles):
        """The style sheet ``widget`` carries for the ``roles`` it draws."""
        shades = Shades(widget)
        return "".join(getattr(cls, role)(shades) for role in roles)


class PaletteStyle:
    """A mixin that keeps a widget's style sheet current with the palette.

    A class mixes this in ahead of the Qt widget it derives from, builds its
    style sheet in :meth:`_apply_style`, and calls it once its children exist;
    the mixin re-applies it whenever the application palette changes.
    """

    # Both palette events matter. The theme manager pairs a new application
    # palette with a fresh global style sheet, whose re-polish arrives here as
    # PaletteChange; under the system look it sets the palette alone, and
    # ApplicationPaletteChange is what carries that.
    #
    # The icons rasterize for the screen they are on, so a move between screens
    # of different scale has to re-render them just as a theme switch does.
    # Where Qt cannot report that move, the icons keep the scale they were
    # rendered at until the next palette change.
    _RESTYLE_EVENTS = (
        (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange)
        + ((_DPR_CHANGE_EVENT,) if _DPR_CHANGE_EVENT is not None else ()))

    def event(self, event):
        if event.type() in self._RESTYLE_EVENTS:
            self._apply_style()
            self.update()
        return super().event(event)

    def _apply_style(self):
        raise NotImplementedError


class PaletteStyled(PaletteStyle, QtWidgets.QWidget):
    """A plain widget whose child controls are styled from the palette."""

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
