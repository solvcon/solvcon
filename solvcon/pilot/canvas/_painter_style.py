# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Palette-derived styling shared by the pieces of the Painter panel.

The design's greys are all a step from the panel color toward the text color,
so a piece that mixes its own shades follows a light/dark switch without
naming a single hex value.
"""

from PySide6 import QtCore, QtGui, QtWidgets

__all__ = [
    'blend',
    'shade',
    'rule',
    'PaletteStyled',
]


# Qt 6.6 added this event, and the theme backends still guard for older Qt.
# Naming it in a class body would fail the import of the whole pilot there.
_DPR_CHANGE_EVENT = getattr(QtCore.QEvent, 'DevicePixelRatioChange', None)


def rule(shape):
    """A hairline divider between two areas of the panel."""
    line = QtWidgets.QFrame()
    line.setFrameShape(shape)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


def blend(color, other, ratio):
    """Mix ``ratio`` of ``other`` into ``color``."""
    keep = 1.0 - ratio
    return QtGui.QColor(
        round(color.red() * keep + other.red() * ratio),
        round(color.green() * keep + other.green() * ratio),
        round(color.blue() * keep + other.blue() * ratio))


def shade(widget, ratio):
    """The panel color of ``widget`` moved ``ratio`` toward its text color.

    Blending toward the text is what makes a shade follow the theme: it steps
    darker under a light palette and lighter under a dark one, which is how the
    design separates the selector from the inspector and the checked tab from
    both.
    """
    palette = widget.palette()
    return blend(palette.color(QtGui.QPalette.Window),
                 palette.color(QtGui.QPalette.WindowText), ratio)


class PaletteStyled(QtWidgets.QWidget):
    """A widget whose child controls are styled from the palette.

    A subclass builds its style sheet in :meth:`_apply_style` and calls it once
    its children exist; the base re-applies it whenever the application palette
    changes, so the piece follows a light/dark switch.
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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
