# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Style and colors."""

import numpy as np

from PySide6.QtGui import QColor

from ... import pilot

__all__ = [
    'Ink',
    'Rules',
    'colormap',
]

# Viridis at nine stops; perceptually even and readable on both grounds.
_VIRIDIS = np.array([[0x44, 0x01, 0x54], [0x48, 0x28, 0x78],
                     [0x3e, 0x4a, 0x89], [0x31, 0x68, 0x8e],
                     [0x26, 0x82, 0x8e], [0x1f, 0x9e, 0x89],
                     [0x35, 0xb7, 0x79], [0x6e, 0xce, 0x58],
                     [0xfd, 0xe7, 0x25]], dtype='float64') / 255.0


def colormap(fractions):
    """Return the viridis colors, rows of red, green, and blue in [0, 1],
    at ``fractions`` in [0, 1]; a fraction that is not finite takes the
    low end."""
    fractions = np.nan_to_num(np.asarray(fractions, dtype='float64'),
                              nan=0.0, posinf=1.0, neginf=0.0)
    stops = np.linspace(0.0, 1.0, len(_VIRIDIS), dtype='float64')
    fractions = np.clip(fractions, 0.0, 1.0)
    return np.stack([np.interp(fractions, stops, _VIRIDIS[:, it])
                     for it in range(3)], axis=-1)


class Ink(pilot.Ink):
    """The shades the panels draw with on the base ground, and ``march``,
    the data color of the numerical march, which shows as a mark at 3:1
    against the base."""

    def __init__(self, widget):
        super().__init__(widget)
        self.march = pilot.legible(
            self, QColor(pilot.DATA[1 if self.dark else 0]), self.base,
            floor=pilot.GRAPHIC_CONTRAST)


class Rules(pilot.AppRules):
    """The style rules of the shock-tube window, one method per role it
    draws."""

    #: Without the room :class:`~solvcon.pilot.AppRules` keeps for an
    #: icon: these buttons carry none.
    PADDING = ("4px 14px", "3px 10px")

    @staticmethod
    def chips(shades):
        """Style the toggles under the panels: the chips that each turn one
        thing on or off, and the two-way switch whose halves join into one
        pill."""
        edge = pilot.legible(shades, shades.border, shades.base,
                             floor=pilot.GRAPHIC_CONTRAST)
        checked = pilot.legible(shades, shades.accent, shades.base,
                                floor=pilot.GRAPHIC_CONTRAST)
        muted = pilot.legible(shades, shades.muted, shades.base)
        return f"""
            QToolButton#chip, QToolButton#switch {{
                border: 1px solid {edge.name()};
                padding: 1px 9px;
                font-size: {pilot.SMALL_PX}px;
                color: {muted.name()};
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
                background: {shades.raised(pilot.HOVER_MIX).name()};
                color: {shades.on_base.name()};
            }}
            QToolButton#chip:checked, QToolButton#switch:checked {{
                background: {shades.tinted(pilot.TINT_MIX).name()};
                color: {shades.on_base.name()};
            }}
            QToolButton#chip:checked {{ border-color: {checked.name()}; }}
            QLabel#switchLabel {{
                font-size: {pilot.SMALL_PX}px;
                color: {muted.name()};
            }}
            """

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
