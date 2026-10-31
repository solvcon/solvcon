# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Style and colors."""

import functools

import numpy as np

from PySide6.QtGui import QColor, QFont

from ... import pilot

__all__ = [
    'Ink',
    'Rules',
    'colormap',
    'number_font',
]

#: (light, dark) data color of the numerical march, of the Okabe-Ito
#: palette (M. Okabe and K. Ito, Color Universal Design, 2008,
#: https://jfly.uni-koeln.de/color/); the exact solution takes the ink.
MARCH = ('#0072B2', '#56B4E9')
#: (light, dark) color of the good tone of the state pill.
GOOD = ('#007A5A', '#3DDC97')
#: The contrast WCAG 2.1 asks of text (1.4.3), and of what shows where a
#: control is or what state it is in (1.4.11).
TEXT_CONTRAST = 4.5
GRAPHIC_CONTRAST = 3.0

# Viridis at nine stops; perceptually even and readable on both grounds.
_VIRIDIS = np.array([[0x44, 0x01, 0x54], [0x48, 0x28, 0x78],
                     [0x3e, 0x4a, 0x89], [0x31, 0x68, 0x8e],
                     [0x26, 0x82, 0x8e], [0x1f, 0x9e, 0x89],
                     [0x35, 0xb7, 0x79], [0x6e, 0xce, 0x58],
                     [0xfd, 0xe7, 0x25]], dtype='float64') / 255.0

_SMALL_PX = 11
_RADIUS = 6
# The steps pilot.Shades leaves to its users, each from the color a shade
# starts at toward the one it mixes in.
_HOVER_MIX = 0.07
_FILL_MIX = 0.12
_TINT_MIX = 0.16
_DEEPEN_MIX = 0.12
_GUIDE_MIX = 0.42
_GRID_MIX = (0.07, 0.085)

# Qt 6.7 added the font features; an older Qt keeps the figures the font
# sets its digits in.
_TABULAR = QFont.Tag("tnum") if hasattr(QFont, 'Tag') else None


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


def number_font(base, pixel_size=None):
    """Return a copy of ``base`` that sets digits in tabular figures, so a
    column of numbers keeps its alignment while the values change."""
    font = QFont(base)
    if pixel_size is not None:
        font.setPixelSize(pixel_size)
    if _TABULAR is not None:
        font.setFeature(_TABULAR, 1)
    return font


def contrast(one, other):
    """Return the contrast ratio of two colors as WCAG 2.1 defines it, from
    1 for a color on itself to 21 for black on white
    (https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio)."""
    def luminance(color):
        red, green, blue = (value / 12.92 if value <= 0.04045
                            else ((value + 0.055) / 1.055) ** 2.4
                            for value in color.getRgbF()[:3])
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue
    high, low = sorted((luminance(one), luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _ends(shades):
    """Return the palette colors a shade can step toward."""
    return (shades.text, shades.panel, shades.base, shades.on_base)


def _farthest(ends, grounds):
    """Return the one of ``ends`` that stands farthest from all
    ``grounds``."""
    return max(ends, key=lambda end: min(contrast(end, ground)
                                         for ground in grounds))


def legible(shades, color, *grounds, floor=TEXT_CONTRAST, steps=50):
    """Return ``color`` moved, in ``steps`` even steps, toward the palette
    color farthest from ``grounds`` just far enough to stand ``floor`` to 1
    against each, or unchanged where it already does."""
    return QColor.fromRgb(_legible(
        color.rgb(), tuple(ground.rgb() for ground in grounds),
        tuple(end.rgb() for end in _ends(shades)), floor, steps))


# Cached by the values of the colors, since a paint builds its shades anew
# each time it runs.
@functools.lru_cache(maxsize=256)
def _legible(color, grounds, ends, floor, steps):
    grounds = [QColor.fromRgb(ground) for ground in grounds]
    far = _farthest([QColor.fromRgb(end) for end in ends], grounds)
    for step in range(steps + 1):
        mixed = pilot.Shades.blend(QColor.fromRgb(color), far, step / steps)
        if min(contrast(mixed, ground) for ground in grounds) >= floor:
            return mixed.rgb()
    return far.rgb()


def _dark(shades):
    """Return whether ``shades`` come off a dark palette."""
    return shades.base.lightnessF() < 0.5


class Ink(pilot.Shades):
    """The shades the panels draw with on the base ground.

    Every color is read off the palette of the widget that draws it, or
    picked from a pair (light, dark) by the lightness of that palette, so a
    theme switch recolors the window without a hex value surviving the
    switch.  A color that draws text or a mark steps toward the far end of
    the palette until it holds its contrast against the ground.

    ``ink`` and ``muted`` read as text, at 4.5:1 against the base, and
    ``guide`` and ``march`` show as marks, at 3:1; ``frame`` and ``grid`` are
    decoration and hold no floor.
    """

    def __init__(self, widget):
        super().__init__(widget)
        self.dark = _dark(self)
        self.ink = legible(self, self.on_base, self.base)
        self.muted = legible(self, super().muted, self.base)
        self.guide = legible(self, self.raised(_GUIDE_MIX), self.base,
                             floor=GRAPHIC_CONTRAST)
        self.frame = self.border
        self.grid = self.raised(_GRID_MIX[1 if self.dark else 0])
        self.march = legible(self, QColor(MARCH[1 if self.dark else 0]),
                             self.base, floor=GRAPHIC_CONTRAST)


def _tone(shades, name):
    """Return the color of the tone ``name`` on the ground of ``shades``."""
    if 'good' == name:
        return QColor(GOOD[1 if _dark(shades) else 0])
    return shades.accent if 'accent' == name else shades.muted


class Rules(pilot.RuleCatalog):
    """The style rules of the shock-tube window, one method per role it
    draws."""

    @staticmethod
    def buttons(shades):
        """Style the primary accent action, whose fill deepens until its
        label reads, and the secondary outlined ones."""
        fill = legible(shades, shades.accent, shades.on_accent)
        deep = _farthest(_ends(shades), (shades.on_accent,))
        lift, press = (shades.blend(fill, deep, steps * _DEEPEN_MIX)
                       for steps in (1, 2))
        edge = legible(shades, shades.border, shades.base, shades.panel,
                       floor=GRAPHIC_CONTRAST)
        hover = shades.blend(shades.base, shades.on_base, _HOVER_MIX)
        sunken = shades.raised(_FILL_MIX)
        return f"""
            QPushButton#primary {{
                background: {fill.name()};
                color: {shades.on_accent.name()};
                border: none;
                border-radius: {_RADIUS}px;
                padding: 4px 14px;
                font-weight: 600;
                min-height: 20px;
            }}
            QPushButton#primary:hover {{ background: {lift.name()}; }}
            QPushButton#primary:pressed {{ background: {press.name()}; }}
            QPushButton#primary:disabled {{
                background: {sunken.name()};
                color: {shades.greyed.name()};
            }}
            QPushButton#secondary {{
                background: {shades.base.name()};
                color: {shades.on_base.name()};
                border: 1px solid {edge.name()};
                border-radius: {_RADIUS}px;
                padding: 3px 10px;
                min-height: 20px;
            }}
            QPushButton#secondary:hover {{ background: {hover.name()}; }}
            QPushButton#secondary:disabled {{
                color: {shades.greyed.name()};
                border-color: {sunken.name()};
            }}
            """

    @staticmethod
    def pill(shades):
        """Style the run-state pill, one look per tone: the tone tints the
        fill and colors the text, stepped until the text reads on it."""
        rules = [f"""
            QLabel#pill {{
                border-radius: 10px;
                padding: 1px 10px;
                font-weight: 600;
                font-size: {_SMALL_PX}px;
            }}
            """]
        for tone in ('muted', 'accent', 'good'):
            color = _tone(shades, tone)
            fill = shades.blend(shades.panel, color, _TINT_MIX)
            rules.append(f"""
            QLabel#pill[tone="{tone}"] {{
                background: {fill.name()};
                color: {legible(shades, color, fill).name()};
            }}
            """)
        return "".join(rules)

    @staticmethod
    def chips(shades):
        """Style the toggles under the panels: the chips that each turn one
        thing on or off, and the two-way switch whose halves join into one
        pill."""
        edge = legible(shades, shades.border, shades.base,
                       floor=GRAPHIC_CONTRAST)
        checked = legible(shades, shades.accent, shades.base,
                          floor=GRAPHIC_CONTRAST)
        muted = legible(shades, shades.muted, shades.base)
        return f"""
            QToolButton#chip, QToolButton#switch {{
                border: 1px solid {edge.name()};
                padding: 1px 9px;
                font-size: {_SMALL_PX}px;
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
                background: {shades.raised(_HOVER_MIX).name()};
                color: {shades.on_base.name()};
            }}
            QToolButton#chip:checked, QToolButton#switch:checked {{
                background: {shades.tinted(_TINT_MIX).name()};
                color: {shades.on_base.name()};
            }}
            QToolButton#chip:checked {{ border-color: {checked.name()}; }}
            QLabel#switchLabel {{
                font-size: {_SMALL_PX}px;
                color: {muted.name()};
            }}
            """

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
