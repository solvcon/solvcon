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
    'AppRules',
    'DATA',
    'FILL_MIX',
    'GRAPHIC_CONTRAST',
    'HOVER_MIX',
    'Ink',
    'PaletteStyle',
    'PaletteStyled',
    'RuleCatalog',
    'Shades',
    'SMALL_PX',
    'TINT_MIX',
    'legible',
    'number_font',
    'set_style_property',
]

#: The contrast WCAG 2.1 asks of text (1.4.3), and of what shows where a
#: control is or what state it is in (1.4.11).
TEXT_CONTRAST = 4.5
GRAPHIC_CONTRAST = 3.0
#: (light, dark) color of the good tone.
GOOD = ('#007A5A', '#3DDC97')
#: (light, dark) color of a computed data series, the blue of the
#: Okabe-Ito palette (M. Okabe and K. Ito, Color Universal Design, 2008,
#: https://jfly.uni-koeln.de/color/).
DATA = ('#0072B2', '#56B4E9')


# Qt 6.6 added this event, and the theme backends still guard for older Qt.
# Naming it in a class body would fail the import of the whole pilot there.
_DPR_CHANGE_EVENT = getattr(QtCore.QEvent, 'DevicePixelRatioChange', None)

# How far the shades every page shares sit from the color they mix against.
_MUTED_MIX = 0.35
_GREYED_MIX = 0.6
_BORDER_MIX = 0.25
# The steps the shared roles take, each from the color a shade starts at
# toward the one it mixes in.
HOVER_MIX = 0.07
FILL_MIX = 0.12
TINT_MIX = 0.16
_DEEPEN_MIX = 0.12
_GUIDE_MIX = 0.42
_GRID_MIX = (0.07, 0.085)

#: The size of small text, in device-independent pixels.
SMALL_PX = 11
_RADIUS = 6

# Qt 6.7 added the font features; an older Qt keeps the figures the font
# sets its digits in.
_TABULAR = QtGui.QFont.Tag("tnum") if hasattr(QtGui.QFont, 'Tag') else None


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
        self.bright_text = palette.color(QtGui.QPalette.BrightText)
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


def set_style_property(widget, name, value):
    """Set the property ``name`` a style sheet selects on, and polish
    ``widget`` again so the sheet picks it up."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def number_font(base, pixel_size=None, weight=None):
    """Return a copy of ``base`` that sets digits in tabular figures, so a
    column of numbers keeps its alignment while the values change."""
    font = QtGui.QFont(base)
    if pixel_size is not None:
        font.setPixelSize(pixel_size)
    if weight is not None:
        font.setWeight(weight)
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


def _is_dark(shades):
    return shades.base.lightnessF() < 0.5


def _ends(shades):
    """Return the palette colors a shade can step toward."""
    return (shades.text, shades.panel, shades.base, shades.on_base)


def _farthest(ends, grounds):
    return max(ends, key=lambda end: min(contrast(end, ground)
                                         for ground in grounds))


def legible(shades, color, *grounds, floor=TEXT_CONTRAST, steps=50):
    """Return ``color`` moved, in ``steps`` even steps, toward the palette
    color farthest from ``grounds`` just far enough to stand ``floor`` to 1
    against each, or unchanged where it already does."""
    return QtGui.QColor.fromRgb(_legible(
        color.rgb(), tuple(ground.rgb() for ground in grounds),
        tuple(end.rgb() for end in _ends(shades)), floor, steps))


# Cached by the values of the colors, since a paint builds its shades anew
# each time it runs.
@functools.lru_cache(maxsize=256)
def _legible(color, grounds, ends, floor, steps):
    grounds = [QtGui.QColor.fromRgb(ground) for ground in grounds]
    far = _farthest([QtGui.QColor.fromRgb(end) for end in ends], grounds)
    for step in range(steps + 1):
        mixed = Shades.blend(QtGui.QColor.fromRgb(color), far, step / steps)
        if min(contrast(mixed, ground) for ground in grounds) >= floor:
            return mixed.rgb()
    return far.rgb()


class Ink(Shades):
    """The shades a widget draws with on the base ground.

    A color that draws text or a mark steps toward the far end of the
    palette until it holds its contrast against :meth:`grounds`: ``ink``
    and ``muted`` read as text, at 4.5:1, and ``guide`` shows as a mark, at
    3:1; ``frame`` and ``grid`` are decoration and hold no floor.

    :ivar dark: Whether the palette is a dark one.
    """

    def __init__(self, widget):
        super().__init__(widget)
        grounds = self.grounds()
        self.dark = _is_dark(self)
        self.ink = legible(self, self.on_base, *grounds)
        self.muted = legible(self, self.muted, *grounds)
        self.guide = legible(self, self.raised(_GUIDE_MIX), *grounds,
                             floor=GRAPHIC_CONTRAST)
        self.frame = self.border
        self.grid = self.raised(_GRID_MIX[1 if self.dark else 0])

    def grounds(self):
        """Return the grounds the text and the marks hold their floors
        against."""
        return (self.base,)


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


class AppRules(RuleCatalog):
    """The roles of the controls of an app window: the primary and the
    secondary buttons and the run-state pill.

    An app subclasses this for the rules of its own roles.
    """

    #: The padding of the primary and of the secondary buttons.
    PADDING = ("4px 14px", "3px 10px")
    #: The tones the pill takes, all that :meth:`tone` names.
    TONES = ('muted', 'accent', 'good', 'bad')

    @staticmethod
    def tone(shades, name):
        """Return the color of the tone ``name``, 'good', 'bad', 'accent',
        or any other for muted, on the ground of ``shades``, before it is
        stepped to read."""
        if 'good' == name:
            return QtGui.QColor(GOOD[1 if _is_dark(shades) else 0])
        if 'bad' == name:
            return shades.bright_text
        return shades.accent if 'accent' == name else shades.muted

    @classmethod
    def buttons(cls, shades):
        """Style the primary accent action, whose fill deepens until its
        label reads, and the secondary outlined ones, which an
        ``attention`` property outlines in the accent."""
        fill = legible(shades, shades.accent, shades.on_accent)
        deep = _farthest(_ends(shades), (shades.on_accent,))
        lift, press = (shades.blend(fill, deep, steps * _DEEPEN_MIX)
                       for steps in (1, 2))
        edge = legible(shades, shades.border, shades.base, shades.panel,
                       floor=GRAPHIC_CONTRAST)
        attention = legible(shades, shades.accent, shades.base)
        hover = shades.blend(shades.base, shades.on_base, HOVER_MIX)
        sunken = shades.raised(FILL_MIX)
        primary, secondary = cls.PADDING
        return f"""
            QPushButton#primary {{
                background: {fill.name()};
                color: {shades.on_accent.name()};
                border: none;
                border-radius: {_RADIUS}px;
                padding: {primary};
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
                padding: {secondary};
                min-height: 20px;
            }}
            QPushButton#secondary:hover {{ background: {hover.name()}; }}
            QPushButton#secondary:disabled {{
                color: {shades.greyed.name()};
                border-color: {sunken.name()};
            }}
            QPushButton#secondary[attention="true"]:enabled {{
                border: 1px solid {attention.name()};
                color: {attention.name()};
            }}
            """

    @classmethod
    def pill(cls, shades):
        """Style the run-state pill, one look per tone: the tone tints the
        fill and colors the text, stepped until the text reads on it."""
        rules = [f"""
            QLabel#pill {{
                border-radius: 10px;
                padding: 1px 10px;
                font-weight: 600;
                font-size: {SMALL_PX}px;
            }}
            """]
        for tone in cls.TONES:
            color = cls.tone(shades, tone)
            fill = shades.blend(shades.panel, color, TINT_MIX)
            rules.append(f"""
            QLabel#pill[tone="{tone}"] {{
                background: {fill.name()};
                color: {legible(shades, color, fill).name()};
            }}
            """)
        return "".join(rules)


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
