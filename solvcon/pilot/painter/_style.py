# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The style rules the Painter is drawn with, and the small conversions its pages
share.

The design draws the same handful of things on every page: an object row, a
boxed value, a section head, a pill. Each is a role here, so two pages that
draw one of them cannot drift apart.
"""

import math

from PySide6 import QtGui, QtWidgets

from .._style import RuleCatalog, Shades

__all__ = [
    'Parts',
    'Rules',
]

# Text sizes from the design, in device-independent pixels.
_NAME_PX = 12
_VALUE_PX = 11
_SMALL_PX = 10
_ENTRY_PX = 9

_RADIUS = 5
_BADGE_RADIUS = 4
_CHIP_RADIUS = 11
_ENTRY_RADIUS = 7

# The pill and the unchecked tab label move far enough for the pill to read
# as raised and the rest as secondary.
_PILL_MIX = 0.15
_SELECTED_MIX = 0.15
_HOVER_MIX = 0.08
_ENTRY_HOVER_MIX = 0.12
_TAB_LABEL_MIX = 0.45
_ENTRY_LABEL_MIX = 0.25
_ENTRY_OFF_MIX = 0.55


class Parts:
    """The widgets and measurements a page builds itself out of."""

    @staticmethod
    def rule(shape):
        """A hairline divider between two areas of the panel."""
        line = QtWidgets.QFrame()
        line.setFrameShape(shape)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    @staticmethod
    def mono_font():
        """The stand-in for the mono font the design gives every number."""
        return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)

    @staticmethod
    def obb_metrics(obb):
        """Return the center, width, and height of an oriented bounding box.

        The corners run top-left, top-right, bottom-right, bottom-left, so the
        two sides give the shape's own width and height rather than the wider
        span a rotated shape covers along the axes.
        """
        xs = obb[0::2]
        ys = obb[1::2]
        # Divided before summed: four corners far enough out add up past the
        # double range, and the center would come back infinite.
        return (sum(x / 4 for x in xs), sum(y / 4 for y in ys),
                math.hypot(xs[1] - xs[0], ys[1] - ys[0]),
                math.hypot(xs[3] - xs[0], ys[3] - ys[0]))


class Rules(RuleCatalog):
    """The Painter's roles, and the icon colors that must match them."""

    @staticmethod
    def name(shades):
        """The object name a row and the Design header both write."""
        return f"""
            QLabel#name {{
                font-size: {_NAME_PX}px;
            }}
            """

    @staticmethod
    def row(shades):
        """The object row the Layers list is filled with."""
        return f"""
            QFrame#row {{
                border-radius: {_RADIUS}px;
            }}
            QFrame#row[selected="true"] {{
                background: {shades.tinted(_SELECTED_MIX).name()};
            }}
            QLabel#metric {{
                font-size: {_SMALL_PX}px;
                color: {shades.muted.name()};
            }}
            QFrame#row[selected="true"] QLabel#metric {{
                color: {shades.accent.name()};
            }}
            """

    @staticmethod
    def box(shades):
        """The bordered box the design lays on the base color."""
        return f"""
            QFrame#field, QFrame#search, QFrame#box, QToolButton#footer {{
                border: 1px solid {shades.border.name()};
                border-radius: {_RADIUS}px;
                background: {shades.base.name()};
            }}
            """

    @staticmethod
    def editor(shades):
        """The text editor inside a box, which draws the border around it."""
        return f"""
            QLineEdit {{
                border: none;
                background: transparent;
                font-size: {_VALUE_PX}px;
            }}
            QLineEdit:read-only {{
                color: {shades.muted.name()};
            }}
            """

    @staticmethod
    def axis(shades):
        """The axis letter heading a position field."""
        return f"""
            QLabel#axis {{
                font-size: {_SMALL_PX}px;
                color: {shades.muted.name()};
            }}
            QLabel#axis:disabled {{
                color: {shades.greyed.name()};
            }}
            """

    @staticmethod
    def badge(shades):
        """The selection count beside the Design header."""
        return f"""
            QLabel#badge {{
                border-radius: {_BADGE_RADIUS}px;
                padding: 2px 6px;
                font-size: {_SMALL_PX}px;
                background: {shades.raised(_PILL_MIX).name()};
                color: {shades.muted.name()};
            }}
            """

    @staticmethod
    def section(shades):
        """The small uppercase head over one section of a page."""
        return f"""
            QLabel#section {{
                color: {shades.muted.name()};
            }}
            QLabel#section:disabled {{
                color: {shades.greyed.name()};
            }}
            """

    @staticmethod
    def readout(shades):
        """What a Canvas box measures, then the number it reads."""
        return f"""
            QLabel#label {{
                font-size: {_SMALL_PX}px;
                color: {shades.muted.name()};
            }}
            QLabel#value {{
                font-size: {_VALUE_PX}px;
            }}
            """

    @staticmethod
    def action(shades):
        """A Canvas button that acts on the view."""
        return f"""
            QPushButton#action {{
                border: 1px solid {shades.border.name()};
                border-radius: {_RADIUS}px;
                font-size: {_VALUE_PX}px;
                background: {shades.base.name()};
            }}
            QPushButton#action:hover {{
                background: {shades.raised(_HOVER_MIX).name()};
            }}
            QPushButton#action:disabled {{
                color: {shades.greyed.name()};
            }}
            """

    @staticmethod
    def chip(shades):
        """A Layers filter chip."""
        return f"""
            QPushButton#chip {{
                border: none;
                border-radius: {_CHIP_RADIUS}px;
                padding: 3px 8px;
                font-size: {_SMALL_PX}px;
                background: transparent;
                color: {shades.greyed.name()};
            }}
            QPushButton#chip:checked {{
                background: {shades.raised(_PILL_MIX).name()};
            }}
            """

    @staticmethod
    def empty(shades):
        """The line the Layers list shows while it holds no rows."""
        return f"""
            QLabel#empty {{
                font-size: {_NAME_PX}px;
                color: {shades.greyed.name()};
            }}
            """

    @staticmethod
    def count(shades):
        """The object count in the Layers footer."""
        return f"""
            QLabel#count {{
                font-size: {_SMALL_PX}px;
                color: {shades.muted.name()};
            }}
            """

    @staticmethod
    def tab(shades):
        """One segment of the inspector's page selector."""
        return f"""
            QPushButton {{
                border: none;
                border-radius: {_RADIUS}px;
                padding: 5px 0;
                font-size: {_VALUE_PX}px;
                background: transparent;
                color: {shades.dimmed(_TAB_LABEL_MIX).name()};
            }}
            QPushButton:hover {{
                color: {shades.text.name()};
            }}
            QPushButton:checked {{
                background: {shades.raised(_PILL_MIX).name()};
                color: {shades.text.name()};
                font-weight: 500;
            }}
            """

    @staticmethod
    def entry(shades):
        """One entry of the draw tool selector.

        The disabled rule comes last so a greyed entry keeps its color even
        under the pointer, where the hover rule would otherwise light it up.
        """
        return f"""
            QToolButton {{
                border: none;
                border-radius: {_ENTRY_RADIUS}px;
                padding: 7px 0 6px;
                font-size: {_ENTRY_PX}px;
                background: transparent;
                color: {shades.dimmed(_ENTRY_LABEL_MIX).name()};
            }}
            QToolButton:hover {{
                background: {shades.raised(_ENTRY_HOVER_MIX).name()};
                color: {shades.text.name()};
            }}
            QToolButton:checked {{
                background: {shades.accent.name()};
                color: {shades.on_accent.name()};
            }}
            QToolButton:disabled {{
                background: transparent;
                color: {shades.dimmed(_ENTRY_OFF_MIX).name()};
            }}
            """

    @staticmethod
    def row_icon_color(widget, selected):
        """The color a row on ``widget`` strokes its type icon in."""
        shades = Shades(widget)
        return shades.accent if selected else shades.muted

    @staticmethod
    def entry_icon_colors(widget):
        """The colors a draw tool entry on ``widget`` strokes its icon in.

        They come from here rather than from the selector so that an icon
        reads in the color the "entry" role gives the label beside it.

        :return: The label color, the greyed-out one, and the one an entry
            wears over the accent pill.
        :rtype: tuple
        """
        shades = Shades(widget)
        return (shades.dimmed(_ENTRY_LABEL_MIX), shades.dimmed(_ENTRY_OFF_MIX),
                shades.on_accent)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
