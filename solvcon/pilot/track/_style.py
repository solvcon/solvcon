# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The style rules the pilot's track panels are drawn with.

A rule here says only what the palette does not already give the widget.
The delegate paints its rows itself and takes its colors from
:func:`row_colors` rather than from a rule.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontDatabase

from .._style import RuleCatalog, Shades

__all__ = [
    'Rules',
    'font',
    'row_colors',
    'header_colors',
    'ROW_PAD',
    'ROW_GAP',
    'BODY_TEXT_PIXEL_SIZE',
    'CAPTION_TEXT_PIXEL_SIZE',
    'TOPIC_NAME_PIXEL_SIZE',
    'TOPIC_TYPE_PIXEL_SIZE',
]

BODY_TEXT_PIXEL_SIZE = 12
CAPTION_TEXT_PIXEL_SIZE = 11
TOPIC_NAME_PIXEL_SIZE = 12
TOPIC_TYPE_PIXEL_SIZE = 10
ROW_PAD = 4
ROW_GAP = 8

# How far the hairlines, the header, and the faint text sit from the
# surface each is mixed against.
_HAIRLINE_STEP_TOWARD_TEXT = 0.10
_HEADER_UNDERLINE_STEP_TOWARD_TEXT = 0.06
_CONTROL_BORDER_STEP_TOWARD_TEXT = 0.17
_HEADER_SURFACE_STEP_TOWARD_TEXT = 0.05
_HOVERED_ROW_STEP_TOWARD_ACCENT_ON_LIGHT = 0.06
_HOVERED_ROW_STEP_TOWARD_ACCENT_ON_DARK = 0.2
_CAPTION_STEP_TOWARD_PANEL = 0.5
_TOPIC_NAME_COLOR_ON_LIGHT = QColor(Qt.black)
_TOPIC_NAME_COLOR_ON_DARK = QColor(Qt.white)
_TOPIC_TYPE_STEP_TOWARD_SURFACE_ON_LIGHT = 0.5
_TOPIC_TYPE_STEP_TOWARD_SURFACE_ON_DARK = 0.5
_SELECTED_TYPE_STEP_TOWARD_ACCENT = 0.2
_BAR_SURFACE_STEP_TOWARD_TEXT = 0.02
_ALTERNATE_ROW_STEP_TOWARD_TEXT = 0.02
_BADGE_SURFACE_STEP_TOWARD_TEXT = 0.08


def _is_dark(shades):
    """Whether ``shades`` come from a palette with light text on dark."""
    return shades.base.lightness() < shades.on_base.lightness()


def font(px, mono=False, bold=False):
    """The system font at ``px``, fixed-width when ``mono``."""
    got = QFontDatabase.systemFont(
        QFontDatabase.FixedFont if mono else QFontDatabase.GeneralFont)
    got.setPixelSize(px)
    got.setBold(bold)
    return got


def row_colors(widget, selected):
    """The ``(name, type)`` colors a topic row paints in.

    A selected row sits on the accent and a plain one on the list surface,
    so the type line is faded toward whichever of the two is behind it.
    """
    shades = Shades(widget)
    if selected:
        name = shades.on_accent
        type_ = Shades.blend(name, shades.accent,
                             _SELECTED_TYPE_STEP_TOWARD_ACCENT)
        return name, type_
    if _is_dark(shades):
        name = _TOPIC_NAME_COLOR_ON_DARK
        type_step = _TOPIC_TYPE_STEP_TOWARD_SURFACE_ON_DARK
    else:
        name = _TOPIC_NAME_COLOR_ON_LIGHT
        type_step = _TOPIC_TYPE_STEP_TOWARD_SURFACE_ON_LIGHT
    type_ = Shades.blend(shades.on_base, shades.base, type_step)
    return name, type_


def _header_surface(shades):
    return Shades.blend(shades.base, shades.on_base,
                        _HEADER_SURFACE_STEP_TOWARD_TEXT)


def header_colors(widget):
    """The ``(surface, line)`` colors a column header paints in."""
    shades = Shades(widget)
    return _header_surface(shades), shades.raised(_HAIRLINE_STEP_TOWARD_TEXT)


def _control(shades, selector, radius, padding):
    """A control on the base surface: the button and the field share it."""
    border = shades.raised(_CONTROL_BORDER_STEP_TOWARD_TEXT)
    return f"""
        {selector} {{
            background: {shades.base.name()};
            color: {shades.on_base.name()};
            border: 1px solid {border.name()};
            border-radius: {radius}px;
            padding: {padding};
        }}
        """


class Rules(RuleCatalog):
    """One method per role the pilot's track panels draw."""

    @staticmethod
    def label(shades):
        """The key of a summary row and the header of a topic column."""
        return f"QLabel {{ color: {shades.muted.name()}; }}"

    @staticmethod
    def caption(shades):
        """The heading the topic list is filed under."""
        color = shades.dimmed(_CAPTION_STEP_TOWARD_PANEL)
        return (f"QLabel {{ color: {color.name()};"
                f" letter-spacing: 1px; }}")

    @staticmethod
    def faint(shades):
        """A hint the reader can do without."""
        return f"QLabel {{ color: {shades.greyed.name()}; }}"

    @staticmethod
    def badge(shades):
        """The schema type beside the topic name."""
        line = shades.raised(_HAIRLINE_STEP_TOWARD_TEXT)
        chip = shades.raised(_BADGE_SURFACE_STEP_TOWARD_TEXT)
        return f"""
            QLabel {{
                color: {shades.muted.name()};
                background: {chip.name()};
                border: 1px solid {line.name()};
                border-radius: 4px;
                padding: 1px 6px;
            }}
            """

    @staticmethod
    def bar(shades):
        """The strips over and under the body of the main window."""
        line = shades.raised(_HEADER_UNDERLINE_STEP_TOWARD_TEXT)
        surface = shades.raised(_BAR_SURFACE_STEP_TOWARD_TEXT)
        return f"""
            QWidget#toolbar, QWidget#footer {{
                background: {surface.name()};
            }}
            QWidget#toolbar {{ border-bottom: 1px solid {line.name()}; }}
            QWidget#footer {{ border-top: 1px solid {line.name()}; }}
            """

    @staticmethod
    def field(shades):
        """The field that takes the page number."""
        return _control(shades, "QLineEdit", 4, "1px 4px")

    @staticmethod
    def table(shades):
        """The paged table of one topic; the header paints itself."""
        alternate = Shades.blend(shades.base, shades.on_base,
                                 _ALTERNATE_ROW_STEP_TOWARD_TEXT)
        return f"""
            QTableView {{
                background: {shades.base.name()};
                alternate-background-color: {alternate.name()};
                border: none;
            }}
            QTableView::item:selected {{
                background-color: {shades.accent.name()};
                color: {shades.on_accent.name()};
            }}
            """

    @staticmethod
    def rule(shades):
        """The hairline the summary is closed with."""
        line = shades.raised(_HAIRLINE_STEP_TOWARD_TEXT)
        return f"QFrame {{ background: {line.name()}; }}"

    @staticmethod
    def button(shades):
        """The control the file dialog is opened from."""
        return _control(shades, "QPushButton", 5, "2px 12px")

    @staticmethod
    def topics(shades):
        """The box the topic list is read in.

        The list is transparent so that the box paints the surface under
        every row, which leaves the rounded corners to the box alone.
        """
        line = shades.raised(_HAIRLINE_STEP_TOWARD_TEXT)
        underline = shades.raised(_HEADER_UNDERLINE_STEP_TOWARD_TEXT)
        head = _header_surface(shades)
        if _is_dark(shades):
            hover_step = _HOVERED_ROW_STEP_TOWARD_ACCENT_ON_DARK
        else:
            hover_step = _HOVERED_ROW_STEP_TOWARD_ACCENT_ON_LIGHT
        hover = Shades.blend(shades.base, shades.accent, hover_step)
        return f"""
            QFrame#topics {{
                background: {shades.base.name()};
                border: 1px solid {line.name()};
                border-radius: 4px;
            }}
            QWidget#header {{
                background: {head.name()};
                border-bottom: 1px solid {underline.name()};
            }}
            QListWidget {{
                background: transparent;
                border: none;
            }}
            QListWidget::item:hover {{
                background-color: {hover.name()};
            }}
            QListWidget::item:selected {{
                background-color: {shades.accent.name()};
            }}
            """

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
