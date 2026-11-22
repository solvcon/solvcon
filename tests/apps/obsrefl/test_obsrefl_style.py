# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The style sheets of the reflection window and its control column."""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtCore import qInstallMessageHandler
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QWidget
        from solvcon.apps.obsrefl import _column, _window
except ImportError:
    pilot = None

#: A light and a dark palette, as (window, text, base, bright text,
#: highlight, highlighted text).
PALETTES = (('#f2f2f3', '#1c1e21', '#ffffff', '#d32f2f', '#3574f0',
             '#ffffff'),
            ('#2d2f33', '#e6e6e7', '#232427', '#ff6b68', '#3d82e0',
             '#ffffff'))


def _palette(colors):
    palette = QPalette()
    for role, color in zip((QPalette.Window, QPalette.WindowText,
                            QPalette.Base, QPalette.BrightText,
                            QPalette.Highlight, QPalette.HighlightedText),
                           colors):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Text, QColor(colors[1]))
    return palette


@testing.requires_pilot
class StyledWidgetsTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_the_sheets_parse_and_follow_a_palette_switch(self):
        widgets = (_window.ReflectionWindow(QWidget()),
                   _column.ControlColumn())
        for widget in widgets:
            self.addCleanup(testing.dispose_widget, widget)
        messages = []
        qInstallMessageHandler(lambda mode, context, text:
                               messages.append(text))
        self.addCleanup(qInstallMessageHandler, None)

        for widget in widgets:
            sheets = []
            for colors in PALETTES:
                widget.setPalette(_palette(colors))
                for child in [widget] + widget.findChildren(QWidget):
                    child.ensurePolished()
                sheets.append(widget.styleSheet())
            with self.subTest(widget=type(widget).__name__):
                self.assertNotEqual(*sheets)
        self.assertEqual([], messages)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
