# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The style sheets of the reflection window and its control column."""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtCore import Qt, qInstallMessageHandler
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QApplication, QWidget
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

    def test_a_theme_switch_redraws_the_strip_icons(self):
        # The icons are pixmaps drawn in the ink of the palette, which no
        # sheet recolors, so only restyle() can follow the switch.
        hints = QApplication.styleHints()
        if not hasattr(hints, 'setColorScheme'):
            self.skipTest("the platform does not switch the theme")
        window = _window.ReflectionWindow(QWidget())
        self.addCleanup(testing.dispose_widget, window)
        self.addCleanup(hints.unsetColorScheme)
        step = window.strip.buttons['step']
        images = []
        for scheme in (Qt.ColorScheme.Light, Qt.ColorScheme.Dark):
            hints.setColorScheme(scheme)
            QApplication.processEvents()
            images.append(step.icon().pixmap(16, 16).toImage())
        if QApplication.palette().window().color().lightness() > 128:
            self.skipTest("the platform kept the light palette")
        self.assertNotEqual(*images)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
