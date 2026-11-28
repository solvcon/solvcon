# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The bar of a sub-window as drawn on a live window: its colors in
either theme."""

import unittest

from solvcon import testing

try:
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon import apps  # noqa: F401
except ImportError:
    pilot = None


def _settle():
    for _ in range(10):
        QApplication.processEvents()


def _bar_color(widget):
    image = widget.grab().toImage()
    if image.isNull() or 0 == image.width():
        raise unittest.SkipTest("the window gave no frame")
    return image.pixelColor(image.width() // 2, 4)


@unittest.skipIf(pilot is None or not pilot.enable, "pilot is disabled")
@testing.requires_live_window
class ChromeLiveTC(unittest.TestCase):

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.mgr.show()
        self.area = self.mgr.mdiArea
        self.area.closeAllSubWindows()
        self.app = _gui.controller.apps["shocktube"]
        self.app.open()
        self.sub = self.app.subwindow
        _settle()

    def tearDown(self):
        self.mgr.set_theme('light')
        self.area.closeAllSubWindows()
        _settle()

    def test_the_bar_follows_the_theme(self):
        for theme in ('light', 'dark'):
            self.mgr.set_theme(theme)
            _settle()
            window = self.sub.palette().color(QPalette.Window)
            bar = _bar_color(self.sub)
            self.assertLess(abs(bar.lightness() - window.lightness()), 40,
                            theme)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
