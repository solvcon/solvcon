# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The chrome of the sub-windows: one bar drawn from the palette and its
height on every platform."""

import unittest

try:
    from PySide6.QtWidgets import QApplication
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon import apps  # noqa: F401
except ImportError:
    pilot = None


def _settle():
    for _ in range(10):
        QApplication.processEvents()


@unittest.skipIf(pilot is None or not pilot.enable, "pilot is disabled")
class ChromeTC(unittest.TestCase):

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

    def test_the_bar_takes_its_height_on_every_platform(self):
        self.sub.showNormal()
        _settle()
        self.assertEqual(24, self.sub.widget().geometry().top())
        self.sub.showMaximized()
        _settle()
        self.assertEqual(24, self.sub.widget().geometry().top())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
