# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Two app windows open in the main window keep the area they fill.

The area keeps one sub-window maximized, and puts the one it leaves back
to its normal size, so these checks need the sub-windows on screen.
"""

import unittest

from solvcon import testing

try:
    from PySide6.QtCore import QMargins
    from PySide6.QtWidgets import QApplication
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon import apps  # noqa: F401
except ImportError:
    pilot = None


def _settle():
    for _ in range(10):
        QApplication.processEvents()


@testing.requires_live_window
class AppSubwindowSizeTC(unittest.TestCase):

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.mgr.show()
        self.area = self.mgr.mdiArea
        self.area.closeAllSubWindows()
        _settle()
        self.apps = [_gui.controller.apps[name]
                     for name in ('obsrefl', 'shocktube')]

    def tearDown(self):
        for app in self.apps:
            app.close()
        self.area.closeAllSubWindows()
        _settle()

    def test_a_second_app_leaves_the_first_filling_the_area(self):
        first, second = self.apps
        first.open()
        _settle()
        second.open()
        _settle()
        full = self.area.viewport().rect().size()
        self.assertFalse(first.subwindow.isMaximized())
        self.assertEqual(full, first.subwindow.size())
        # Back to the first, the second, opened over a maximized one, keeps
        # the area too.
        first.open()
        _settle()
        self.assertFalse(second.subwindow.isMaximized())
        self.assertEqual(full, second.subwindow.size())
        # A restore of the window in front is the user's, and keeps the
        # size the user gave it.
        smaller = full.shrunkBy(QMargins(0, 0, 20, 20))
        first.subwindow.showNormal()
        first.subwindow.resize(smaller)
        first.subwindow.showMaximized()
        _settle()
        first.subwindow.showNormal()
        _settle()
        self.assertEqual(smaller, first.subwindow.size())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
