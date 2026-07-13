# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6 import QtWidgets
    from PySide6.QtGui import QPalette
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ThemeManagerTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def tearDown(self):
        # The manager is a shared singleton, so restore the default mode to
        # keep the tests independent of the order they run in.
        self.mgr.set_theme("system")

    def _window_lightness(self):
        app = QtWidgets.QApplication.instance()
        return app.palette().color(QPalette.Window).lightness()

    def test_platform_is_recognized(self):
        self.assertIn(self.mgr.theme_platform, ("linux", "mac", "windows"))

    def test_dark_mode_paints_a_dark_window(self):
        self.mgr.set_theme("dark")
        self.assertEqual(self.mgr.theme_mode, "dark")
        self.assertEqual(self.mgr.theme_variant, "dark")
        self.assertLess(self._window_lightness(), 100)

    def test_light_mode_paints_a_light_window(self):
        self.mgr.set_theme("light")
        self.assertEqual(self.mgr.theme_mode, "light")
        self.assertEqual(self.mgr.theme_variant, "light")
        self.assertGreater(self._window_lightness(), 150)

    def test_switching_back_and_forth_repaints_each_time(self):
        self.mgr.set_theme("dark")
        dark = self._window_lightness()
        self.mgr.set_theme("light")
        light = self._window_lightness()
        self.assertLess(dark, light)

    def test_unknown_mode_falls_back_to_system(self):
        self.mgr.set_theme("solarized")
        self.assertEqual(self.mgr.theme_mode, "system")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
