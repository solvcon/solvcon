# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot import _gui
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


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ThemeMenuTC(unittest.TestCase):
    def setUp(self):
        # The theme menu is built in Python by the controller, so build the
        # full bar rather than only the C++ manager.
        self.mgr = _gui.controller.build()
        self.model = self.mgr.menu_model

    def tearDown(self):
        # The manager is a shared singleton, so restore the default mode to
        # keep the tests independent of the order they run in.
        self.mgr.set_theme("system")

    def _window_lightness(self):
        app = QtWidgets.QApplication.instance()
        return app.palette().color(QPalette.Window).lightness()

    def test_theme_menu_lists_the_three_modes(self):
        items = [a.text() for a in self.model.menu("View/Theme").actions()]
        self.assertEqual(items, ["Follow system", "Light", "Dark"])

    def test_theme_group_is_exclusive_and_defaults_to_system(self):
        group = self.model.group("theme.mode")
        self.assertEqual(len(group.actions()), 3)
        self.assertTrue(group.isExclusive())
        checked = group.checkedAction()
        self.assertIsNotNone(checked)
        self.assertEqual(checked.objectName(), "theme.mode_system")

    def test_honored_modes_are_enabled_on_this_platform(self):
        # Linux, macOS, and Windows all follow the system and can force a
        # variant, so every mode is offered rather than greyed.
        for mode in ("system", "light", "dark"):
            action = self.model.action("theme.mode_" + mode)
            self.assertTrue(action.isEnabled())

    def test_dark_action_paints_a_dark_window(self):
        self.model.action("theme.mode_dark").trigger()
        self.assertEqual(self.mgr.theme_mode, "dark")
        self.assertEqual(self.mgr.theme_variant, "dark")
        self.assertLess(self._window_lightness(), 100)

    def test_light_action_paints_a_light_window(self):
        self.model.action("theme.mode_light").trigger()
        self.assertEqual(self.mgr.theme_mode, "light")
        self.assertEqual(self.mgr.theme_variant, "light")
        self.assertGreater(self._window_lightness(), 150)

    def test_set_theme_keeps_the_menu_check_in_step(self):
        self.mgr.set_theme("dark")
        group = self.model.group("theme.mode")
        # Scripting the theme should move the radio check too, so the menu
        # never disagrees with the applied palette.
        checked = group.checkedAction()
        self.assertEqual(checked.objectName(), "theme.mode_dark")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
