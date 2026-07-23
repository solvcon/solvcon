# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from PySide6 import QtWidgets
    from PySide6.QtCore import QSettings, QStandardPaths
    from PySide6.QtGui import QPalette
    # Redirect QSettings to a throwaway location so the theme's persistence
    # does not touch the developer's real configuration during the tests.
    QStandardPaths.setTestModeEnabled(True)
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
        # The Theme menu also carries the look choices, so filter to the mode
        # actions by their object-name prefix.
        items = [a.text() for a in self.model.menu("View/Theme").actions()
                 if a.objectName().startswith("theme.mode_")]
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


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ThemeLookTC(unittest.TestCase):
    def setUp(self):
        # The theme menu is built in Python by the controller, so build the
        # full bar rather than only the C++ manager.
        self.mgr = _gui.controller.build()
        self.model = self.mgr.menu_model
        self.app = QtWidgets.QApplication.instance()

    def tearDown(self):
        # The manager is a shared singleton, so restore the controlled default
        # to keep the tests independent of the order they run in.
        self.mgr.set_look("curated")
        self.mgr.set_theme("system")

    def test_look_menu_lists_the_two_looks(self):
        items = [a.text() for a in self.model.menu("View/Theme").actions()
                 if a.objectName().startswith("theme.look_")]
        self.assertEqual(items, ["System colors", "Curated colors"])

    def test_look_group_is_exclusive_and_defaults_to_curated(self):
        group = self.model.group("theme.look")
        self.assertEqual(len(group.actions()), 2)
        self.assertTrue(group.isExclusive())
        self.assertEqual(group.checkedAction().objectName(),
                         "theme.look_curated")

    def test_both_looks_are_enabled_on_this_platform(self):
        for look in ("system", "curated"):
            self.assertTrue(
                self.model.action("theme.look_" + look).isEnabled())

    def test_curated_look_paints_the_curated_dark_window(self):
        self.mgr.set_look("curated")
        self.mgr.set_theme("dark")
        self.assertEqual(self.mgr.theme_look, "curated")
        self.assertLess(
            self.app.palette().color(QPalette.Window).lightness(), 100)

    def test_system_look_uses_the_native_style_palette(self):
        self.mgr.set_look("system")
        self.assertEqual(self.mgr.theme_look, "system")
        native = self.app.style().standardPalette().color(QPalette.Window)
        self.assertEqual(self.app.palette().color(QPalette.Window), native)

    def test_look_menu_check_follows_scripting(self):
        self.mgr.set_look("system")
        group = self.model.group("theme.look")
        self.assertEqual(group.checkedAction().objectName(),
                         "theme.look_system")


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ThemePolishTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.app = QtWidgets.QApplication.instance()

    def tearDown(self):
        self.mgr.set_look("curated")
        self.mgr.set_theme("system")

    def test_accent_role_matches_the_highlight(self):
        self.mgr.set_look("curated")
        self.mgr.set_theme("light")
        pal = self.app.palette()
        self.assertEqual(pal.color(QPalette.Accent),
                         pal.color(QPalette.Highlight))

    def test_curated_look_adds_a_supplemental_stylesheet(self):
        self.mgr.set_look("curated")
        self.assertIn("QToolTip", self.app.styleSheet())

    def test_system_look_clears_the_stylesheet(self):
        self.mgr.set_look("system")
        self.assertEqual(self.app.styleSheet(), "")

    def test_mode_and_look_persist_across_sessions(self):
        self.mgr.set_theme("dark")
        self.mgr.set_look("system")
        # A new session reads these back through the same store to start on the
        # last chosen theme.
        settings = QSettings("solvcon", "pilot")
        self.assertEqual(settings.value("theme/mode"), "dark")
        self.assertEqual(settings.value("theme/look"), "system")


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ThemeConsoleTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.app = QtWidgets.QApplication.instance()

    def tearDown(self):
        # The manager is a shared singleton, so restore the default mode to
        # keep the tests independent of the order they run in.
        self.mgr.set_theme("system")

    def _console_edits(self):
        # The console's transcript and command editors are the QTextEdit
        # instances in the pilot. Deliver the posted palette-change events and
        # polish each so its palette resolves against the current application
        # palette before it is read.
        self.app.processEvents()
        edits = [w for w in self.app.allWidgets()
                 if isinstance(w, QtWidgets.QTextEdit)]
        for edit in edits:
            edit.ensurePolished()
        return edits

    def test_console_does_not_force_its_own_text_palette(self):
        # The crux of the change: the console dropped its hardcoded
        # black-on-white palette, so it never overrides the Text or Base brush
        # and follows the application theme like every other widget.
        active = QPalette.Active
        for edit in self._console_edits():
            palette = edit.palette()
            self.assertFalse(palette.isBrushSet(active, QPalette.Text))
            self.assertFalse(palette.isBrushSet(active, QPalette.Base))

    def test_console_text_is_light_under_the_dark_theme(self):
        self.mgr.set_theme("dark")
        for edit in self._console_edits():
            self.assertGreater(
                edit.palette().color(QPalette.Text).lightness(), 150)

    def test_console_text_is_dark_under_the_light_theme(self):
        self.mgr.set_theme("light")
        for edit in self._console_edits():
            self.assertLess(
                edit.palette().color(QPalette.Text).lightness(), 100)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
