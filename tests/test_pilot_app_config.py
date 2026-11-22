# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the App config panel, which holds the config of the active app.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtWidgets import QApplication, QLabel
except ImportError:
    pilot = None


@testing.requires_pilot
class AppConfigTC(unittest.TestCase):

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.area = self.mgr.mdiArea
        self.panel = pilot.panel.AppConfig(mgr=self.mgr)
        self.panel.populate_menu()

    def tearDown(self):
        self.area.closeAllSubWindows()
        QApplication.processEvents()

    def test_the_panel_shows_the_config_of_the_active_app(self):
        subwindows = [self.area.addSubWindow(QLabel(name))
                      for name in ("one", "two")]
        configs = [QLabel("config one"), QLabel("config two")]
        for subwindow, config in zip(subwindows, configs):
            subwindow.show()
            self.panel.add(subwindow, config)
        stack = self.panel.dock.widget()
        self.assertIs(configs[1], stack.currentWidget())
        self.area.setActiveSubWindow(subwindows[0])
        self.assertIs(configs[0], stack.currentWidget())
        self.panel.remove(subwindows[0])
        self.assertEqual(1, stack.count())
        self.panel.remove(subwindows[1])
        self.assertIsNone(self.panel.dock)
        self.assertFalse(self.panel.action.isEnabled())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
