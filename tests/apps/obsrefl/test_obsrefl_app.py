# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The reflection app wired into a pilot that is not on screen.

The menu item opens the reflection window as a maximized sub-window of the
pilot and puts the control column in the App config panel; closing the
window stops its run and takes the column along.  None of it needs a
window surface, so these checks build the app on a pilot that is not
shown and read the sub-windows, the docks, and the menu items back.
What needs the pilot on screen is checked in
``tests/gui/test_app_obsrefl.py``.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtWidgets import QApplication, QDockWidget
        from solvcon.apps.obsrefl import _app
except ImportError:
    pilot = None


def _delete_later_now():
    """Delete the objects whose deletion waits for the event loop."""
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@testing.requires_pilot
class AppWiringTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # The app is built on the bare manager rather than through the
        # controller, which later tests expect to build last.
        cls.mgr = pilot.RManager.instance.setUp()
        cls.app_config = pilot.panel.AppConfig(mgr=cls.mgr)
        cls.app_config.populate_menu()
        cls.app = _app.ObsreflApp(mgr=cls.mgr,
                                  app_config=cls.app_config)
        cls.app.populate_menu()
        cls.item = cls.app._mgr.menu_model.action("app.obsrefl")

    def setUp(self):
        self.main = self.mgr.mainWindow
        # Held: a sub-window reached through a dropped wrapper of the MDI
        # area goes with it.
        self.area = self.mgr.mdiArea
        self.area.closeAllSubWindows()
        QApplication.processEvents()

    def tearDown(self):
        self.app.close()
        self.area.closeAllSubWindows()
        QApplication.processEvents()
        _delete_later_now()

    def _open(self):
        self.app.open()
        QApplication.processEvents()

    def _docks(self):
        _delete_later_now()
        return [dock for dock in self.main.findChildren(QDockWidget)
                if self.app_config.TITLE == dock.windowTitle()]

    def test_the_menu_opens_one_maximized_window(self):
        self.assertEqual("Oblique shock reflection", self.item.text())
        self.assertIsNone(self.mgr.menu_model.action("panel.obsrefl"))
        self._open()
        subwin = self.app.subwindow
        self.assertIn(subwin, self.area.subWindowList())
        self.assertIs(self.app.window, subwin.widget())
        self.assertTrue(subwin.isMaximized())
        self.assertIn("64 x 16", subwin.windowTitle())
        # A second pick brings the open window back instead of a copy.
        subwin.showNormal()
        self._open()
        self.assertIs(subwin, self.app.subwindow)
        self.assertEqual(1, self.area.subWindowList().count(subwin))
        self.assertTrue(subwin.isMaximized())
        # The title follows the run the window shows.
        self.app.column.numerics.fields['nx'].setText("20")
        self.app.column.numerics.fields['ny'].setText("5")
        self.assertTrue(self.app.controller.reset())
        self.assertIn("20 x 5", subwin.windowTitle())

    def test_the_column_goes_in_the_app_config(self):
        self._open()
        dock = self.app_config.dock
        self.assertIs(self.app.column, dock.widget().currentWidget())
        self.assertEqual(Qt.LeftDockWidgetArea, self.main.dockWidgetArea(dock))
        self.assertFalse(dock.isHidden())
        self.assertEqual([dock], self._docks())
        self.app.close()
        self.assertIsNone(self.app_config.dock)
        self.assertEqual([], self._docks())

    def test_closing_the_window_stops_its_march(self):
        self._open()
        ctl = self.app.controller
        ctl.run_or_pause()
        self.assertTrue(ctl._timer.isActive())
        # The sub-window deletes the window on the next turn of the event
        # loop; the march has to stop before that, with the close.
        self.app.close()
        self.assertFalse(ctl._timer.isActive())
        self.assertIsNone(self.app.controller)

    def test_a_new_mesh_nudges_the_inspector(self):
        nudges = []
        self.app.mesh_updated = lambda: nudges.append(True)
        self.addCleanup(setattr, self.app, 'mesh_updated', None)
        self._open()
        self.assertTrue(nudges)
        del nudges[:]
        self.assertTrue(self.app.controller.reset())
        self.assertTrue(nudges)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
