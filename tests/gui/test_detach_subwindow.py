# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Move a sub-window out of the main window into a window of its own and
back.

A window of its own is a real top-level window, so these checks live in
the window lane.
"""

import unittest

from solvcon import testing

try:
    from PySide6.QtWidgets import QApplication, QLabel, QSizeGrip
    from solvcon import pilot
    from solvcon.pilot import base
    from solvcon.pilot.base import _gui
    from solvcon.pilot.panel._window_manager import WindowManager
except ImportError:
    pilot = None


def _settle():
    for _ in range(10):
        QApplication.processEvents()


class _PilotFixture(object):

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.mgr.show()
        self.area = self.mgr.mdiArea
        self.area.closeAllSubWindows()
        _settle()

    def tearDown(self):
        self.area.closeAllSubWindows()
        _settle()

    def _detach_action(self):
        return self.mgr.menu_model.action(WindowManager.DETACH_ID)

    def _refresh_menu(self):
        self.mgr.menu_model.menu("Window").aboutToShow.emit()


@testing.requires_live_window
class SubWindowDetacherTC(_PilotFixture, unittest.TestCase):

    def _subwindow(self, text="held"):
        label = QLabel(text)
        subwindow = self.mgr.addSubWindow(label)
        subwindow.setWindowTitle(text)
        subwindow.showMaximized()
        detacher = base.SubWindowDetacher(subwindow, self.mgr)
        _settle()
        return subwindow, label, detacher

    def test_the_widget_goes_out_and_comes_back(self):
        subwindow, label, detacher = self._subwindow()
        self.assertIs(detacher, base.SubWindowDetacher.of(subwindow))
        detacher.detach()
        _settle()
        self.assertTrue(detacher.detached)
        self.assertIs(detacher.window, label.window())
        self.assertFalse(subwindow.isVisible())
        # The window of its own follows the title of the sub-window.
        subwindow.setWindowTitle("renamed")
        self.assertEqual("renamed", detacher.window.windowTitle())
        detacher.attach()
        _settle()
        self.assertFalse(detacher.detached)
        self.assertIs(label, subwindow.widget())
        self.assertTrue(subwindow.isVisible())
        self.assertTrue(subwindow.isMaximized())

    def test_a_grip_stays_over_the_widget_that_comes_back(self):
        subwindow, label, detacher = self._subwindow()
        subwindow.showNormal()
        subwindow.resize(320, 240)
        self.mgr.addSubWindowGrip(subwindow)
        _settle()
        detacher.detach()
        _settle()
        # The sub-window comes back at its size, so no resize raises the
        # grip over the widget.
        detacher.attach()
        _settle()
        self.assertEqual((320, 240), subwindow.size().toTuple())
        corner = subwindow.rect().bottomRight()
        self.assertIs(subwindow.findChild(QSizeGrip),
                      subwindow.childAt(corner))

    def test_closing_the_window_of_its_own_moves_the_widget_back(self):
        subwindow, label, detacher = self._subwindow()
        closed = []
        subwindow.installEventFilter(
            base.SubWindowCloseFilter(lambda: closed.append(1), subwindow))
        detacher.detach()
        _settle()
        detacher.window.close()
        _settle()
        self.assertEqual([], closed)
        self.assertFalse(detacher.detached)
        self.assertIs(label, subwindow.widget())
        self.assertTrue(subwindow.isVisible())

    def test_closing_the_subwindow_while_out_takes_its_window(self):
        subwindow, label, detacher = self._subwindow()
        detacher.detach()
        _settle()
        window = detacher.window
        destroyed = []
        window.destroyed.connect(lambda: destroyed.append(1))
        subwindow.close()
        _settle()
        self.assertEqual([1], destroyed)
        self.assertNotIn(subwindow, self.area.subWindowList())

    def test_the_window_menu_moves_the_active_window(self):
        subwindow, label, detacher = self._subwindow("menu")
        plain = self.mgr.addSubWindow(QLabel("plain"))
        plain.show()
        _settle()
        # A sub-window without a detacher cannot move.
        self.area.setActiveSubWindow(plain)
        self._refresh_menu()
        self.assertFalse(self._detach_action().isEnabled())
        self.area.setActiveSubWindow(subwindow)
        self._refresh_menu()
        action = self._detach_action()
        self.assertTrue(action.isEnabled())
        self.assertEqual(WindowManager.DETACH_TEXT, action.text())
        action.trigger()
        _settle()
        self.assertTrue(detacher.detached)
        # The window out stays listed, and its entry brings it forward.
        self._refresh_menu()
        items = [a for a in self.mgr.menu_model.menu("Window").actions()
                 if a.objectName() == WindowManager.ITEM_ID]
        self.assertIn("menu", [a.text() for a in items])
        plain.close()
        _settle()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
