# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Move a sub-window out of the main window into a window of its own and
back.

A window of its own is a real top-level window, and the reflection app
draws its domain only on a live surface, so these checks live in the
window lane.
"""

import unittest

from solvcon import testing

try:
    from PySide6.QtWidgets import QApplication, QLabel, QSizeGrip
    from solvcon import pilot
    from solvcon.pilot import base
    from solvcon.pilot.base import _gui
    from solvcon.pilot.panel._window_manager import WindowManager
    from solvcon import apps  # noqa: F401
except ImportError:
    pilot = None


def _settle():
    for _ in range(10):
        QApplication.processEvents()


def _colored(widget):
    """Count the pixels of a grab of ``widget`` that carry a color rather
    than a grey, skipping where the grab gives nothing."""
    image = widget.grab().toImage()
    if image.isNull() or 0 == image.width():
        raise unittest.SkipTest("the window gave no frame")
    count = 0
    for iy in range(0, image.height(), 3):
        for ix in range(0, image.width(), 3):
            color = image.pixelColor(ix, iy)
            channels = (color.red(), color.green(), color.blue())
            count += max(channels) - min(channels) > 60
    return count


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


@testing.requires_live_window
class DetachedReflectionTC(_PilotFixture, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.app = _gui.controller.apps['obsrefl']
        self.app.open()
        _settle()

    def tearDown(self):
        self.app.close()
        super().tearDown()

    def test_the_app_window_has_a_size_grip(self):
        grips = self.app.subwindow.findChildren(QSizeGrip)
        self.assertEqual(1, len(grips))
        self.assertIs(self.app.subwindow, grips[0].parent())

    def test_the_domain_draws_in_a_window_of_its_own(self):
        self.app.detacher.detach()
        self.app.controller.step()
        _settle()
        self.assertIs(self.app.detacher.window, self.app.window.window())
        self.assertGreater(_colored(self.app.window.figure.domain.widget),
                           0)
        # A second pick of the menu item finds the window where it is.
        self.app.open()
        self.assertTrue(self.app.detacher.detached)
        # Out of the area, the window still shows its config in front.
        self.app.detacher.activated.emit()
        self.assertIs(self.app.column,
                      _gui.controller.app_config._stack.currentWidget())

    def test_closing_a_window_out_moves_it_back_and_keeps_the_run(self):
        controller = self.app.controller
        self.app.detacher.detach()
        controller.run()
        _settle()
        self.app.detacher.window.close()
        _settle()
        self.assertFalse(self.app.detacher.detached)
        self.assertTrue(self.app.subwindow.isMaximized())
        self.assertTrue(controller._timer.isActive())
        self.assertGreater(_colored(self.app.window.figure.domain.widget),
                           0)

    def test_closing_the_app_while_out_stops_the_run(self):
        controller = self.app.controller
        self.app.detacher.detach()
        controller.run()
        _settle()
        self.assertTrue(controller._timer.isActive())
        # The domain view goes with its window of its own; moving it back
        # into the closing sub-window crashed.
        self.app.close()
        # The timer goes with the window, so it is read before that.
        self.assertFalse(controller._timer.isActive())
        self.assertIsNone(self.app.subwindow)
        _settle()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
