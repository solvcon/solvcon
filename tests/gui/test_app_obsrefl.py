# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The oblique-shock reflection app in the live pilot window.

These checks need the pilot on screen: the 3D view in the figure draws only
on a live surface, a run marches and paints on the timer, and no pilot
shortcut takes the run keys.  The wiring that needs no window surface (the
menu item, the dock, closing) is checked in the fast lane, in
``tests/apps/obsrefl/test_obsrefl_app.py``, and the controller in
``tests/apps/obsrefl/test_obsrefl_controller.py``.
"""

import time
import unittest

from solvcon import testing

try:
    from PySide6.QtCore import QEvent, QEventLoop, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication, QDockWidget
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    # Import the apps as the entry point does, so they are registered
    # whichever test builds the controller first.
    from solvcon import apps  # noqa: F401
except ImportError:
    pilot = None


def _settle():
    """Let the layouts that requested a pass take it."""
    for _ in range(5):
        QApplication.processEvents()


def _pump(until, seconds=30.0):
    """Run the event loop until ``until()`` holds or the time runs out."""
    deadline = time.monotonic() + seconds
    while not until() and time.monotonic() < deadline:
        QApplication.processEvents(QEventLoop.AllEvents, 20)
        time.sleep(0.002)
    return until()


def _press(widget, key, text=""):
    # Posted rather than sent: a key sent into the pilot binary from a
    # test crashes it.
    QApplication.postEvent(widget, QKeyEvent(QEvent.KeyPress, key,
                                             Qt.NoModifier, text))
    QApplication.postEvent(widget, QKeyEvent(QEvent.KeyRelease, key,
                                             Qt.NoModifier, text))
    QApplication.processEvents()


def _colored(widget):
    """Return the count of pixels of a grab of ``widget`` that carry a
    color rather than a grey, skipping where the grab gives nothing."""
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


class _AppFixture(object):
    """Open the app on the pilot on screen, on a coarse mesh."""

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.mgr.show()
        self.area = self.mgr.mdiArea
        self.area.closeAllSubWindows()
        QApplication.processEvents()
        self.app = _gui.controller.apps['obsrefl']

    def tearDown(self):
        self.app.close()
        self.area.closeAllSubWindows()
        QApplication.processEvents()

    def _open(self, nx=10, ny=4):
        self.mgr.menu_model.action("app.obsrefl").trigger()
        _settle()
        numerics = self.app.column.numerics
        numerics.fields['nx'].setText(str(nx))
        numerics.fields['ny'].setText(str(ny))
        self.assertTrue(self.app.controller.reset())
        _settle()
        return self.app.controller


@testing.requires_live_window
class ReflectionAppTC(_AppFixture, unittest.TestCase):

    def test_a_run_marches_on_the_timer_and_finishes(self):
        ctl = self._open()
        self.app.column.numerics._cap.setValue(20)
        ctl.run_or_pause()
        self.assertTrue(_pump(lambda: 'finished' == ctl.state))
        self.assertEqual(20, ctl.session.step)
        strip = self.app.window.strip
        self.assertEqual(("Finished", 'good'), strip.pill())
        self.assertEqual("Run again", strip.buttons['run'].text())
        self.assertTrue(self.app.window.figure.caption.text().startswith(
            "density, step 20:"))

    def test_no_pilot_shortcut_takes_the_run_keys(self):
        # A posted key is offered to the shortcuts of the pilot first, as a
        # typed one is, so this is where a menu shortcut would steal it.
        ctl = self._open()
        window = self.app.window
        _press(window, Qt.Key_Space, " ")
        self.assertTrue(_pump(lambda: 'running' == ctl.state, 5.0))
        _press(window, Qt.Key_Space, " ")
        self.assertTrue(_pump(lambda: 'paused' == ctl.state, 5.0))
        step = ctl.session.step
        _press(window, Qt.Key_Right)
        self.assertTrue(_pump(lambda: ctl.session.step > step, 5.0))
        _press(window, Qt.Key_R, "r")
        self.assertTrue(_pump(lambda: 'ready' == ctl.state, 5.0))
        self.assertEqual(0, ctl.session.step)

    def test_the_figure_draws_the_field_in_the_window(self):
        # A dock an earlier test left open can squeeze the figure to a
        # sliver.
        for dock in self.mgr.mainWindow.findChildren(QDockWidget):
            if not dock.isHidden() and not dock.isFloating():
                dock.hide()
                self.addCleanup(dock.show)
        self._open()
        domain = self.app.window.figure.domain
        if not domain.available:
            self.skipTest("the pilot builds no 3D view")
        self.assertTrue(_pump(lambda: _colored(domain) > 50, 5.0))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
