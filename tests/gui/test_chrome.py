# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The bar of a sub-window as drawn on a live window: its colors in
either theme and clicks on its buttons."""

import unittest

from solvcon import testing

try:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent, QPalette
    from PySide6.QtWidgets import QApplication
    from solvcon import pilot
    from solvcon.pilot import _pilot_core, _style
    from solvcon.pilot.base import _gui
    from solvcon import apps  # noqa: F401
except ImportError:
    pilot = None


def _settle():
    for _ in range(10):
        QApplication.processEvents()


def _bar_color(widget):
    image = widget.grab().toImage()
    if image.isNull() or 0 == image.width():
        raise unittest.SkipTest("the window gave no frame")
    return image.pixelColor(image.width() // 2, 4)


def _distance(one, other):
    return max(abs(one.red() - other.red()),
               abs(one.green() - other.green()),
               abs(one.blue() - other.blue()))


@unittest.skipIf(pilot is None or not pilot.enable, "pilot is disabled")
@testing.requires_live_window
class ChromeLiveTC(unittest.TestCase):

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

    def test_the_bar_follows_the_theme(self):
        for theme in ('light', 'dark'):
            self.mgr.set_theme(theme)
            _settle()
            window = self.sub.palette().color(QPalette.Window)
            bar = _bar_color(self.sub)
            self.assertLess(abs(bar.lightness() - window.lightness()), 40,
                            theme)

    def test_the_bar_is_mixed_as_the_python_side_reads_the_mixes(self):
        # One source for both halves: the ground the C++ style paints is a
        # mix Python computes from the same metrics. Whether the window
        # counts as active depends on the window system, so either ground
        # will do.
        metrics = _style.chrome_metrics()
        for theme in ('light', 'dark'):
            self.mgr.set_theme(theme)
            _settle()
            shades = _style.Shades(self.sub)
            bar = _bar_color(self.sub)
            self.assertLessEqual(min(
                _distance(shades.raised(metrics[key]), bar)
                for key in ('ground_active_mix', 'ground_inactive_mix')),
                2, theme)

    def test_a_click_on_the_detach_button_moves_the_window_out(self):
        self.sub.showNormal()
        _settle()
        x, y, width, height = _pilot_core.chrome_detach_rect(
            self.sub.width())
        point = QPointF(x + width / 2, y + height / 2)
        for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            QApplication.postEvent(self.sub, QMouseEvent(
                kind, point, self.sub.mapToGlobal(point), Qt.LeftButton,
                Qt.LeftButton if QEvent.MouseButtonPress == kind
                else Qt.NoButton, Qt.NoModifier))
        _settle()
        self.assertTrue(self.app.detacher.detached)
        self.app.detacher.attach()
        _settle()

    def test_one_click_on_maximize_maximizes(self):
        # The pointer arrives over the button before the press, as a real
        # mouse does; a move region that still covers the button would
        # turn the press into the start of a drag.
        self.sub.showNormal()
        _settle()
        metrics = _style.chrome_metrics()
        side = metrics['button_side']
        # The maximize button is the second from the right end.
        right = (self.sub.width() - metrics['bar_margin']
                 - side - metrics['button_gap'])
        point = QPointF(right - side / 2, metrics['bar_height'] / 2)
        for kind, button, buttons in (
                (QEvent.MouseMove, Qt.NoButton, Qt.NoButton),
                (QEvent.MouseButtonPress, Qt.LeftButton, Qt.LeftButton),
                (QEvent.MouseButtonRelease, Qt.LeftButton, Qt.NoButton)):
            QApplication.postEvent(self.sub, QMouseEvent(
                kind, point, self.sub.mapToGlobal(point), button, buttons,
                Qt.NoModifier))
            _settle()
        self.assertTrue(self.sub.isMaximized())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
