# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The chrome of the sub-windows and of the dock panels: one bar drawn
from the palette, its buttons, and the note of a panel."""

import unittest

try:
    from PySide6.QtWidgets import QApplication, QDockWidget, QWidget
    from solvcon import pilot
    from solvcon.pilot import _pilot_core, _style, panel
    from solvcon.pilot.base import _gui
    from solvcon import apps  # noqa: F401
except ImportError:
    pilot = None


def _settle():
    for _ in range(10):
        QApplication.processEvents()


@unittest.skipIf(pilot is None or not pilot.enable, "pilot is disabled")
class ChromeTC(unittest.TestCase):

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

    def test_python_reads_the_metrics_the_style_paints_from(self):
        metrics = _style.chrome_metrics()
        self.assertEqual(24, metrics['bar_height'])
        self.assertEqual(_style.TEXT_CONTRAST, metrics['text_contrast'])
        self.assertEqual(_style.GRAPHIC_CONTRAST,
                         metrics['graphic_contrast'])
        self.assertTrue(_pilot_core.chrome_enabled())

    def test_the_bar_takes_its_height_on_every_platform(self):
        height = _style.chrome_metrics()['bar_height']
        self.sub.showNormal()
        _settle()
        self.assertEqual(height, self.sub.widget().geometry().top())
        self.sub.showMaximized()
        _settle()
        self.assertEqual(height, self.sub.widget().geometry().top())

    def test_the_square_bar_takes_no_frame_mask(self):
        # A frame mask rounds the corners on Fusion, and a sub-window resized
        # while hidden keeps the mask of its old size.
        sub = self.mgr.addSubWindow(QWidget())
        sub.resize(320, 240)
        self.assertTrue(sub.mask().isEmpty())

    def test_the_detach_button_sits_left_of_maximize_and_close(self):
        x, y, width, height = _pilot_core.chrome_detach_rect(400)
        metrics = _style.chrome_metrics()
        self.assertEqual(metrics['button_side'], width)
        self.assertEqual(metrics['button_side'], height)
        room = 400 - (x + width)
        self.assertEqual(metrics['bar_margin'] + 2 * (
            metrics['button_side'] + metrics['button_gap']), room)
        self.assertEqual((metrics['bar_height'] - height) // 2, y)

    def test_the_strip_leaves_the_problem_to_the_bar(self):
        self.assertFalse(self.sub.widget().strip.problem_shown())
        self.assertIn("Shock tube", self.sub.windowTitle())

    def test_a_window_of_its_own_shows_the_problem_in_its_strip(self):
        strip = self.sub.widget().strip
        self.app.detacher.detach()
        _settle()
        self.assertTrue(strip.problem_shown())
        self.app.detacher.attach()
        _settle()
        self.assertFalse(strip.problem_shown())

    def test_detach_heads_the_system_menu(self):
        first = self.sub.systemMenu().actions()[0]
        self.assertEqual("Detach to Own Window", first.text())

    def test_a_docked_panel_wears_the_bar_and_a_floating_one_does_not(self):
        main = self.mgr.mainWindow
        dock = next(d for d in main.findChildren(QDockWidget)
                    if "Console" == d.windowTitle())
        bar = dock.titleBarWidget()
        self.assertIsInstance(bar, panel.DockTitleBar)
        self.assertEqual("Console", dock.accessibleName())
        self.assertEqual("Close Console",
                         bar.buttons['close'].accessibleName())
        self.assertEqual(_style.chrome_metrics()['bar_height'],
                         bar.height())
        dock.setFloating(True)
        _settle()
        self.assertIsNone(dock.titleBarWidget())
        dock.setFloating(False)
        _settle()
        self.assertIs(bar, dock.titleBarWidget())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
