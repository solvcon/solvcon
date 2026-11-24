# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The reflection window and its control column laid out on a real window.

A layout settles only on a window that is shown, so these checks need a
window surface: the column beside the window, as the pilot docks it, fits
a 1280 by 800 screen without scrolling whichever group is open.  The
passive widgets alone are checked in
``tests/apps/obsrefl/test_obsrefl_strip.py`` and ``test_obsrefl_column.py``.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtWidgets import QApplication, QHBoxLayout, QWidget
        from solvcon.apps.obsrefl import _column, _viewmodel, _window
except ImportError:
    pilot = None


def _settle():
    """Let the layouts that requested a pass take it."""
    for _ in range(5):
        QApplication.processEvents()


@testing.requires_live_window
class ReflectionWindowTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.host = QWidget()
        self.column = _column.ControlColumn()
        self.figure = QWidget()
        self.window = _window.ReflectionWindow(self.figure)
        layout = QHBoxLayout(self.host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.column)
        layout.addWidget(self.window, 1)
        self.window.strip.set_problem("Mach 3, 10 deg, 64 x 16")
        self.window.strip.show_state(pilot.panel.StripState(
            'paused', 120, 2000, 0.24, 4.0))
        self.column.card.show_results(_viewmodel.Results(
            field="density",
            zones=tuple(_viewmodel.ResultRow(
                f"zone {it}", _viewmodel.Cell("1.000"),
                _viewmodel.Cell("1.004"), _viewmodel.Cell("+0.42%", 'good'))
                for it in (1, 2, 3)),
            fits=(_viewmodel.ResultRow(
                "shock angle", _viewmodel.Cell("29.31"),
                _viewmodel.Cell("29.32"), _viewmodel.Cell("-0.03%")),
                  _viewmodel.ResultRow(
                "wall x", _viewmodel.Cell("2.850"),
                _viewmodel.Cell("2.851"), _viewmodel.Cell("-0.04%"))),
            rows=(("mass drift", "+1.2e-9"), ("CFL", "0.41-0.52")),
            caption=""))
        self.host.resize(1280, 800)
        self.host.show()
        _settle()

    def tearDown(self):
        self.host.close()
        self.host.deleteLater()
        _settle()

    def test_every_group_opens_in_the_room_left(self):
        for group in self.column.groups:
            with self.subTest(group=group.title):
                group._head.click()
                _settle()
                self.assertTrue(group.is_open)
                self.assertEqual(
                    0, self.column.verticalScrollBar().maximum())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
