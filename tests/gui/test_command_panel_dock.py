# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the command panel docked in the pilot.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    from solvcon.pilot.panel import _command
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel
except ImportError:
    pilot = None


@testing.requires_live_window
class CommandPanelTC(unittest.TestCase):

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.feature = _command.CommandPanel(mgr=self.mgr)
        self.feature.populate_menu()
        self.feature._action.setChecked(True)
        self.panel = self.feature.panel

    def tearDown(self):
        self.panel.clear()

    def test_python_block_holds_its_output(self):
        block = self.feature.run_python("print('hello')\n6 * 7")
        texts = [label.text() for label in block.findChildren(QLabel)]
        self.assertIn("hello\n42", texts)

    def test_output_that_looks_like_markup_shows_as_written(self):
        block = self.feature.run_python("print('<b>bold</b>')")
        [label] = [label for label in block.findChildren(QLabel)
                   if label.text() == "<b>bold</b>"]
        self.assertEqual(label.textFormat(), Qt.PlainText)

    def test_a_dock_tabbed_behind_another_stays_open(self):
        dock = self.feature._dock
        dock.show()
        self.feature._action.setChecked(True)
        # What Qt says when another dock's tab comes to the front.
        dock.visibilityChanged.emit(False)
        self.assertTrue(self.feature._action.isChecked())
        self.assertFalse(dock.isHidden())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
