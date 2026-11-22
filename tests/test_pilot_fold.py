# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the fold group of a config column, built without a window.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.pilot import panel
except ImportError:
    pilot = None


@testing.requires_pilot
class FoldGroupTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_a_group_folds_itself_without_an_owner(self):
        group = panel.FoldGroup("Grid")
        self.assertTrue(group.is_open)
        group._head.click()
        self.assertFalse(group.is_open)
        self.assertTrue(group.body.isHidden())
        # An owner takes the click instead, to fold the other groups.
        clicked = []
        group.toggled = clicked.append
        group._head.click()
        self.assertEqual([group], clicked)
        self.assertFalse(group.is_open)

    def test_a_summary_marks_edits_that_wait(self):
        group = panel.FoldGroup("Grid")
        group.set_summary("201 on [-10, 10]", dirty=True)
        self.assertEqual("201 on [-10, 10]", group.summary())
        self.assertEqual("true", group._summary.property("dirty"))
        group.set_summary("201 on [-10, 10]")
        self.assertEqual("false", group._summary.property("dirty"))

    def test_a_long_summary_elides_short_of_the_title(self):
        group = panel.FoldGroup("Grid")
        group.GAP = 28
        group.resize(200, 30)
        text = "Mach 3, 10 deg, gamma 1.4, density 1, pressure 1  (pending)"
        group.set_summary(text)
        self.assertEqual(text, group.summary())
        self.assertEqual(text, group._head.toolTip())
        self.assertTrue(group._summary.text().endswith(chr(0x2026)))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
