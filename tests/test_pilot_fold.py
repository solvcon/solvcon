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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
