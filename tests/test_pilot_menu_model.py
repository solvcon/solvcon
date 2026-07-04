# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class MenuModelTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.model = self.mgr.menu_model
        # The model persists on the singleton across tests; start each test
        # from a bar holding only the built-in menus.
        self.model.clear()

    def _top_level(self):
        return [a.text() for a in self.mgr.mainWindow.menuBar().actions()]

    def _children(self, menu):
        return [a.text() for a in menu.actions()]

    def test_menu_creates_top_level_node(self):
        menu = self.model.menu("Alpha")
        self.assertEqual(menu.title(), "Alpha")
        self.assertIn("Alpha", self._top_level())

    def test_menu_is_idempotent(self):
        first = self.model.menu("Alpha")
        second = self.model.menu("Alpha")
        # The same path resolves to the same menu, not a duplicate on the bar.
        self.assertEqual(first, second)
        self.assertEqual(self._top_level().count("Alpha"), 1)

    def test_menu_creates_ancestors_on_demand(self):
        leaf = self.model.menu("Beta/Gamma")
        self.assertEqual(leaf.title(), "Gamma")
        self.assertIn("Beta", self._top_level())
        parent = self.model.menu("Beta")
        self.assertIn("Gamma", self._children(parent))

    def test_node_weight_orders_the_bar(self):
        # Declare out of order; the weights, not arrival, fix the bar order.
        self.model.menu("Later", weight=40)
        self.model.menu("Early", weight=10)
        self.model.menu("Middle", weight=25)
        top = [t for t in self._top_level() if t in ("Early", "Middle",
                                                     "Later")]
        self.assertEqual(top, ["Early", "Middle", "Later"])

    def test_equal_weight_keeps_arrival_order(self):
        self.model.menu("Uno", weight=10)
        self.model.menu("Dos", weight=10)
        self.model.menu("Tres", weight=10)
        top = [t for t in self._top_level() if t in ("Uno", "Dos", "Tres")]
        self.assertEqual(top, ["Uno", "Dos", "Tres"])

    def test_submenu_weight_orders_children(self):
        self.model.menu("Root", weight=90)
        self.model.menu("Root/Second", weight=20)
        self.model.menu("Root/First", weight=10)
        parent = self.model.menu("Root")
        self.assertEqual(self._children(parent), ["First", "Second"])

    def test_reweight_moves_an_existing_node(self):
        self.model.menu("A", weight=10)
        self.model.menu("B", weight=20)
        # Re-declaring with a larger weight repositions the same node.
        self.model.menu("A", weight=30)
        top = [t for t in self._top_level() if t in ("A", "B")]
        self.assertEqual(top, ["B", "A"])

    def test_clear_removes_model_menus(self):
        self.model.menu("Temp")
        self.assertIn("Temp", self._top_level())
        self.model.clear()
        self.assertNotIn("Temp", self._top_level())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
