# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6 import QtGui
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


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class MenuPlacementTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.model = self.mgr.menu_model
        self.model.clear()

    def _act(self, text, oid):
        act = QtGui.QAction(text, self.mgr.mainWindow)
        act.setObjectName(oid)
        return act

    def _items(self, path):
        return [a.text() for a in self.model.menu(path).actions()]

    def test_place_orders_items_by_weight(self):
        self.model.place("Box", self._act("Later", "b.later"), weight=40)
        self.model.place("Box", self._act("Early", "b.early"), weight=10)
        self.model.place("Box", self._act("Middle", "b.middle"), weight=25)
        self.assertEqual(self._items("Box"), ["Early", "Middle", "Later"])

    def test_place_equal_weight_keeps_arrival_order(self):
        self.model.place("Box", self._act("First", "b.1"), weight=10)
        self.model.place("Box", self._act("Second", "b.2"), weight=10)
        self.model.place("Box", self._act("Third", "b.3"), weight=10)
        self.assertEqual(self._items("Box"), ["First", "Second", "Third"])

    def test_items_and_submenu_interleave_by_weight(self):
        self.model.place("Box", self._act("Top", "b.top"), weight=5)
        self.model.menu("Box/Sub", weight=10)
        self.model.place("Box", self._act("Bottom", "b.bottom"), weight=20)
        self.assertEqual(self._items("Box"), ["Top", "Sub", "Bottom"])

    def test_action_id_round_trip(self):
        act = self._act("Named", "box.named")
        self.model.place("Box", act, weight=10)
        self.assertEqual(self.model.action("box.named"), act)
        self.assertIsNone(self.model.action("box.missing"))

    def test_remove_takes_item_out(self):
        self.model.place("Box", self._act("Gone", "box.gone"), weight=10)
        self.assertIn("Gone", self._items("Box"))
        self.model.remove("box.gone")
        self.assertIsNone(self.model.action("box.gone"))
        self.assertNotIn("Gone", self._items("Box"))

    def test_remove_finds_item_in_a_submenu(self):
        self.model.place("Box/Sub", self._act("Deep", "box.deep"), weight=10)
        self.assertIn("Deep", self._items("Box/Sub"))
        self.model.remove("box.deep")
        self.assertNotIn("Deep", self._items("Box/Sub"))

    def test_place_separator(self):
        self.model.place("Box", self._act("A", "box.a"), weight=10)
        self.model.place_separator("Box", weight=15)
        self.model.place("Box", self._act("B", "box.b"), weight=20)
        actions = self.model.menu("Box").actions()
        self.assertTrue(actions[1].isSeparator())
        self.assertEqual([actions[0].text(), actions[2].text()], ["A", "B"])

    def test_group_is_created_once(self):
        first = self.model.group("modes")
        second = self.model.group("modes")
        self.assertEqual(first, second)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
