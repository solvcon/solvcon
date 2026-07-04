# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import itertools
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6 import QtGui
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)

# A fresh tag per test keeps each test's menus and ids from colliding on the
# shared RManager singleton, whose menu model now owns the real bar; clearing
# the model would delete the built-in menus other tests rely on.
_TAG = itertools.count()


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class MenuModelTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.model = self.mgr.menu_model
        self.tag = "m%d_" % next(_TAG)

    def _top_level(self):
        return [a.text() for a in self.mgr.mainWindow.menuBar().actions()]

    def _children(self, menu):
        return [a.text() for a in menu.actions()]

    def _mine(self, *names):
        wanted = {self.tag + n for n in names}
        return [t for t in self._top_level() if t in wanted]

    def test_menu_creates_top_level_node(self):
        menu = self.model.menu(self.tag + "Alpha")
        self.assertEqual(menu.title(), self.tag + "Alpha")
        self.assertIn(self.tag + "Alpha", self._top_level())

    def test_menu_is_idempotent(self):
        first = self.model.menu(self.tag + "Alpha")
        second = self.model.menu(self.tag + "Alpha")
        # The same path resolves to the same menu, not a duplicate on the bar.
        self.assertEqual(first, second)
        self.assertEqual(self._top_level().count(self.tag + "Alpha"), 1)

    def test_menu_creates_ancestors_on_demand(self):
        leaf = self.model.menu(self.tag + "Beta/Gamma")
        self.assertEqual(leaf.title(), "Gamma")
        self.assertIn(self.tag + "Beta", self._top_level())
        parent = self.model.menu(self.tag + "Beta")
        self.assertIn("Gamma", self._children(parent))

    def test_node_weight_orders_the_bar(self):
        # Declare out of order; the weights, not arrival, fix the bar order.
        self.model.menu(self.tag + "Later", weight=40)
        self.model.menu(self.tag + "Early", weight=10)
        self.model.menu(self.tag + "Middle", weight=25)
        self.assertEqual(self._mine("Later", "Early", "Middle"),
                         [self.tag + "Early", self.tag + "Middle",
                          self.tag + "Later"])

    def test_equal_weight_keeps_arrival_order(self):
        self.model.menu(self.tag + "Uno", weight=10)
        self.model.menu(self.tag + "Dos", weight=10)
        self.model.menu(self.tag + "Tres", weight=10)
        self.assertEqual(self._mine("Uno", "Dos", "Tres"),
                         [self.tag + "Uno", self.tag + "Dos",
                          self.tag + "Tres"])

    def test_submenu_weight_orders_children(self):
        self.model.menu(self.tag + "Root", weight=90)
        self.model.menu(self.tag + "Root/Second", weight=20)
        self.model.menu(self.tag + "Root/First", weight=10)
        parent = self.model.menu(self.tag + "Root")
        self.assertEqual(self._children(parent), ["First", "Second"])

    def test_reweight_moves_an_existing_node(self):
        self.model.menu(self.tag + "A", weight=10)
        self.model.menu(self.tag + "B", weight=20)
        # Re-declaring with a larger weight repositions the same node.
        self.model.menu(self.tag + "A", weight=30)
        self.assertEqual(self._mine("A", "B"),
                         [self.tag + "B", self.tag + "A"])


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class MenuPlacementTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.model = self.mgr.menu_model
        self.tag = "p%d_" % next(_TAG)
        self.box = self.tag + "Box"

    def _act(self, text, oid):
        act = QtGui.QAction(text, self.mgr.mainWindow)
        act.setObjectName(oid)
        return act

    def _items(self, path):
        return [a.text() for a in self.model.menu(path).actions()]

    def test_place_orders_items_by_weight(self):
        self.model.place(self.box, self._act("Later", "b.later"), weight=40)
        self.model.place(self.box, self._act("Early", "b.early"), weight=10)
        self.model.place(self.box, self._act("Middle", "b.middle"), weight=25)
        self.assertEqual(self._items(self.box), ["Early", "Middle", "Later"])

    def test_place_equal_weight_keeps_arrival_order(self):
        self.model.place(self.box, self._act("First", "b.1"), weight=10)
        self.model.place(self.box, self._act("Second", "b.2"), weight=10)
        self.model.place(self.box, self._act("Third", "b.3"), weight=10)
        self.assertEqual(self._items(self.box), ["First", "Second", "Third"])

    def test_items_and_submenu_interleave_by_weight(self):
        self.model.place(self.box, self._act("Top", "b.top"), weight=5)
        self.model.menu(self.box + "/Sub", weight=10)
        self.model.place(self.box, self._act("Bottom", "b.bottom"), weight=20)
        self.assertEqual(self._items(self.box), ["Top", "Sub", "Bottom"])

    def test_action_id_round_trip(self):
        act = self._act("Named", self.tag + "named")
        self.model.place(self.box, act, weight=10)
        self.assertEqual(self.model.action(self.tag + "named"), act)
        self.assertIsNone(self.model.action(self.tag + "missing"))

    def test_remove_takes_item_out(self):
        self.model.place(self.box, self._act("Gone", self.tag + "gone"),
                         weight=10)
        self.assertIn("Gone", self._items(self.box))
        self.model.remove(self.tag + "gone")
        self.assertIsNone(self.model.action(self.tag + "gone"))
        self.assertNotIn("Gone", self._items(self.box))

    def test_remove_finds_item_in_a_submenu(self):
        self.model.place(self.box + "/Sub",
                         self._act("Deep", self.tag + "deep"), weight=10)
        self.assertIn("Deep", self._items(self.box + "/Sub"))
        self.model.remove(self.tag + "deep")
        self.assertNotIn("Deep", self._items(self.box + "/Sub"))

    def test_place_separator(self):
        self.model.place(self.box, self._act("A", "box.a"), weight=10)
        self.model.place_separator(self.box, weight=15)
        self.model.place(self.box, self._act("B", "box.b"), weight=20)
        actions = self.model.menu(self.box).actions()
        self.assertTrue(actions[1].isSeparator())
        self.assertEqual([actions[0].text(), actions[2].text()], ["A", "B"])

    def test_group_is_created_once(self):
        first = self.model.group(self.tag + "modes")
        second = self.model.group(self.tag + "modes")
        self.assertEqual(first, second)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
