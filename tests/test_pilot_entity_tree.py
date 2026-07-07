# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot import _entity_tree
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


def _crossing_world():
    """Two lines crossing at (1, 1): shape 0 and shape 1."""
    world = solvcon.WorldFp64()
    world.add_line(0, 0, 2, 2)
    world.add_line(0, 2, 2, 0)
    return world


def _all_item_texts(tree):
    """Every item's text in the tree, walked depth first."""
    texts = []

    def walk(item):
        texts.append(item.text(0))
        for it in range(item.childCount()):
            walk(item.child(it))

    for it in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(it))
    return texts


class _CountingWorld:
    """Wraps a real world, counting describe_state calls per level so a test
    can assert the panel caches instead of resweeping on every poll."""

    def __init__(self, real):
        self._real = real
        self.calls = {"basic": 0, "diagnostics": 0}

    def describe_state(self, level="basic"):
        self.calls[level] += 1
        return self._real.describe_state(level=level)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class EntityTreePanelTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def _panel_on(self):
        feature = _entity_tree.EntityTreePanel(mgr=self.mgr)
        feature.populate_menu()
        feature._action.setChecked(True)
        return feature

    def test_toggle_is_placed_under_view_panels(self):
        feature = _entity_tree.EntityTreePanel(mgr=self.mgr)
        feature.populate_menu()
        panels = self.mgr.menu_model.menu("View/Panels")
        self.assertIn(feature._action, panels.actions())

    def test_run_through(self):
        # End to end: the active canvas's world reaches the panel through the
        # R2DWidget.world binding and renders geometry plus diagnostics.
        world = _crossing_world()
        world.add_triangle(10, 0, 11, 0, 12, 0)  # collinear: a degeneracy
        widget = self.mgr.add2DWidget()
        widget.updateWorld(world)
        texts = _all_item_texts(self._panel_on()._panel._tree)
        self.assertIn("World (2D)", texts)
        self.assertIn("shape: 3", texts)
        self.assertIn("shape 0 crosses shape 1 at (1, 1)", texts)
        self.assertIn("shape 2: triangle (collinear)", texts)

    def test_level_selector_gates_diagnostics(self):
        tree = _entity_tree.EntityTreeWidget(_crossing_world())
        self.assertIn("Diagnostics", _all_item_texts(tree._tree))
        tree._levels["basic"].setChecked(True)
        texts = _all_item_texts(tree._tree)
        self.assertNotIn("Diagnostics", texts)
        self.assertIn("shape: 2", texts)  # geometry stays

    def test_diagnostics_cached_until_world_changes(self):
        real = _crossing_world()
        world = _CountingWorld(real)
        tree = _entity_tree.EntityTreeWidget()
        tree.set_world(world)
        tree.set_world(world)  # unchanged poll reuses the cache
        self.assertEqual(world.calls["diagnostics"], 1)
        real.add_line(5, 5, 6, 6)  # a real edit flips the fingerprint
        tree.set_world(world)
        self.assertEqual(world.calls["diagnostics"], 2)

    def test_no_world_shows_placeholder(self):
        self.mgr.add2DWidget()  # fresh canvas becomes current, no world
        root = self._panel_on()._panel._tree.topLevelItem(0)
        self.assertIn("No world", root.text(0))
        self.assertEqual(root.childCount(), 0)

    def test_poll_timer_runs_only_while_visible(self):
        feature = self._panel_on()
        self.assertEqual(feature._timer.interval(), feature._POLL_MS)
        feature._on_visibility_changed(True)
        self.assertTrue(feature._timer.isActive())
        feature._on_visibility_changed(False)
        self.assertFalse(feature._timer.isActive())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
