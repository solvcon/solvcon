# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6 import QtWidgets
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class CameraMenuTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.model = self.mgr.menu_model

    def test_camera_move_items_are_data(self):
        items = [a.text()
                 for a in self.model.menu("View/Camera move").actions()]
        self.assertEqual(items[0], "Reset camera")
        self.assertEqual(len(items), 11)
        # The false "(key)" hints are gone from every label.
        for text in items:
            self.assertNotIn("(", text)

    def test_reset_carries_the_escape_shortcut(self):
        reset = self.model.action("camera.reset")
        self.assertIsNotNone(reset)
        self.assertEqual(reset.shortcut().toString(), "Esc")

    def test_camera_mode_group_is_exclusive_and_defaults_to_orbit(self):
        group = self.model.group("camera.mode")
        self.assertEqual(len(group.actions()), 3)
        self.assertTrue(group.isExclusive())
        checked = group.checkedAction()
        self.assertIsNotNone(checked)
        self.assertEqual(checked.objectName(), "camera.mode_orbit")

    def test_reset_attaches_to_each_new_viewer(self):
        reset = self.model.action("camera.reset")
        before = len(reset.associatedObjects())
        viewer = self.mgr.add3DWidget()
        self.assertIsNotNone(viewer)
        after = reset.associatedObjects()
        # The new viewer joins the reset action so its widget-context shortcut
        # can fire; without this association the Escape binding stays inert.
        self.assertEqual(len(after), before + 1)
        self.assertIsInstance(after[-1], QtWidgets.QWidget)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
