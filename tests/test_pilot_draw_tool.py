# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot._pilot_core import draw_tool_names
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class DrawToolTC(unittest.TestCase):
    def setUp(self):
        self.mgr = _gui.controller.build()
        self.model = self.mgr.menu_model
        self.painter = _gui.controller.painter

    def test_one_radio_item_per_registered_tool(self):
        items = [a.text()
                 for a in self.model.menu("Canvas/Draw tool").actions()]
        self.assertEqual(len(items), len(draw_tool_names()))

    def test_samples_moved_under_their_own_submenu(self):
        samples = [a.text()
                   for a in self.model.menu("Canvas/Samples").actions()]
        self.assertEqual(len(samples), 7)
        # The samples no longer sit at the Canvas top level.
        top = [a.text() for a in self.model.menu("Canvas").actions()
               if not a.isSeparator()]
        self.assertNotIn("Sample: Parabola", top)

    def test_menu_radio_drives_manager_and_toolbox(self):
        self.painter._ensure_dock()
        self.model.action("draw.tool.line").trigger()
        self.assertEqual(self.mgr.drawTool, "line")
        # The toolbox button is a view of the same action.
        self.assertTrue(
            self.painter._buttons["line"].defaultAction().isChecked())
        self.assertFalse(
            self.painter._buttons["pan"].defaultAction().isChecked())

    def test_console_set_draw_tool_checks_the_menu(self):
        self.mgr.setDrawTool("rectangle")
        group = self.model.group("draw.tool")
        self.assertEqual(group.checkedAction().objectName(),
                         "draw.tool.rectangle")
        # The group is exclusive, so the previous choice clears.
        self.assertFalse(self.model.action("draw.tool.line").isChecked())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
