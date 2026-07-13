# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot import _gui
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)

BUILTINS = ["File", "Edit", "View", "One", "Mesh", "Canvas", "Profiling",
            "Window"]


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class BarStructureTC(unittest.TestCase):
    def test_full_bar_is_assembled_from_the_model(self):
        mgr = _gui.controller.build()
        model = mgr.menu_model

        # The eight built-in menus keep their declared order. Other tests may
        # append scratch menus to the shared singleton, so filter to these.
        bar = [a.text() for a in mgr.mainWindow.menuBar().actions()]
        self.assertEqual([t for t in bar if t in BUILTINS], BUILTINS)

        # A known feature item lands under its intended path and id.
        self.assertIsNotNone(model.action("mesh.sample_dialog"))
        self.assertIn("Sample mesh dialog",
                      [a.text() for a in model.menu("Mesh").actions()])
        self.assertIsNotNone(model.action("file.save_2d_canvas"))
        self.assertIn("Save 2D canvas",
                      [a.text() for a in model.menu("File").actions()])

        # Panels sits first in View and holds the four dock toggles. Other
        # tests build their own panel features on the shared singleton, so a
        # toggle can repeat; assert the distinct entries in their declared
        # order.
        view = [a.text() for a in model.menu("View").actions()]
        self.assertEqual(view[0], "Panels")
        panels = [a.text() for a in model.menu("View/Panels").actions()]
        self.assertEqual(list(dict.fromkeys(panels)),
                         ["Inspector", "Painter", "Console", "Agent Console"])

        # The Console toggle lives with the other panel toggles; the Window
        # menu holds only the dynamic sub-window list.
        self.assertIsNotNone(model.action("panel.console"))
        self.assertNotIn("Console",
                         [a.text() for a in model.menu("Window").actions()])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
