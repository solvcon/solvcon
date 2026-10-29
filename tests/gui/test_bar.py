# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

from solvcon import testing

try:
    from PySide6 import QtCore
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    # Import the apps as the entry point does, so their menu entries are in
    # the bar whichever test builds the controller first.
    from solvcon import apps  # noqa: F401
except ImportError:
    pilot = None

BUILTINS = ["File", "Edit", "View", "App", "Mesh", "Canvas", "Profiling",
            "Track", "Window"]


@testing.requires_live_window
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

        # Panels sits first in View and holds the dock toggles. Other tests
        # build their own panel features on the shared singleton, so a toggle
        # can repeat; assert the distinct entries in their declared order.
        view = [a.text() for a in model.menu("View").actions()]
        self.assertEqual(view[0], "Panels")
        panels = [a.text() for a in model.menu("View/Panels").actions()]
        self.assertEqual(
            list(dict.fromkeys(panels)),
            ["Inspector", "Oblique shock reflection", "App config",
             "Painter", "MCAP", "Console", "Terminal", "Agent Console"])

        # The Console and Terminal toggles live with the other panel toggles;
        # the Window menu holds only the dynamic sub-window list.
        self.assertIsNotNone(model.action("window.console"))
        self.assertIsNotNone(model.action("window.terminal"))
        window_items = [a.text() for a in model.menu("Window").actions()]
        self.assertNotIn("Console", window_items)
        self.assertNotIn("Terminal", window_items)

    def test_benchmark_menu(self):
        mgr = _gui.controller.build()
        action = mgr.menu_model.action('profiling.benchmark')
        self.assertIn(action, mgr.menu_model.menu('Profiling').actions())
        action.trigger()
        inspector = _gui.controller.benchmark
        try:
            self.assertIn(inspector, mgr.mdiArea.subWindowList())
            inspector.close()
            action.trigger()
            self.assertIs(_gui.controller.benchmark, inspector)
            self.assertFalse(inspector.isHidden())
            self.assertFalse(inspector.widget().isHidden())
            inspector.showMinimized()
            action.trigger()
            self.assertFalse(inspector.isMinimized())
        finally:
            inspector.close()
            mgr.mdiArea.removeSubWindow(inspector)
            inspector.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(
                inspector, QtCore.QEvent.Type.DeferredDelete)
            _gui.controller.benchmark = None

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
