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

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class DrawToolTC(unittest.TestCase):
    def setUp(self):
        self.mgr = _gui.controller.build()
        self.model = self.mgr.menu_model
        self.painter = _gui.controller.painter

    def _panel(self):
        """The Painter dock body, building the dock on first use."""
        self.painter._ensure_dock()
        return self.painter.panel

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
        panel = self._panel()
        self.model.action("draw.tool.line").trigger()
        self.assertEqual(self.mgr.drawTool, "line")
        # The toolbox button is a view of the same action.
        self.assertTrue(
            panel.tool_buttons["line"].defaultAction().isChecked())
        self.assertFalse(
            panel.tool_buttons["pan"].defaultAction().isChecked())

    def test_console_set_draw_tool_checks_the_menu(self):
        self.mgr.setDrawTool("rectangle")
        group = self.model.group("draw.tool")
        self.assertEqual(group.checkedAction().objectName(),
                         "draw.tool.rectangle")
        # The group is exclusive, so the previous choice clears.
        self.assertFalse(self.model.action("draw.tool.line").isChecked())

    def test_dock_follows_the_panel_toggle(self):
        action = self.model.action("panel.painter")
        # The controller is a shared singleton, so drive the toggle from a
        # known-off state rather than from whatever a previous test left.
        action.setChecked(False)
        action.setChecked(True)
        self.assertIsNotNone(self.painter._dock)
        # The main window is not shown in the test, so ask whether the dock
        # is hidden rather than whether it is on screen.
        self.assertFalse(self.painter._dock.isHidden())
        action.setChecked(False)
        self.assertTrue(self.painter._dock.isHidden())

    def test_selector_holds_one_button_per_registered_tool(self):
        panel = self._panel()
        self.assertEqual(set(panel.tool_buttons), set(draw_tool_names()))

    def test_inspector_stacks_one_page_per_tab(self):
        panel = self._panel()
        self.assertEqual(list(panel.tabs), ["Design", "Layers", "Canvas"])
        self.assertEqual(panel._stack.count(), len(panel.PAGES))
        self.assertEqual(panel.current_page(), "Design")

    def test_inspector_tab_selects_its_page(self):
        panel = self._panel()
        panel.tabs["Canvas"].click()
        self.assertEqual(panel.current_page(), "Canvas")
        # The tabs are one exclusive group, so the previous choice clears.
        self.assertFalse(panel.tabs["Design"].isChecked())
        panel.show_page("Layers")
        self.assertEqual(panel.current_page(), "Layers")
        self.assertTrue(panel.tabs["Layers"].isChecked())
        # The panel outlives the test, so leave it on the default page.
        panel.show_page("Design")

    def test_selector_shade_follows_the_theme(self):
        # The selector paints its own shade of the panel colour. Reading it
        # back from a grab is what catches a shade captured once and never
        # refreshed, which is how a set-on-palette-change colour behaves.
        tools = self._panel()._tools
        try:
            self.mgr.set_theme("light")
            light = tools.grab().toImage().pixelColor(2, 2).lightness()
            self.mgr.set_theme("dark")
            dark = tools.grab().toImage().pixelColor(2, 2).lightness()
        finally:
            self.mgr.set_theme("system")
        self.assertGreater(light, dark)

    def test_widening_the_panel_grows_the_inspector(self):
        # The selector keeps its designed width and the inspector takes the
        # rest, so a dock wider than the design leaves no bands beside it.
        panel = self._panel()
        panel.resize(500, 400)
        panel.layout().activate()
        self.assertEqual(panel._tools.width(), 64)
        self.assertGreater(panel._inspector.width(), 264)

    def test_tab_row_restyles_with_the_theme(self):
        tabs = self._panel()._tabs
        try:
            self.mgr.set_theme("light")
            light = tabs.styleSheet()
            self.mgr.set_theme("dark")
            dark = tabs.styleSheet()
        finally:
            self.mgr.set_theme("system")
        self.assertIn("border-radius", light)
        self.assertNotEqual(light, dark)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
