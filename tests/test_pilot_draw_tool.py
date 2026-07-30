# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon
from pilot_ci import SKIP_PILOT_WIDGETS

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot._pilot_core import draw_tool_names
    from solvcon.pilot.painter import _icons
    from PySide6 import QtGui, QtWidgets
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


def _drawn_pixels(image):
    """The (x, y) of every pixel an icon actually painted."""
    return [(x, y)
            for x in range(image.width())
            for y in range(image.height())
            if image.pixelColor(x, y).alpha() > 0]


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PainterIconTC(unittest.TestCase):
    """The tool icons rasterize to the box they are asked for.

    An unscaled test screen reports ratio 1, which hides the viewport trap
    ``render`` guards against, so the ratio is driven explicitly here rather
    than taken from wherever the suite happens to run.
    """

    _SIZE = 17

    def setUp(self):
        # No window needed, only a live QGuiApplication to hold a QPixmap.
        pilot.RManager.instance.setUp()

    def _ink(self, name, ratio):
        """The bounding box of an icon's drawn pixels as (x0, y0, x1, y1)."""
        image = _icons.render(
            name, QtGui.QColor("black"), self._SIZE, ratio).toImage()
        marks = _drawn_pixels(image)
        return (min(x for x, _y in marks), min(y for _x, y in marks),
                max(x for x, _y in marks), max(y for _x, y in marks))

    def test_icon_scales_with_the_device_pixel_ratio(self):
        for name in _icons.ICONS:
            with self.subTest(name=name):
                plain = self._ink(name, 1.0)
                scaled = self._ink(name, 2.0)
                # An icon drawn too large loses its far side, so the edge
                # checks below are the half that catches a bad viewport.
                for flat, deep in zip(plain, scaled):
                    self.assertAlmostEqual(deep, 2 * flat, delta=2)
                self.assertLess(scaled[2], 2 * self._SIZE - 1)
                self.assertLess(scaled[3], 2 * self._SIZE - 1)


@unittest.skipIf(NO_LIVE_WINDOW or SKIP_PILOT_WIDGETS or not solvcon.HAS_PILOT,
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
            panel.tool_buttons["select"].defaultAction().isChecked())

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

    def test_selector_entries_carry_a_tinted_icon_and_the_short_name(self):
        panel = self._panel()
        selector = panel._tools
        for tool, button in panel.tool_buttons.items():
            with self.subTest(tool=tool):
                self.assertFalse(button.icon().isNull())
                self.assertEqual(button.iconSize().width(), selector._ICON_PX)
                self.assertEqual(button.width(), selector._ENTRY_WIDTH)
        # Select keeps its menu label because it fits; Triangle does not, and
        # that is what the short name is for.
        self.assertEqual(panel.tool_buttons["select"].text(), "Select")
        self.assertEqual(
            self.model.action("draw.tool.select").text(), "Select")
        self.assertEqual(panel.tool_buttons["triangle"].text(), "Tri")
        self.assertEqual(
            self.model.action("draw.tool.triangle").text(), "Triangle")

    def test_selector_entry_survives_a_tool_switch(self):
        # Pins the icon and short name that _SelectorEntry restores after Qt
        # re-reads them from the action.
        panel = self._panel()
        button = panel.tool_buttons["rectangle"]
        self.model.action("draw.tool.rectangle").trigger()
        QtWidgets.QApplication.processEvents()
        self.assertEqual(button.text(), "Rect")
        self.assertFalse(button.icon().isNull())

    def test_selector_tip_names_the_tool_and_its_key(self):
        panel = self._panel()
        for tool, key in (("line", "L"), ("circle", "C")):
            with self.subTest(tool=tool):
                action = panel.tool_buttons[tool].defaultAction()
                self.assertEqual(
                    action.shortcut().toString(QtGui.QKeySequence.NativeText),
                    key)
                self.assertEqual(action.toolTip(), f"{action.text()}  {key}")

    def test_tool_placeholders_are_visible_but_greyed_out(self):
        # The Text entry and the Grid toggle hold their designed places so the
        # column matches the design; their contents wait on the model.
        panel = self._panel()
        self.assertEqual(set(panel.tool_placeholders), {"text", "grid"})
        for name, button in panel.tool_placeholders.items():
            with self.subTest(name=name):
                self.assertFalse(button.isEnabled())
                self.assertFalse(button.icon().isNull())
                self.assertIn("needs", button.toolTip())
        self.assertEqual(panel.tool_placeholders["text"].text(), "Text")
        self.assertEqual(panel.tool_placeholders["grid"].text(), "Grid")

    def test_grid_toggle_is_pinned_below_the_tools(self):
        # The design parks the Grid toggle at the column bottom, away from the
        # tools, so it sits after the layout's stretch rather than under Text.
        panel = self._panel()
        layout = panel._tools.layout()
        items = [layout.itemAt(i) for i in range(layout.count())]
        self.assertIs(items[-1].widget(), panel.tool_placeholders["grid"])
        self.assertIsNone(items[-2].widget())

    def test_selector_icons_retint_with_the_theme(self):
        # A colour captured once would survive a theme switch as is.
        panel = self._panel()
        button = panel.tool_buttons["circle"]
        size = panel._tools._ICON_PX

        def stroke():
            image = button.icon().pixmap(size, size).toImage()
            return max(image.pixelColor(x, y).lightness()
                       for x, y in _drawn_pixels(image))

        try:
            self.mgr.set_theme("light")
            light = stroke()
            self.mgr.set_theme("dark")
            dark = stroke()
        finally:
            self.mgr.set_theme("system")
        self.assertGreater(dark, light)

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
