# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon import agent
    from solvcon.pilot import _agent_gui
    from PySide6.QtCore import Qt
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


class _CircleBackend:
    """Test backend that emits one real Agent Draw command without a CLI."""

    name = "circle (test)"

    def available(self):
        return True

    def send(self, prompt, scene_context, tool_surface):
        return agent.BackendResponse(text="circle added", commands=[
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0}])


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class AgentPanelTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def _panel_on(self):
        feature = _agent_gui.AgentPanel(mgr=self.mgr)
        feature.populate_menu()
        feature._action.setChecked(True)
        return feature

    def _select_echo(self, widget):
        combo = widget._backend_combo
        for i in range(combo.count()):
            if combo.itemText(i).startswith("echo"):
                combo.setCurrentIndex(i)
                return
        self.fail("echo backend is not in the selector")

    def test_toggle_is_placed_under_view_panels(self):
        feature = _agent_gui.AgentPanel(mgr=self.mgr)
        feature.populate_menu()
        panels = self.mgr.menu_model.menu("View/Panels")
        self.assertIn(feature._action, panels.actions())

    def test_appears_by_default_titled_agent(self):
        feature = _agent_gui.AgentPanel(mgr=self.mgr)
        feature.populate_menu()
        self.assertTrue(feature._action.isChecked())
        self.assertIsNotNone(feature._dock)
        self.assertEqual(feature._dock.windowTitle(), "Agent")

    def test_dock_sits_in_the_bottom_area(self):
        # The console owns the bottom-left; the agent takes the bottom-right.
        feature = self._panel_on()
        area = self.mgr.mainWindow.dockWidgetArea(feature._dock)
        self.assertEqual(area, Qt.BottomDockWidgetArea)

    def test_single_turn_echo_round_trip(self):
        feature = self._panel_on()
        widget = feature._panel
        self._select_echo(widget)
        widget._input.setText("draw a circle")
        widget._emit()
        text = widget._transcript.toPlainText()
        self.assertIn("You: draw a circle", text)
        self.assertIn("Agent: echo: draw a circle", text)
        # The prompt box is cleared and re-enabled for the next turn.
        self.assertEqual(widget._input.text(), "")
        self.assertTrue(widget._input.isEnabled())

    def test_drives_the_active_canvas_world(self):
        # A real draw command pins active-world binding and command dispatch,
        # which the echo round-trip does not exercise.
        feature = self._panel_on()
        widget = self.mgr.add2DWidget()
        world = solvcon.WorldFp64()
        widget.updateWorld(world)
        panel = feature._panel
        backend = _CircleBackend()
        panel._backend_combo.addItem(backend.name, backend)
        panel._backend_combo.setCurrentIndex(panel._backend_combo.count() - 1)
        panel._input.setText("draw a circle")
        panel._emit()
        self.assertIs(feature._session.world, world)
        self.assertEqual(world.nshape, 1)
        text = panel._transcript.toPlainText()
        self.assertIn("You: draw a circle", text)
        self.assertIn("circle added", text)
        self.assertIn("add_circle: ok", text)

    def test_blank_prompt_is_ignored(self):
        feature = self._panel_on()
        widget = feature._panel
        widget._input.setText("   ")
        widget._emit()
        self.assertEqual(widget._transcript.toPlainText(), "")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
