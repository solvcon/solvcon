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


class _PingBackend:
    """Test backend that always emits the dummy ``ping`` command, so a GUI
    test can drive a real command through the session without a live CLI."""

    name = "ping (test)"

    def available(self):
        return True

    def send(self, prompt, scene_context, tool_surface):
        return agent.BackendResponse(text="pong", commands=[{"op": "ping"}])


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
        # A real canvas world reaches the session and its command is executed
        # and rendered. This pins the canvas-driving path (world binding plus
        # command dispatch) that the echo round-trip cannot.
        feature = self._panel_on()
        widget = self.mgr.add2DWidget()
        widget.updateWorld(solvcon.WorldFp64())
        panel = feature._panel
        panel._backend_combo.addItem(_PingBackend().name, _PingBackend())
        panel._backend_combo.setCurrentIndex(panel._backend_combo.count() - 1)
        panel._input.setText("do a ping")
        panel._emit()
        # The session bound the active canvas world, not None.
        self.assertIsNotNone(feature._session.world)
        text = panel._transcript.toPlainText()
        self.assertIn("You: do a ping", text)
        self.assertIn("pong", text)
        self.assertIn("ping: ok", text)

    def test_blank_prompt_is_ignored(self):
        feature = self._panel_on()
        widget = feature._panel
        widget._input.setText("   ")
        widget._emit()
        self.assertEqual(widget._transcript.toPlainText(), "")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
