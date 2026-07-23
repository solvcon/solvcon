# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon import agent
    from solvcon.pilot.agent import _agent_gui
    from PySide6.QtCore import Qt, QCoreApplication
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


class _TranslateBackend:
    """Test backend that asks to translate a fixed shape id, so a GUI test can
    drive a by-id command whose target may have been removed meanwhile.  It
    uses an update op, not a delete, so the session's destructive gating does
    not intercept it before the shape-liveness check."""

    name = "translate (test)"

    def __init__(self, shape_id):
        self._shape_id = shape_id

    def available(self):
        return True

    def send(self, prompt, scene_context, tool_surface):
        return agent.BackendResponse(
            text="translating", commands=[
                {"op": "translate_shape", "shape_id": self._shape_id,
                 "dx": 1.0, "dy": 0.0}])


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

    def _select_backend(self, panel, backend):
        panel._backend_combo.addItem(backend.name, backend)
        panel._backend_combo.setCurrentIndex(panel._backend_combo.count() - 1)

    def _finish_turn(self, feature):
        """Drive the pending async turn to completion: wait for the backend
        worker, then pump the event loop so its queued reply reaches the main
        thread and finishes the turn."""
        worker = feature._worker
        if worker is not None:
            self.assertTrue(worker.wait(5000))
        QCoreApplication.processEvents()

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
        self._finish_turn(feature)
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
        self._select_backend(panel, _CircleBackend())
        panel._input.setText("draw a circle")
        panel._emit()
        self.assertIs(feature._session.world, world)
        self._finish_turn(feature)
        self.assertEqual(world.nshape, 1)
        text = panel._transcript.toPlainText()
        self.assertIn("You: draw a circle", text)
        self.assertIn("circle added", text)
        self.assertIn("add_circle: ok", text)

    def test_turn_runs_off_the_main_thread(self):
        feature = self._panel_on()
        panel = feature._panel
        self._select_echo(panel)
        panel._input.setText("draw a circle")
        panel._emit()
        # A worker is live and the prompt is locked, but no reply has landed:
        # the main thread was never blocked on the backend call.
        self.assertIsNotNone(feature._worker)
        self.assertFalse(panel._input.isEnabled())
        self.assertNotIn("Agent:", panel._transcript.toPlainText())
        self._finish_turn(feature)
        self.assertIsNone(feature._worker)
        self.assertTrue(panel._input.isEnabled())
        self.assertIn("Agent: echo: draw a circle",
                      panel._transcript.toPlainText())

    def test_working_indicator_shows_while_a_turn_runs(self):
        # Assert on the animation timer and text rather than isVisible(), which
        # is false for an unshown dock in a headless test.
        feature = self._panel_on()
        panel = feature._panel
        self._select_echo(panel)
        panel._input.setText("draw a circle")
        panel._emit()
        self.assertTrue(panel._working_timer.isActive())
        self.assertIn("working", panel._status.text())
        self._finish_turn(feature)
        self.assertFalse(panel._working_timer.isActive())
        self.assertEqual(panel._status.text(), "")

    def test_second_submit_is_dropped_while_a_turn_runs(self):
        feature = self._panel_on()
        panel = feature._panel
        self._select_echo(panel)
        panel._input.setText("first")
        panel._emit()
        running = feature._worker
        panel.submitted.emit("second")
        self.assertIs(feature._worker, running)
        self._finish_turn(feature)
        text = panel._transcript.toPlainText()
        self.assertIn("You: first", text)
        self.assertNotIn("You: second", text)

    def test_shutdown_joins_the_running_worker(self):
        # The teardown path waits for an in-flight worker so its QThread is
        # never destroyed while still running (which would abort the process).
        feature = self._panel_on()
        panel = feature._panel
        self._select_echo(panel)
        panel._input.setText("draw a circle")
        panel._emit()
        worker = feature._worker
        self.assertIsNotNone(worker)
        feature._join_worker()
        self.assertTrue(worker.isFinished())
        QCoreApplication.processEvents()

    def test_stale_by_id_command_fails_cleanly(self):
        # The race workaround: a command that names a shape the user removed
        # while the model was thinking fails as a not-live shape rather than
        # crashing the turn.  The empty world stands in for that removal.
        feature = self._panel_on()
        widget = self.mgr.add2DWidget()
        world = solvcon.WorldFp64()
        widget.updateWorld(world)
        panel = feature._panel
        self._select_backend(panel, _TranslateBackend(4242))
        panel._input.setText("move shape 4242 right")
        panel._emit()
        self._finish_turn(feature)
        text = panel._transcript.toPlainText()
        self.assertIn("translate_shape", text)
        self.assertNotIn("ok", text.split("translate_shape", 1)[1])
        self.assertTrue(panel._input.isEnabled())

    def test_blank_prompt_is_ignored(self):
        feature = self._panel_on()
        widget = feature._panel
        widget._input.setText("   ")
        widget._emit()
        self.assertEqual(widget._transcript.toPlainText(), "")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
