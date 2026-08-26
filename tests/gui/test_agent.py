# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon import agent
    from solvcon.pilot.agent import _agent_gui
    from solvcon.pilot.agent import _agent_settings
    from PySide6.QtCore import Qt, QCoreApplication
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


class _CircleBackend:
    """Test backend that emits one real Agent Draw command without a CLI."""

    name = "circle (test)"

    def available(self):
        return True

    def send(self, prompt, scene_context, tool_surface, history=()):
        return agent.BackendResponse(text="circle added", commands=[
            {"op": "log", "message": "Add a unit circle at the origin"},
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

    def send(self, prompt, scene_context, tool_surface, history=()):
        return agent.BackendResponse(
            text="translating", commands=[
                {"op": "translate_shape", "shape_id": self._shape_id,
                 "dx": 1.0, "dy": 0.0}])


@unittest.skipIf(not solvcon.HAS_PILOT, "pilot is not built")
class AgentTurnFormatTC(unittest.TestCase):
    def test_log_messages_are_the_user_facing_reply(self):
        turn = agent.TranscriptTurn(
            role="agent", text="ignored prose",
            commands=[{"op": "log", "message": "Shift car right by 25"},
                      {"op": "translate_shape", "shape_id": 0,
                       "dx": 25, "dy": 0}],
            results=[agent.CommandResult("log", True),
                     agent.CommandResult("translate_shape", True)])
        self.assertEqual(
            _agent_gui.AgentPanel._format_turn(turn),
            "Shift car right by 25")

    def test_a_failed_command_is_reported_under_the_reply(self):
        turn = agent.TranscriptTurn(
            role="agent", text="moving the car",
            commands=[{"op": "translate_shape", "shape_id": 0,
                       "dx": 25, "dy": 0}, {"op": "add_circle"}],
            results=[agent.CommandResult(
                "translate_shape", False, error="shape 0 is not live")])
        self.assertEqual(
            _agent_gui.AgentPanel._format_turn(turn).splitlines(),
            ["moving the car",
             "  - translate_shape: shape 0 is not live",
             "  - add_circle: not run"])

    def test_a_log_that_failed_is_not_announced_as_the_reply(self):
        # The message describes work the turn went on to plan; with the batch
        # dead it would announce something that never happened.
        turn = agent.TranscriptTurn(
            role="agent", text="drawing a car",
            commands=[{"op": "log", "message": "body, roof, wheels"},
                      {"op": "add_circle"}],
            results=[agent.CommandResult("log", False, error="no canvas"),
                     agent.CommandResult("add_circle", False,
                                         error="no canvas")])
        self.assertEqual(
            _agent_gui.AgentPanel._format_turn(turn).splitlines()[0],
            "drawing a car")

    def test_successful_commands_stay_out_of_the_reply(self):
        turn = agent.TranscriptTurn(
            role="agent", text="drew it",
            commands=[{"op": "add_line", "x0": 0, "y0": 0, "x1": 1, "y1": 1}],
            results=[agent.CommandResult("add_line", True, {"shape_id": 0})])
        self.assertEqual(_agent_gui.AgentPanel._turn_error_lines(turn), [])
        self.assertEqual(_agent_gui.AgentPanel._format_turn(turn), "drew it")

    def test_a_log_message_does_not_hide_a_later_failure(self):
        turn = agent.TranscriptTurn(
            role="agent", text="",
            commands=[{"op": "log", "message": "Car plan"},
                      {"op": "add_line"}],
            results=[agent.CommandResult("log", True),
                     agent.CommandResult("add_line", False,
                                         error="x0 is required")])
        self.assertEqual(
            _agent_gui.AgentPanel._format_turn(turn).splitlines(),
            ["Car plan", "  - add_line: x0 is required"])

    def test_backend_prose_fills_in_when_there_is_no_log(self):
        turn = agent.TranscriptTurn(role="agent", text="echo: hi")
        self.assertEqual(_agent_gui.AgentPanel._format_turn(turn), "echo: hi")


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class AgentPanelTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def _panel_on(self):
        feature = _agent_gui.AgentPanel(mgr=self.mgr)
        feature.populate_menu()
        feature._action.setChecked(True)
        return feature

    def _select_echo(self, widget):
        """Add the offline echo double and pick it: the selector itself lists
        only real backends, none of which a test may run."""
        self._select_backend(widget, agent.EchoBackend())

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

    def test_hidden_by_default(self):
        # Pilot opens with no 2D canvas, so the console has nothing to act on
        # and stays out of the way until a canvas opens.
        feature = _agent_gui.AgentPanel(mgr=self.mgr)
        feature.populate_menu()
        self.assertFalse(feature._action.isChecked())
        self.assertIsNone(feature._dock)

    def test_present_opens_the_dock_titled_agent(self):
        feature = _agent_gui.AgentPanel(mgr=self.mgr)
        feature.populate_menu()
        feature.present()
        self.assertTrue(feature._action.isChecked())
        self.assertEqual(feature._dock.windowTitle(), "Agent")
        self.assertFalse(feature._dock.isHidden())

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
        self.assertIn("Add a unit circle at the origin", text)
        self.assertNotIn("add_circle", text)

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
        self.assertIn("- translate_shape:", text)
        self.assertIn("4242", text)
        self.assertTrue(panel._input.isEnabled())

    def test_blank_prompt_is_ignored(self):
        feature = self._panel_on()
        widget = feature._panel
        widget._input.setText("   ")
        widget._emit()
        self.assertEqual(widget._transcript.toPlainText(), "")

    def test_backend_settings_reach_the_cli(self):
        # The whole path in one go: the dialog is built from the backend's
        # spec, accepting it stores the picks, and a later turn puts them on
        # the CLI command line.
        backend = agent.ClaudeCliBackend()
        dialog = _agent_settings.AgentBackendSettingsDialog(backend)
        dialog._editors["model"].setCurrentText("opus")
        dialog._editors["effort"].setCurrentText("high")
        dialog.accept()
        argv = backend._build_argv("/usr/bin/claude", "draw", "system")
        self.assertIn("--model=opus", argv)
        self.assertIn("--effort=high", argv)

    def test_codex_backend_settings_reach_the_cli(self):
        backend = agent.CodexCliBackend()
        dialog = _agent_settings.AgentBackendSettingsDialog(backend)
        dialog._editors["model"].setCurrentText("gpt-5.6-sol")
        dialog._editors["effort"].setCurrentText("high")
        dialog.accept()
        argv = backend._build_argv("/usr/bin/codex", "draw", "system")
        self.assertIn("--model=gpt-5.6-sol", argv)
        self.assertIn('--config=model_reasoning_effort="high"', argv)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
