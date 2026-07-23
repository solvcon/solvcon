# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The Agent Window and Agent View real-target suite: the command families
driving live pilot windows and the active canvas view through the pilot
RManager, its QMdiArea, and the R2DWidget. Skipped where no GUI is available;
run with ``make run_pilot_pytest HEADLESS=1``."""

import os
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot, agent
    from solvcon.pilot.agent import _agent_gui
    from solvcon.pilot.agent._agent_control import (
        PilotWindowManager, LiveViewExecutor, build_control_dispatcher,
        pilot_scene_context)
    from solvcon.agent import window
    from PySide6 import QtWidgets
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class _PilotTC(unittest.TestCase):
    """Shared setup: a shown main window with an empty MDI area, so every
    freshly added sub-window reports visible and earlier tests' windows do
    not linger in the list."""

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.area = self.mgr.mdiArea
        self.mgr.show()
        self.area.closeAllSubWindows()
        QtWidgets.QApplication.processEvents()

    def tearDown(self):
        self.area.closeAllSubWindows()
        QtWidgets.QApplication.processEvents()


class PilotWindowManagerTC(_PilotTC):
    """The window command family runs against the live RManager/QMdiArea."""

    def setUp(self):
        super().setUp()
        self.wm = PilotWindowManager(self.mgr)
        self.ex = window.Executor(self.wm)

    def test_new_canvas_opens_a_real_canvas_and_lists_it(self):
        res = self.ex.run({"op": "new_canvas"})
        self.assertTrue(res.ok, res.error)
        window_id = res.value["window_id"]
        windows = self.ex.run({"op": "list_windows"}).value["windows"]
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["id"], window_id)
        self.assertTrue(windows[0]["active"])
        self.assertTrue(windows[0]["title"].endswith("2D canvas"))
        self.assertIsNotNone(self.mgr.currentR2DWidget())

    def test_list_reflects_the_newest_active_canvas(self):
        first = self.ex.run({"op": "new_canvas"}).value["window_id"]
        second = self.ex.run({"op": "new_canvas"}).value["window_id"]
        windows = self.ex.run({"op": "list_windows"}).value["windows"]
        self.assertEqual({w["id"] for w in windows}, {first, second})
        active = [w["id"] for w in windows if w["active"]]
        self.assertEqual(active, [second])

    def test_activate_window_switches_the_active_one(self):
        first = self.ex.run({"op": "new_canvas"}).value["window_id"]
        self.ex.run({"op": "new_canvas"})
        self.assertTrue(
            self.ex.run({"op": "activate_window", "window_id": first}).ok)
        windows = self.ex.run({"op": "list_windows"}).value["windows"]
        self.assertEqual([w["id"] for w in windows if w["active"]], [first])

    def test_close_window_removes_it(self):
        first = self.ex.run({"op": "new_canvas"}).value["window_id"]
        second = self.ex.run({"op": "new_canvas"}).value["window_id"]
        self.assertTrue(
            self.ex.run({"op": "close_window", "window_id": first}).ok)
        QtWidgets.QApplication.processEvents()
        ids = [w["id"]
               for w in self.ex.run({"op": "list_windows"}).value["windows"]]
        self.assertEqual(ids, [second])

    def test_save_image_writes_a_real_file(self):
        window_id = self.ex.run({"op": "new_canvas"}).value["window_id"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "canvas.png")
            res = self.ex.run(
                {"op": "save_image", "window_id": window_id, "path": path})
            self.assertTrue(res.ok, res.error)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_by_id_commands_fail_cleanly_on_a_missing_window(self):
        self.ex.run({"op": "new_canvas"})
        for op in ("activate_window", "close_window"):
            with self.subTest(op=op):
                res = self.ex.run({"op": op, "window_id": 999})
                self.assertFalse(res.ok)
                self.assertIn("999", res.error)
        res = self.ex.run(
            {"op": "save_image", "window_id": 999, "path": "x.png"})
        self.assertFalse(res.ok)


class LiveViewExecutorTC(_PilotTC):
    """The view command family steers the active canvas's live transform."""

    def setUp(self):
        super().setUp()
        self.ex = LiveViewExecutor(self.mgr)

    def test_set_view_writes_through_to_the_canvas(self):
        self.mgr.add2DWidget()
        res = self.ex.run(
            {"op": "set_view", "pan_x": 10.0, "pan_y": 20.0, "zoom": 4.0})
        self.assertTrue(res.ok, res.error)
        transform = self.mgr.currentR2DWidget().viewTransform
        self.assertEqual(transform.pan_x, 10.0)
        self.assertEqual(transform.pan_y, 20.0)
        self.assertEqual(transform.zoom, 4.0)
        self.assertEqual(self.ex.run({"op": "get_view"}).value["view"],
                         {"pan_x": 10.0, "pan_y": 20.0, "zoom": 4.0})

    def test_pan_then_reset_returns_to_identity(self):
        self.mgr.add2DWidget()
        self.ex.run({"op": "set_view", "pan_x": 0, "pan_y": 0, "zoom": 1})
        self.assertTrue(self.ex.run(
            {"op": "pan", "dx_screen": 100.0, "dy_screen": 50.0}).ok)
        transform = self.mgr.currentR2DWidget().viewTransform
        self.assertEqual((transform.pan_x, transform.pan_y), (100.0, 50.0))
        self.assertTrue(self.ex.run({"op": "reset_view"}).ok)
        self.assertEqual(self.ex.run({"op": "get_view"}).value["view"],
                         {"pan_x": 0.0, "pan_y": 0.0, "zoom": 1.0})

    def test_commands_fail_cleanly_without_an_active_canvas(self):
        res = self.ex.run({"op": "get_view"})
        self.assertFalse(res.ok)
        self.assertIn("no active 2D canvas", res.error)


class DispatcherIntegrationTC(_PilotTC):
    """One dispatcher lets a session drive windows, the view, and the world."""

    def test_dispatcher_drives_windows_view_and_world(self):
        dispatcher = build_control_dispatcher(self.mgr)
        self.assertTrue(dispatcher.run({"op": "new_canvas"}).ok)
        self.assertTrue(dispatcher.run(
            {"op": "set_view", "pan_x": 5.0, "pan_y": 6.0, "zoom": 2.0}).ok)
        self.assertTrue(dispatcher.run(
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0}).ok)
        widget = self.mgr.currentR2DWidget()
        self.assertEqual(widget.viewTransform.zoom, 2.0)
        self.assertEqual(widget.world.nshape, 1)

    def test_open_then_draw_lands_on_the_new_canvas(self):
        # The batch opens a canvas and draws on it in one go: the draw target
        # must follow the window the batch just opened, not the world (if any)
        # that was active when the batch began.
        dispatcher = build_control_dispatcher(self.mgr)
        results = dispatcher.run_script([
            {"op": "new_canvas"},
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0}])
        self.assertTrue(all(r.ok for r in results), [r.error for r in results])
        self.assertEqual(self.mgr.currentR2DWidget().world.nshape, 1)

    def test_draw_fails_cleanly_without_a_canvas(self):
        dispatcher = build_control_dispatcher(self.mgr)
        res = dispatcher.run(
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0})
        self.assertFalse(res.ok)
        self.assertIn("no active 2D canvas", res.error)

    def test_scene_context_lists_windows_and_the_active_view(self):
        dispatcher = build_control_dispatcher(self.mgr)
        dispatcher.run({"op": "new_canvas"})
        scene = pilot_scene_context(dispatcher, "world with 0 shapes")
        self.assertIn("world with 0 shapes", scene)
        self.assertIn("2D canvas", scene)
        self.assertIn("(active)", scene)
        self.assertIn("view:", scene)

    def test_scene_context_notes_when_no_window_is_open(self):
        dispatcher = build_control_dispatcher(self.mgr)
        scene = pilot_scene_context(dispatcher, "base")
        self.assertIn("windows: none open", scene)

    def test_session_tool_surface_unions_the_families(self):
        dispatcher = build_control_dispatcher(self.mgr)
        session = agent.AgentSession(runner=dispatcher)
        names = {tool["name"] for tool in session.tool_surface()}
        self.assertLessEqual(
            {"add_circle", "new_canvas", "list_windows", "get_view",
             "set_view"}, names)
        # Delete ops stay hidden by default, across every family.
        self.assertNotIn("close_window", names)
        self.assertNotIn("clear", names)
        self.assertNotIn("remove_shape", names)

    def test_opt_in_exposes_the_delete_ops_of_every_family(self):
        dispatcher = build_control_dispatcher(self.mgr)
        session = agent.AgentSession(runner=dispatcher, allow_destructive=True)
        names = {tool["name"] for tool in session.tool_surface()}
        self.assertLessEqual(
            {"close_window", "clear", "remove_shape"}, names)


class AgentPanelControlTC(_PilotTC):
    """The console panel wires the composite dispatcher into its session."""

    def test_panel_session_exposes_window_and_view_tools(self):
        feature = _agent_gui.AgentPanel(mgr=self.mgr)
        names = {tool["name"] for tool in feature._session.tool_surface()}
        self.assertLessEqual(
            {"add_circle", "new_canvas", "list_windows", "get_view",
             "set_view"}, names)
        self.assertNotIn("close_window", names)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
